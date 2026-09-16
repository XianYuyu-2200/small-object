"""Desktop console: camera capture -> image analysis -> JSON result."""

from __future__ import annotations

import json
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import yaml
from PIL import Image, ImageTk

from swallow_yolo.calibration import load_calibration
from swallow_yolo.camera_profile import resolve_camera_profile
from swallow_yolo.measurement import MeasurementError, measure_objects
from swallow_yolo.mindvision import MindVisionCamera
from swallow_yolo.vlm import analyze_frame, load_vlm_config


def project_root() -> Path:
    if getattr(sys, "frozen", False):
        executable_root = Path(sys.executable).resolve().parent
        if (executable_root / "config").exists():
            return executable_root
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[1]


def write_error_log(name: str, traceback_text: str) -> Path | None:
    try:
        path = project_root() / "runs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(traceback_text, encoding="utf-8")
        return path
    except Exception:
        return None


def crop_measurement(frame, measurement, margin_ratio: float = 0.35):
    """Crop one measured contour with enough surrounding context for naming."""

    x, y, width, height = cv2.boundingRect(measurement.contour_xy.astype("float32"))
    margin = max(36, int(max(width, height) * margin_ratio))
    frame_height, frame_width = frame.shape[:2]
    left = max(0, x - margin)
    top = max(0, y - margin)
    right = min(frame_width, x + width + margin)
    bottom = min(frame_height, y + height + margin)
    if right <= left or bottom <= top:
        raise MeasurementError("目标裁剪区域无效")
    return frame[top:bottom, left:right].copy()


class CameraWorker(threading.Thread):
    """Continuously preview the MindVision camera and retain the latest frame."""

    def __init__(self, events: queue.Queue, stop_event: threading.Event, profile_path: Path, preview_width: int):
        super().__init__(daemon=True)
        self.events = events
        self.stop_event = stop_event
        self.profile_path = profile_path
        self.preview_width = preview_width
        self.latest_lock = threading.Lock()
        self.latest_frame = None
        self.device_name = ""

    def emit(self, kind: str, value) -> None:
        try:
            if kind == "frame":
                self.events.put_nowait((kind, value))
            else:
                self.events.put((kind, value), timeout=1.0)
        except queue.Full:
            pass

    def snapshot(self):
        with self.latest_lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def run(self) -> None:
        camera = None
        try:
            profile = yaml.safe_load(self.profile_path.read_text(encoding="utf-8")) if self.profile_path.exists() else {}
            camera_values = resolve_camera_profile(profile or {}, {})
            sdk_path = Path(str(camera_values["sdk_path"])).expanduser()
            if not sdk_path.is_absolute():
                sdk_path = (project_root() / sdk_path).resolve()
            camera = MindVisionCamera(
                sdk_path,
                camera_values["camera"],
                camera_values["resolution_index"],
                camera_values["frame_speed_index"],
                camera_values["exposure_us"],
                camera_values["gain_x"],
            )
            self.device_name = camera.device_name
            self.emit("camera_ready", self.device_name)
            while not self.stop_event.is_set():
                frame = camera.read()
                if frame is None or frame.size == 0:
                    raise RuntimeError("相机返回了空画面")
                with self.latest_lock:
                    self.latest_frame = frame
                scale = min(1.0, float(self.preview_width) / max(frame.shape[:2]))
                preview = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
                self.emit("frame", preview)
        except Exception as error:
            log_path = write_error_log("gui_error.log", traceback.format_exc())
            self.emit("camera_error", (str(error), str(log_path) if log_path else ""))
        finally:
            if camera is not None:
                try:
                    camera.close()
                except Exception:
                    pass
            self.emit("camera_stopped", None)


class AnalysisWorker(threading.Thread):
    """Analyze one already-captured frame without blocking Tkinter."""

    def __init__(self, events: queue.Queue, frame, config_path: Path, image_path: Path, calibration_path: Path, background_path: Path):
        super().__init__(daemon=True)
        self.events = events
        self.frame = frame.copy()
        self.config_path = config_path
        self.image_path = image_path
        self.calibration_path = calibration_path
        self.background_path = background_path

    def emit(self, kind: str, value) -> None:
        try:
            self.events.put((kind, value), timeout=1.0)
        except queue.Full:
            pass

    def run(self) -> None:
        try:
            self.emit("analysis_status", "正在测量目标尺寸…")
            calibration = load_calibration(self.calibration_path)
            background = cv2.imread(str(self.background_path))
            if background is None:
                raise MeasurementError(f"无法读取空台背景图：{self.background_path}")
            measurements = measure_objects(self.frame, background, calibration)
            config = load_vlm_config(self.config_path)
            undistorted = calibration.undistort(self.frame)
            total = len(measurements)
            completed: dict[int, dict] = {}
            errors: list[Exception] = []

            def analyze_one(index: int, measurement_result):
                measurement = measurement_result.to_dict()
                crop = crop_measurement(undistorted, measurement_result)
                result = analyze_frame(crop, config, measurement=measurement)
                item = result.to_dict()
                item["object_id"] = index
                item["measurement"] = measurement
                return index, item

            workers = min(5, total)
            self.emit("analysis_status", f"正在并行分析 {total} 个目标…")
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="object-analysis") as executor:
                futures = {
                    executor.submit(analyze_one, index, measurement_result): index
                    for index, measurement_result in enumerate(measurements, start=1)
                }
                finished = 0
                for future in as_completed(futures):
                    try:
                        index, item = future.result()
                        completed[index] = item
                    except Exception as error:
                        errors.append(error)
                    finished += 1
                    self.emit("analysis_status", f"目标分析完成 {finished}/{total}…")

            if not completed:
                raise errors[0] if errors else RuntimeError("没有完成任何目标分析")
            objects = [completed[index] for index in sorted(completed)]

            record = {
                "objects": objects,
                "analyzed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "image_path": str(self.image_path),
            }
            output = project_root() / "runs" / "analysis"
            output.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps(record, ensure_ascii=False, indent=2)
            (output / "latest.json").write_text(encoded, encoding="utf-8")
            (output / f"{self.image_path.stem}.json").write_text(encoded, encoding="utf-8")
            self.emit("analysis_result", record)
        except Exception as error:
            log_path = write_error_log("analysis_error.log", traceback.format_exc())
            self.emit("analysis_error", (str(error), str(log_path) if log_path else ""))


class App(tk.Tk):
    BG = "#0f172a"
    PANEL = "#172033"
    PANEL_2 = "#1e293b"
    TEXT = "#f8fafc"
    MUTED = "#94a3b8"
    BLUE = "#2563eb"
    GREEN = "#16a34a"
    RED = "#dc2626"
    AMBER = "#f59e0b"
    DECISION_COLORS = {
        "无法吞咽": "#16a34a",
        "不容易吞咽": "#4d7c0f",
        "可能吞咽": "#b45309",
        "容易吞咽": "#c2410c",
        "极易吞咽": "#991b1b",
    }

    def __init__(self):
        super().__init__()
        self.title("小物件吞咽识别控制台")
        self.geometry("1500x920")
        self.minsize(1120, 820)
        self.configure(bg=self.BG)
        self.events: queue.Queue = queue.Queue(maxsize=16)
        self.stop_event = threading.Event()
        self.camera_worker: CameraWorker | None = None
        self.analysis_worker: AnalysisWorker | None = None
        self.captured_frame = None
        self.captured_path: Path | None = None
        self.preview_hold_until = 0.0
        self.photo = None
        self._build_style()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(50, self.poll_events)

    def _build_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TButton", font=("Segoe UI", 11), padding=(14, 10), background=self.PANEL_2, foreground=self.TEXT)
        style.map("TButton", background=[("active", "#334155")], foreground=[("disabled", "#64748b")])
        style.configure("Accent.TButton", background=self.BLUE, foreground="#ffffff")
        style.map("Accent.TButton", background=[("active", "#1d4ed8"), ("disabled", "#334155")])
        style.configure("Capture.TButton", background=self.GREEN, foreground="#ffffff")
        style.map("Capture.TButton", background=[("active", "#15803d"), ("disabled", "#334155")])

    def _build_ui(self) -> None:
        header = tk.Frame(self, bg=self.BG)
        header.pack(fill="x", padx=28, pady=(22, 14))
        tk.Label(header, text="小物件吞咽识别", font=("Segoe UI", 25, "bold"), bg=self.BG, fg=self.TEXT).pack(side="left")
        tk.Label(header, text="手动拍照 · 图像分析", font=("Segoe UI", 11), bg=self.BG, fg=self.MUTED).pack(side="left", padx=18, pady=(8, 0))
        self.status = tk.Label(header, text="● 相机未启动", font=("Segoe UI", 12, "bold"), bg=self.BG, fg=self.MUTED)
        self.status.pack(side="right", pady=(8, 0))

        body = tk.Frame(self, bg=self.BG)
        body.pack(fill="both", expand=True, padx=28, pady=(0, 22))
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=0, minsize=560)
        body.grid_rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=self.PANEL, highlightthickness=1, highlightbackground="#334155")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        tk.Label(left, text="实时相机画面", font=("Segoe UI", 14, "bold"), bg=self.PANEL, fg=self.TEXT, anchor="w").pack(fill="x", padx=20, pady=(18, 10))
        self.video = tk.Label(left, text="点击“开始相机”连接迈德威视相机", font=("Segoe UI", 16), bg="#020617", fg=self.MUTED)
        self.video.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        right = tk.Frame(
            body,
            width=560,
            height=720,
            bg=self.PANEL,
            highlightthickness=1,
            highlightbackground="#334155",
        )
        right.grid(row=0, column=1, sticky="n")
        right.pack_propagate(False)
        tk.Label(right, text="分析结果", font=("Segoe UI", 14, "bold"), bg=self.PANEL, fg=self.TEXT, anchor="w").pack(fill="x", padx=20, pady=(18, 4))
        tk.Label(right, text="吞咽等级判定", font=("Segoe UI", 9), bg=self.PANEL, fg=self.MUTED, anchor="w").pack(fill="x", padx=20, pady=(0, 12))

        result_box = tk.Frame(right, bg=self.PANEL_2)
        result_box.pack(fill="both", expand=True, padx=20, pady=(0, 14))
        self.decision_label = tk.Label(result_box, text="等待拍照", font=("Segoe UI", 22, "bold"), bg=self.PANEL_2, fg=self.MUTED, anchor="w")
        self.decision_label.pack(fill="x", padx=16, pady=(16, 4))
        self.meta_label = tk.Label(result_box, text="拍照后点击“开始分析”", font=("Segoe UI", 10), bg=self.PANEL_2, fg=self.MUTED, anchor="w", justify="left", wraplength=520)
        self.meta_label.pack(fill="x", padx=16, pady=(0, 6))
        self.result_rows = tk.Frame(result_box, bg=self.PANEL_2)
        self.result_rows.pack(fill="both", expand=True, padx=16, pady=(0, 10))

        controls = tk.Frame(right, bg=self.PANEL, highlightthickness=1, highlightbackground="#334155")
        controls.pack(fill="x", padx=20, pady=(0, 20))
        tk.Label(controls, text="操作", font=("Segoe UI", 11, "bold"), bg=self.PANEL, fg=self.TEXT, anchor="w").pack(fill="x", padx=14, pady=(12, 8))

        camera_buttons = tk.Frame(controls, bg=self.PANEL)
        camera_buttons.pack(fill="x", padx=14, pady=(0, 8))
        camera_buttons.grid_columnconfigure(0, weight=1)
        camera_buttons.grid_columnconfigure(1, weight=1)
        self.start_button = ttk.Button(camera_buttons, text="开始相机", command=self.start_camera, style="Accent.TButton")
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.stop_button = ttk.Button(camera_buttons, text="停止相机", command=self.stop_camera, state="disabled")
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        self.background_button = ttk.Button(camera_buttons, text="采集空台背景", command=self.capture_background, state="disabled")
        self.background_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        action_buttons = tk.Frame(controls, bg=self.PANEL)
        action_buttons.pack(fill="x", padx=14, pady=(0, 10))
        action_buttons.grid_columnconfigure(0, weight=1)
        action_buttons.grid_columnconfigure(1, weight=1)
        self.capture_button = ttk.Button(action_buttons, text="手动拍照", command=self.capture_photo, state="disabled", style="Capture.TButton")
        self.capture_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.analyze_button = ttk.Button(action_buttons, text="开始分析", command=self.analyze_photo, state="disabled", style="Accent.TButton")
        self.analyze_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        self.action_status = tk.Label(controls, text="请先启动相机并放置目标物", font=("Segoe UI", 9), bg=self.PANEL, fg=self.MUTED, anchor="w", justify="left", wraplength=430)
        self.action_status.pack(fill="x", padx=14, pady=(0, 6))
        tk.Label(controls, text="相机参数：config/camera_profile.yaml", font=("Segoe UI", 9), bg=self.PANEL, fg=self.MUTED, anchor="w", justify="left", wraplength=430).pack(fill="x", padx=14, pady=(0, 12))

    def _camera_alive(self) -> bool:
        return self.camera_worker is not None and self.camera_worker.is_alive()

    def _analysis_alive(self) -> bool:
        return self.analysis_worker is not None and self.analysis_worker.is_alive()

    def start_camera(self) -> None:
        if self._analysis_alive():
            return
        if self._camera_alive():
            return
        profile = project_root() / "config" / "camera_profile.yaml"
        if not profile.exists():
            messagebox.showerror("配置不存在", f"找不到相机配置文件：\n{profile}")
            return
        self.captured_frame = None
        self.captured_path = None
        self._clear_result("等待拍照", "拍照后点击“开始分析”")
        self.analyze_button.configure(state="disabled")
        self.stop_event.clear()
        preview_width = 1280
        try:
            raw_profile = yaml.safe_load(profile.read_text(encoding="utf-8")) or {}
            preview_width = int(raw_profile.get("preview_width", 1280))
        except Exception:
            pass
        self.camera_worker = CameraWorker(self.events, self.stop_event, profile, preview_width)
        self.camera_worker.start()
        self.status.configure(text="● 相机正在启动…", fg=self.AMBER)
        self.action_status.configure(text="正在连接相机…", fg=self.AMBER)
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.capture_button.configure(state="disabled")
        self.background_button.configure(state="disabled")

    def stop_camera(self) -> None:
        if not self._camera_alive():
            self.start_button.configure(state="disabled" if self._analysis_alive() else "normal")
            self.stop_button.configure(state="disabled")
            self.capture_button.configure(state="disabled")
            self.background_button.configure(state="disabled")
            return
        self.stop_event.set()
        self.status.configure(text="● 相机正在停止…", fg=self.AMBER)
        self.stop_button.configure(state="disabled")
        self.background_button.configure(state="disabled")

    def capture_photo(self) -> None:
        if self._analysis_alive():
            messagebox.showwarning("正在分析", "请等待当前分析完成后再重新拍照。")
            return
        if not self._camera_alive() or self.camera_worker is None:
            messagebox.showwarning("相机未运行", "请先启动相机。")
            return
        frame = self.camera_worker.snapshot()
        if frame is None:
            messagebox.showwarning("暂无画面", "相机还没有产生可拍摄的画面，请稍后重试。")
            return
        captures = project_root() / "runs" / "captures"
        captures.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = captures / f"capture_{stamp}.jpg"
        try:
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError(f"无法写入照片：{path}")
            cv2.imwrite(str(captures / "latest.jpg"), frame)
        except Exception as error:
            messagebox.showerror("拍照失败", str(error))
            return
        self.captured_frame = frame.copy()
        self.captured_path = path
        self.preview_hold_until = time.monotonic() + 1.2
        self.show_frame(frame)
        self._clear_result("已拍照，等待分析", f"照片：{path.name}")
        self.analyze_button.configure(state="normal")
        self.action_status.configure(text=f"照片已保存：{path}", fg=self.MUTED)

    def capture_background(self) -> None:
        if self._analysis_alive():
            messagebox.showwarning("正在分析", "请等待当前分析完成后再采集背景。")
            return
        if not self._camera_alive() or self.camera_worker is None:
            messagebox.showwarning("相机未运行", "请先启动相机。")
            return
        frame = self.camera_worker.snapshot()
        if frame is None:
            messagebox.showwarning("暂无画面", "相机还没有产生可拍摄的画面，请稍后重试。")
            return
        if not messagebox.askyesno(
            "采集空台背景",
            "请确认实验台上没有任何目标物，并且当前光照与实验时保持一致。\n\n是否用当前画面更新空台背景？",
        ):
            return
        path = project_root() / "runs" / "measurement" / "background.jpg"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                raise RuntimeError(f"无法写入背景图：{path}")
        except Exception as error:
            messagebox.showerror("背景采集失败", str(error))
            return
        self.action_status.configure(text=f"空台背景已更新：{path}", fg=self.GREEN)
        messagebox.showinfo("背景已更新", "空台背景采集完成。现在放置目标物并点击“手动拍照”。")
    def analyze_photo(self) -> None:
        if self._analysis_alive():
            return
        if self.captured_frame is None or self.captured_path is None:
            messagebox.showwarning("尚未拍照", "请先点击“手动拍照”。")
            return
        config = project_root() / "config" / "vlm.yaml"
        if not config.exists():
            messagebox.showerror("配置不存在", "分析配置未就绪，请检查部署文件。")
            return
        calibration = project_root() / "data" / "calibration" / "calibration_new.json"
        if not calibration.exists():
            messagebox.showerror("标定不存在", "尚未完成尺寸标定。")
            return
        background = project_root() / "runs" / "measurement" / "background.jpg"
        if not background.exists():
            messagebox.showerror("背景图不存在", "请先采集空实验台背景图。")
            return
        self._clear_result("分析中…", "正在生成分析结果")
        self.action_status.configure(text="正在进行图像分析，请稍候…", fg=self.AMBER)
        self.start_button.configure(state="disabled")
        self.capture_button.configure(state="disabled")
        self.background_button.configure(state="disabled")
        self.analyze_button.configure(state="disabled")
        self.analysis_worker = AnalysisWorker(self.events, self.captured_frame, config, self.captured_path, calibration, background)
        self.analysis_worker.start()

    def poll_events(self) -> None:
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "frame":
                    if time.monotonic() >= self.preview_hold_until:
                        self.show_frame(value)
                elif kind == "camera_ready":
                    self.status.configure(text=f"● 相机运行中 · {value}", fg=self.GREEN)
                    self.action_status.configure(text="请放置目标物，稳定后点击“手动拍照”", fg=self.MUTED)
                    if not self._analysis_alive():
                        self.capture_button.configure(state="normal")
                        self.background_button.configure(state="normal")
                elif kind == "camera_error":
                    self.status.configure(text="● 相机启动失败", fg=self.RED)
                    self.action_status.configure(text="相机启动失败，请查看错误日志", fg=self.RED)
                    message, log_path = value
                    hint = "\n\n请确认已关闭 MVDCP2、相机已连接，且 config/camera_profile.yaml 正确。"
                    if log_path:
                        hint += f"\n\n错误日志：{log_path}"
                    messagebox.showerror("相机启动失败", message + hint)
                    self.background_button.configure(state="disabled")
                elif kind == "camera_stopped":
                    self.camera_worker = None
                    self.start_button.configure(state="disabled" if self._analysis_alive() else "normal")
                    self.stop_button.configure(state="disabled")
                    self.capture_button.configure(state="disabled")
                    self.background_button.configure(state="disabled")
                    if self.captured_frame is None:
                        self.status.configure(text="● 相机已停止", fg=self.MUTED)
                    else:
                        self.status.configure(text="● 相机已停止 · 已保留照片", fg=self.MUTED)
                elif kind == "analysis_status":
                    self.action_status.configure(text=f"● {value}", fg=self.AMBER)
                elif kind == "analysis_result":
                    self.analysis_worker = None
                    self.show_analysis_result(value)
                    self.action_status.configure(text="分析完成；可重新拍照后再次分析", fg=self.GREEN)
                    self.analyze_button.configure(state="normal")
                    if not self._camera_alive():
                        self.start_button.configure(state="normal")
                    if self._camera_alive():
                        self.capture_button.configure(state="normal")
                        self.background_button.configure(state="normal")
                elif kind == "analysis_error":
                    self.analysis_worker = None
                    self.status.configure(text="● 分析失败", fg=self.RED)
                    self.action_status.configure(text="分析未完成", fg=self.RED)
                    message, log_path = value
                    hint = "\n\n请检查网络连接或稍后重试。"
                    if log_path:
                        hint += f"\n\n错误日志：{log_path}"
                    messagebox.showerror("分析失败", "本次分析未能完成。" + hint)
                    self.analyze_button.configure(state="normal")
                    if not self._camera_alive():
                        self.start_button.configure(state="normal")
                    if self._camera_alive():
                        self.capture_button.configure(state="normal")
                        self.background_button.configure(state="normal")
        except queue.Empty:
            pass
        self.after(50, self.poll_events)

    def show_frame(self, frame) -> None:
        width = max(400, self.video.winfo_width())
        height = max(300, self.video.winfo_height())
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.video.configure(image=self.photo, text="")

    def _clear_result_rows(self) -> None:
        for child in self.result_rows.winfo_children():
            child.destroy()

    def _clear_result(self, title: str, detail: str) -> None:
        self._clear_result_rows()
        self.decision_label.configure(text=title, fg=self.MUTED)
        self.meta_label.configure(text=detail)

    def show_analysis_result(self, record: dict) -> None:
        objects = record.get("objects")
        if not isinstance(objects, list) or not objects:
            objects = [record]
        self._clear_result_rows()
        self.decision_label.configure(text=f"检测到 {len(objects)} 个目标", fg=self.MUTED)
        self.meta_label.configure(text="各目标按实测尺寸独立分级" if len(objects) > 1 else "")
        for index, item in enumerate(objects, start=1):
            if not isinstance(item, dict):
                continue
            decision = str(item.get("decision", "可能吞咽"))
            color = self.DECISION_COLORS.get(decision, self.MUTED)
            object_name = str(item.get("object_name", "未识别")).strip() or "未识别"
            confidence = item.get("confidence")
            measurement = item.get("measurement")
            row = tk.Frame(self.result_rows, bg=self.PANEL_2)
            row.pack(fill="x", pady=(0, 6))
            header = tk.Frame(row, bg=self.PANEL_2)
            header.pack(fill="x")
            tk.Label(header, text=f"{index}. {object_name}", font=("Segoe UI", 11, "bold"), bg=self.PANEL_2, fg=self.TEXT, anchor="w").pack(side="left")
            tk.Label(header, text=decision, font=("Segoe UI", 11, "bold"), bg=self.PANEL_2, fg=color, anchor="e").pack(side="right")
            details = []
            if isinstance(measurement, dict):
                length = measurement.get("length_mm")
                width = measurement.get("width_mm")
                if isinstance(length, (int, float)) and isinstance(width, (int, float)):
                    details.append(f"尺寸：{float(length):.1f} × {float(width):.1f} mm")
            if isinstance(confidence, (int, float)):
                details.append(f"置信度：{float(confidence):.0%}")
            if details:
                tk.Label(row, text="  ".join(details), font=("Segoe UI", 9), bg=self.PANEL_2, fg=self.MUTED, anchor="w").pack(fill="x", pady=(1, 0))
        self.status.configure(text="● 分析完成", fg=self.GREEN)

    def on_close(self) -> None:
        self.stop_event.set()
        worker = self.camera_worker
        if worker is not None and worker.is_alive():
            # Give the SDK worker time to call CameraUnInit before the process exits.
            worker.join(timeout=5.0)
        self.destroy()


if __name__ == "__main__":
    App().mainloop()

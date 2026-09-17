"""Dataset capture console: press Space to save full-resolution frames for labelling."""

from __future__ import annotations

import queue
import re
import sys
import threading
import traceback
from pathlib import Path

import cv2
import yaml
import tkinter as tk
from tkinter import messagebox, ttk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from swallow_yolo.camera_profile import resolve_camera_profile
from swallow_yolo.mindvision import MindVisionCamera

JPEG_QUALITY = 95
IMAGE_PREFIX = "img"


def project_root() -> Path:
    """Resolve the runtime root for both source runs and frozen executables."""
    if getattr(sys, "frozen", False):
        executable_root = Path(sys.executable).resolve().parent
        if (executable_root / "config").exists():
            return executable_root
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return PROJECT_ROOT


def write_error_log(name: str, traceback_text: str) -> Path | None:
    try:
        path = project_root() / "runs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(traceback_text, encoding="utf-8")
        return path
    except Exception:
        return None


def next_capture_path(folder: Path) -> Path:
    """Return the next unused img_NNNN.jpg, continuing after any existing files."""
    folder.mkdir(parents=True, exist_ok=True)
    highest = 0
    for path in folder.glob(f"{IMAGE_PREFIX}_*.jpg"):
        match = re.fullmatch(rf"{IMAGE_PREFIX}_(\d+)", path.stem)
        if match:
            highest = max(highest, int(match.group(1)))
    return folder / f"{IMAGE_PREFIX}_{highest + 1:04d}.jpg"


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for path in folder.glob("*.jpg") if path.is_file())


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
            values = resolve_camera_profile(profile or {}, {})
            sdk_path = Path(str(values["sdk_path"])).expanduser()
            if not sdk_path.is_absolute():
                sdk_path = (project_root() / sdk_path).resolve()
            camera = MindVisionCamera(
                sdk_path,
                values["camera"],
                values["resolution_index"],
                values["frame_speed_index"],
                values["exposure_us"],
                values["gain_x"],
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
        except Exception:
            log_path = write_error_log("capture_error.log", traceback.format_exc())
            self.emit("camera_error", str(log_path) if log_path else "")
        finally:
            if camera is not None:
                try:
                    camera.close()
                except Exception:
                    pass
            self.emit("camera_stopped", None)


class DatasetCaptureApp:
    BG = "#10151d"
    PANEL = "#18202b"
    PANEL_2 = "#202b39"
    TEXT = "#eef3f8"
    MUTED = "#93a3b5"
    BLUE = "#2f7fd6"
    GREEN = "#2f9e63"
    AMBER = "#c9861f"
    RED = "#c0432f"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("数据集采集")
        self.root.configure(bg=self.BG)
        self.root.geometry("1280x780")
        self.root.minsize(1024, 640)

        self.output_dir = project_root() / "dataset" / "images"
        self.history: list[Path] = []
        self.events: queue.Queue = queue.Queue(maxsize=4)
        self.stop_event = threading.Event()
        self.camera_worker: CameraWorker | None = None
        self.photo = None
        self._photo_ref = None

        self._build_ui()
        self.refresh_count()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<space>", lambda _event: self.capture())
        self.root.bind("<Control-z>", lambda _event: self.undo())
        self.root.bind("<Escape>", lambda _event: self.on_close())
        self.start_camera()
        self.root.after(50, self.poll_events)

    def _build_ui(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TButton", font=("Segoe UI", 11), padding=(12, 9))
        style.configure("Capture.TButton", background=self.GREEN, foreground="#ffffff", font=("Segoe UI", 14, "bold"), padding=(18, 16))

        body = tk.Frame(self.root, bg=self.BG)
        body.pack(fill="both", expand=True, padx=14, pady=14)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=0, minsize=300)
        body.grid_rowconfigure(0, weight=1)

        video_frame = tk.Frame(body, bg=self.PANEL)
        video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        video_frame.grid_rowconfigure(0, weight=1)
        video_frame.grid_columnconfigure(0, weight=1)
        self.video = tk.Label(video_frame, bg="#000000", text="正在连接相机…", fg=self.MUTED, font=("Segoe UI", 13))
        self.video.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        panel = tk.Frame(body, bg=self.PANEL)
        panel.grid(row=0, column=1, sticky="nsew")

        self.capture_button = ttk.Button(panel, text="采集 (空格)", command=self.capture, style="Capture.TButton")
        self.capture_button.pack(fill="x", padx=16, pady=(18, 8))
        self.undo_button = ttk.Button(panel, text="撤销上一张 (Ctrl+Z)", command=self.undo)
        self.undo_button.pack(fill="x", padx=16, pady=(0, 14))

        self.count_label = tk.Label(panel, text="", font=("Segoe UI", 20, "bold"), bg=self.PANEL, fg=self.TEXT, anchor="w")
        self.count_label.pack(fill="x", padx=16, pady=(6, 0))
        tk.Label(panel, text="已采集张数", font=("Segoe UI", 10), bg=self.PANEL, fg=self.MUTED, anchor="w").pack(
            fill="x", padx=16, pady=(0, 12)
        )

        tk.Label(panel, text="保存位置", font=("Segoe UI", 11, "bold"), bg=self.PANEL, fg=self.TEXT, anchor="w").pack(
            fill="x", padx=16, pady=(4, 2)
        )
        tk.Label(
            panel,
            text=str(self.output_dir),
            font=("Segoe UI", 9),
            bg=self.PANEL,
            fg=self.MUTED,
            anchor="w",
            justify="left",
            wraplength=270,
        ).pack(fill="x", padx=16, pady=(0, 14))

        tk.Label(
            panel,
            text=(
                "拍照建议\n"
                "· 覆盖不同位置、角度、光照\n"
                "· 变换远近、加入部分遮挡\n"
                "· 也可以把多个目标放在一起拍\n"
                "· 每个目标建议至少 150 张\n\n"
                "快捷键\n"
                "空格        采集\n"
                "Ctrl+Z    撤销上一张\n"
                "Esc          退出"
            ),
            font=("Segoe UI", 9),
            bg=self.PANEL,
            fg=self.MUTED,
            anchor="w",
            justify="left",
            wraplength=270,
        ).pack(fill="x", padx=16, pady=(0, 12))

        self.status = tk.Label(self.root, text="● 正在连接相机…", font=("Segoe UI", 10), bg=self.BG, fg=self.AMBER, anchor="w")
        self.status.pack(fill="x", padx=18, pady=(0, 4))
        self.action_status = tk.Label(self.root, text="", font=("Segoe UI", 10), bg=self.BG, fg=self.MUTED, anchor="w")
        self.action_status.pack(fill="x", padx=18, pady=(0, 10))

    def refresh_count(self) -> None:
        total = count_images(self.output_dir)
        self.count_label.configure(text=str(total))

    def start_camera(self) -> None:
        profile = project_root() / "config" / "camera_profile.yaml"
        profile_values = {}
        if profile.exists():
            profile_values = yaml.safe_load(profile.read_text(encoding="utf-8")) or {}
        preview_width = int(profile_values.get("preview_width", 1280))
        self.stop_event.clear()
        self.camera_worker = CameraWorker(self.events, self.stop_event, profile, preview_width)
        self.camera_worker.start()

    def poll_events(self) -> None:
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "frame":
                    self.show_frame(value)
                elif kind == "camera_ready":
                    self.status.configure(text=f"● 相机就绪 · {value}", fg=self.GREEN)
                    self.action_status.configure(text="按空格开始采集", fg=self.MUTED)
                elif kind == "camera_error":
                    self.status.configure(text="● 相机启动失败", fg=self.RED)
                    self.action_status.configure(text="请查看 runs/capture_error.log", fg=self.RED)
                    self.capture_button.configure(state="disabled")
                    hint = f"\n\n错误日志：{value}" if value else ""
                    messagebox.showerror("相机启动失败", "无法打开相机，请确认相机已连接且未被其他程序占用。" + hint)
                elif kind == "camera_stopped":
                    if not self.stop_event.is_set():
                        self.status.configure(text="● 相机已停止", fg=self.MUTED)
        except queue.Empty:
            pass
        self.root.after(50, self.poll_events)

    def show_frame(self, frame) -> None:
        width = max(400, self.video.winfo_width() - 20)
        height = max(300, self.video.winfo_height() - 20)
        h, w = frame.shape[:2]
        scale = min(width / w, height / h)
        if scale < 1.0:
            frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.photo = tk.PhotoImage(data=cv2.imencode(".png", rgb)[1].tobytes(), format="png")
        self.video.configure(image=self.photo, text="")

    def snapshot(self):
        if self.camera_worker is None:
            return None
        return self.camera_worker.snapshot()

    def capture(self) -> None:
        frame = self.snapshot()
        if frame is None:
            self.action_status.configure(text="尚未收到相机画面，请稍候", fg=self.AMBER)
            return
        target = next_capture_path(self.output_dir)
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok or not buffer.tofile(str(target)):
            self.action_status.configure(text="保存失败", fg=self.RED)
            return
        self.history.append(target)
        self.refresh_count()
        self.action_status.configure(text=f"已保存 {target.name}", fg=self.GREEN)

    def undo(self) -> None:
        if not self.history:
            self.action_status.configure(text="没有可撤销的采集", fg=self.MUTED)
            return
        path = self.history.pop()
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            self.action_status.configure(text=f"无法删除 {path.name}", fg=self.RED)
            return
        self.refresh_count()
        self.action_status.configure(text=f"已撤销 {path.name}", fg=self.AMBER)

    def on_close(self) -> None:
        self.stop_event.set()
        worker = self.camera_worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=5.0)
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    DatasetCaptureApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

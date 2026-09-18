"""Local YOLO backend for the desktop console.

The desktop UI and deterministic size grading stay unchanged.  This module
only replaces the remote image-understanding step with a local Ultralytics
model.  Both detection and segmentation checkpoints are supported.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import yaml

from swallow_yolo.calibration import Calibration
from swallow_yolo.geometry import estimate_mask_size_mm, estimate_size_mm
from swallow_yolo.vlm import classify_size_level


class InferenceConfigError(RuntimeError):
    """Raised when the local inference configuration is invalid."""


class YoloDetectorError(RuntimeError):
    """Raised when the local YOLO model cannot produce a usable result."""


@dataclass(frozen=True)
class YoloBackendConfig:
    model_path: Path
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.50
    image_size: int = 1280
    device: str = "cpu"
    max_detections: int = 20
    min_mask_area_px: float = 200.0
    max_length_mm: float = 500.0
    border_margin_px: float = 8.0
    class_names: Mapping[str, str] = field(default_factory=dict)

    def display_name(self, raw_name: str) -> str:
        return str(self.class_names.get(raw_name, raw_name)).strip() or raw_name


@dataclass(frozen=True)
class InferenceConfig:
    backend: str
    yolo: YoloBackendConfig | None = None


def no_detection_item() -> dict[str, Any]:
    """Return the UI/JSON result used when none of the trained classes are found."""

    return {
        "object_name": "未识别到",
        "decision": "未识别到",
        "confidence": None,
        "reasons": (),
        "measurement": None,
    }


@dataclass(frozen=True)
class YoloDetection:
    object_name: str
    confidence: float
    length_mm: float
    width_mm: float
    box_xyxy: tuple[float, float, float, float]
    measurement_source: str

    def to_analysis_item(self) -> dict[str, Any]:
        decision, effective_mm = classify_size_level(self.length_mm, self.width_mm)
        measurement = {
            "length_mm": round(self.length_mm, 2),
            "width_mm": round(self.width_mm, 2),
            "characteristic_mm": round(max(self.length_mm, self.width_mm), 2),
        }
        return {
            "object_name": self.object_name,
            "decision": decision,
            "confidence": round(float(self.confidence), 4),
            "reasons": (
                f"标定尺寸 {self.length_mm:.1f} × {self.width_mm:.1f} mm",
                f"有效尺寸 {effective_mm:.1f} mm",
            ),
            "measurement": measurement,
            "box_xyxy": [round(float(value), 1) for value in self.box_xyxy],
            "measurement_source": self.measurement_source,
        }


_MODEL_CACHE: dict[tuple[str, int, int], Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _is_border_clipped(box: tuple[float, float, float, float], frame_shape: tuple[int, int], margin_px: float) -> bool:
    """Reject clipped detections that are not fully inside the camera frame."""

    frame_height, frame_width = frame_shape[:2]
    return (
        box[0] <= margin_px
        or box[1] <= margin_px
        or box[2] >= frame_width - 1 - margin_px
        or box[3] >= frame_height - 1 - margin_px
    )


def _positive_number(value: Any, name: str, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise InferenceConfigError(f"{name} 必须是数字") from error
    if not minimum <= number <= maximum:
        raise InferenceConfigError(f"{name} 必须在 {minimum:g} 到 {maximum:g} 之间")
    return number


def _positive_integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise InferenceConfigError(f"{name} 必须是整数") from error
    if not minimum <= number <= maximum:
        raise InferenceConfigError(f"{name} 必须在 {minimum} 到 {maximum} 之间")
    return number


def load_inference_config(path: str | Path, project_root: str | Path | None = None) -> InferenceConfig:
    """Load the active backend without importing torch or ultralytics."""

    config_path = Path(path)
    if not config_path.exists():
        raise InferenceConfigError(f"缺少推理配置文件：{config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as error:
        raise InferenceConfigError(f"无法读取推理配置：{config_path}") from error
    if not isinstance(raw, Mapping):
        raise InferenceConfigError("推理配置根节点必须是 YAML 对象")

    backend_value = str(raw.get("backend", "yolo")).strip().lower()
    backend_aliases = {
        "yolo": "yolo",
        "local": "yolo",
        "local_yolo": "yolo",
        "vlm": "vlm",
        "api": "vlm",
        "remote": "vlm",
    }
    backend = backend_aliases.get(backend_value)
    if backend is None:
        raise InferenceConfigError("backend 只能是 yolo 或 vlm")
    if backend == "vlm":
        return InferenceConfig(backend="vlm")

    root = Path(project_root) if project_root is not None else config_path.resolve().parents[1]
    model_value = str(raw.get("model_path", "best.pt")).strip() or "best.pt"
    model_path = Path(model_value).expanduser()
    if not model_path.is_absolute():
        model_path = (root / model_path).resolve()
    if not model_path.is_file():
        raise InferenceConfigError(f"找不到本地模型文件：{model_path}")

    raw_names = raw.get("class_names", {})
    if raw_names is None:
        raw_names = {}
    if not isinstance(raw_names, Mapping):
        raise InferenceConfigError("class_names 必须是 YAML 对象")
    class_names = {str(key).strip(): str(value).strip() for key, value in raw_names.items() if str(key).strip()}

    return InferenceConfig(
        backend="yolo",
        yolo=YoloBackendConfig(
            model_path=model_path,
            confidence_threshold=_positive_number(raw.get("confidence_threshold", 0.25), "confidence_threshold", 0.0, 1.0),
            iou_threshold=_positive_number(raw.get("iou_threshold", 0.50), "iou_threshold", 0.0, 1.0),
            image_size=_positive_integer(raw.get("image_size", 1280), "image_size", 320, 4096),
            device=str(raw.get("device", "cpu")).strip() or "cpu",
            max_detections=_positive_integer(raw.get("max_detections", 20), "max_detections", 1, 100),
            min_mask_area_px=_positive_number(raw.get("min_mask_area_px", 200.0), "min_mask_area_px", 0.0, 10_000_000.0),
            max_length_mm=_positive_number(raw.get("max_length_mm", 500.0), "max_length_mm", 1.0, 10000.0),
            border_margin_px=_positive_number(raw.get("border_margin_px", 8.0), "border_margin_px", 0.0, 500.0),
            class_names=class_names,
        ),
    )


def _load_cached_model(model_path: Path):
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise YoloDetectorError("本地推理组件未安装，请先安装 ultralytics") from error

    stat = model_path.stat()
    cache_key = (str(model_path), int(stat.st_mtime_ns), int(stat.st_size))
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(cache_key)
        if model is None:
            try:
                model = YOLO(str(model_path))
            except Exception as error:
                raise YoloDetectorError(f"无法加载本地模型：{error}") from error
            _MODEL_CACHE.clear()
            _MODEL_CACHE[cache_key] = model
    return model


class YoloDetector:
    """Run one local model over a calibrated camera frame."""

    def __init__(self, config: YoloBackendConfig):
        self.config = config
        self.model = _load_cached_model(config.model_path)
        self.task = str(getattr(self.model, "task", "") or "").lower()

    def detect(self, frame: np.ndarray, calibration: Calibration) -> tuple[YoloDetection, ...]:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise YoloDetectorError("没有可识别的相机画面")
        if not calibration.validate_frame(frame):
            raise YoloDetectorError("目标照片分辨率与标定不一致")

        undistorted = calibration.undistort(frame)
        predict_options: dict[str, Any] = {
            "source": undistorted,
            "conf": self.config.confidence_threshold,
            "iou": self.config.iou_threshold,
            "imgsz": self.config.image_size,
            "device": self.config.device,
            "max_det": self.config.max_detections,
            "verbose": False,
        }
        if self.task == "segment":
            predict_options["retina_masks"] = True
        try:
            results = self.model.predict(**predict_options)
        except Exception as error:
            raise YoloDetectorError(f"本地模型推理失败：{error}") from error
        if not results:
            return ()

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return ()

        names = getattr(result, "names", None) or getattr(self.model, "names", {}) or {}
        masks = getattr(result, "masks", None)
        mask_polygons = getattr(masks, "xy", None) if masks is not None else None
        detections: list[tuple[float, YoloDetection]] = []
        frame_height, frame_width = undistorted.shape[:2]
        border_margin = float(self.config.border_margin_px)

        for index in range(len(boxes)):
            try:
                class_id = int(boxes.cls[index].item())
                confidence = float(boxes.conf[index].item())
                box = tuple(float(value) for value in boxes.xyxy[index].tolist())
            except Exception as error:
                raise YoloDetectorError("本地模型返回的检测框格式无效") from error
            if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                continue
            if _is_border_clipped(box, (frame_height, frame_width), border_margin):
                continue

            raw_name = str(names.get(class_id, class_id))
            if raw_name not in self.config.class_names:
                continue
            mask_xy = None
            if mask_polygons is not None and index < len(mask_polygons):
                candidate = np.asarray(mask_polygons[index], dtype=np.float32).reshape(-1, 2)
                if len(candidate) >= 3 and abs(float(cv2.contourArea(candidate))) >= self.config.min_mask_area_px:
                    mask_xy = candidate

            try:
                if mask_xy is not None:
                    length_mm, width_mm = estimate_mask_size_mm(mask_xy, calibration.table_homography_px_to_mm)
                    source = "mask"
                else:
                    length_mm, width_mm = estimate_size_mm(box, calibration.table_homography_px_to_mm)
                    source = "box"
            except (TypeError, ValueError):
                continue
            if not (0.0 < length_mm <= self.config.max_length_mm and 0.0 < width_mm <= self.config.max_length_mm):
                continue

            box_area = (box[2] - box[0]) * (box[3] - box[1])
            detections.append(
                (
                    box_area,
                    YoloDetection(
                        object_name=self.config.display_name(raw_name),
                        confidence=confidence,
                        length_mm=length_mm,
                        width_mm=width_mm,
                        box_xyxy=box,
                        measurement_source=source,
                    ),
                )
            )

        detections.sort(key=lambda item: item[0], reverse=True)
        return tuple(detection for _, detection in detections)

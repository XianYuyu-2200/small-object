"""OpenCV planar object measurement for a calibrated fixed camera."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from swallow_yolo.calibration import Calibration
from swallow_yolo.geometry import estimate_mask_size_mm


class MeasurementError(RuntimeError):
    """Raised when a reliable object contour cannot be measured."""


@dataclass(frozen=True)
class MeasurementConfig:
    blur_kernel: int = 21
    threshold: int = 15
    morph_kernel: int = 21
    min_contour_area_px: float = 2000.0
    max_length_mm: float = 500.0
    max_objects: int = 5
    max_changed_area_ratio: float = 0.30


@dataclass(frozen=True)
class MeasurementResult:
    length_mm: float
    width_mm: float
    contour_area_px: float
    contour_xy: np.ndarray
    mask: np.ndarray

    def to_dict(self) -> dict[str, float]:
        return {
            "length_mm": round(self.length_mm, 2),
            "width_mm": round(self.width_mm, 2),
            "characteristic_mm": round(max(self.length_mm, self.width_mm), 2),
        }


def _odd_kernel(value: int, name: str) -> int:
    if value < 1:
        raise MeasurementError(f"{name} 必须大于 0")
    if value % 2 == 0:
        value += 1
    return value


def _undistorted_pair(frame: np.ndarray, background: np.ndarray, calibration: Calibration) -> tuple[np.ndarray, np.ndarray]:
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        raise MeasurementError("没有可测量的目标照片")
    if background is None or not isinstance(background, np.ndarray) or background.size == 0:
        raise MeasurementError("缺少空台背景图")
    if not calibration.validate_frame(frame):
        raise MeasurementError("目标照片分辨率与标定不一致")
    if background.shape[:2] != frame.shape[:2]:
        raise MeasurementError("空台背景图与目标照片分辨率不一致")
    return calibration.undistort(frame), calibration.undistort(background)


def measure_objects(
    frame: np.ndarray,
    background: np.ndarray,
    calibration: Calibration,
    config: MeasurementConfig | None = None,
) -> tuple[MeasurementResult, ...]:
    """Measure all sufficiently large changed planar contours.

    Results are ordered from largest to smallest contour so object numbering is
    stable for a fixed scene.
    """

    config = config or MeasurementConfig()
    if config.max_objects < 1:
        raise MeasurementError("max_objects 必须大于 0")
    undistorted_frame, undistorted_background = _undistorted_pair(frame, background, calibration)

    gray_frame = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
    gray_background = cv2.cvtColor(undistorted_background, cv2.COLOR_BGR2GRAY)
    blur_kernel = _odd_kernel(config.blur_kernel, "blur_kernel")
    morph_kernel = _odd_kernel(config.morph_kernel, "morph_kernel")

    gray_frame = cv2.GaussianBlur(gray_frame, (blur_kernel, blur_kernel), 0)
    gray_background = cv2.GaussianBlur(gray_background, (blur_kernel, blur_kernel), 0)
    difference = cv2.absdiff(gray_background, gray_frame)

    threshold = int(max(1, min(254, config.threshold)))
    _, mask = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    if not 0.0 < config.max_changed_area_ratio <= 1.0:
        raise MeasurementError("max_changed_area_ratio 必须在 0 到 1 之间")
    changed_area_ratio = cv2.countNonZero(mask) / float(mask.size)
    if changed_area_ratio > config.max_changed_area_ratio:
        raise MeasurementError("背景与当前画面差异过大，请清空台面后重新采集空台背景")

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise MeasurementError("背景差分后没有检测到目标轮廓")

    results: list[MeasurementResult] = []
    small_only = True
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = float(cv2.contourArea(contour))
        if area < config.min_contour_area_px:
            continue
        small_only = False
        contour_xy = contour.reshape(-1, 2).astype(np.float64)
        try:
            length_mm, width_mm = estimate_mask_size_mm(contour_xy, calibration.table_homography_px_to_mm)
        except ValueError:
            continue
        if length_mm > config.max_length_mm or width_mm > config.max_length_mm:
            continue
        results.append(
            MeasurementResult(
                length_mm=length_mm,
                width_mm=width_mm,
                contour_area_px=area,
                contour_xy=contour_xy,
                mask=mask,
            )
        )
        if len(results) >= config.max_objects:
            break

    if not results:
        if small_only:
            raise MeasurementError("目标轮廓过小或与台面差异不足")
        raise MeasurementError("测得尺寸异常，目标可能未被完整分割或台面发生变化")
    return tuple(results)


def measure_object(
    frame: np.ndarray,
    background: np.ndarray,
    calibration: Calibration,
    config: MeasurementConfig | None = None,
) -> MeasurementResult:
    """Measure the largest changed planar contour after background subtraction."""

    return measure_objects(frame, background, calibration, config)[0]

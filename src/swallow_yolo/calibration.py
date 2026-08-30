"""Calibration file validation and image undistortion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Calibration:
    image_size: tuple[int, int]
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    table_homography_px_to_mm: np.ndarray
    reprojection_error_px: float
    metadata: dict

    def validate_frame(self, frame: np.ndarray) -> bool:
        height, width = frame.shape[:2]
        return (width, height) == self.image_size

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        if not self.validate_frame(frame):
            raise ValueError(f"frame size {(frame.shape[1], frame.shape[0])} does not match calibration {self.image_size}")
        return cv2.undistort(frame, self.camera_matrix, self.distortion_coefficients)


def load_calibration(path: str | Path) -> Calibration:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"image_size", "camera_matrix", "distortion_coefficients", "table_homography_px_to_mm", "reprojection_error_px"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"calibration missing fields: {', '.join(sorted(missing))}")
    image_size = tuple(raw["image_size"])
    if len(image_size) != 2 or min(image_size) <= 0:
        raise ValueError("image_size must be [width, height]")
    homography = np.asarray(raw["table_homography_px_to_mm"], dtype=float)
    if homography.shape != (3, 3):
        raise ValueError("table_homography_px_to_mm must be 3x3")
    return Calibration(
        image_size=(int(image_size[0]), int(image_size[1])),
        camera_matrix=np.asarray(raw["camera_matrix"], dtype=float),
        distortion_coefficients=np.asarray(raw["distortion_coefficients"], dtype=float),
        table_homography_px_to_mm=homography,
        reprojection_error_px=float(raw["reprojection_error_px"]),
        metadata=dict(raw.get("metadata", {})),
    )

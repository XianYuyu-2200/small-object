"""Planar geometry helpers for a pre-calibrated, fixed camera."""

from __future__ import annotations

import numpy as np


def apply_homography(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Map Nx2 image points into calibrated table-plane coordinates in millimetres."""
    points = np.asarray(points, dtype=float)
    homography = np.asarray(homography, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if homography.shape != (3, 3):
        raise ValueError("homography must have shape (3, 3)")
    homogeneous = np.column_stack((points, np.ones(len(points))))
    mapped = homogeneous @ homography.T
    if np.any(np.isclose(mapped[:, 2], 0)):
        raise ValueError("homography maps a point to infinity")
    return mapped[:, :2] / mapped[:, 2:3]


def estimate_size_mm(box_xyxy: tuple[float, float, float, float], homography: np.ndarray) -> tuple[float, float]:
    """Return width and height of an axis-aligned detection box on the table plane.

    These are projected planar dimensions, not an object's 3D dimensions.
    """
    x1, y1, x2, y2 = box_xyxy
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box must satisfy x2 > x1 and y2 > y1")
    corners = apply_homography(np.array([[x1, y1], [x2, y1], [x1, y2]], dtype=float), homography)
    width = float(np.linalg.norm(corners[1] - corners[0]))
    height = float(np.linalg.norm(corners[2] - corners[0]))
    return width, height

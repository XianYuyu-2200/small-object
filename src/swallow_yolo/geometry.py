"""Planar geometry helpers for a pre-calibrated, fixed camera."""

from __future__ import annotations

import numpy as np
import cv2


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


def estimate_mask_size_mm(mask_xy: np.ndarray, homography: np.ndarray) -> tuple[float, float]:
    """Estimate oriented planar dimensions from a segmentation polygon."""
    points = apply_homography(np.asarray(mask_xy, dtype=float), homography).astype(np.float32)
    return oriented_size_mm(points)


def oriented_size_mm(points_mm: np.ndarray) -> tuple[float, float]:
    """Return (length, width) of the minimum-area rectangle around table-plane points."""
    points = np.asarray(points_mm, dtype=np.float32).reshape(-1, 2)
    if len(points) < 3:
        raise ValueError("polygon must contain at least three points")
    width, height = cv2.minAreaRect(points)[1]
    if width <= 0 or height <= 0:
        raise ValueError("polygon has zero area")
    return float(max(width, height)), float(min(width, height))


def nadir_point_mm(homography: np.ndarray, camera_matrix: np.ndarray | None = None, nadir_px: tuple[float, float] | None = None) -> np.ndarray:
    """Table-plane point directly below the camera, used as the centre of parallax.

    For a near-vertical camera the principal point projected onto the table is
    a good approximation of the nadir.  ``nadir_px`` overrides this when the
    nadir has been measured (e.g. with a plumb line).  A few millimetres of
    error here changes the corrected size by well under 1 mm.
    """
    if nadir_px is not None:
        pixel = np.asarray(nadir_px, dtype=float).reshape(1, 2)
    elif camera_matrix is not None:
        camera_matrix = np.asarray(camera_matrix, dtype=float)
        pixel = np.array([[camera_matrix[0, 2], camera_matrix[1, 2]]], dtype=float)
    else:
        raise ValueError("either camera_matrix or nadir_px is required")
    return apply_homography(pixel, homography)[0]


def parallax_scale(object_height_mm: float, camera_height_mm: float) -> float:
    """Factor by which the top face of an object appears enlarged on the table plane."""
    if camera_height_mm <= 0:
        raise ValueError("camera_height_mm must be positive")
    if object_height_mm < 0 or object_height_mm >= camera_height_mm:
        raise ValueError("object_height_mm must be within [0, camera_height_mm)")
    return camera_height_mm / (camera_height_mm - object_height_mm)


def correct_prism_footprint_mm(contour_mm: np.ndarray, object_height_mm: float, camera_height_mm: float, nadir_mm: np.ndarray) -> np.ndarray:
    """Recover the table footprint of a right prism from its table-plane silhouette.

    The silhouette S observed through the table homography is the union of the
    footprint F and the top face F scaled about the nadir by ``k = H / (H - h)``.
    Hence ``F = S ∩ scale(S, 1/k)``.  This is exact for objects with vertical
    sides and a flat top; it is not valid for spheres or tilted objects.
    """
    hull = cv2.convexHull(np.asarray(contour_mm, dtype=np.float32).reshape(-1, 2)).reshape(-1, 2)
    if len(hull) < 3:
        raise ValueError("silhouette must contain at least three points")
    scale = parallax_scale(object_height_mm, camera_height_mm)
    if np.isclose(scale, 1.0):
        return hull
    nadir = np.asarray(nadir_mm, dtype=np.float32).reshape(1, 2)
    shrunk = (nadir + (hull - nadir) / scale).astype(np.float32)
    area, footprint = cv2.intersectConvexConvex(hull, shrunk)
    if footprint is None or area <= 0 or len(footprint) < 3:
        raise ValueError("silhouette is inconsistent with the configured object height")
    return footprint.reshape(-1, 2)


def distance_from_nadir_mm(polygon_mm: np.ndarray, nadir_mm: np.ndarray) -> float:
    """Distance from the polygon centroid to the nadir on the table plane."""
    points = np.asarray(polygon_mm, dtype=float).reshape(-1, 2)
    return float(np.linalg.norm(points.mean(axis=0) - np.asarray(nadir_mm, dtype=float)))

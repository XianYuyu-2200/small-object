import cv2
import numpy as np
import pytest

from swallow_yolo.geometry import (
    apply_homography,
    correct_prism_footprint_mm,
    distance_from_nadir_mm,
    estimate_size_mm,
    nadir_point_mm,
    oriented_size_mm,
    parallax_scale,
)


def test_identity_homography_preserves_points():
    points = np.array([[0, 0], [10, 5], [100, 50]], dtype=float)
    result = apply_homography(points, np.eye(3))
    np.testing.assert_allclose(result, points)


def test_estimate_size_uses_projected_box_corners():
    homography = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]], dtype=float)
    length, width = estimate_size_mm((0, 0, 40, 20), homography)
    assert length == 20
    assert width == 10


def test_nadir_defaults_to_projected_principal_point():
    homography = np.array([[0.1, 0, -5], [0, 0.1, -7], [0, 0, 1]], dtype=float)
    camera_matrix = np.array([[1000, 0, 200], [0, 1000, 150], [0, 0, 1]], dtype=float)
    np.testing.assert_allclose(nadir_point_mm(homography, camera_matrix), [15.0, 8.0])
    np.testing.assert_allclose(nadir_point_mm(homography, camera_matrix, nadir_px=(100, 100)), [5.0, 3.0])


def test_parallax_scale_rejects_impossible_heights():
    assert parallax_scale(0, 328) == 1.0
    assert parallax_scale(28, 328) == pytest.approx(328 / 300)
    with pytest.raises(ValueError):
        parallax_scale(400, 328)


def _rectangle(cx, cy, w, d):
    return np.array([[cx - w / 2, cy - d / 2], [cx + w / 2, cy - d / 2], [cx + w / 2, cy + d / 2], [cx - w / 2, cy + d / 2]], np.float32)


def _silhouette(footprint, height, camera_height, nadir):
    """What the camera sees through the table homography: footprint ∪ enlarged top face."""
    scale = camera_height / (camera_height - height)
    top = nadir + (footprint - nadir) * scale
    return cv2.convexHull(np.vstack([footprint, top]).astype(np.float32)).reshape(-1, 2)


@pytest.mark.parametrize("centre", [(94.0, 63.0), (150.0, 100.0), (-20.0, -40.0), (260.0, 200.0)])
def test_prism_footprint_recovered_anywhere_on_table(centre):
    nadir = np.array([94.0, 63.0])
    footprint = _rectangle(*centre, 60, 40)
    silhouette = _silhouette(footprint, 30, 328, nadir)
    raw_length, raw_width = oriented_size_mm(silhouette)
    assert raw_length > 60.5  # uncorrected silhouette is inflated

    corrected = correct_prism_footprint_mm(silhouette, 30, 328, nadir)
    length, width = oriented_size_mm(corrected)
    assert length == pytest.approx(60, abs=0.05)
    assert width == pytest.approx(40, abs=0.05)


def test_zero_height_returns_silhouette_unchanged():
    nadir = np.array([0.0, 0.0])
    footprint = _rectangle(50, 50, 20, 10)
    corrected = correct_prism_footprint_mm(footprint, 0, 328, nadir)
    assert oriented_size_mm(corrected) == pytest.approx((20, 10))


def test_distance_from_nadir_uses_centroid():
    polygon = _rectangle(30, 40, 10, 10)
    assert distance_from_nadir_mm(polygon, np.array([0.0, 0.0])) == pytest.approx(50.0)

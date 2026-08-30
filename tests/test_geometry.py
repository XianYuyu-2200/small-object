import numpy as np

from swallow_yolo.geometry import apply_homography, estimate_size_mm


def test_identity_homography_preserves_points():
    points = np.array([[0, 0], [10, 5], [100, 50]], dtype=float)
    result = apply_homography(points, np.eye(3))
    np.testing.assert_allclose(result, points)


def test_estimate_size_uses_projected_box_corners():
    homography = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]], dtype=float)
    length, width = estimate_size_mm((0, 0, 40, 20), homography)
    assert length == 20
    assert width == 10

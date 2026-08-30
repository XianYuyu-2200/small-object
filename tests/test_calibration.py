import json

import numpy as np

from swallow_yolo.calibration import load_calibration


def test_load_calibration_rejects_missing_homography(tmp_path):
    calibration_path = tmp_path / "bad.json"
    calibration_path.write_text(json.dumps({"image_size": [1920, 1080]}), encoding="utf-8")
    try:
        load_calibration(calibration_path)
    except ValueError as error:
        assert "table_homography_px_to_mm" in str(error)
    else:
        raise AssertionError("invalid calibration must be rejected")


def test_load_calibration_validates_matching_frame_size(tmp_path):
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "image_size": [4, 3],
                "camera_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "distortion_coefficients": [0, 0, 0, 0, 0],
                "table_homography_px_to_mm": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "reprojection_error_px": 0.1,
            }
        ),
        encoding="utf-8",
    )
    calibration = load_calibration(calibration_path)
    assert calibration.validate_frame(np.zeros((3, 4, 3), dtype=np.uint8))
    assert not calibration.validate_frame(np.zeros((4, 3, 3), dtype=np.uint8))

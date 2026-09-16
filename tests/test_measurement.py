import numpy as np
import pytest

from swallow_yolo.calibration import Calibration
from swallow_yolo.measurement import MeasurementError, MeasurementConfig, measure_object, measure_objects


def _calibration() -> Calibration:
    return Calibration(
        image_size=(200, 200),
        camera_matrix=np.eye(3, dtype=float),
        distortion_coefficients=np.zeros(5, dtype=float),
        table_homography_px_to_mm=np.eye(3, dtype=float),
        reprojection_error_px=0.0,
        metadata={},
    )


def test_measure_object_uses_background_contour_and_homography():
    background = np.zeros((200, 200, 3), dtype=np.uint8)
    frame = background.copy()
    frame[50:150, 80:120] = 255

    result = measure_object(
        frame,
        background,
        _calibration(),
        MeasurementConfig(blur_kernel=5, morph_kernel=5, min_contour_area_px=100),
    )

    assert result.length_mm == pytest.approx(100, abs=3)
    assert result.width_mm == pytest.approx(40, abs=3)
    assert result.to_dict()["characteristic_mm"] == pytest.approx(100, abs=3)


def test_measure_objects_detects_multiple_separated_objects():
    background = np.zeros((200, 200, 3), dtype=np.uint8)
    frame = background.copy()
    frame[30:70, 20:60] = 255
    frame[120:170, 130:180] = 255

    results = measure_objects(
        frame,
        background,
        _calibration(),
        MeasurementConfig(blur_kernel=5, morph_kernel=5, min_contour_area_px=100),
    )

    assert len(results) == 2
    assert results[0].contour_area_px > results[1].contour_area_px


def test_measure_object_rejects_empty_difference():
    background = np.zeros((200, 200, 3), dtype=np.uint8)
    with pytest.raises(MeasurementError):
        measure_object(background.copy(), background, _calibration())

def test_measure_rejects_invalid_background():
    background = np.zeros((200, 200, 3), dtype=np.uint8)
    frame = np.full((200, 200, 3), 255, dtype=np.uint8)

    with pytest.raises(MeasurementError, match="重新采集空台背景"):
        measure_object(
            frame,
            background,
            _calibration(),
            MeasurementConfig(blur_kernel=5, morph_kernel=5, min_contour_area_px=100),
        )
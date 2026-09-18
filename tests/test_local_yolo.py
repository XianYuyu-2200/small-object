from pathlib import Path

import pytest
import yaml

import numpy as np

from swallow_yolo.calibration import Calibration
from swallow_yolo.local_yolo import InferenceConfigError, YoloDetection, YoloDetector, _is_border_clipped, load_inference_config, no_detection_item


def test_load_yolo_inference_config_resolves_model_and_names(tmp_path):
    (tmp_path / 'best.pt').write_bytes(b'model')
    config_path = tmp_path / 'config' / 'inference.yaml'
    config_path.parent.mkdir()
    config_path.write_text(
        yaml.safe_dump(
            {
                'backend': 'yolo',
                'model_path': 'best.pt',
                'confidence_threshold': 0.3,
                'image_size': 1280,
                'class_names': {'raisins': '葡萄干'},
            },
            allow_unicode=True,
        ),
        encoding='utf-8',
    )

    config = load_inference_config(config_path, tmp_path)

    assert config.backend == 'yolo'
    assert config.yolo is not None
    assert config.yolo.model_path == (tmp_path / 'best.pt').resolve()
    assert config.yolo.confidence_threshold == pytest.approx(0.3)
    assert config.yolo.display_name('raisins') == '葡萄干'


def test_load_vlm_inference_config_does_not_require_model(tmp_path):
    config_path = tmp_path / 'inference.yaml'
    config_path.write_text('backend: vlm\n', encoding='utf-8')

    config = load_inference_config(config_path, tmp_path)

    assert config.backend == 'vlm'
    assert config.yolo is None


def test_no_detection_item_is_an_explicit_ui_result():
    item = no_detection_item()

    assert item["object_name"] == "未识别到"
    assert item["decision"] == "未识别到"
    assert item["measurement"] is None


def test_detector_returns_empty_tuple_when_model_finds_nothing():
    class FakeModel:
        task = "segment"
        names = {}

        def predict(self, **_):
            class Result:
                boxes = None
                masks = None
                names = {}

            return [Result()]

    calibration = Calibration(
        image_size=(10, 10),
        camera_matrix=np.eye(3, dtype=float),
        distortion_coefficients=np.zeros(5, dtype=float),
        table_homography_px_to_mm=np.eye(3, dtype=float),
        reprojection_error_px=0.0,
        metadata={},
    )
    detector = object.__new__(YoloDetector)
    detector.config = type(
        "Config",
        (),
        {
            "confidence_threshold": 0.25,
            "iou_threshold": 0.5,
            "image_size": 640,
            "device": "cpu",
            "max_detections": 10,
            "min_mask_area_px": 1.0,
            "max_length_mm": 500.0,
            "border_margin_px": 8.0,
        },
    )()
    detector.model = FakeModel()
    detector.task = "segment"

    assert detector.detect(np.zeros((10, 10, 3), dtype="uint8"), calibration) == ()


def test_detector_ignores_class_not_declared_in_config():
    class Value:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

        def tolist(self):
            return list(self.value) if isinstance(self.value, tuple) else self.value

    class Boxes:
        cls = [Value(99)]
        conf = [Value(0.99)]
        xyxy = [Value((20.0, 20.0, 80.0, 80.0))]

        def __len__(self):
            return 1

    class Masks:
        xy = [np.asarray([[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [20.0, 80.0]], dtype=np.float32)]

    class Result:
        boxes = Boxes()
        masks = Masks()
        names = {99: "unknown"}

    class FakeModel:
        task = "segment"

        def predict(self, **_):
            return [Result()]

    calibration = Calibration(
        image_size=(100, 100),
        camera_matrix=np.eye(3, dtype=float),
        distortion_coefficients=np.zeros(5, dtype=float),
        table_homography_px_to_mm=np.eye(3, dtype=float),
        reprojection_error_px=0.0,
        metadata={},
    )
    detector = object.__new__(YoloDetector)
    detector.config = type(
        "Config",
        (),
        {
            "confidence_threshold": 0.25,
            "iou_threshold": 0.5,
            "image_size": 640,
            "device": "cpu",
            "max_detections": 10,
            "min_mask_area_px": 1.0,
            "max_length_mm": 500.0,
            "border_margin_px": 8.0,
            "class_names": {"known": "已知"},
        },
    )()
    detector.model = FakeModel()
    detector.task = "segment"

    assert detector.detect(np.zeros((100, 100, 3), dtype="uint8"), calibration) == ()


def test_border_clipped_detection_is_rejected():
    assert _is_border_clipped((2.9, 0.0, 1028.0, 2773.0), (3672, 5488), 8.0) is True
    assert _is_border_clipped((1400.0, 900.0, 2200.0, 1600.0), (3672, 5488), 8.0) is False


def test_yolo_detection_reuses_five_level_size_decision():
    detection = YoloDetection(
        object_name='黄豆',
        confidence=0.91,
        length_mm=8.0,
        width_mm=7.0,
        box_xyxy=(1.0, 2.0, 9.0, 10.0),
        measurement_source='mask',
    )

    item = detection.to_analysis_item()

    assert item['object_name'] == '黄豆'
    assert item['decision'] == '极易吞咽'
    assert item['measurement']['length_mm'] == 8.0
    assert item['measurement_source'] == 'mask'


def test_load_inference_config_rejects_unknown_backend(tmp_path):
    config_path = tmp_path / 'inference.yaml'
    config_path.write_text('backend: unsupported\n', encoding='utf-8')

    with pytest.raises(InferenceConfigError):
        load_inference_config(config_path, tmp_path)

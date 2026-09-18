from pathlib import Path

import pytest
import yaml

from swallow_yolo.local_yolo import InferenceConfigError, YoloDetection, load_inference_config


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

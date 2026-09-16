import pytest

from swallow_yolo.risk import RiskConfig, RiskInput, classify_risk


def test_small_flat_object_is_easy_to_swallow():
    config = RiskConfig(ingestible_max_mm=20, buffer_mm=2)
    result = classify_risk(config, RiskInput(label="bead", confidence=0.95, length_mm=12, width_mm=8))
    assert result.category == "易误食"
    assert result.reason == "尺寸小于易误食阈值"


def test_measurement_near_threshold_is_unknown():
    config = RiskConfig(ingestible_max_mm=20, buffer_mm=2)
    result = classify_risk(config, RiskInput(label="coin", confidence=0.95, length_mm=21, width_mm=10))
    assert result.category == "无法判断"
    assert "缓冲区" in result.reason


def test_low_confidence_is_unknown():
    config = RiskConfig(ingestible_max_mm=20, buffer_mm=2, min_confidence=0.7)
    result = classify_risk(config, RiskInput(label="coin", confidence=0.4, length_mm=12, width_mm=8))
    assert result.category == "无法判断"
    assert "置信度" in result.reason


def test_unknown_object_height_is_unknown_when_required():
    config = RiskConfig(ingestible_max_mm=20, buffer_mm=2, require_object_height=True)
    item = RiskInput(label="block", confidence=0.95, length_mm=12, width_mm=8, object_height_known=False)
    result = classify_risk(config, item)
    assert result.category == "无法判断"
    assert "物体高度" in result.reason
    relaxed = RiskConfig(ingestible_max_mm=20, buffer_mm=2, require_object_height=False)
    assert classify_risk(relaxed, item).category == "易误食"


def test_object_outside_measure_zone_is_unknown():
    config = RiskConfig(ingestible_max_mm=20, buffer_mm=2, measure_zone_radius_mm=120)
    far = RiskInput(label="block", confidence=0.95, length_mm=12, width_mm=8, distance_from_nadir_mm=150)
    assert "测量区域" in classify_risk(config, far).reason
    near = RiskInput(label="block", confidence=0.95, length_mm=12, width_mm=8, distance_from_nadir_mm=50)
    assert classify_risk(config, near).category == "易误食"
    missing = RiskInput(label="block", confidence=0.95, length_mm=12, width_mm=8)
    assert classify_risk(config, missing).category == "无法判断"


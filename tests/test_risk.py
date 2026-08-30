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


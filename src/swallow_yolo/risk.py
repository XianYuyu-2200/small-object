"""Conservative, configurable triage; not a product-safety certification."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RiskConfig:
    """Project-specific rule values, supplied from a reviewed teaching standard."""

    ingestible_max_mm: float | None = None
    buffer_mm: float = 0.0
    min_confidence: float = 0.70
    manual_review_labels: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class RiskInput:
    label: str
    confidence: float
    length_mm: float | None
    width_mm: float | None
    calibration_valid: bool = True
    touches_frame: bool = False
    occluded_or_stacked: bool = False
    flat_on_table: bool = True


@dataclass(frozen=True)
class RiskResult:
    category: str
    reason: str
    characteristic_mm: float | None


def _unknown(reason: str, characteristic_mm: float | None = None) -> RiskResult:
    return RiskResult("无法判断", reason, characteristic_mm)


def classify_risk(config: RiskConfig, item: RiskInput) -> RiskResult:
    """Classify a detection only when the measurement conditions are trustworthy.

    The characteristic dimension is the larger planar box dimension.  It is a
    transparent project rule, not a claim about an infant-safety standard.
    """
    if config.ingestible_max_mm is None:
        return _unknown("未配置经审核的尺寸规则")
    if not item.calibration_valid:
        return _unknown("标定无效或输入分辨率不匹配")
    if item.confidence < config.min_confidence:
        return _unknown("检测置信度不足")
    if item.touches_frame:
        return _unknown("物件贴近画面边缘")
    if item.occluded_or_stacked:
        return _unknown("物件遮挡、重叠或叠放")
    if not item.flat_on_table:
        return _unknown("物件未平贴台面，二维尺寸不可靠")
    if item.label in config.manual_review_labels:
        return _unknown("该类别要求人工复核")
    if item.length_mm is None or item.width_mm is None:
        return _unknown("缺少平面尺寸")
    if item.length_mm <= 0 or item.width_mm <= 0:
        return _unknown("平面尺寸无效")

    characteristic = max(item.length_mm, item.width_mm)
    lower = config.ingestible_max_mm - config.buffer_mm
    upper = config.ingestible_max_mm + config.buffer_mm
    if characteristic <= lower:
        return RiskResult("易误食", "尺寸小于易误食阈值", characteristic)
    if characteristic >= upper:
        return RiskResult("不易误食", "尺寸大于易误食阈值", characteristic)
    return _unknown("测量值位于阈值缓冲区", characteristic)

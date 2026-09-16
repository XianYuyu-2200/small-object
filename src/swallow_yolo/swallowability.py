"""Class-to-swallowability lookup for the teaching demonstration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SwallowabilityResult:
    can_swallow: bool | None
    text: str
    reason: str


def load_swallowability(path: str | Path) -> tuple[dict[str, bool], float]:
    """Load exact model-label -> bool mappings and a confidence threshold."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"缺少吞咽属性配置文件：{path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = data.get("swallowable", data.get("can_swallow"))
    if not isinstance(raw, dict):
        raise SystemExit(f"{path} 必须包含 swallowable: {{类别名: true/false}}")
    mapping: dict[str, bool] = {}
    for label, value in raw.items():
        if not isinstance(value, bool):
            raise SystemExit(f"类别 {label!r} 的吞咽属性必须是 true 或 false")
        mapping[str(label).strip()] = value
    threshold = float(data.get("min_confidence", 0.0))
    if not 0.0 <= threshold <= 1.0:
        raise SystemExit("min_confidence 必须在 0 到 1 之间")
    return mapping, threshold


def classify_swallowability(mapping: dict[str, bool], label: str, confidence: float, min_confidence: float = 0.0) -> SwallowabilityResult:
    configured_label = label.strip()
    if configured_label not in mapping:
        return SwallowabilityResult(None, "无法判断", "该类别未配置吞咽属性")
    # Once a class is explicitly configured, the requested decision is a
    # deterministic lookup. Confidence remains in the output for auditing but
    # does not override the class-level true/false rule.
    can_swallow = mapping[configured_label]
    return SwallowabilityResult(can_swallow, "能吞咽" if can_swallow else "不能吞咽", "按类别配置判定")

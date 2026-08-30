"""Deterministic timing summaries for camera diagnostics."""

from __future__ import annotations


def summarize_durations(seconds: list[float]) -> dict[str, float | int | None]:
    if not seconds:
        return {"count": 0, "mean_ms": None, "fps": 0.0}
    mean_seconds = sum(seconds) / len(seconds)
    return {
        "count": len(seconds),
        "mean_ms": round(mean_seconds * 1000, 2),
        "fps": round(1 / mean_seconds, 2) if mean_seconds > 0 else 0.0,
    }


def summarize_stages(stages: dict[str, list[float]]) -> dict[str, dict[str, float | int | None]]:
    return {name: summarize_durations(durations) for name, durations in stages.items()}

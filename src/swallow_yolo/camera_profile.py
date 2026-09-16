"""Reproducible acquisition profile for the fixed competition camera."""

from __future__ import annotations


DEFAULT_CAMERA_PROFILE = {
    "backend": "mindvision",
    "sdk_path": "mindvision",
    "camera": 0,
    "resolution_index": 0,
    "frame_speed_index": 2,
    "exposure_us": 90000.0,
    "gain_x": 3.0,
    "preview_width": 1280,
}


def resolve_camera_profile(profile: dict, overrides: dict) -> dict:
    """Merge defaults, profile values, and explicit overrides (last wins)."""
    resolved = dict(DEFAULT_CAMERA_PROFILE)
    resolved.update(profile)
    resolved.update({key: value for key, value in overrides.items() if value is not None})
    return resolved

"""Preview-only image scaling helpers."""

from __future__ import annotations

import cv2
import numpy as np


def fit_for_display(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Shrink a frame to fit a preview width; never alter the source frame."""
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame.copy()
    scale = max_width / width
    return cv2.resize(frame, (max_width, round(height * scale)), interpolation=cv2.INTER_AREA)

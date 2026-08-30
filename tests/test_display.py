import numpy as np

from swallow_yolo.display import fit_for_display


def test_fit_for_display_preserves_small_frame():
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    displayed = fit_for_display(frame, 1280)
    assert displayed.shape == frame.shape


def test_fit_for_display_scales_large_frame_without_upscaling():
    frame = np.zeros((3672, 5488, 3), dtype=np.uint8)
    displayed = fit_for_display(frame, 1280)
    assert displayed.shape[:2] == (856, 1280)

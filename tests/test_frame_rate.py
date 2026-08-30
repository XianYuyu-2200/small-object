from swallow_yolo.frame_rate import FrameRateMonitor


def test_frame_rate_uses_elapsed_interval():
    monitor = FrameRateMonitor(start_time=10.0)
    monitor.record(10.5)
    monitor.record(11.0)
    monitor.record(11.5)

    assert monitor.fps == 2.0


def test_frame_rate_returns_zero_before_a_frame():
    assert FrameRateMonitor(start_time=10.0).fps == 0.0

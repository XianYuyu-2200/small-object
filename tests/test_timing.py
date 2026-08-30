from swallow_yolo.timing import summarize_durations


def test_summarize_durations_reports_mean_and_fps():
    result = summarize_durations([0.1, 0.2, 0.1])

    assert result["count"] == 3
    assert result["mean_ms"] == 133.33
    assert result["fps"] == 7.5


def test_summarize_durations_handles_empty_samples():
    assert summarize_durations([]) == {"count": 0, "mean_ms": None, "fps": 0.0}


def test_summarize_stages_summarizes_each_named_stage():
    from swallow_yolo.timing import summarize_stages

    assert summarize_stages({"acquire": [0.1], "process": [0.2]}) == {
        "acquire": {"count": 1, "mean_ms": 100.0, "fps": 10.0},
        "process": {"count": 1, "mean_ms": 200.0, "fps": 5.0},
    }

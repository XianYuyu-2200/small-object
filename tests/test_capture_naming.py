from swallow_yolo.capture_naming import next_capture_path


def test_first_capture_is_named_one(tmp_path):
    assert next_capture_path(tmp_path).name == "1.jpg"


def test_next_capture_uses_largest_numeric_name_without_overwriting(tmp_path):
    (tmp_path / "1.jpg").touch()
    (tmp_path / "3.jpg").touch()
    (tmp_path / "notes.jpg").touch()

    assert next_capture_path(tmp_path).name == "4.jpg"

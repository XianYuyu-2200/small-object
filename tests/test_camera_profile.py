from swallow_yolo.camera_profile import resolve_camera_profile


def test_profile_supplies_mindvision_defaults():
    profile = {
        "backend": "mindvision",
        "sdk_path": "G:/mindvision",
        "camera": 0,
        "resolution_index": 0,
        "frame_speed_index": 2,
        "exposure_us": 90000,
        "gain_x": 3,
        "preview_width": 1280,
    }

    result = resolve_camera_profile(profile, {})

    assert result["exposure_us"] == 90000
    assert result["gain_x"] == 3
    assert result["backend"] == "mindvision"


def test_explicit_value_overrides_profile_value():
    result = resolve_camera_profile({"exposure_us": 90000}, {"exposure_us": 40000})
    assert result["exposure_us"] == 40000

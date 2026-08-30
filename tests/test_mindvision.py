from pathlib import Path

from swallow_yolo.mindvision import configure_camera, resolve_sdk_paths


def test_resolve_sdk_paths_uses_x64_sdk_layout(tmp_path):
    root = tmp_path / "mindvision"
    demo = root / "Demo" / "Python" / "Basic"
    binary = root / "SDK" / "X64"
    demo.mkdir(parents=True)
    binary.mkdir(parents=True)
    (demo / "mvsdk.py").write_text("# shim", encoding="utf-8")
    (binary / "MVCAMSDK_X64.dll").write_bytes(b"")

    result = resolve_sdk_paths(root)

    assert result.python_dir == demo
    assert result.binary_dir == binary


def test_resolve_sdk_paths_explains_missing_python_binding(tmp_path):
    root = tmp_path / "mindvision"
    root.mkdir()
    try:
        resolve_sdk_paths(root)
    except FileNotFoundError as error:
        assert "mvsdk.py" in str(error)
    else:
        raise AssertionError("incomplete SDK must be rejected")


def test_configure_camera_disables_auto_exposure_for_manual_exposure():
    calls = []

    class FakeSdk:
        def CameraSetFrameSpeed(self, handle, value): calls.append(("speed", handle, value))
        def CameraSetAeState(self, handle, value): calls.append(("ae", handle, value))
        def CameraSetExposureTime(self, handle, value): calls.append(("exposure", handle, value))

    configure_camera(FakeSdk(), 3, frame_speed_index=2, exposure_us=10_000)

    assert calls == [("speed", 3, 2), ("ae", 3, False), ("exposure", 3, 10_000)]


def test_configure_camera_can_set_analog_gain_multiplier():
    calls = []

    class FakeSdk:
        def CameraSetAnalogGainX(self, handle, value): calls.append((handle, value))

    configure_camera(FakeSdk(), 3, gain_x=4.0)

    assert calls == [(3, 4.0)]

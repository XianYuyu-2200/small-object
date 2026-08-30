from pathlib import Path

from swallow_yolo.mindvision import resolve_sdk_paths


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

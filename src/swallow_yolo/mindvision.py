"""Minimal adapter for MindVision's official Windows x64 Python SDK binding."""

from __future__ import annotations

import ctypes
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MindVisionSdkPaths:
    python_dir: Path
    binary_dir: Path


def resolve_sdk_paths(root: str | Path) -> MindVisionSdkPaths:
    root = Path(root)
    python_dir = root / "Demo" / "Python" / "Basic"
    binary_dir = root / "SDK" / "X64"
    if not (python_dir / "mvsdk.py").is_file():
        raise FileNotFoundError(f"未找到官方 Python 封装 mvsdk.py：{python_dir}")
    if not (binary_dir / "MVCAMSDK_X64.dll").is_file():
        raise FileNotFoundError(f"未找到 64 位 SDK DLL：{binary_dir / 'MVCAMSDK_X64.dll'}")
    return MindVisionSdkPaths(python_dir, binary_dir)


def load_mvsdk(root: str | Path):
    """Load the vendor binding without copying vendor SDK files into the repository."""
    paths = resolve_sdk_paths(root)
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(paths.binary_dir))
    if str(paths.python_dir) not in sys.path:
        sys.path.insert(0, str(paths.python_dir))
    if "mvsdk" in sys.modules:
        del sys.modules["mvsdk"]
    return importlib.import_module("mvsdk")


def camera_diagnostics(mvsdk, handle: int, capability) -> dict[str, object]:
    """Read current SDK capture state without changing any camera setting."""
    resolution = mvsdk.CameraGetImageResolution(handle)
    limits = capability.sResolutionRange
    exposure_min, exposure_max, exposure_step = mvsdk.CameraGetExposureTimeRange(handle)
    return {
        "resolution": f"{resolution.iWidth}x{resolution.iHeight}",
        "resolution_description": resolution.GetDescription(),
        "trigger_mode": mvsdk.CameraGetTriggerMode(handle),
        "auto_exposure": bool(mvsdk.CameraGetAeState(handle)),
        "exposure_us": mvsdk.CameraGetExposureTime(handle),
        "exposure_range_us": [exposure_min, exposure_max, exposure_step],
        "frame_speed_index": mvsdk.CameraGetFrameSpeed(handle),
        "resolution_range": f"{limits.iWidthMin}x{limits.iHeightMin}..{limits.iWidthMax}x{limits.iHeightMax}",
        "skip_mode_mask": limits.uSkipModeMask,
        "bin_average_mode_mask": limits.uBinAverageModeMask,
        "resample_mask": limits.uResampleMask,
        "preset_resolutions": [
            {
                "index": int(capability.pImageSizeDesc[i].iIndex),
                "description": capability.pImageSizeDesc[i].GetDescription(),
                "output": f"{capability.pImageSizeDesc[i].iWidth}x{capability.pImageSizeDesc[i].iHeight}",
                "fov": f"{capability.pImageSizeDesc[i].iWidthFOV}x{capability.pImageSizeDesc[i].iHeightFOV}",
            }
            for i in range(capability.iImageSizeDesc)
        ],
    }


def resolution_by_index(capability, preset_index: int):
    """Find a vendor resolution preset by its SDK index, not array position."""
    for position in range(capability.iImageSizeDesc):
        preset = capability.pImageSizeDesc[position]
        if preset.iIndex == preset_index:
            return preset
    available = [capability.pImageSizeDesc[position].iIndex for position in range(capability.iImageSizeDesc)]
    raise IndexError(f"分辨率预设 {preset_index} 不存在；可用编号为 {available}")


class MindVisionCamera:
    """One MindVision camera, returning processed BGR NumPy frames."""

    def __init__(self, sdk_root: str | Path, device_index: int = 0, resolution_index: int | None = None):
        self.mvsdk = load_mvsdk(sdk_root)
        devices = self.mvsdk.CameraEnumerateDevice()
        if not devices:
            raise RuntimeError("迈德威视 SDK 未发现相机；请关闭占用相机的 MVDCP2 后重试")
        if not 0 <= device_index < len(devices):
            raise IndexError(f"迈德威视相机索引 {device_index} 不存在；可用范围为 0..{len(devices) - 1}")
        self.device_name = devices[device_index].GetFriendlyName()
        self.handle = self.mvsdk.CameraInit(devices[device_index], -1, -1)
        self.capability = self.mvsdk.CameraGetCapability(self.handle)
        if resolution_index is not None:
            try:
                self.mvsdk.CameraSetImageResolution(self.handle, resolution_by_index(self.capability, resolution_index))
            except Exception:
                self.close()
                raise
        self.channels = 1 if self.capability.sIspCapacity.bMonoSensor else 3
        output_format = self.mvsdk.CAMERA_MEDIA_TYPE_MONO8 if self.channels == 1 else self.mvsdk.CAMERA_MEDIA_TYPE_BGR8
        self.mvsdk.CameraSetIspOutFormat(self.handle, output_format)
        self.mvsdk.CameraSetTriggerMode(self.handle, 0)
        max_width = self.capability.sResolutionRange.iWidthMax
        max_height = self.capability.sResolutionRange.iHeightMax
        self.frame_buffer = self.mvsdk.CameraAlignMalloc(max_width * max_height * self.channels, 16)
        self.mvsdk.CameraPlay(self.handle)

    def read(self, timeout_ms: int = 2000) -> np.ndarray:
        raw = None
        try:
            raw, header = self.mvsdk.CameraGetImageBuffer(self.handle, timeout_ms)
            self.mvsdk.CameraImageProcess(self.handle, raw, self.frame_buffer, header)
            byte_count = header.iWidth * header.iHeight * self.channels
            image = np.frombuffer(ctypes.string_at(self.frame_buffer, byte_count), dtype=np.uint8)
            if self.channels == 1:
                return image.reshape((header.iHeight, header.iWidth)).copy()
            return image.reshape((header.iHeight, header.iWidth, self.channels)).copy()
        finally:
            if raw:
                self.mvsdk.CameraReleaseImageBuffer(self.handle, raw)

    def close(self) -> None:
        if getattr(self, "handle", None) is not None:
            self.mvsdk.CameraUnInit(self.handle)
            self.handle = None
        if getattr(self, "frame_buffer", None):
            self.mvsdk.CameraAlignFree(self.frame_buffer)
            self.frame_buffer = None

    def diagnostics(self) -> dict[str, object]:
        return camera_diagnostics(self.mvsdk, self.handle, self.capability)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

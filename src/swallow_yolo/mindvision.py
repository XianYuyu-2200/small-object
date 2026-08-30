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


class MindVisionCamera:
    """One MindVision camera, returning processed BGR NumPy frames."""

    def __init__(self, sdk_root: str | Path, device_index: int = 0):
        self.mvsdk = load_mvsdk(sdk_root)
        devices = self.mvsdk.CameraEnumerateDevice()
        if not devices:
            raise RuntimeError("迈德威视 SDK 未发现相机；请关闭占用相机的 MVDCP2 后重试")
        if not 0 <= device_index < len(devices):
            raise IndexError(f"迈德威视相机索引 {device_index} 不存在；可用范围为 0..{len(devices) - 1}")
        self.device_name = devices[device_index].GetFriendlyName()
        self.handle = self.mvsdk.CameraInit(devices[device_index], -1, -1)
        self.capability = self.mvsdk.CameraGetCapability(self.handle)
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

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

# -*- mode: python ; coding: utf-8 -*-

import importlib.util
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs

_torchvision_spec = importlib.util.find_spec("torchvision")
if _torchvision_spec is None or not _torchvision_spec.origin:
    raise RuntimeError("缺少 torchvision，无法构建本地 YOLO 程序")
_torchvision_root = Path(_torchvision_spec.origin).resolve().parent
_torchvision_extension = _torchvision_root / "_C_stable.pyd"
if not _torchvision_extension.is_file():
    raise RuntimeError(f"缺少 torchvision C++ 扩展：{_torchvision_extension}")
torchvision_binaries = collect_dynamic_libs("torchvision") + [
    (str(_torchvision_extension), "torchvision")
]

a = Analysis(
    ['scripts\\app.py'],
    pathex=['src'],
    binaries=torchvision_binaries,
    datas=[('config', 'config')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SwallowabilityConsole',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SwallowabilityConsole',
)

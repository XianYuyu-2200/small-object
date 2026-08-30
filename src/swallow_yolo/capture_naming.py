"""Sequential, non-overwriting names for captured images."""

from __future__ import annotations

from pathlib import Path


def next_capture_path(directory: str | Path, suffix: str = ".jpg") -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in directory.glob("*.jpg"):
        try:
            numbers.append(int(path.stem))
        except ValueError:
            continue
    return directory / f"{max(numbers, default=0) + 1}{suffix}"

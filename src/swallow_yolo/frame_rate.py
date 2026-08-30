"""Small dependency-free FPS monitor for camera diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FrameRateMonitor:
    start_time: float
    _timestamps: list[float] = field(default_factory=list)

    def record(self, timestamp: float) -> None:
        self._timestamps.append(float(timestamp))

    @property
    def fps(self) -> float:
        if not self._timestamps:
            return 0.0
        elapsed = self._timestamps[-1] - self.start_time
        if elapsed <= 0:
            return 0.0
        return len(self._timestamps) / elapsed

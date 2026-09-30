"""Drift-compensated cadence and runtime embedding telemetry."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math


class CumulativeDeadlineScheduler:
    """Schedule work against a fixed time grid instead of ``now + interval``.

    A slow frame can skip one or more deadlines, but it cannot move the grid.
    This prevents phase drift and makes recorded-video replay deterministic.
    """

    def __init__(self, target_fps: float, *, epsilon: float = 1e-9) -> None:
        if not math.isfinite(target_fps) or target_fps <= 0:
            raise ValueError("target_fps must be a finite value greater than zero")
        self.target_fps = float(target_fps)
        self.interval = 1.0 / self.target_fps
        self.epsilon = max(0.0, float(epsilon))
        self.next_deadline: float | None = None

    def reset(self, start_time: float | None = None) -> None:
        self.next_deadline = None if start_time is None else float(start_time)

    def due(self, now: float) -> bool:
        now = float(now)
        if self.next_deadline is None:
            self.next_deadline = now
        if now + self.epsilon < self.next_deadline:
            return False
        missed = max(1, math.floor((now - self.next_deadline + self.epsilon) / self.interval) + 1)
        self.next_deadline += missed * self.interval
        return True


@dataclass(frozen=True)
class EmbeddingTelemetrySnapshot:
    actual_fps: float
    window_duration_s: float
    samples: int


class EmbeddingTelemetry:
    """Measure delivered embedding cadence and the real LSTM window width."""

    def __init__(self, sequence_length: int, *, horizon_s: float = 3.0) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")
        if horizon_s <= 0:
            raise ValueError("horizon_s must be > 0")
        self.sequence_length = sequence_length
        self.horizon_s = float(horizon_s)
        self._times: deque[float] = deque()

    def reset(self) -> None:
        self._times.clear()

    def mark(self, timestamp_s: float) -> None:
        timestamp_s = float(timestamp_s)
        self._times.append(timestamp_s)
        cutoff = timestamp_s - max(self.horizon_s, self.sequence_length / 2)
        while len(self._times) > self.sequence_length * 4 and self._times[1] < cutoff:
            self._times.popleft()

    @property
    def snapshot(self) -> EmbeddingTelemetrySnapshot:
        times = list(self._times)
        recent = times
        if times:
            cutoff = times[-1] - self.horizon_s
            recent = [value for value in times if value >= cutoff]
        actual_fps = 0.0
        if len(recent) >= 2 and recent[-1] > recent[0]:
            actual_fps = (len(recent) - 1) / (recent[-1] - recent[0])
        window = times[-self.sequence_length:]
        duration = window[-1] - window[0] if len(window) >= 2 else 0.0
        return EmbeddingTelemetrySnapshot(actual_fps, duration, len(window))

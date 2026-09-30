"""Small elapsed-time evidence accumulator used by temporal fusion."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceSnapshot:
    key: str
    active: bool
    duration_s: float


class TemporalEvidenceAccumulator:
    def __init__(self, *, max_gap_s: float = 0.35) -> None:
        if max_gap_s <= 0:
            raise ValueError("max_gap_s must be > 0")
        self.max_gap_s = float(max_gap_s)
        self._start: dict[str, float] = {}
        self._last: dict[str, float] = {}

    def reset(self, key: str | None = None) -> None:
        if key is None:
            self._start.clear()
            self._last.clear()
            return
        self._start.pop(key, None)
        self._last.pop(key, None)

    def observe(self, key: str, active: bool, timestamp_s: float) -> EvidenceSnapshot:
        now = float(timestamp_s)
        if not active:
            self.reset(key)
            return EvidenceSnapshot(key, False, 0.0)
        previous = self._last.get(key)
        if previous is None or now < previous or now - previous > self.max_gap_s:
            self._start[key] = now
        self._last[key] = now
        return EvidenceSnapshot(key, True, max(0.0, now - self._start[key]))

    def duration(self, key: str, timestamp_s: float | None = None) -> float:
        if key not in self._start or key not in self._last:
            return 0.0
        end = self._last[key] if timestamp_s is None else min(float(timestamp_s), self._last[key])
        return max(0.0, end - self._start[key])

    def stable(self, key: str, required_s: float) -> bool:
        return self.duration(key) + 1e-9 >= required_s

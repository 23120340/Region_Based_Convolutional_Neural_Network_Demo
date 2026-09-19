from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

from .model_contract import Prediction


@dataclass(frozen=True)
class StablePrediction:
    action: str
    vote_ratio: float
    mean_confidence: float


class TemporalDebouncer:
    """Turn noisy frame predictions into one event per stable gesture.

    A stable ``idle`` segment rearms the same action for a later assembly cycle.
    Sustained predictions are latched, preventing one physical gesture from being
    submitted to the FSM repeatedly.
    """

    def __init__(
        self,
        window_size: int = 5,
        min_votes: int = 3,
        min_confidence: float = 0.70,
        idle_actions: frozenset[str] | set[str] = frozenset({"idle", "uncertain"}),
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size phải >= 1")
        if not 1 <= min_votes <= window_size:
            raise ValueError("min_votes phải nằm trong [1, window_size]")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence phải nằm trong [0, 1]")
        self.window_size = window_size
        self.min_votes = min_votes
        self.min_confidence = min_confidence
        self.idle_actions = frozenset(idle_actions)
        self._history: deque[Prediction | None] = deque(maxlen=window_size)
        self._last_emitted: str | None = None

    def reset(self) -> None:
        self._history.clear()
        self._last_emitted = None

    def update(self, prediction: Prediction) -> StablePrediction | None:
        accepted = prediction if prediction.confidence >= self.min_confidence else None
        self._history.append(accepted)

        labels = [item.action for item in self._history if item is not None]
        if not labels:
            return None
        action, votes = Counter(labels).most_common(1)[0]
        if votes < self.min_votes:
            return None

        matching = [
            item.confidence
            for item in self._history
            if item is not None and item.action == action
        ]
        stable = StablePrediction(
            action=action,
            vote_ratio=votes / len(self._history),
            mean_confidence=sum(matching) / len(matching),
        )

        if action in self.idle_actions:
            self._last_emitted = None
            return None
        if action == self._last_emitted:
            return None

        self._last_emitted = action
        return stable


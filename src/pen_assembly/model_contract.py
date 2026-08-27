from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class Prediction:
    """Output contract shared by the simulator and the future ViT-LSTM model."""

    action: str
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence phải nằm trong [0, 1]")


@runtime_checkable
class ActionRecognizer(Protocol):
    """A future camera recognizer only needs to implement this interface."""

    def predict(self, frames: Sequence[object]) -> Prediction:
        ...


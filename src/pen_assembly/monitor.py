from __future__ import annotations

import json
from pathlib import Path

from .fsm import ConfigurableAssemblyTracker, FsmOutcome
from .model_contract import Prediction
from .smoother import TemporalDebouncer


class JsonlEventLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, outcome: FsmOutcome, prediction: Prediction | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = outcome.to_dict()
        if prediction is not None:
            record["prediction"] = {
                "action": prediction.action,
                "confidence": prediction.confidence,
            }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


class AssemblyMonitor:
    """Orchestrate recognizer outputs, temporal filtering, FSM and logging."""

    def __init__(
        self,
        tracker: ConfigurableAssemblyTracker,
        debouncer: TemporalDebouncer,
        logger: JsonlEventLogger | None = None,
    ) -> None:
        self.tracker = tracker
        self.debouncer = debouncer
        self.logger = logger

    def submit_prediction(self, prediction: Prediction) -> FsmOutcome | None:
        stable = self.debouncer.update(prediction)
        if stable is None:
            return None
        outcome = self.tracker.process(stable.action)
        if self.logger is not None:
            self.logger.append(outcome, prediction)
        return outcome

    def submit_stable_action(self, action: str) -> FsmOutcome:
        """Submit a confirmed action, useful for deterministic scenario tests."""

        outcome = self.tracker.process(action)
        if self.logger is not None:
            self.logger.append(outcome)
        return outcome

    def reset(self) -> FsmOutcome:
        self.debouncer.reset()
        outcome = self.tracker.reset()
        if self.logger is not None:
            self.logger.append(outcome)
        return outcome


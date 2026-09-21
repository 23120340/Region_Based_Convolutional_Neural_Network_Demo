from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from .config import AssemblyConfig


@dataclass(frozen=True)
class FsmOutcome:
    type: str
    action: str
    previous_state: str
    state: str
    message: str
    cycle_id: int
    completed_steps: tuple[str, ...]
    is_complete: bool
    timestamp: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ConfigurableAssemblyTracker:
    """Finite-state workflow verifier with no pen-specific transition code."""

    def __init__(self, config: AssemblyConfig) -> None:
        self.config = config
        self.state = config.initial_state
        self.completed_steps: list[str] = []
        self.cycle_id = 0
        self._last_accepted_action: str | None = None

    @property
    def is_complete(self) -> bool:
        return self.state == self.config.completed_state

    @property
    def expected_actions(self) -> tuple[str, ...]:
        allowed = self.config.states[self.state].get("allowed", {})
        return tuple(allowed.keys())

    @property
    def instruction(self) -> str:
        return str(self.config.states[self.state].get("message", ""))

    def reset(self) -> FsmOutcome:
        previous = self.state
        self.state = self.config.initial_state
        self.completed_steps.clear()
        self._last_accepted_action = None
        return self._outcome(
            "RESET",
            "reset",
            previous,
            "Đã đặt lại hệ thống; sẵn sàng cho một chu trình mới.",
        )

    def process(self, action: str) -> FsmOutcome:
        previous = self.state

        if action in self.config.idle_actions:
            return self._outcome("INFO", action, previous, self.instruction)

        if action not in self.config.actions:
            return self._outcome(
                "VIOLATION",
                action,
                previous,
                f"Nhãn hành động không tồn tại trong cấu hình: {action!r}.",
            )

        if action == self._last_accepted_action:
            return self._outcome(
                "INFO",
                action,
                previous,
                f"Hành động {action!r} vẫn đang diễn ra; không ghi nhận lặp.",
            )

        rule = self.config.states[self.state]
        allowed: dict[str, str] = rule.get("allowed", {})
        violations: dict[str, str | dict[str, str]] = rule.get("violations", {})

        if action in allowed:
            if self.state in {self.config.initial_state, self.config.completed_state}:
                self.cycle_id += 1
                self.completed_steps.clear()
            self.state = allowed[action]
            self.completed_steps.append(action)
            self._last_accepted_action = action
            message = f"Đúng quy trình: {self.config.actions[action]}. {self.instruction}"
            return self._outcome("PASS", action, previous, message)

        if action in violations:
            violation = violations[action]
            if isinstance(violation, str):
                return self._outcome("VIOLATION", action, previous, violation)

            # A physical regression (for example, an earbud being removed)
            # remains a VIOLATION, but the FSM must also follow the real scene.
            # Otherwise a later close_case could incorrectly PASS from the old
            # state even though an already-confirmed component is now missing.
            target_state = violation["target_state"]
            self.state = target_state
            self._restore_completed_steps(target_state)
            return self._outcome(
                "VIOLATION",
                action,
                previous,
                violation["message"],
            )

        expected = ", ".join(self.expected_actions) or "không có"
        return self._outcome(
            "VIOLATION",
            action,
            previous,
            f"Không được thực hiện {action!r} tại {self.state}; bước hợp lệ: {expected}.",
        )

    def _restore_completed_steps(self, target_state: str) -> None:
        """Keep progress consistent after a configured violation rollback."""

        completed: list[str] = []
        for step in self.config.workflow:
            completed.append(step.action)
            if step.state == target_state:
                break
        else:
            completed = []
        self.completed_steps = completed
        self._last_accepted_action = completed[-1] if completed else None

    def _outcome(
        self,
        result_type: str,
        action: str,
        previous_state: str,
        message: str,
    ) -> FsmOutcome:
        return FsmOutcome(
            type=result_type,
            action=action,
            previous_state=previous_state,
            state=self.state,
            message=message,
            cycle_id=self.cycle_id,
            completed_steps=tuple(self.completed_steps),
            is_complete=self.is_complete,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorkflowStep:
    state: str
    action: str
    label: str


@dataclass(frozen=True)
class AssemblyConfig:
    project: str
    initial_state: str
    completed_state: str
    idle_actions: frozenset[str]
    actions: dict[str, str]
    workflow: tuple[WorkflowStep, ...]
    states: dict[str, dict[str, Any]]

    @property
    def workflow_actions(self) -> tuple[str, ...]:
        return tuple(step.action for step in self.workflow)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Cấu hình FSM không hợp lệ: {message}")


def load_config(path: str | Path) -> AssemblyConfig:
    """Load and validate a JSON workflow configuration.

    JSON is intentionally used for the dependency-free MVP. The engine consumes
    the resulting dataclass, so a YAML loader can be added later without changing
    tracking logic.
    """

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    _require(raw.get("schema_version") == 1, "schema_version phải bằng 1")
    actions = raw.get("actions")
    states = raw.get("states")
    workflow_raw = raw.get("workflow")
    _require(isinstance(actions, dict) and bool(actions), "thiếu actions")
    _require(isinstance(states, dict) and bool(states), "thiếu states")
    _require(isinstance(workflow_raw, list) and bool(workflow_raw), "thiếu workflow")

    initial_state = raw.get("initial_state")
    completed_state = raw.get("completed_state")
    _require(initial_state in states, "initial_state không tồn tại trong states")
    _require(completed_state in states, "completed_state không tồn tại trong states")

    workflow: list[WorkflowStep] = []
    seen_actions: set[str] = set()
    for index, item in enumerate(workflow_raw):
        _require(isinstance(item, dict), f"workflow[{index}] phải là object")
        step = WorkflowStep(
            state=str(item.get("state", "")),
            action=str(item.get("action", "")),
            label=str(item.get("label", "")),
        )
        _require(step.state in states, f"state {step.state!r} trong workflow không tồn tại")
        _require(step.action in actions, f"action {step.action!r} trong workflow không tồn tại")
        _require(bool(step.label), f"workflow[{index}] thiếu label")
        _require(step.action not in seen_actions, f"action {step.action!r} bị lặp trong workflow")
        seen_actions.add(step.action)
        workflow.append(step)

    idle_actions = frozenset(str(item) for item in raw.get("idle_actions", []))
    _require(idle_actions.issubset(actions.keys() | {"uncertain"}), "idle_actions có nhãn lạ")

    for state_name, rule in states.items():
        _require(isinstance(rule, dict), f"rule của {state_name} phải là object")
        allowed = rule.get("allowed", {})
        violations = rule.get("violations", {})
        _require(isinstance(allowed, dict), f"allowed của {state_name} phải là object")
        _require(isinstance(violations, dict), f"violations của {state_name} phải là object")
        for action, target in allowed.items():
            _require(action in actions, f"{state_name} dùng action lạ {action!r}")
            _require(target in states, f"{state_name} trỏ tới state lạ {target!r}")
        for action, message in violations.items():
            _require(action in actions, f"{state_name} có violation action lạ {action!r}")
            _require(isinstance(message, str) and bool(message), f"violation {action!r} thiếu message")

    return AssemblyConfig(
        project=str(raw.get("project", config_path.stem)),
        initial_state=initial_state,
        completed_state=completed_state,
        idle_actions=idle_actions,
        actions={str(key): str(value) for key, value in actions.items()},
        workflow=tuple(workflow),
        states=states,
    )


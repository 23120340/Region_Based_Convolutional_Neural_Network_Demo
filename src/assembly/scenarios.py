"""Simulation scenarios derived from the selected workflow, not a product name."""

from .config import AssemblyConfig


SCENARIO_LABELS = {
    "correct": "Đúng quy trình",
    "missing_step": "Bỏ một bước",
    "wrong_order": "Sai thứ tự",
    "premature_finish": "Kết thúc sớm",
}


def build_scenarios(config: AssemblyConfig) -> dict[str, tuple[str, ...]]:
    actions = config.workflow_actions
    scenarios = {"correct": actions}
    if len(actions) >= 2:
        scenarios["wrong_order"] = (actions[1], actions[0])
    if len(actions) >= 3:
        scenarios["missing_step"] = actions[:-2] + actions[-1:]
        scenarios["premature_finish"] = (actions[0], actions[-1])
    return scenarios

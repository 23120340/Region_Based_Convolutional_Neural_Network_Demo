import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pen_assembly.config import load_config
from pen_assembly.fsm import ConfigurableAssemblyTracker
from pen_assembly.paths import DEFAULT_CONFIG
from pen_assembly.scenarios import SCENARIOS


class FsmTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = ConfigurableAssemblyTracker(load_config(DEFAULT_CONFIG))

    def run_actions(self, *actions: str):
        return [self.tracker.process(action) for action in actions]

    def test_correct_sequence_completes(self) -> None:
        outcomes = self.run_actions(*SCENARIOS["correct"])
        self.assertTrue(all(item.type == "PASS" for item in outcomes))
        self.assertTrue(self.tracker.is_complete)
        self.assertEqual(self.tracker.cycle_id, 1)

    def test_missing_spring_is_rejected_without_state_change(self) -> None:
        self.run_actions("pick_barrel", "insert_refill")
        outcome = self.tracker.process("screw_cap")
        self.assertEqual(outcome.type, "VIOLATION")
        self.assertIn("quên lắp lò xo", outcome.message)
        self.assertEqual(self.tracker.state, "S2_REFILL_INSERTED")

    def test_missing_refill_is_rejected(self) -> None:
        self.run_actions("pick_barrel")
        outcome = self.tracker.process("screw_cap")
        self.assertEqual(outcome.type, "VIOLATION")
        self.assertIn("chưa có ruột", outcome.message)

    def test_wrong_order_is_rejected(self) -> None:
        self.run_actions("pick_barrel")
        outcome = self.tracker.process("insert_spring")
        self.assertEqual(outcome.type, "VIOLATION")
        self.assertIn("Sai thứ tự", outcome.message)

    def test_premature_test_is_rejected(self) -> None:
        self.run_actions("pick_barrel", "insert_refill", "insert_spring")
        outcome = self.tracker.process("test_click")
        self.assertEqual(outcome.type, "VIOLATION")
        self.assertEqual(self.tracker.state, "S3_SPRING_INSERTED")

    def test_sustained_action_is_not_a_violation(self) -> None:
        self.tracker.process("pick_barrel")
        repeated = self.tracker.process("pick_barrel")
        self.assertEqual(repeated.type, "INFO")
        self.assertEqual(self.tracker.completed_steps, ["pick_barrel"])

    def test_new_pen_can_start_after_completion(self) -> None:
        self.run_actions(*SCENARIOS["correct"])
        outcome = self.tracker.process("pick_barrel")
        self.assertEqual(outcome.type, "PASS")
        self.assertEqual(outcome.cycle_id, 2)
        self.assertEqual(self.tracker.completed_steps, ["pick_barrel"])

    def test_reset_returns_to_initial_state(self) -> None:
        self.run_actions("pick_barrel", "insert_refill")
        outcome = self.tracker.reset()
        self.assertEqual(outcome.type, "RESET")
        self.assertEqual(self.tracker.state, "S0_IDLE")
        self.assertEqual(self.tracker.completed_steps, [])


if __name__ == "__main__":
    unittest.main()


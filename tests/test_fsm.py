import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assembly.config import load_config
from assembly.fsm import ConfigurableAssemblyTracker
from assembly.paths import DEFAULT_CONFIG


class FsmTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = ConfigurableAssemblyTracker(load_config(DEFAULT_CONFIG))

    def run_actions(self, *actions: str):
        return [self.tracker.process(action) for action in actions]

    def test_correct_sequence_completes(self) -> None:
        outcomes = self.run_actions("pick_case", "insert_earbud", "close_case")
        self.assertTrue(all(item.type == "PASS" for item in outcomes))
        self.assertTrue(self.tracker.is_complete)
        self.assertEqual(self.tracker.cycle_id, 1)

    def test_earbud_fsm_workflow_and_violation(self) -> None:
        # Violation test: insert_earbud before pick_case
        bad_outcome = self.tracker.process("insert_earbud")
        self.assertEqual(bad_outcome.type, "VIOLATION")
        self.assertIn("Chưa đặt hộp sạc", bad_outcome.message)

        # Correct workflow
        o1 = self.tracker.process("pick_case")
        self.assertEqual(o1.type, "PASS")
        self.assertEqual(self.tracker.state, "S1_CASE_READY")

        o2 = self.tracker.process("insert_earbud")
        self.assertEqual(o2.type, "PASS")
        self.assertEqual(self.tracker.state, "S2_EARBUD_INSERTED")

        o3 = self.tracker.process("close_case")
        self.assertEqual(o3.type, "PASS")
        self.assertEqual(self.tracker.state, "S3_COMPLETED")
        self.assertTrue(self.tracker.is_complete)

    def test_sustained_action_is_not_a_violation(self) -> None:
        self.tracker.process("pick_case")
        repeated = self.tracker.process("pick_case")
        self.assertEqual(repeated.type, "INFO")
        self.assertEqual(self.tracker.completed_steps, ["pick_case"])

    def test_new_case_can_start_after_completion(self) -> None:
        self.run_actions("pick_case", "insert_earbud", "close_case")
        outcome = self.tracker.process("pick_case")
        self.assertEqual(outcome.type, "PASS")
        self.assertEqual(outcome.cycle_id, 2)
        self.assertEqual(self.tracker.completed_steps, ["pick_case"])

    def test_reset_returns_to_initial_state(self) -> None:
        self.run_actions("pick_case", "insert_earbud")
        outcome = self.tracker.reset()
        self.assertEqual(outcome.type, "RESET")
        self.assertEqual(self.tracker.state, "S0_IDLE")
        self.assertEqual(self.tracker.completed_steps, [])


if __name__ == "__main__":
    unittest.main()




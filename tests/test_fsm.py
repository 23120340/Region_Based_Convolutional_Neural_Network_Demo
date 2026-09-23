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
        outcomes = self.run_actions(
            "open_case", "insert_earbud_1", "insert_earbud_2", "close_case"
        )
        self.assertTrue(all(item.type == "PASS" for item in outcomes))
        self.assertTrue(self.tracker.is_complete)
        self.assertEqual(self.tracker.cycle_id, 1)

    def test_earbud_fsm_workflow_and_violation(self) -> None:
        # Violation test: first insert before the open case baseline.
        bad_outcome = self.tracker.process("insert_earbud_1")
        self.assertEqual(bad_outcome.type, "VIOLATION")
        self.assertIn("chưa xác minh hộp mở", bad_outcome.message)

        # Correct workflow
        o1 = self.tracker.process("open_case")
        self.assertEqual(o1.type, "PASS")
        self.assertEqual(self.tracker.state, "S1_CASE_OPEN_EMPTY")

        o2 = self.tracker.process("insert_earbud_1")
        self.assertEqual(o2.type, "PASS")
        self.assertEqual(self.tracker.state, "S2_FIRST_EARBUD_INSERTED")

        o3 = self.tracker.process("insert_earbud_2")
        self.assertEqual(o3.type, "PASS")
        self.assertEqual(self.tracker.state, "S3_TWO_EARBUDS_INSERTED")

        o4 = self.tracker.process("close_case")
        self.assertEqual(o4.type, "PASS")
        self.assertEqual(self.tracker.state, "S4_COMPLETED")
        self.assertTrue(self.tracker.is_complete)

    def test_closing_after_only_one_earbud_is_rejected(self) -> None:
        self.run_actions("open_case", "insert_earbud_1")
        outcome = self.tracker.process("close_case")
        self.assertEqual(outcome.type, "VIOLATION")
        self.assertIn("mới xác minh 1/2 tai", outcome.message)
        self.assertEqual(self.tracker.state, "S2_FIRST_EARBUD_INSERTED")

    def test_wrong_earbud_side_is_violation_without_advancing(self) -> None:
        self.tracker.process("open_case")

        outcome = self.tracker.process("wrong_earbud_side")

        self.assertEqual(outcome.type, "VIOLATION")
        self.assertEqual(self.tracker.state, "S1_CASE_OPEN_EMPTY")
        self.assertEqual(self.tracker.completed_steps, ["open_case"])

    def test_removal_violation_rolls_back_to_physical_state(self) -> None:
        self.run_actions("open_case", "insert_earbud_1", "insert_earbud_2")

        removed_one = self.tracker.process("remove_earbud_to_one")
        self.assertEqual(removed_one.type, "VIOLATION")
        self.assertEqual(self.tracker.state, "S2_FIRST_EARBUD_INSERTED")
        self.assertEqual(
            self.tracker.completed_steps,
            ["open_case", "insert_earbud_1"],
        )

        removed_last = self.tracker.process("remove_earbud_to_zero")
        self.assertEqual(removed_last.type, "VIOLATION")
        self.assertEqual(self.tracker.state, "S1_CASE_OPEN_EMPTY")
        self.assertEqual(self.tracker.completed_steps, ["open_case"])

    def test_close_cannot_pass_after_removal_rollback(self) -> None:
        self.run_actions("open_case", "insert_earbud_1", "insert_earbud_2")
        self.tracker.process("remove_earbud_to_one")

        outcome = self.tracker.process("close_case")

        self.assertEqual(outcome.type, "VIOLATION")
        self.assertEqual(self.tracker.state, "S2_FIRST_EARBUD_INSERTED")

    def test_sustained_action_is_not_a_violation(self) -> None:
        self.tracker.process("open_case")
        repeated = self.tracker.process("open_case")
        self.assertEqual(repeated.type, "INFO")
        self.assertEqual(self.tracker.completed_steps, ["open_case"])

    def test_new_case_can_start_after_completion(self) -> None:
        self.run_actions("open_case", "insert_earbud_1", "insert_earbud_2", "close_case")
        outcome = self.tracker.process("open_case")
        self.assertEqual(outcome.type, "PASS")
        self.assertEqual(outcome.cycle_id, 2)
        self.assertEqual(self.tracker.completed_steps, ["open_case"])

    def test_reset_returns_to_initial_state(self) -> None:
        self.run_actions("open_case", "insert_earbud_1")
        outcome = self.tracker.reset()
        self.assertEqual(outcome.type, "RESET")
        self.assertEqual(self.tracker.state, "S0_IDLE")
        self.assertEqual(self.tracker.completed_steps, [])


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assembly.vision import (
    ComponentDwellGate,
    Detection,
    EarbudAssemblyGate,
    NormalizedZone,
)


class VisionLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.zone = NormalizedZone(0.25, 0.25, 0.75, 0.75)
        self.gate = ComponentDwellGate({"case": "open_case", "earbud": "insert_earbud_1"}, dwell_frames=3)

    def detection(self, label: str, box=(40, 40, 60, 60)) -> Detection:
        return Detection(label, label, 0.9, box)

    def test_expected_component_emits_after_dwell(self) -> None:
        self.assertIsNone(self.gate.update([self.detection("case")], ["open_case"], self.zone, 100, 100))
        self.assertIsNone(self.gate.update([self.detection("case")], ["open_case"], self.zone, 100, 100))
        result = self.gate.update([self.detection("case")], ["open_case"], self.zone, 100, 100)
        self.assertEqual(result.action, "open_case")
        self.assertTrue(result.is_expected)

    def test_wrong_component_is_reported_for_fsm_validation(self) -> None:
        result = None
        for _ in range(3):
            result = self.gate.update([self.detection("earbud")], ["open_case"], self.zone, 100, 100)
        self.assertIsNotNone(result)
        self.assertEqual(result.action, "insert_earbud_1")
        self.assertFalse(result.is_expected)

    def test_held_component_is_not_reemitted_after_acknowledgement(self) -> None:
        case = self.detection("case")
        for _ in range(3):
            self.gate.update([case], ["open_case"], self.zone, 100, 100)
        self.gate.acknowledge()
        results = [
            self.gate.update([case], ["insert_earbud_1"], self.zone, 100, 100)
            for _ in range(5)
        ]
        self.assertTrue(all(result is None for result in results))

    def test_new_component_emits_while_completed_part_remains_visible(self) -> None:
        case = self.detection("case")
        earbud = self.detection("earbud", (45, 45, 55, 55))
        for _ in range(3):
            self.gate.update([case], ["open_case"], self.zone, 100, 100)
        self.gate.acknowledge()
        result = None
        for _ in range(3):
            result = self.gate.update([case, earbud], ["insert_earbud_1"], self.zone, 100, 100) or result
        self.assertEqual(result.action, "insert_earbud_1")

    def test_component_outside_zone_resets_dwell(self) -> None:
        inside = self.detection("case")
        outside = self.detection("case", (0, 0, 10, 10))
        self.gate.update([inside], ["open_case"], self.zone, 100, 100)
        self.gate.update([inside], ["open_case"], self.zone, 100, 100)
        self.assertIsNone(self.gate.update([outside], ["open_case"], self.zone, 100, 100))
        self.assertIsNone(self.gate.current_suggestion)
        self.assertIsNone(self.gate.update([inside], ["open_case"], self.zone, 100, 100))

    def test_zone_rejects_invalid_coordinates(self) -> None:
        with self.assertRaises(ValueError):
            NormalizedZone(0.8, 0.2, 0.4, 0.9)


class EarbudGeometryGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.zone = NormalizedZone(0.05, 0.05, 0.95, 0.95)
        self.gate = EarbudAssemblyGate(
            open_case_label="open_case",
            closed_case_label="close_case",
            earbud_labels=["left_earbud", "right_earbud"],
            empty_slot_labels=["empty_left", "empty_right"],
            earbud_slot_pairs={
                "left_earbud": "empty_left",
                "right_earbud": "empty_right",
            },
            dwell_frames=2,
            containment_threshold=0.6,
        )
        self.case = self.detection("open_case", (10, 10, 90, 90))
        self.left_slot = self.detection("empty_left", (20, 40, 38, 68))
        self.right_slot = self.detection("empty_right", (62, 40, 80, 68))
        self.left_earbud = self.detection("left_earbud", (20, 40, 38, 68))
        self.right_earbud = self.detection("right_earbud", (62, 40, 80, 68), 0.88)

    @staticmethod
    def detection(
        label: str,
        box: tuple[int, int, int, int],
        confidence: float = 0.9,
    ) -> Detection:
        return Detection(label, label, confidence, box)

    def stable_update(self, detections, expected):
        result = None
        for _ in range(2):
            result = self.gate.update(detections, expected, self.zone, 100, 100) or result
        return result

    def test_full_two_earbud_geometry_sequence(self) -> None:
        opened = self.stable_update(
            [self.case, self.left_slot, self.right_slot], ["open_case"]
        )
        self.assertEqual(opened.action, "open_case")
        self.gate.acknowledge()

        first = self.stable_update(
            [self.case, self.left_earbud, self.right_slot], ["insert_earbud_1"]
        )
        self.assertEqual(first.action, "insert_earbud_1")
        self.assertEqual(self.gate.status.empty_slots, 1)
        self.assertEqual(self.gate.status.earbuds_inside, 1)
        self.gate.acknowledge()

        second = self.stable_update(
            [self.case, self.left_earbud, self.right_earbud], ["insert_earbud_2"]
        )
        self.assertEqual(second.action, "insert_earbud_2")
        self.assertEqual(self.gate.status.empty_slots, 0)
        self.assertEqual(self.gate.status.earbuds_inside, 2)

    def test_earbud_outside_case_never_counts_as_inserted(self) -> None:
        outside = self.detection("left_earbud", (92, 40, 99, 60))
        result = self.stable_update(
            [self.case, outside, self.right_slot], ["insert_earbud_1"]
        )
        self.assertIsNone(result)
        self.assertEqual(self.gate.status.earbuds_inside, 0)
        self.assertTrue(self.gate.status.is_error)
        self.assertIn("chưa nằm bên trong", self.gate.status.message)

    def test_removing_first_earbud_emits_violation_candidate(self) -> None:
        self.stable_update(
            [self.case, self.left_slot, self.right_slot], ["open_case"]
        )
        self.gate.acknowledge()
        self.stable_update(
            [self.case, self.left_earbud, self.right_slot], ["insert_earbud_1"]
        )
        self.gate.acknowledge()

        removed = self.stable_update(
            [self.case, self.left_slot, self.right_slot], ["insert_earbud_2"]
        )

        self.assertEqual(removed.action, "remove_earbud_to_zero")
        self.assertFalse(removed.is_expected)
        self.assertTrue(self.gate.status.is_error)
        self.assertIn("tháo ra", self.gate.status.message)

        self.gate.acknowledge(rearm=True)
        reinserted = self.stable_update(
            [self.case, self.left_earbud, self.right_slot], ["insert_earbud_1"]
        )
        self.assertEqual(reinserted.action, "insert_earbud_1")

    def test_removing_one_of_two_earbuds_emits_violation_candidate(self) -> None:
        removed = self.stable_update(
            [self.case, self.left_earbud, self.right_slot], ["close_case"]
        )

        self.assertEqual(removed.action, "remove_earbud_to_one")
        self.assertFalse(removed.is_expected)
        self.assertEqual(self.gate.status.empty_slots, 1)
        self.assertTrue(self.gate.status.is_error)

        self.gate.acknowledge(rearm=True)
        reinserted = self.stable_update(
            [self.case, self.left_earbud, self.right_earbud], ["insert_earbud_2"]
        )
        self.assertEqual(reinserted.action, "insert_earbud_2")

    def test_left_earbud_in_right_slot_is_a_violation(self) -> None:
        left_earbud_in_right_slot = self.detection(
            "left_earbud", (62, 40, 80, 68)
        )
        result = self.stable_update(
            [self.case, left_earbud_in_right_slot, self.left_slot],
            ["insert_earbud_1"],
        )
        self.assertEqual(result.action, "wrong_earbud_side")
        self.assertFalse(result.is_expected)
        self.assertTrue(self.gate.status.is_error)
        self.assertIn("left_earbud", self.gate.status.message)

    def test_right_earbud_in_left_slot_is_a_violation(self) -> None:
        right_earbud_in_left_slot = self.detection(
            "right_earbud", (20, 40, 38, 68)
        )
        result = self.stable_update(
            [self.case, right_earbud_in_left_slot, self.right_slot],
            ["insert_earbud_1"],
        )
        self.assertEqual(result.action, "wrong_earbud_side")
        self.assertFalse(result.is_expected)
        self.assertIn("right_earbud", self.gate.status.message)

    def test_removing_both_earbuds_one_by_one_emits_two_violations(self) -> None:
        self.stable_update(
            [self.case, self.left_slot, self.right_slot], ["open_case"]
        )
        self.gate.acknowledge()
        self.stable_update(
            [self.case, self.left_earbud, self.right_slot], ["insert_earbud_1"]
        )
        self.gate.acknowledge()
        self.stable_update(
            [self.case, self.left_earbud, self.right_earbud], ["insert_earbud_2"]
        )
        self.gate.acknowledge()

        removed_right = self.stable_update(
            [self.case, self.left_earbud, self.right_slot], ["close_case"]
        )
        self.assertEqual(removed_right.action, "remove_earbud_to_one")
        self.assertFalse(removed_right.is_expected)
        self.gate.acknowledge(rearm=True)

        removed_left = self.stable_update(
            [self.case, self.left_slot, self.right_slot], ["insert_earbud_2"]
        )
        self.assertEqual(removed_left.action, "remove_earbud_to_zero")
        self.assertFalse(removed_left.is_expected)
        self.assertTrue(self.gate.status.is_error)

    def test_missing_slots_do_not_pass_without_two_inside_earbuds(self) -> None:
        result = self.stable_update([self.case], ["insert_earbud_2"])
        self.assertIsNone(result)
        self.assertIn("mới xác minh 0/2", self.gate.status.message)

    def test_closing_early_emits_violation_candidate(self) -> None:
        closed = self.detection("close_case", (10, 10, 90, 90))
        result = self.stable_update([closed], ["insert_earbud_1"])
        self.assertEqual(result.action, "close_case")
        self.assertFalse(result.is_expected)
        self.assertTrue(self.gate.status.is_error)

    def test_violation_rearms_only_after_visual_state_changes(self) -> None:
        closed = self.detection("close_case", (10, 10, 90, 90))
        first = self.stable_update([closed], ["insert_earbud_1"])
        self.assertIsNotNone(first)
        self.gate.acknowledge(rearm=True)
        held = self.stable_update([closed], ["insert_earbud_1"])
        self.assertIsNone(held)
        self.stable_update(
            [self.case, self.left_slot, self.right_slot], ["insert_earbud_1"]
        )
        second = self.stable_update([closed], ["insert_earbud_1"])
        self.assertIsNotNone(second)


if __name__ == "__main__":
    unittest.main()

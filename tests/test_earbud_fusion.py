import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assembly.earbud_fusion import EarbudFusionEngine
from assembly.model_contract import Prediction
from assembly.vision import Detection


def detection(label: str, box: tuple[int, int, int, int], confidence: float = 0.9) -> Detection:
    return Detection(label, label, confidence, box)


CASE = detection("Earphone_Case", (0, 0, 200, 200))
SLOT_LEFT = detection("Empty_Slot", (30, 60, 80, 130))
SLOT_RIGHT = detection("Empty_Slot", (120, 60, 170, 130))
EARBUD_LEFT = detection("Left_Earbud", (35, 65, 75, 125))
EARBUD_LEFT_DUPLICATE = detection("Earbud", (36, 66, 74, 124), 0.8)
EARBUD_RIGHT = detection("Right_Earbud", (125, 65, 165, 125))
SLOT_LEFT_V2 = detection("empty_left", (30, 60, 80, 130))
SLOT_RIGHT_V2 = detection("empty_right", (120, 60, 170, 130))
EARBUD_LEFT_V2 = detection("left_earbud", (120, 65, 165, 125))


class EarbudFusionTests(unittest.TestCase):
    def stable_update(self, engine, detections, prediction=None):
        result = None
        for _ in range(engine.stable_frames):
            result = engine.update(detections, prediction) or result
        return result

    def test_two_insertions_require_geometry_and_action_evidence(self) -> None:
        engine = EarbudFusionEngine(stable_frames=2)
        placed = self.stable_update(engine, [CASE, SLOT_LEFT, SLOT_RIGHT])
        self.assertEqual(placed.action, "pick_case")
        self.assertIn("confirmed=0/2", engine.status_text)

        first = self.stable_update(
            engine,
            [CASE, EARBUD_LEFT, EARBUD_LEFT_DUPLICATE, SLOT_RIGHT],
            Prediction("insert_earbud", 0.92),
        )
        self.assertEqual(first.action, "insert_first_earbud")
        self.assertEqual(first.scene.earbuds_inside_case, 1)

        second = self.stable_update(
            engine,
            [CASE, EARBUD_LEFT, EARBUD_LEFT_DUPLICATE, EARBUD_RIGHT],
            Prediction("insert_earbud", 0.94),
        )
        self.assertEqual(second.action, "insert_second_earbud")
        self.assertEqual(engine.confirmed_insertions, 2)

        closed = engine.update(
            [CASE, EARBUD_LEFT, EARBUD_RIGHT],
            Prediction("close_case", 0.91),
        )
        self.assertEqual(closed.action, "close_case")
        self.assertIn("đủ hai", closed.reason)

    def test_geometry_alone_does_not_confirm_insertion(self) -> None:
        engine = EarbudFusionEngine(stable_frames=1)
        engine.update([CASE, SLOT_LEFT, SLOT_RIGHT])
        result = engine.update([CASE, EARBUD_LEFT, SLOT_RIGHT])
        self.assertIsNone(result)
        self.assertEqual(engine.confirmed_insertions, 0)

    def test_v2_requires_open_and_each_insertion_action(self) -> None:
        engine = EarbudFusionEngine(
            stable_frames=1,
            case_action="open_case",
            require_case_action=True,
        )
        self.assertIsNone(engine.update([CASE, SLOT_LEFT, SLOT_RIGHT]))
        opened = engine.update(
            [CASE, SLOT_LEFT, SLOT_RIGHT],
            Prediction("open_case", 0.94),
        )
        self.assertEqual(opened.action, "open_case")

        self.assertIsNone(
            engine.update(
                [CASE, EARBUD_LEFT, SLOT_RIGHT],
                Prediction("insert_second_earbud", 0.92),
            )
        )
        first = engine.update(
            [CASE, EARBUD_LEFT, SLOT_RIGHT],
            Prediction("insert_first_earbud", 0.92),
        )
        self.assertEqual(first.action, "insert_first_earbud")

        second = engine.update(
            [CASE, EARBUD_LEFT, EARBUD_RIGHT],
            Prediction("insert_second_earbud", 0.93),
        )
        self.assertEqual(second.action, "insert_second_earbud")

    def test_early_close_is_forwarded_for_fsm_violation(self) -> None:
        engine = EarbudFusionEngine(stable_frames=1)
        engine.update([CASE, SLOT_LEFT, SLOT_RIGHT])
        event = engine.update(
            [CASE, SLOT_LEFT, SLOT_RIGHT],
            Prediction("close_case", 0.95),
        )
        self.assertEqual(event.action, "close_case")
        self.assertIn("chưa xác nhận", event.reason)

    def test_reappearing_empty_slot_emits_removal_and_reduces_occupancy(self) -> None:
        engine = EarbudFusionEngine(stable_frames=1)
        engine.update([CASE, SLOT_LEFT, SLOT_RIGHT])
        engine.update(
            [CASE, EARBUD_LEFT, SLOT_RIGHT],
            Prediction("insert_earbud", 0.92),
        )
        engine.update(
            [CASE, EARBUD_LEFT, EARBUD_RIGHT],
            Prediction("insert_earbud", 0.94),
        )

        removed = engine.update([CASE, EARBUD_LEFT, SLOT_RIGHT])

        self.assertEqual(removed.action, "remove_earbud_to_one")
        self.assertEqual(engine.confirmed_insertions, 1)
        self.assertIn("giảm từ 2 xuống 1", removed.reason)

    def test_removing_only_inserted_earbud_returns_to_zero(self) -> None:
        engine = EarbudFusionEngine(stable_frames=1)
        engine.update([CASE, SLOT_LEFT, SLOT_RIGHT])
        engine.update(
            [CASE, EARBUD_LEFT, SLOT_RIGHT],
            Prediction("insert_earbud", 0.92),
        )

        removed = engine.update([CASE, SLOT_LEFT, SLOT_RIGHT])

        self.assertEqual(removed.action, "remove_earbud_to_zero")
        self.assertEqual(engine.confirmed_insertions, 0)

    def test_case_removal_resets_cycle(self) -> None:
        engine = EarbudFusionEngine(stable_frames=1)
        engine.update([CASE, SLOT_LEFT, SLOT_RIGHT])
        engine.update([], None)
        self.assertEqual(engine.confirmed_insertions, 0)
        placed_again = engine.update([CASE, SLOT_LEFT, SLOT_RIGHT])
        self.assertEqual(placed_again.action, "pick_case")

    def test_wrong_side_is_emitted_once_until_scene_changes(self) -> None:
        engine = EarbudFusionEngine(
            stable_frames=1,
            case_action="open_case",
            earbud_slot_pairs={
                "left_earbud": "empty_left",
                "right_earbud": "empty_right",
            },
        )
        engine.update([CASE, SLOT_LEFT_V2, SLOT_RIGHT_V2])

        wrong = engine.update([CASE, EARBUD_LEFT_V2, SLOT_LEFT_V2])
        held = engine.update([CASE, EARBUD_LEFT_V2, SLOT_LEFT_V2])

        self.assertEqual(wrong.action, "wrong_earbud_side")
        self.assertIn("nhầm khe", wrong.reason)
        self.assertIsNone(held)

    def test_remove_action_is_supporting_evidence_for_geometry_removal(self) -> None:
        engine = EarbudFusionEngine(stable_frames=1)
        engine.update([CASE, SLOT_LEFT, SLOT_RIGHT])
        engine.update(
            [CASE, EARBUD_LEFT, SLOT_RIGHT],
            Prediction("insert_earbud", 0.92),
        )

        removed = engine.update(
            [CASE, SLOT_LEFT, SLOT_RIGHT],
            Prediction("remove_earbud", 0.91),
        )

        self.assertEqual(removed.action, "remove_earbud_to_zero")
        self.assertIn("ViT-LSTM", removed.reason)


if __name__ == "__main__":
    unittest.main()

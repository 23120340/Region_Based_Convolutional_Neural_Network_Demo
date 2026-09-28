import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assembly.config import load_config
from assembly.fsm import ConfigurableAssemblyTracker
from assembly.model_contract import Prediction
from assembly.slot_fusion import PairedEarbudFusionEngine
from assembly.vision import Detection


def box(label, xyxy, confidence=0.95):
    return Detection(label, label, confidence, xyxy)


CASE = box("open_case", (0, 0, 200, 200))
CLOSED = box("close_case", (0, 0, 200, 200))
LEFT = box("empty_left", (30, 60, 80, 130))
RIGHT = box("empty_right", (120, 60, 170, 130))
LE = box("left_earbud", (35, 65, 75, 125))
RE = box("right_earbud", (125, 65, 165, 125))


class SlotFusionTests(unittest.TestCase):
    def setUp(self):
        self.engine = PairedEarbudFusionEngine(
            stable_frames=3, min_action_confidence=0.5,
            case_action="open_case", require_case_action=True,
            earbud_slot_pairs={"left_earbud": "empty_left", "right_earbud": "empty_right"},
        )
        self.tracker = ConfigurableAssemblyTracker(load_config(ROOT / "configs/earbud_v2_fsm_config.json"))
        self.last_outcome = None

    def feed(self, detections, action=None, confidence=0.95, count=3):
        events = []
        prediction = Prediction(action, confidence) if action else None
        for _ in range(count):
            event = self.engine.update(detections, prediction)
            if event:
                events.append(event)
                self.last_outcome = self.tracker.process(event.action)
        self.assertLessEqual(len(events), 1)
        return events[0] if events else None

    def opened(self):
        self.assertEqual(self.feed([CASE, LEFT, RIGHT], "open_case").action, "open_case")

    def both(self):
        self.opened()
        self.feed([CASE, LE, RIGHT], "insert_first_earbud")
        self.feed([CASE, LE, RE], "insert_second_earbud")
        self.assertEqual(self.engine.confirmed_insertions, 2)

    def test_matching_overlap_is_insertion_even_if_empty_box_still_visible(self):
        self.opened()
        first = self.feed([CASE, LEFT, RIGHT, RE], "insert_first_earbud")
        self.assertEqual(first.action, "insert_first_earbud")
        self.assertEqual(self.last_outcome.type, "PASS")
        self.assertIn("phải", first.reason)
        # Simultaneous empty and earbud predictions must not create a removal loop.
        self.assertIsNone(self.feed([CASE, LEFT, RIGHT, RE], "insert_first_earbud", count=6))
        self.assertEqual(self.engine.confirmed_insertions, 1)

    def test_remember_slot_when_empty_detection_disappears(self):
        self.opened()
        event = self.feed([CASE, LEFT, RE], "insert_first_earbud")
        self.assertEqual(event.action, "insert_first_earbud")

    def test_geometry_and_confidence_boundary(self):
        self.opened()
        self.assertIsNone(self.feed([CASE, LEFT, RE]))
        self.assertIsNone(self.feed([CASE, LEFT, RE], "insert_first_earbud", 0.5))
        self.assertIsNotNone(self.feed([CASE, LEFT, RE], "insert_first_earbud", 0.5001))

    def test_missing_slot_alone_does_not_insert(self):
        self.opened()
        self.assertIsNone(self.feed([CASE], "insert_first_earbud", count=10))
        self.assertEqual(self.engine.confirmed_insertions, 0)

    def test_earbud_in_case_but_not_over_slot_is_not_inserted(self):
        self.opened()
        floating = box("right_earbud", (95, 5, 140, 40))
        self.assertIsNone(self.feed([CASE, LEFT, floating], "insert_first_earbud"))

    def test_brief_overlap_and_detector_dropout_do_not_change_state(self):
        self.opened()
        self.assertIsNone(self.feed([CASE, LEFT, RE], "insert_first_earbud", count=2))
        self.feed([CASE, LEFT, RIGHT], "idle")
        self.assertEqual(self.engine.confirmed_insertions, 0)
        self.feed([CASE, LEFT, RE], "insert_first_earbud")
        self.assertIsNone(self.feed([CASE, LEFT], count=8))
        self.assertIsNone(self.feed([], count=8))
        self.assertEqual(self.engine.confirmed_insertions, 1)
        self.assertEqual(self.tracker.state, "S2_FIRST_EARBUD_INSERTED")

    def test_wrong_side_requires_actual_overlap_with_opposite_slot(self):
        self.opened()
        wrong = box("right_earbud", LE.box_xyxy)
        event = self.feed([CASE, RIGHT, wrong], "insert_first_earbud")
        self.assertEqual(event.action, "wrong_earbud_side")
        self.assertEqual(self.last_outcome.type, "VIOLATION")
        self.assertEqual(self.engine.confirmed_insertions, 0)
        self.assertIsNone(self.feed([CASE, RIGHT, wrong], "insert_first_earbud"))

    def test_correct_slot_visible_is_not_itself_wrong_side(self):
        self.opened()
        outside = box("right_earbud", (220, 60, 250, 130))
        self.assertIsNone(self.feed([CASE, LEFT, RIGHT, outside], "insert_first_earbud"))

    def test_two_overlapping_earbud_labels_cannot_fill_both_slots(self):
        self.opened()
        duplicate_wrong = box("left_earbud", RE.box_xyxy)
        self.feed([CASE, LEFT, RE, duplicate_wrong], "insert_first_earbud")
        self.assertEqual(self.engine.confirmed_insertions, 0)

    def test_removal_rolls_back_blocks_close_and_requires_reinsertion(self):
        self.both()
        self.assertIsNone(self.feed([CASE, LE, RIGHT], count=2))
        event = self.feed([CASE, LE, RIGHT], count=1)
        self.assertEqual(event.action, "remove_earbud_to_one")
        self.assertEqual(self.last_outcome.type, "VIOLATION")
        self.assertIn("phải", self.engine.instruction)
        self.assertEqual(self.tracker.state, "S2_FIRST_EARBUD_INSERTED")
        self.feed([CLOSED], "close_case")
        self.assertEqual(self.last_outcome.type, "VIOLATION")
        self.feed([CASE, LE, RIGHT], "idle")
        self.feed([CASE, LE, RE], "insert_second_earbud")
        self.assertEqual(self.last_outcome.type, "PASS")
        self.feed([CLOSED], "close_case")
        self.assertTrue(self.tracker.is_complete)

    def test_removing_last_earbud_returns_to_first_step(self):
        self.opened()
        self.feed([CASE, LEFT, RE], "insert_first_earbud")
        self.assertEqual(self.feed([CASE, LEFT, RIGHT]).action, "remove_earbud_to_zero")
        self.assertEqual(self.tracker.state, "S1_CASE_READY")
        self.assertEqual(self.tracker.completed_steps, ["open_case"])
        self.feed([CASE, LEFT, RE], "insert_first_earbud")
        self.assertEqual(self.last_outcome.type, "PASS")

    def test_simultaneous_removal_returns_to_zero(self):
        self.both()
        self.assertEqual(self.feed([CASE, LEFT, RIGHT]).action, "remove_earbud_to_zero")
        self.assertEqual(self.engine.confirmed_insertions, 0)
        self.assertIn("trái", self.engine.instruction)
        self.assertIn("phải", self.engine.instruction)

    def test_wrong_action_order_is_forwarded_without_counting(self):
        self.opened()
        event = self.feed([CASE, LEFT, RE], "insert_second_earbud")
        self.assertEqual(event.action, "insert_second_earbud")
        self.assertEqual(self.last_outcome.type, "VIOLATION")
        self.assertEqual(self.engine.confirmed_insertions, 0)

    def test_close_requires_yolo_closed_case_and_lstm(self):
        self.both()
        self.assertIsNone(self.feed([CASE, LE, RE], "close_case"))
        self.assertIsNone(self.feed([CLOSED], "idle"))
        self.assertIsNotNone(self.feed([CLOSED], "close_case"))
        self.assertTrue(self.tracker.is_complete)
        self.assertIsNone(self.feed([CLOSED], "idle"))
        self.assertIsNone(self.feed([CLOSED], "close_case"))

    def test_removal_after_completion_is_still_violation(self):
        self.both()
        self.feed([CLOSED], "close_case")
        self.feed([CASE, LE, RIGHT], "remove_earbud")
        self.assertEqual(self.last_outcome.type, "VIOLATION")
        self.assertFalse(self.tracker.is_complete)
        self.assertEqual(self.tracker.state, "S2_FIRST_EARBUD_INSERTED")

    def test_small_case_translation_uses_relative_anchors(self):
        self.opened()
        shifted = [box(d.label, tuple(v+20 for v in d.box_xyxy)) for d in [CASE, LEFT, RE]]
        event = self.feed(shifted, "insert_first_earbud")
        self.assertEqual(event.action, "insert_first_earbud")

    def test_low_confidence_detection_does_not_confirm(self):
        self.opened()
        weak = box("right_earbud", RE.box_xyxy, 0.3)
        self.assertIsNone(self.feed([CASE, LEFT, weak], "insert_first_earbud"))

    def test_reset_clears_anchors_and_confirmed_sides(self):
        self.both()
        self.engine.reset()
        self.assertEqual(self.engine.confirmed_insertions, 0)
        self.assertEqual(self.engine.slot_views, ())
        self.assertIsNone(self.feed([CASE, LE, RE], "insert_first_earbud"))

    def test_insertion_before_open_confirmation_is_violation(self):
        self.feed([CASE, LEFT, RIGHT])
        event = self.feed([CASE, LEFT, RE], "insert_first_earbud")
        self.assertEqual(event.action, "insert_first_earbud")
        self.assertEqual(self.last_outcome.type, "VIOLATION")
        self.assertEqual(self.engine.confirmed_insertions, 0)
        self.assertEqual(self.tracker.state, "S0_IDLE")

    def test_empty_calibration_tolerates_short_dropouts_without_simultaneous_runs(self):
        event = None
        for detections in ([CASE, LEFT], [CASE, RIGHT], [CASE, LEFT, RIGHT],
                           [CASE, LEFT], [CASE, RIGHT]):
            event = self.engine.update(detections, Prediction("open_case", 0.95))
        self.assertIsNotNone(event)
        self.assertEqual(event.action, "open_case")
        self.assertEqual(self.engine.calibration_progress, {"empty_left": 3, "empty_right": 3})
        self.tracker.process(event.action)
        self.assertEqual(self.feed([CASE, LEFT, RE], "insert_first_earbud").action, "insert_first_earbud")

    def test_one_empty_detection_does_not_calibrate(self):
        self.assertIsNone(self.feed([CASE, LEFT, RIGHT], "open_case", count=1))
        self.assertIsNone(self.feed([CASE], "open_case", count=8))
        self.assertEqual(self.engine.calibration_progress, {"empty_left": 0, "empty_right": 0})
        self.assertEqual(self.tracker.completed_steps, [])

    def test_stale_empty_calibration_cannot_authorize_open(self):
        self.feed([CASE, LEFT, RIGHT])
        self.feed([CASE], count=6)
        self.assertIsNone(self.feed([CASE], "open_case", count=6))
        self.assertIn("0/3", self.engine.instruction)
        self.assertEqual(self.tracker.state, "S0_IDLE")

    def test_missing_case_discards_initial_calibration(self):
        self.feed([CASE, LEFT, RIGHT])
        self.feed([], count=1)
        self.assertIsNone(self.feed([CASE], "open_case"))
        self.assertEqual(self.tracker.state, "S0_IDLE")

    def test_empty_boxes_under_an_earbud_cannot_calibrate(self):
        self.assertIsNone(self.feed([CASE, LEFT, RIGHT, RE], "open_case", count=10))
        self.assertEqual(self.engine.confirmed_insertions, 0)
        self.assertFalse(self.engine._case_registered)

    def test_profile_detection_threshold_matches_display_without_weakening_lstm(self):
        from assembly.camera_config import load_camera_config
        from assembly.project_config import load_project_config, build_fusion_engine
        project = load_project_config(ROOT / "configs/projects/earbud_v2.json", ROOT)
        engine = build_fusion_engine(project.fusion)
        camera = load_camera_config(project.camera_config)
        self.assertEqual(engine.min_detection_confidence, camera.confidence)
        self.assertEqual(engine.min_action_confidence, 0.5)
        weak = [box(d.label, d.box_xyxy, 0.4) for d in [CASE, LEFT, RIGHT]]
        for _ in range(3):
            event = engine.update(weak, Prediction("open_case", 0.5))
        self.assertIsNone(event)
        self.assertEqual(engine.update(weak, Prediction("open_case", 0.95)).action, "open_case")

    def test_invalid_calibration_window_is_rejected(self):
        with self.assertRaises(ValueError):
            PairedEarbudFusionEngine(stable_frames=3, calibration_window=2,
                earbud_slot_pairs={"left_earbud": "empty_left", "right_earbud": "empty_right"})

    def test_confident_model_alone_explains_missing_geometry(self):
        self.opened()
        self.assertIsNone(self.feed([CASE], "insert_first_earbud"))
        self.assertIn("chờ tai đúng khe", self.engine.instruction)
        self.assertEqual(self.tracker.state, "S1_CASE_READY")

    def test_close_waiting_explains_which_model_is_missing(self):
        self.both()
        self.assertIsNone(self.feed([CASE, LE, RE], "close_case"))
        self.assertIn("chờ YOLO close_case", self.engine.instruction)
        self.assertIsNone(self.feed([CLOSED], "idle"))
        self.assertIn("chờ LSTM: close_case", self.engine.instruction)
        self.assertFalse(self.tracker.is_complete)


if __name__ == "__main__":
    unittest.main()

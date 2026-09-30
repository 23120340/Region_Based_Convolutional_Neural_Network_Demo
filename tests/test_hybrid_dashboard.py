import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assembly.config import load_config
from assembly.fsm import ConfigurableAssemblyTracker
from assembly.hybrid_dashboard import draw_dashboard, confirmation_view, slot_coverage_text
from assembly.slot_fusion import SlotView
from assembly.vision import Detection
from assembly.model_contract import Prediction
from assembly.project_config import load_project_config, build_fusion_engine


class DashboardTests(unittest.TestCase):
    def test_coverage_distinguishes_missing_boxes_partial_overlap_and_zero(self):
        box = (0, 0, 100, 100)
        self.assertIn("thiếu bbox tai", slot_coverage_text(
            SlotView("empty_right", "empty", 0, 3, 0.8, False, box,
                     coverage_available=False, empty_confidence=0.9)))
        self.assertIn("thiếu bbox khe", slot_coverage_text(
            SlotView("empty_right", "unknown", 0, 0, 0.15, False, None,
                     coverage_available=False)))
        self.assertEqual(slot_coverage_text(
            SlotView("empty_right", "unknown", 0.27, 2, 0.15, False, box)), "Tai phủ khe 27%")
        self.assertEqual(slot_coverage_text(
            SlotView("empty_right", "empty", 0, 3, 0.8, False, box)), "Tai phủ khe 0%")
        self.assertIn("nắp đóng", slot_coverage_text(
            SlotView("empty_right", "closed", 0, 3, 0.3, False, box,
                     coverage_available=False)))

    def setUp(self):
        self.tracker = ConfigurableAssemblyTracker(load_config(ROOT / "configs/earbud_v2_fsm_config.json"))

    def test_model_recognition_does_not_confirm_a_step(self):
        view = confirmation_view(self.tracker, prediction=Prediction("open_case", 0.95), prediction_fresh=True)
        self.assertEqual(view.kind, "waiting")
        self.assertIn("CHƯA XÁC NHẬN", view.title)
        self.assertEqual(view.steps[0][1], "expected")
        self.assertNotIn("confirmed", [status for _, status in view.steps])

    def test_uncertain_stale_or_idle_prediction_is_not_confirmation(self):
        for prediction, fresh in [(Prediction("open_case", 0.5), True),
                                  (Prediction("open_case", 0.95), False),
                                  (Prediction("idle", 0.95), True)]:
            view = confirmation_view(self.tracker, prediction=prediction, prediction_fresh=fresh)
            self.assertEqual(view.kind, "waiting")
            self.assertNotIn("ĐÃ NHẬN DIỆN", view.title)

    def test_pass_is_persistent_while_waiting_for_next_action(self):
        result = self.tracker.process("open_case")
        view = confirmation_view(self.tracker, outcome=result, recent_outcomes=(result,))
        self.assertEqual(view.kind, "confirmed")
        self.assertIn("ĐÃ XÁC NHẬN", view.title)
        self.assertEqual([status for _, status in view.steps], ["confirmed", "expected", "pending", "pending"])
        # An INFO outcome and a new prediction cannot erase the accepted step.
        info = self.tracker.process("idle")
        later = confirmation_view(self.tracker, outcome=info, recent_outcomes=(result,),
                                  prediction=Prediction("insert_first_earbud", 0.95), prediction_fresh=True)
        self.assertEqual(later.title, view.title)

    def test_completion_requires_all_fsm_steps(self):
        history = [self.tracker.process(action) for action in
                   ("open_case", "insert_first_earbud", "insert_second_earbud", "close_case")]
        view = confirmation_view(self.tracker, outcome=history[-1], recent_outcomes=history)
        self.assertEqual(view.kind, "complete")
        self.assertIn("HOÀN TẤT", view.title)
        self.assertTrue(all(status == "confirmed" for _, status in view.steps))

    def test_removal_invalidates_completion_and_marks_rework(self):
        history = [self.tracker.process(action) for action in
                   ("open_case", "insert_first_earbud", "insert_second_earbud", "close_case")]
        history.append(self.tracker.process("remove_earbud_to_one"))
        view = confirmation_view(self.tracker, outcome=history[-1], recent_outcomes=history)
        self.assertEqual(view.kind, "violation")
        self.assertEqual([status for _, status in view.steps], ["confirmed", "confirmed", "rework", "rework"])
        history.append(self.tracker.process("insert_second_earbud"))
        restored = confirmation_view(self.tracker, outcome=history[-1], recent_outcomes=history)
        self.assertEqual(restored.kind, "confirmed")
        self.assertEqual(restored.steps[2][1], "confirmed")
        self.assertEqual(restored.steps[3][1], "rework")

    def test_reset_returns_to_unconfirmed_state(self):
        self.tracker.process("open_case")
        result = self.tracker.reset()
        view = confirmation_view(self.tracker, outcome=result, recent_outcomes=())
        self.assertEqual(view.kind, "waiting")
        self.assertEqual([status for _, status in view.steps], ["expected", "pending", "pending", "pending"])

    def test_out_of_order_violation_never_marks_step_confirmed(self):
        result = self.tracker.process("close_case")
        view = confirmation_view(self.tracker, outcome=result, recent_outcomes=(result,))
        self.assertEqual(view.kind, "violation")
        self.assertNotIn("confirmed", [status for _, status in view.steps])

    def test_render_keeps_inference_frame_unmodified_at_different_camera_sizes(self):
        project = load_project_config(ROOT / "configs/projects/earbud_v2.json", ROOT)
        fusion = build_fusion_engine(project.fusion)
        tracker = ConfigurableAssemblyTracker(load_config(project.fsm_config))
        fusion.update([Detection("open_case", "open_case", 0.95, (0, 0, 200, 200)),
                       Detection("empty_left", "empty_left", 0.9, (30, 60, 80, 130)),
                       Detection("empty_right", "empty_right", 0.8, (120, 60, 170, 130))])
        for shape in ((480, 640, 3), (720, 1280, 3), (1280, 720, 3)):
            frame = np.full(shape, 120, dtype=np.uint8)
            original = frame.copy()
            screen = draw_dashboard(
                frame, tracker=tracker, fusion=fusion,
                prediction=Prediction("insert_first_earbud", 0.5),
                prediction_fresh=True, action_threshold=0.5,
            )
            self.assertEqual(screen.shape, (900, 1440, 3))
            self.assertTrue(np.array_equal(frame, original))


if __name__ == "__main__":
    unittest.main()

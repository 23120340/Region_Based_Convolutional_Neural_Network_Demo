import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assembly.config import load_config
from assembly.fsm import ConfigurableAssemblyTracker, ObservationState
from assembly.model_contract import Prediction
from assembly.slot_fusion import PairedEarbudFusionEngine
from assembly.temporal_config import default_temporal_fusion_config
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
            temporal_config=default_temporal_fusion_config(),
            case_action="open_case", require_case_action=True,
            earbud_slot_pairs={"left_earbud": "empty_left", "right_earbud": "empty_right"},
        )
        self.tracker = ConfigurableAssemblyTracker(load_config(ROOT / "configs/earbud_v2_fsm_config.json"))
        self.now = 0.0
        self.outcomes = []

    def sample(self, detections, action=None, confidence=0.95, step=0.1):
        prediction = Prediction(action, confidence) if action else None
        event = self.engine.update(detections, prediction, timestamp_s=self.now)
        self.tracker.set_observation_state(self.engine.observation_state)
        if event:
            self.outcomes.append(self.tracker.process(event.action))
        self.now += step
        return event

    def feed_for(self, detections, seconds, action=None, confidence=0.95, step=0.1):
        events = []
        end = self.now + seconds
        while self.now <= end + 1e-9:
            event = self.sample(detections, action, confidence, step)
            if event:
                events.append(event)
        self.assertLessEqual(len(events), 1)
        return events[0] if events else None

    def opened(self):
        event = self.feed_for([CASE, LEFT, RIGHT], 0.4, "open_case")
        self.assertIsNotNone(event)
        self.assertEqual(event.action, "open_case")

    def insert(self, detections):
        event = self.feed_for(detections, 0.35, "insert_earbud")
        self.assertIsNotNone(event)
        return event

    def both(self):
        self.opened()
        self.assertEqual(self.insert([CASE, LE, RIGHT]).action, "insert_first_earbud")
        self.assertEqual(self.insert([CASE, LE, RE]).action, "insert_second_earbud")

    def test_closed_start_waits_without_violation(self):
        self.assertIsNone(self.feed_for([CLOSED], 1.0, "close_case"))
        self.assertEqual(self.engine.observation_state, ObservationState.WAIT_FOR_OPEN)
        self.assertEqual(self.tracker.state, "WAIT_FOR_OPEN")
        self.assertEqual(self.outcomes, [])

    def test_insert_uses_elapsed_time_and_generic_lstm_label(self):
        self.opened()
        self.assertIsNone(self.feed_for([CASE, LEFT, RIGHT, RE], 0.15, "insert_earbud"))
        event = self.feed_for([CASE, LEFT, RIGHT, RE], 0.15, "insert_earbud")
        self.assertEqual(event.action, "insert_first_earbud")
        self.assertEqual(self.outcomes[-1].type, "PASS")
        self.assertEqual(self.engine.confirmed_insertions, 1)

    def test_missing_boxes_are_unknown_and_never_remove(self):
        self.opened()
        self.insert([CASE, LEFT, RE])
        self.assertIsNone(self.feed_for([CASE], 2.0))
        self.assertEqual(self.engine.observation_state, ObservationState.UNKNOWN)
        self.assertEqual(self.engine.confirmed_insertions, 1)

    def test_empty_flicker_does_not_rollback(self):
        self.opened()
        self.insert([CASE, LEFT, RE])
        self.assertIsNone(self.feed_for([CASE, LEFT, RIGHT], 0.6))
        self.sample([CASE, LEFT, RE])
        self.assertEqual(self.engine.confirmed_insertions, 1)

    def test_removal_requires_clear_empty_for_point_eight_seconds(self):
        self.both()
        self.assertIsNone(self.feed_for([CASE, LE, RIGHT], 0.7))
        event = self.feed_for([CASE, LE, RIGHT], 0.2)
        self.assertEqual(event.action, "remove_earbud_to_one")
        self.assertEqual(self.outcomes[-1].type, "VIOLATION")
        self.assertEqual(self.tracker.state, "S2_FIRST_EARBUD_INSERTED")

    def test_wrong_side_uses_stricter_confidence_and_half_second(self):
        self.opened()
        weak_wrong = box("right_earbud", LE.box_xyxy, 0.56)
        self.assertIsNone(self.feed_for([CASE, LEFT, RIGHT, weak_wrong], 0.8, "insert_earbud"))
        strong_wrong = box("right_earbud", LE.box_xyxy, 0.75)
        self.assertIsNone(self.feed_for([CASE, LEFT, RIGHT, strong_wrong], 0.4, "insert_earbud"))
        event = self.feed_for([CASE, LEFT, RIGHT, strong_wrong], 0.2, "insert_earbud")
        self.assertEqual(event.action, "wrong_earbud_side")
        self.assertEqual(self.outcomes[-1].type, "VIOLATION")

    def test_close_requires_two_confirmed_earbuds_and_action(self):
        self.both()
        self.assertIsNone(self.feed_for([CLOSED], 0.5, "idle"))
        event = self.feed_for([CLOSED], 0.4, "close_case")
        self.assertEqual(event.action, "close_case")
        self.assertTrue(self.tracker.is_complete)
        # One-frame open/closed detector flicker must not re-emit close_case.
        self.sample([CASE], "idle")
        self.assertIsNone(self.feed_for([CLOSED], 0.4, "close_case"))

    def test_test_mode_open_baseline_is_info_not_pass(self):
        config = replace(default_temporal_fusion_config(), test_start_open=True)
        self.engine = PairedEarbudFusionEngine(
            temporal_config=config, case_action="open_case", require_case_action=True,
            earbud_slot_pairs={"left_earbud": "empty_left", "right_earbud": "empty_right"},
        )
        event = self.feed_for([CASE, LEFT, RIGHT], 0.4)
        self.assertEqual(event.action, "initialize_open_case")
        self.assertEqual(self.outcomes[-1].type, "INFO")
        self.assertEqual(self.tracker.state, "S1_CASE_READY")
        self.assertNotIn("open_case", self.tracker.completed_steps)

    def test_empty_threshold_is_strictly_greater_than_point_five(self):
        weak_left = box("empty_left", LEFT.box_xyxy, 0.5)
        self.assertIsNone(self.feed_for([CASE, weak_left, RIGHT], 0.6, "open_case"))
        self.assertEqual(self.tracker.state, "WAIT_FOR_OPEN")

    def test_same_case_translation_keeps_relative_anchors(self):
        self.opened()
        shifted = [box(item.label, tuple(value+20 for value in item.box_xyxy))
                   for item in (CASE, LEFT, RE)]
        event = self.feed_for(shifted, 0.35, "insert_earbud")
        self.assertEqual(event.action, "insert_first_earbud")


if __name__ == "__main__":
    unittest.main()

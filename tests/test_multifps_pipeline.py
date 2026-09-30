import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assembly.cadence import CumulativeDeadlineScheduler
from assembly.config import load_config
from assembly.fsm import ConfigurableAssemblyTracker
from assembly.model_contract import Prediction
from assembly.slot_fusion import PairedEarbudFusionEngine
from assembly.temporal_config import default_temporal_fusion_config
from assembly.vision import Detection


def detection(label, box):
    return Detection(label, label, 0.95, box)


CASE = detection("open_case", (0, 0, 200, 200))
CLOSED = detection("close_case", (0, 0, 200, 200))
LEFT = detection("empty_left", (30, 60, 80, 130))
RIGHT = detection("empty_right", (120, 60, 170, 130))
LE = detection("left_earbud", (35, 65, 75, 125))
RE = detection("right_earbud", (125, 65, 165, 125))


def scene(timestamp):
    if timestamp < 0.6:
        return [CASE, LEFT, RIGHT], "open_case"
    if timestamp < 1.1:
        return [CASE, LEFT, RE], "insert_earbud"
    if timestamp < 1.6:
        return [CASE, LE, RE], "insert_earbud"
    return [CLOSED], "close_case"


class MultiFpsPipelineTests(unittest.TestCase):
    def test_same_workflow_for_nonstandard_source_fps(self):
        for source_fps in (12.49, 15.0, 30.0008):
            with self.subTest(source_fps=source_fps):
                fusion = PairedEarbudFusionEngine(
                    temporal_config=default_temporal_fusion_config(),
                    case_action="open_case", require_case_action=True,
                    earbud_slot_pairs={"left_earbud": "empty_left", "right_earbud": "empty_right"},
                )
                tracker = ConfigurableAssemblyTracker(
                    load_config(ROOT / "configs/earbud_v2_fsm_config.json")
                )
                scheduler = CumulativeDeadlineScheduler(10.0)
                latest = None
                actions = []
                for index in range(round(source_fps * 2.2)):
                    timestamp = index / source_fps
                    detections, action = scene(timestamp)
                    if scheduler.due(timestamp):
                        latest = Prediction(action, 0.95)
                    event = fusion.update(detections, latest, timestamp_s=timestamp)
                    if event:
                        outcome = tracker.process(event.action)
                        actions.append((outcome.type, event.action))
                self.assertEqual(actions, [
                    ("PASS", "open_case"),
                    ("PASS", "insert_first_earbud"),
                    ("PASS", "insert_second_earbud"),
                    ("PASS", "close_case"),
                ])
                self.assertTrue(tracker.is_complete)


if __name__ == "__main__":
    unittest.main()

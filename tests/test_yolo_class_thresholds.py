"""Class filtering before boxes reach either the overlay or fusion."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from assembly.camera_config import load_camera_config
from assembly.yolo_world_detector import YoloWorldDetector


class YoloClassThresholdTests(unittest.TestCase):
    def test_empty_boundary_filters_actual_detector_output_by_name_not_class_id(self):
        detector = object.__new__(YoloWorldDetector)
        detector.config = load_camera_config(ROOT / "configs/camera_earbud_config.json")
        detector.device, detector.use_half, detector.open_vocabulary = "cpu", False, False
        result = SimpleNamespace(
            names={0: "Empty Left", 1: "empty-right", 2: "Left_Earbud"},
            boxes=SimpleNamespace(
                xyxy=torch.tensor([[0, 0, 10, 10]] * 4),
                conf=torch.tensor([0.49, 0.5, 0.5001, 0.36]),
                cls=torch.tensor([0, 1, 1, 2]),
            ),
        )
        detector.model = SimpleNamespace(predict=Mock(return_value=[result]))
        detections = detector.predict(None)
        self.assertEqual([d.label for d in detections], ["empty_right", "left_earbud"])
        self.assertGreater(detections[0].confidence, 0.5)
        self.assertEqual(detector.model.predict.call_args.kwargs["conf"], 0.35)


if __name__ == "__main__":
    unittest.main()

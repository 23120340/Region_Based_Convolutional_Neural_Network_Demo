import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assembly.camera_config import load_camera_config
from assembly.yolo_world_detector import _normalise_class_name


class CameraConfigTests(unittest.TestCase):
    def test_earbud_camera_config_loads_properly(self) -> None:
        config = load_camera_config(ROOT / "configs" / "camera_earbud_config.json")
        self.assertEqual(config.action_map["open_case"], "open_case")
        self.assertEqual(config.action_map["close_case"], "close_case")
        self.assertNotIn("left_earbud", config.action_map)
        self.assertNotIn("right_earbud", config.action_map)
        self.assertNotIn("empty_left", config.action_map)
        self.assertEqual(len(config.classes), 6)
        self.assertIsNotNone(config.earbud_geometry)
        self.assertEqual(
            config.earbud_geometry.empty_slot_labels,
            ("empty_left", "empty_right"),
        )
        self.assertEqual(
            dict(config.earbud_geometry.earbud_slot_pairs),
            {"left_earbud": "empty_left", "right_earbud": "empty_right"},
        )
        self.assertEqual(config.image_size, 512)
        self.assertEqual((config.capture_width, config.capture_height), (960, 540))
        self.assertEqual(config.capture_buffer_size, 1)
        self.assertTrue(config.half_precision)
        self.assertEqual(config.max_detections, 20)

    def test_detection_spatial_containment(self) -> None:
        from assembly.vision import Detection

        case = Detection(
            label="Case",
            prompt="case",
            confidence=0.9,
            box_xyxy=(100, 100, 300, 300),
        )
        earbud_inside = Detection(
            label="Earbud",
            prompt="earbud",
            confidence=0.85,
            box_xyxy=(150, 150, 200, 200),
        )
        earbud_outside = Detection(
            label="Earbud",
            prompt="earbud",
            confidence=0.85,
            box_xyxy=(400, 400, 450, 450),
        )

        self.assertEqual(case.area, 40000.0)
        self.assertEqual(earbud_inside.area, 2500.0)
        self.assertTrue(case.contains_point((150, 150)))
        self.assertFalse(case.contains_point((450, 450)))
        self.assertTrue(earbud_inside.is_inside(case, threshold=0.9))
        self.assertFalse(earbud_outside.is_inside(case, threshold=0.1))
        self.assertEqual(earbud_outside.intersection_area(case), 0.0)

    def test_closed_set_class_names_are_case_normalised(self) -> None:
        self.assertEqual(_normalise_class_name("Left_Earbud"), "left_earbud")
        self.assertEqual(_normalise_class_name("Right Earbud"), "right_earbud")


if __name__ == "__main__":
    unittest.main()

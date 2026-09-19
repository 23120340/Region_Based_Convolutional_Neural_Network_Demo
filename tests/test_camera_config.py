import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assembly.camera_config import load_camera_config


class CameraConfigTests(unittest.TestCase):
    def test_earbud_camera_config_loads_properly(self) -> None:
        config = load_camera_config(ROOT / "configs" / "camera_earbud_config.json")
        self.assertEqual(config.action_map["Earphone_Case"], "pick_case")
        self.assertEqual(config.action_map["Earbud"], "insert_earbud")
        self.assertEqual(config.action_map["Left_Earbud"], "insert_earbud")
        self.assertNotIn("Empty_Slot", config.action_map)
        self.assertEqual(len(config.classes), 6)

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


if __name__ == "__main__":
    unittest.main()


import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pen_assembly.camera_config import load_camera_config


class CameraConfigTests(unittest.TestCase):
    def test_camera_config_maps_components_to_fsm_actions(self) -> None:
        config = load_camera_config(ROOT / "configs" / "camera_config.json")
        self.assertEqual(config.action_map["barrel"], "pick_barrel")
        self.assertEqual(config.action_map["cap"], "screw_cap")
        self.assertNotIn("assembled_pen", config.action_map)
        self.assertEqual(len(config.prompts), 5)


if __name__ == "__main__":
    unittest.main()

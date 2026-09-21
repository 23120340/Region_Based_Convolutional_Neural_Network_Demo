import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assembly.camera_app import next_camera_index


class CameraSwitchTests(unittest.TestCase):
    def test_cycles_through_available_cameras(self) -> None:
        self.assertEqual(next_camera_index(0, [0, 2, 4]), 2)
        self.assertEqual(next_camera_index(2, [0, 2, 4]), 4)
        self.assertEqual(next_camera_index(4, [0, 2, 4]), 0)

    def test_selects_first_when_current_is_missing(self) -> None:
        self.assertEqual(next_camera_index(1, [0, 3]), 0)

    def test_keeps_current_when_no_camera_is_found(self) -> None:
        self.assertEqual(next_camera_index(2, []), 2)


if __name__ == "__main__":
    unittest.main()


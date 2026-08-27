import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pen_assembly.vision import ComponentDwellGate, Detection, NormalizedZone


class VisionLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.zone = NormalizedZone(0.25, 0.25, 0.75, 0.75)
        self.gate = ComponentDwellGate({"barrel": "pick_barrel", "spring": "insert_spring"}, dwell_frames=3)

    def detection(self, label: str, box=(40, 40, 60, 60)) -> Detection:
        return Detection(label, label, 0.9, box)

    def test_expected_component_emits_after_dwell(self) -> None:
        self.assertIsNone(self.gate.update([self.detection("barrel")], ["pick_barrel"], self.zone, 100, 100))
        self.assertIsNone(self.gate.update([self.detection("barrel")], ["pick_barrel"], self.zone, 100, 100))
        result = self.gate.update([self.detection("barrel")], ["pick_barrel"], self.zone, 100, 100)
        self.assertEqual(result.action, "pick_barrel")

    def test_wrong_component_does_not_emit(self) -> None:
        for _ in range(5):
            result = self.gate.update([self.detection("spring")], ["pick_barrel"], self.zone, 100, 100)
        self.assertIsNone(result)

    def test_component_outside_zone_resets_dwell(self) -> None:
        inside = self.detection("barrel")
        outside = self.detection("barrel", (0, 0, 10, 10))
        self.gate.update([inside], ["pick_barrel"], self.zone, 100, 100)
        self.gate.update([inside], ["pick_barrel"], self.zone, 100, 100)
        self.assertIsNone(self.gate.update([outside], ["pick_barrel"], self.zone, 100, 100))
        self.assertIsNone(self.gate.current_suggestion)
        self.assertIsNone(self.gate.update([inside], ["pick_barrel"], self.zone, 100, 100))

    def test_zone_rejects_invalid_coordinates(self) -> None:
        with self.assertRaises(ValueError):
            NormalizedZone(0.8, 0.2, 0.4, 0.9)


if __name__ == "__main__":
    unittest.main()

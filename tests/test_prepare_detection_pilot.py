import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_detection_pilot.py"
SPEC = importlib.util.spec_from_file_location("prepare_detection_pilot", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PrepareDetectionPilotTests(unittest.TestCase):
    def test_selection_prefers_quality_then_visual_diversity(self) -> None:
        candidates = [
            MODULE.ImageCandidate(Path("best.jpg"), 100.0, 120.0, 0, 1.0),
            MODULE.ImageCandidate(Path("similar.jpg"), 90.0, 120.0, 0, 0.9),
            MODULE.ImageCandidate(Path("different.jpg"), 80.0, 120.0, (1 << 64) - 1, 0.8),
        ]
        selected = MODULE.select_diverse_candidates(candidates, 2)
        self.assertEqual([item.path.name for item in selected], ["best.jpg", "different.jpg"])

    def test_zero_count_returns_empty_selection(self) -> None:
        candidate = MODULE.ImageCandidate(Path("one.jpg"), 1.0, 1.0, 0, 1.0)
        self.assertEqual(MODULE.select_diverse_candidates([candidate], 0), [])


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pen_assembly.model_contract import Prediction
from pen_assembly.smoother import TemporalDebouncer


class DebouncerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.debouncer = TemporalDebouncer(window_size=5, min_votes=3, min_confidence=0.7)

    def test_emits_after_three_votes(self) -> None:
        self.assertIsNone(self.debouncer.update(Prediction("pick_barrel", 0.9)))
        self.assertIsNone(self.debouncer.update(Prediction("idle", 0.9)))
        self.assertIsNone(self.debouncer.update(Prediction("pick_barrel", 0.8)))
        stable = self.debouncer.update(Prediction("pick_barrel", 0.95))
        self.assertIsNotNone(stable)
        self.assertEqual(stable.action, "pick_barrel")

    def test_sustained_action_only_emits_once(self) -> None:
        results = [
            self.debouncer.update(Prediction("pick_barrel", 0.95))
            for _ in range(10)
        ]
        self.assertEqual(sum(item is not None for item in results), 1)

    def test_low_confidence_is_not_counted(self) -> None:
        results = [
            self.debouncer.update(Prediction("pick_barrel", 0.2))
            for _ in range(8)
        ]
        self.assertTrue(all(item is None for item in results))

    def test_idle_rearms_same_action(self) -> None:
        for _ in range(5):
            first = self.debouncer.update(Prediction("pick_barrel", 0.9))
        self.assertIsNone(first)
        for _ in range(5):
            self.debouncer.update(Prediction("idle", 0.9))
        emitted = None
        for _ in range(5):
            emitted = self.debouncer.update(Prediction("pick_barrel", 0.9)) or emitted
        self.assertIsNotNone(emitted)


if __name__ == "__main__":
    unittest.main()


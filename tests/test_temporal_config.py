import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assembly.temporal_config import load_temporal_fusion_config


class TemporalConfigTests(unittest.TestCase):
    def test_tracked_thresholds_match_contract(self):
        config = load_temporal_fusion_config(ROOT / "configs/earbud_temporal_thresholds.json")
        self.assertEqual(config.scheduler.target_embedding_fps, 10.0)
        self.assertGreaterEqual(config.duration.insert_s, 0.2)
        self.assertLessEqual(config.duration.insert_s, 0.3)
        self.assertEqual(config.duration.wrong_side_s, 0.5)
        self.assertEqual(config.duration.removal_s, 0.8)
        self.assertEqual(config.confidence.occupied_earbud_min, 0.5)
        self.assertGreaterEqual(config.confidence.wrong_side_earbud_min, 0.55)
        self.assertFalse(config.test_start_open)

    def test_invalid_confidence_is_rejected(self):
        raw = json.loads((ROOT / "configs/earbud_temporal_thresholds.json").read_text(encoding="utf-8"))
        raw["confidence"]["display_min"] = 1.1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_temporal_fusion_config(path)


if __name__ == "__main__":
    unittest.main()

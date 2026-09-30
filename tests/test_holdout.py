import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assembly.dataset_split import validate_detection_capture_manifest
from assembly.holdout import load_holdout_registry


class HoldoutTests(unittest.TestCase):
    def test_known_regression_video_id_is_rejected(self):
        registry = load_holdout_registry(ROOT / "configs/holdout_regression.json")
        with self.assertRaisesRegex(ValueError, "never enter training"):
            registry.reject_video_ids(["insert_test_1"])

    def test_unrelated_video_is_allowed(self):
        registry = load_holdout_registry(ROOT / "configs/holdout_regression.json")
        registry.reject_video_ids(["session04_correct_01"])


class DetectionSplitTests(unittest.TestCase):
    def write_manifest(self, path, rows):
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["image_id", "video_id", "session_id", "split"])
            writer.writeheader()
            writer.writerows(rows)

    def test_session_level_split_passes(self):
        rows = [
            {"image_id": "a", "video_id": "v1", "session_id": "s1", "split": "train"},
            {"image_id": "b", "video_id": "v2", "session_id": "s2", "split": "val"},
            {"image_id": "c", "video_id": "v3", "session_id": "s3", "split": "test"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            self.write_manifest(path, rows)
            self.assertEqual(validate_detection_capture_manifest(path), {"train": 1, "val": 1, "test": 1})

    def test_same_session_across_splits_is_rejected(self):
        rows = [
            {"image_id": "a", "video_id": "v1", "session_id": "s1", "split": "train"},
            {"image_id": "b", "video_id": "v2", "session_id": "s1", "split": "val"},
            {"image_id": "c", "video_id": "v3", "session_id": "s3", "split": "test"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            self.write_manifest(path, rows)
            with self.assertRaisesRegex(ValueError, "sessions leak"):
                validate_detection_capture_manifest(path)


if __name__ == "__main__":
    unittest.main()

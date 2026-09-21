import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assembly.action_dataset import CachedActionWindowDataset


class ActionDatasetTests(unittest.TestCase):
    def test_short_segment_is_padded_to_sequence_length(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features_dir = root / "features"
            features_dir.mkdir()
            np.save(features_dir / "video01.npy", np.arange(40, dtype=np.float32).reshape(10, 4))
            (features_dir / "video01.json").write_text(
                json.dumps({"sample_fps": 5.0, "feature_frames": 10}), encoding="utf-8"
            )
            annotations = root / "annotations.csv"
            with annotations.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["video_id", "split", "start_time_s", "end_time_s", "action_name"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "video_id": "video01",
                        "split": "train",
                        "start_time_s": "0.0",
                        "end_time_s": "1.0",
                        "action_name": "idle",
                    }
                )
            dataset = CachedActionWindowDataset(
                features_dir,
                annotations,
                "train",
                {"idle": 0},
                sequence_length=8,
                stride=2,
                expected_embedding_dim=4,
            )
            features, label = dataset[0]
            self.assertEqual(tuple(features.shape), (8, 4))
            self.assertEqual(label.item(), 0)
            self.assertTrue(np.array_equal(features[-1].numpy(), features[-2].numpy()))

    def test_rejects_same_session_in_multiple_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features_dir = root / "features"
            features_dir.mkdir()
            for video_id in ("clip_a", "clip_b"):
                np.save(features_dir / f"{video_id}.npy", np.ones((5, 3), dtype=np.float32))
                (features_dir / f"{video_id}.json").write_text(
                    json.dumps({"sample_fps": 2.0, "feature_frames": 5}), encoding="utf-8"
                )
            annotations = root / "annotations.csv"
            annotations.write_text(
                "video_id,person_id,session_id,split,start_time_s,end_time_s,action_name\n"
                "clip_a,p01,s01,train,0,1,idle\n"
                "clip_b,p01,s01,val,0,1,idle\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "person/session"):
                CachedActionWindowDataset(
                    features_dir=features_dir,
                    annotations_csv=annotations,
                    split="train",
                    action_to_id={"idle": 0},
                    sequence_length=4,
                    stride=2,
                    expected_embedding_dim=3,
                )


if __name__ == "__main__":
    unittest.main()


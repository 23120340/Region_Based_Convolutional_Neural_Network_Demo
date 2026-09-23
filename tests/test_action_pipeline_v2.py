"""End-to-end CLI contract test using synthetic embeddings, not accuracy data."""
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class ActionPipelineV2Tests(unittest.TestCase):
    def run_script(self, name, *args, success=True):
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / name), *map(str, args)],
                                cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=60)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_split_verify_train_and_evaluate_six_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads((ROOT / "configs/action_earbud_v2_config.json").read_text(encoding="utf-8"))
            config["temporal"].update(hidden_dim=4, num_layers=1, head_dim=4, dropout=0.0)
            config["training"].update(epochs=1, batch_size=32)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            embedding_dim = config["spatial"]["embedding_dim"]
            features = root / "features"
            features.mkdir()
            annotations = root / "raw.csv"
            fields = ["video_id", "person_id", "session_id", "split", "start_time_s", "end_time_s", "action_name"]
            rows = []
            rng = np.random.default_rng(5)
            for group in range(3):
                video_id = f"per1_{group:02d}_test"
                np.save(features / f"{video_id}.npy", rng.normal(size=(70, embedding_dim)).astype("float32"))
                meta = dict(feature_frames=70, embedding_dim=embedding_dim, sample_fps=10, source_fps=10,
                            source_frames=70, backbone=config["spatial"]["backbone"])
                (features / f"{video_id}.json").write_text(json.dumps(meta), encoding="utf-8")
                for label_id, label in enumerate(config["actions"]):
                    rows.append([video_id, "per1", f"{group:02d}", "train", label_id, label_id+1, label])
            with annotations.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(fields)
                writer.writerows(rows)
            split = root / "split.csv"
            before = annotations.read_bytes()
            self.run_script("split_annotations.py", "--annotations", annotations, "--output", split,
                            "--config", config_path, "--mode", "group-ratio")
            self.assertEqual(annotations.read_bytes(), before)
            shared = ["--annotations", split, "--features-dir", features, "--config", config_path]
            self.run_script("verify_pipeline.py", *shared)
            output = root / "model"
            self.run_script("train_action_model.py", *shared, "--output", output, "--device", "cpu")
            report_path = root / "evaluation.json"
            result = self.run_script("evaluate_action_model.py", *shared, "--checkpoint", output / "best.pt",
                                     "--split", "test", "--device", "cpu", "--output", report_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["actions"], config["actions"])
            self.assertEqual(np.asarray(report["confusion_matrix"]).shape, (6, 6))
            self.assertIn("Confusion matrix", result.stdout)
            self.assertIn("insert_second_earbud", result.stdout)
            # Deliberately overlapping labels must fail before model training.
            split.write_text(split.read_text(encoding="utf-8") + ",".join(map(str, rows[0])) + "\n", encoding="utf-8")
            failed = self.run_script("verify_pipeline.py", *shared, success=False)
            self.assertIn("chồng thời gian", failed.stdout)

    def test_missing_annotation_is_reported_without_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_script("verify_pipeline.py", "--annotations", Path(directory) / "missing.csv", success=False)
        self.assertIn("ERROR:", result.stdout)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()

"""Exercise runtime -> fusion -> FSM -> JSONL without a camera/model download."""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from assembly.model_contract import Prediction
from assembly.vision import Detection


def d(label, coords):
    return Detection(label, label, 0.95, coords)


class HybridRuntimeTests(unittest.TestCase):
    def test_camera_listing_does_not_load_models_or_start_runtime(self):
        spec = importlib.util.spec_from_file_location("hybrid_list_under_test", ROOT / "scripts/run_hybrid.py")
        app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app)
        with patch.object(sys, "argv", ["run_hybrid.py", "--list-cameras", "--max-camera-index", "8"]), \
             patch.object(app, "discover_camera_indices", return_value=[0, 3]) as discover, \
             patch.object(app, "YoloWorldDetector") as detector, \
             patch.object(app, "ViTLstmActionRecognizer") as recognizer, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(app.main(), 0)
        discover.assert_called_once_with(app.cv2, 8)
        detector.assert_not_called()
        recognizer.assert_not_called()
        self.assertIn("0, 3", output.getvalue())

    def test_gui_receives_persistent_confirmations_and_rolling_context_then_reset_clears_history(self):
        from assembly.hybrid_dashboard import confirmation_view
        spec = importlib.util.spec_from_file_location("hybrid_confirmation_under_test", ROOT / "scripts/run_hybrid.py")
        app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app)
        case = d("open_case", (0, 0, 200, 200))
        closed = d("close_case", case.box_xyxy)
        left = d("empty_left", (30, 60, 80, 130))
        right = d("empty_right", (120, 60, 170, 130))
        le = d("left_earbud", (35, 65, 75, 125))
        re = d("right_earbud", (125, 65, 165, 125))
        sequence = ([([case, left, right], "open_case")] * 4
                    + [([case, left, re], "insert_first_earbud")] * 3
                    + [([case, le, re], "insert_second_earbud")] * 3
                    + [([closed], "close_case")] * 3
                    + [([case, left, right], "open_case")] * 3)

        class Capture:
            index = -1
            def isOpened(self): return True
            def get(self, key): return 10
            def read(self):
                self.index += 1
                return (True, np.zeros((200, 200, 3), np.uint8)) if self.index < len(sequence) else (False, None)
            def release(self): pass

        capture = Capture()
        detector = Mock()
        detector.model.names = {i: name for i, name in enumerate(
            ["open_case", "close_case", "left_earbud", "right_earbud", "empty_left", "empty_right"])}
        detector.predict.side_effect = lambda frame: sequence[capture.index][0]
        recognizer = Mock()
        recognizer.device = SimpleNamespace(type="cpu")
        recognizer.encode_frame.return_value = np.zeros(384)
        recognizer.predict_embeddings.side_effect = lambda frames: Prediction(sequence[capture.index][1], 0.95)
        snapshots = []

        def dashboard(frame, **kwargs):
            snapshots.append((confirmation_view(kwargs["tracker"], outcome=kwargs["outcome"],
                recent_outcomes=kwargs["recent_outcomes"]), kwargs["embedding_count"], kwargs["recent_outcomes"]))
            return frame

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads((ROOT / "configs/action_earbud_pilot_config.json").read_text(encoding="utf-8"))
            config["temporal"]["sequence_length"] = 4
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            checkpoint, video, report = root / "best.pt", root / "test.mp4", root / "events.jsonl"
            checkpoint.touch()
            video.touch()
            args = ["run_hybrid.py", "--source", str(video), "--action-config", str(config_path),
                    "--action-model", str(checkpoint), "--yolo-model", str(checkpoint),
                    "--event-log", str(report), "--no-mirror"]
            with patch.object(sys, "argv", args), \
                 patch.object(app, "YoloWorldDetector", return_value=detector), \
                 patch.object(app, "ViTLstmActionRecognizer", return_value=recognizer), \
                 patch.object(app.cv2, "VideoCapture", return_value=capture), \
                 patch.object(app.cv2, "namedWindow"), patch.object(app.cv2, "resizeWindow"), \
                 patch.object(app.cv2, "imshow"), patch.object(app.cv2, "destroyAllWindows"), \
                 patch.object(app.cv2, "waitKey", side_effect=[0]*12 + [ord("r")] + [0]*3), \
                 patch.object(app, "draw_dashboard", side_effect=dashboard), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(app.main(), 0)
            records = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]
        self.assertIn("sample_fps=10", output.getvalue())
        self.assertEqual([(r["type"], r["action"]) for r in records], [
            ("PASS", "open_case"), ("PASS", "insert_first_earbud"),
            ("PASS", "insert_second_earbud"), ("PASS", "close_case"), ("RESET", "reset")])
        self.assertEqual(snapshots[3][0].kind, "confirmed")
        self.assertEqual(snapshots[6][0].steps[1][1], "confirmed")
        self.assertEqual(snapshots[9][0].steps[2][1], "confirmed")
        self.assertEqual(snapshots[12][0].kind, "complete")
        self.assertTrue(all(count == 4 for _, count, _ in snapshots[3:13]))
        self.assertEqual(len(snapshots[12][2]), 4)
        self.assertEqual(snapshots[13][2], ())
        self.assertEqual(snapshots[13][0].kind, "waiting")
        self.assertEqual(snapshots[13][1], 1)

    def test_recovery_sequence_is_logged_and_no_gui_opens_in_headless_mode(self):
        spec = importlib.util.spec_from_file_location("hybrid_runtime_under_test", ROOT / "scripts/run_hybrid.py")
        app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app)
        case = d("open_case", (0, 0, 200, 200))
        closed = d("close_case", case.box_xyxy)
        left = d("empty_left", (30, 60, 80, 130))
        right = d("empty_right", (120, 60, 170, 130))
        le = d("left_earbud", (35, 65, 75, 125))
        re = d("right_earbud", (125, 65, 165, 125))
        sequence = [
            ([case, left, right], "open_case"),
            ([case, left, re], "insert_first_earbud"),
            ([case, le, re], "insert_second_earbud"),
            ([case, le, right], "idle"),
            ([closed], "close_case"),
            ([case, le, re], "insert_second_earbud"),
            ([closed], "close_case"),
        ]

        class Capture:
            index = -1
            released = False
            def isOpened(self): return True
            def get(self, key): return 10
            def read(self):
                self.index += 1
                return (True, np.zeros((200, 200, 3), np.uint8)) if self.index < 21 else (False, None)
            def release(self): self.released = True

        capture = Capture()
        detector = Mock()
        detector.model.names = {i: x for i, x in enumerate(
            ["open_case", "close_case", "left_earbud", "right_earbud", "empty_left", "empty_right"])}
        detector.predict.side_effect = lambda frame: sequence[capture.index // 3][0]
        recognizer = Mock()
        recognizer.device = SimpleNamespace(type="cpu")
        recognizer.encode_frame.return_value = np.zeros(384)
        recognizer.predict_embeddings.side_effect = lambda frames: Prediction(sequence[capture.index // 3][1], 0.95)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads((ROOT / "configs/action_earbud_pilot_config.json").read_text(encoding="utf-8"))
            config["temporal"]["sequence_length"] = 1
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            checkpoint, video, report = root / "best.pt", root / "test.mp4", root / "events.jsonl"
            checkpoint.touch()
            video.touch()
            args = ["run_hybrid.py", "--project", str(ROOT / "configs/projects/earbud_v2.json"),
                    "--source", str(video), "--action-config", str(config_path),
                    "--action-model", str(checkpoint), "--yolo-model", str(checkpoint),
                    "--event-log", str(report), "--yolo-every", "1", "--headless"]
            with patch.object(sys, "argv", args), \
                 patch.object(app, "YoloWorldDetector", return_value=detector), \
                 patch.object(app, "ViTLstmActionRecognizer", return_value=recognizer), \
                 patch.object(app.cv2, "VideoCapture", return_value=capture), \
                 patch.object(app.cv2, "imshow") as imshow, \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(app.main(), 0)
                imshow.assert_not_called()
            records = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]
        self.assertIn("action > 0.5", output.getvalue())
        self.assertTrue(capture.released)
        self.assertEqual([(r["type"], r["action"]) for r in records], [
            ("PASS", "open_case"), ("PASS", "insert_first_earbud"), ("PASS", "insert_second_earbud"),
            ("VIOLATION", "remove_earbud_to_one"), ("VIOLATION", "close_case"),
            ("PASS", "insert_second_earbud"), ("PASS", "close_case"),
        ])
        self.assertTrue(records[-1]["is_complete"])


if __name__ == "__main__":
    unittest.main()

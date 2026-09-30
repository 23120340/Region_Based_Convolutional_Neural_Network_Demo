"""Exercise runtime -> temporal fusion -> FSM -> JSONL without hardware/downloads."""
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
    def test_generic_insert_recovery_sequence_is_logged_headlessly(self):
        spec = importlib.util.spec_from_file_location("hybrid_runtime_under_test", ROOT / "scripts/run_hybrid.py")
        app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app)
        case = d("open_case", (0, 0, 200, 200))
        closed = d("close_case", case.box_xyxy)
        left = d("empty_left", (30, 60, 80, 130))
        right = d("empty_right", (120, 60, 170, 130))
        le = d("left_earbud", (35, 65, 75, 125))
        re = d("right_earbud", (125, 65, 165, 125))

        def scene(index):
            if index <= 4:
                return [case, left, right], "open_case"
            if index <= 8:
                return [case, left, re], "insert_earbud"
            if index <= 12:
                return [case, le, re], "insert_earbud"
            if index <= 15:
                return [case, le], "idle"  # occlusion/missing box -> UNKNOWN only
            if index <= 25:
                return [case, le, right], "idle"  # verified removal after 0.8 s
            if index <= 29:
                return [case, le, re], "insert_earbud"
            return [closed], "close_case"

        class Capture:
            index = -1
            released = False

            def isOpened(self): return True
            def get(self, key): return 10
            def read(self):
                self.index += 1
                return (True, np.zeros((200, 200, 3), np.uint8)) if self.index < 35 else (False, None)
            def release(self): self.released = True

        capture = Capture()
        detector = Mock()
        detector.model.names = {i: value for i, value in enumerate(
            ["open_case", "close_case", "left_earbud", "right_earbud", "empty_left", "empty_right"])}
        detector.predict.side_effect = lambda frame: scene(capture.index)[0]
        recognizer = Mock()
        recognizer.device = SimpleNamespace(type="cpu")
        recognizer.encode_frame.return_value = np.zeros(384)
        recognizer.predict_embeddings.side_effect = lambda frames: Prediction(scene(capture.index)[1], 0.95)

        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            config = json.loads((ROOT / "configs/action_earbud_generic_config.json").read_text(encoding="utf-8"))
            config["temporal"]["sequence_length"] = 1
            config_path = temp / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            checkpoint, video, report = temp / "best.pt", temp / "test.mp4", temp / "events.jsonl"
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

        self.assertIn("sample_fps=10", output.getvalue())
        self.assertTrue(capture.released)
        self.assertEqual([(record["type"], record["action"]) for record in records], [
            ("PASS", "open_case"),
            ("PASS", "insert_first_earbud"),
            ("PASS", "insert_second_earbud"),
            ("VIOLATION", "remove_earbud_to_one"),
            ("PASS", "insert_second_earbud"),
            ("PASS", "close_case"),
        ])
        self.assertTrue(records[-1]["is_complete"])


if __name__ == "__main__":
    unittest.main()

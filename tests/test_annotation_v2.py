import contextlib
import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import annotate_actions as annotator
from assembly.action_config import load_action_model_config


class AnnotationV2Tests(unittest.TestCase):
    def test_seek_updates_display_and_two_insertions_save_distinct_intervals(self):
        fake_cv = MagicMock()
        fake_cv.CAP_PROP_FPS, fake_cv.CAP_PROP_FRAME_COUNT, fake_cv.CAP_PROP_POS_FRAMES = 1, 2, 3
        capture = fake_cv.VideoCapture.return_value
        cursor = [0]

        def get_value(prop):
            return {1: 10, 2: 100, 3: cursor[0]}[prop]

        def read_frame():
            if cursor[0] >= 100:
                return False, None
            frame = np.full((2, 2, 3), cursor[0], dtype=np.uint8)
            cursor[0] += 1
            return True, frame

        capture.get.side_effect = get_value
        capture.read.side_effect = read_frame
        capture.set.side_effect = lambda prop, value: cursor.__setitem__(0, int(value))
        keys = list("sfe.sfeq")
        fake_cv.waitKey.side_effect = [ord(key) for key in keys]
        actions = load_action_model_config(ROOT / "configs/action_earbud_v2_config.json").actions
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "annotations.csv"
            with patch.dict(sys.modules, {"cv2": fake_cv}), patch("builtins.input", side_effect=["2", "3"]), contextlib.redirect_stdout(io.StringIO()):
                count = annotator.annotate_video(Path("per1_02_demo.mp4"), actions, set(), output, True)
            with output.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
        self.assertEqual(count, 2)
        self.assertEqual([r["action_name"] for r in rows], ["insert_first_earbud", "insert_second_earbud"])
        self.assertEqual([(r["start_time_s"], r["end_time_s"]) for r in rows], [("0.0000", "5.1000"), ("5.1000", "10.0000")])
        displays = [int(call.args[1][0, 0, 0]) for call in fake_cv.imshow.call_args_list]
        self.assertIn(50, displays)
        self.assertIn(99, displays)
        capture.release.assert_called_once()

    def test_resume_rejects_old_generic_labels_without_modifying_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "per1_02_test.mp4").touch()
            output = base / "old.csv"
            content = ",".join(annotator.ANNOTATIONS_COLUMNS) + "\nper1_02_test,per1,02,train,0,1,insert_earbud\n"
            output.write_text(content, encoding="utf-8")
            argv = ["annotate_actions.py", "--videos-dir", str(base), "--output", str(output),
                    "--config", str(ROOT / "configs/action_earbud_v2_config.json"), "--resume"]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(SystemExit, "insert_earbud"):
                    annotator.main()
            self.assertEqual(output.read_text(encoding="utf-8"), content)


if __name__ == "__main__":
    unittest.main()

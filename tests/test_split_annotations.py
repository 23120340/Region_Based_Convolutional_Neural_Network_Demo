import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from split_annotations import _split_group_ratio, _split_video_ratio, _validate_assignment


def row(person: str, session: str, video: str) -> dict[str, str]:
    return {
        "video_id": video,
        "person_id": person,
        "session_id": session,
        "split": "train",
        "start_time_s": "0",
        "end_time_s": "1",
        "action_name": "idle",
    }


class SplitAnnotationsTests(unittest.TestCase):
    def test_group_ratio_keeps_person_session_in_one_split(self) -> None:
        rows = [
            row("p01", "s01", "a1"),
            row("p01", "s01", "a2"),
            row("p01", "s02", "b1"),
            row("p02", "s01", "c1"),
            row("p03", "s01", "d1"),
        ]
        result = _split_group_ratio(rows, 0.7, 0.15, 0.15, seed=42)
        _validate_assignment(result)
        self.assertEqual({item["split"] for item in result}, {"train", "val", "test"})
        first_group_splits = {
            item["split"]
            for item in result
            if (item["person_id"], item["session_id"]) == ("p01", "s01")
        }
        self.assertEqual(len(first_group_splits), 1)

    def test_group_ratio_requires_three_independent_groups(self) -> None:
        rows = [row("p01", "s01", "a"), row("p01", "s02", "b")]
        with self.assertRaisesRegex(ValueError, "ít nhất 3"):
            _split_group_ratio(rows, 0.7, 0.15, 0.15, seed=42)

    def test_pilot_17_videos_keeps_segments_together_and_requires_opt_in(self) -> None:
        rows = [row("p01", "02", f"clip{i}") for i in range(17) for _ in range(2)]
        result = _split_video_ratio(rows, 0.7, 0.15, 0.15, seed=42)
        counts = [len({r["video_id"] for r in result if r["split"] == split})
                  for split in ("train", "val", "test")]
        self.assertEqual(counts, [12, 3, 2])
        with self.assertRaisesRegex(ValueError, "person/session"):
            _validate_assignment(result)
        _validate_assignment(result, allow_same_session=True)
        result[0]["split"] = "test" if result[0]["split"] != "test" else "train"
        with self.assertRaisesRegex(ValueError, "video"):
            _validate_assignment(result, allow_same_session=True)


if __name__ == "__main__":
    unittest.main()

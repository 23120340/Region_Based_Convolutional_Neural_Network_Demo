"""Session/video-level split validation for extracted YOLO frames."""
from __future__ import annotations

import csv
from pathlib import Path

REQUIRED_COLUMNS = {"image_id", "video_id", "session_id", "split"}
VALID_SPLITS = {"train", "val", "test"}


def validate_detection_capture_manifest(path: str | Path) -> dict[str, int]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"capture manifest missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("capture manifest is empty")
    image_ids: set[str] = set()
    session_splits: dict[str, set[str]] = {}
    video_splits: dict[str, set[str]] = {}
    counts = {split: 0 for split in VALID_SPLITS}
    for index, row in enumerate(rows, start=2):
        image_id = row["image_id"].strip()
        video_id = row["video_id"].strip()
        session_id = row["session_id"].strip()
        split = row["split"].strip()
        if not image_id or not video_id or not session_id or split not in VALID_SPLITS:
            raise ValueError(f"invalid capture manifest row {index}")
        if image_id in image_ids:
            raise ValueError(f"duplicate image_id: {image_id}")
        image_ids.add(image_id)
        session_splits.setdefault(session_id, set()).add(split)
        video_splits.setdefault(video_id, set()).add(split)
        counts[split] += 1
    leaking_sessions = sorted(key for key, values in session_splits.items() if len(values) > 1)
    leaking_videos = sorted(key for key, values in video_splits.items() if len(values) > 1)
    if leaking_sessions:
        raise ValueError(f"sessions leak across splits: {leaking_sessions}")
    if leaking_videos:
        raise ValueError(f"videos leak across splits: {leaking_videos}")
    missing_splits = sorted(split for split, count in counts.items() if count == 0)
    if missing_splits:
        raise ValueError(f"empty splits: {missing_splits}")
    return counts

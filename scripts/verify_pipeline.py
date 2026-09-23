"""Validate temporal annotations, split boundaries, and cached ViT features."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from assembly.action_config import load_action_model_config
from assembly.paths import DEFAULT_ACTION_ANNOTATIONS, DEFAULT_ACTION_CONFIG, DEFAULT_FEATURE_CACHE


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra annotation, split và feature trước khi train")
    parser.add_argument("--config", type=Path, default=DEFAULT_ACTION_CONFIG)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ACTION_ANNOTATIONS)
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--allow-same-session", action="store_true", help="Pilot: cho phép cùng session, vẫn cấm cùng video ở nhiều split")
    args = parser.parse_args()
    try:
        config = load_action_model_config(args.config)
        with args.annotations.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            required = {"video_id", "person_id", "session_id", "split", "start_time_s", "end_time_s", "action_name"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"Annotation thiếu cột: {sorted(missing)}")
            rows = list(reader)
        if not rows:
            raise ValueError("Annotation rỗng")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: {error}")
        return 1

    print(f"Config: {config.spatial.backbone} | labels={list(config.actions)}")
    errors = []
    intervals = defaultdict(list)
    for index, row in enumerate(rows, start=2):
        if any(not row.get(key, "").strip() for key in required):
            errors.append(f"Dòng {index}: thiếu giá trị bắt buộc")
            continue
        if row["action_name"] not in config.actions:
            errors.append(f"Dòng {index}: nhãn không thuộc config: {row['action_name']}")
        if row["split"] not in {"train", "val", "test"}:
            errors.append(f"Dòng {index}: split không hợp lệ: {row['split']}")
        try:
            start, end = float(row["start_time_s"]), float(row["end_time_s"])
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                raise ValueError()
            intervals[row["video_id"]].append((start, end))
        except ValueError:
            errors.append(f"Dòng {index}: cần 0 <= start_time_s < end_time_s hữu hạn")
    for video_id, segments in intervals.items():
        segments.sort()
        if any(right[0] < left[1] for left, right in zip(segments, segments[1:])):
            errors.append(f"{video_id}: các đoạn trùng/chồng thời gian")

    for split in ("train", "val", "test"):
        counts = Counter(row["action_name"] for row in rows if row["split"] == split)
        print(f"{split}: {dict(counts)}")
        missing = set(config.actions) - set(counts)
        if missing:
            errors.append(f"Split {split!r} thiếu nhãn: {sorted(missing)}")

    try:
        import numpy as np
        for video_id in sorted(intervals):
            metadata = json.loads((args.features_dir / f"{video_id}.json").read_text(encoding="utf-8"))
            array = np.load(args.features_dir / f"{video_id}.npy", allow_pickle=False)
            expected_shape = (int(metadata["feature_frames"]), config.spatial.embedding_dim)
            if array.shape != expected_shape or len(array) == 0 or not np.isfinite(array).all():
                errors.append(f"{video_id}: feature phải có shape {expected_shape}, hữu hạn và không rỗng")
            if metadata.get("backbone") != config.spatial.backbone:
                errors.append(f"{video_id}: backbone cache không khớp config; trích xuất lại")
            source_fps = float(metadata["source_fps"])
            sample_fps = float(metadata["sample_fps"])
            if not (source_fps > 0 and sample_fps > 0 and math.isfinite(source_fps) and math.isfinite(sample_fps)):
                errors.append(f"{video_id}: FPS metadata không hợp lệ")
                continue
            if not math.isclose(sample_fps, min(config.spatial.sample_fps, source_fps), abs_tol=1e-6):
                errors.append(f"{video_id}: sample_fps không khớp config")
            duration = int(metadata["source_frames"]) / source_fps
            if any(end > duration + 0.001 for _, end in intervals[video_id]):
                errors.append(f"{video_id}: annotation vượt thời lượng video")
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(f"Lỗi cache: {error}")

    if not errors:
        from assembly.action_dataset import CachedActionWindowDataset
        try:
            for split in ("train", "val", "test"):
                dataset = CachedActionWindowDataset(
                    args.features_dir, args.annotations, split, config.action_to_id,
                    config.temporal.sequence_length, config.temporal.window_stride,
                    config.spatial.embedding_dim, allow_same_session=args.allow_same_session,
                )
                print(f"{split}: {len(dataset)} temporal windows")
        except (ValueError, OSError, KeyError) as error:
            errors.append(str(error))

    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        return 1
    if args.allow_same_session:
        print("PILOT: các split có thể cùng session; cần session mới để đánh giá tổng quát.")
    print("OK: annotation, feature và split hợp lệ.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

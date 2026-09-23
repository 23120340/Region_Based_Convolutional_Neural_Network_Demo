"""Split temporal annotations without leaking a recording session across sets.

The recommended mode is ``group-ratio``. Its split unit is
``(person_id, session_id)`` rather than an annotation row or a video. This
keeps every video recorded by the same person in the same session together.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

from assembly.paths import DEFAULT_ACTION_ANNOTATIONS
from assembly.action_config import load_action_model_config


COLUMNS = [
    "video_id",
    "person_id",
    "session_id",
    "split",
    "start_time_s",
    "end_time_s",
    "action_name",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = set(COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"File annotation thiếu cột: {sorted(missing)}")
        return list(reader)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _validate_ratios(train: float, val: float, test: float) -> None:
    ratios = (train, val, test)
    if any(value <= 0 for value in ratios):
        raise ValueError("Tỷ lệ train/val/test đều phải lớn hơn 0")
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-6):
        raise ValueError("Tổng tỷ lệ train + val + test phải bằng 1")


def _group_key(row: dict[str, str]) -> tuple[str, str]:
    person = row["person_id"].strip()
    session = row["session_id"].strip()
    if not person or not session:
        raise ValueError("person_id và session_id không được để trống")
    return person, session


def _split_group_ratio(
    rows: list[dict[str, str]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> list[dict[str, str]]:
    """Assign whole person/session groups to train, validation, or test."""

    _validate_ratios(train_ratio, val_ratio, test_ratio)
    groups = sorted({_group_key(row) for row in rows})
    if len(groups) < 3:
        raise ValueError(
            "Cần ít nhất 3 nhóm person/session độc lập để tạo đủ train, val và test. "
            f"Hiện chỉ có {len(groups)} nhóm: {groups}"
        )

    random.Random(seed).shuffle(groups)
    count = len(groups)
    val_count = max(1, round(count * val_ratio))
    test_count = max(1, round(count * test_ratio))
    while val_count + test_count >= count:
        if val_count >= test_count and val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            raise ValueError("Không thể tạo ba split không rỗng từ số group hiện tại")
    train_count = count - val_count - test_count

    train_groups = set(groups[:train_count])
    val_groups = set(groups[train_count : train_count + val_count])
    test_groups = set(groups[train_count + val_count :])
    assignment = {
        **{group: "train" for group in train_groups},
        **{group: "val" for group in val_groups},
        **{group: "test" for group in test_groups},
    }
    for row in rows:
        row["split"] = assignment[_group_key(row)]

    print(f"  seed={seed}, groups={count}")
    for split_name, selected in (
        ("train", train_groups),
        ("val", val_groups),
        ("test", test_groups),
    ):
        print(f"  {split_name:5s}: {sorted(selected)}")
    return rows


def _split_manual_video(
    rows: list[dict[str, str]],
    val_videos: list[str],
    test_videos: list[str],
) -> list[dict[str, str]]:
    val_set = set(val_videos)
    test_set = set(test_videos)
    overlap = val_set & test_set
    if overlap:
        raise ValueError(f"Video xuất hiện đồng thời trong val và test: {sorted(overlap)}")
    for row in rows:
        video_id = row["video_id"].strip()
        row["split"] = "test" if video_id in test_set else "val" if video_id in val_set else "train"
    return rows


def _split_video_ratio(rows, train_ratio, val_ratio, test_ratio, seed):
    """Pilot only: hold out whole videos while permitting a shared session."""
    _validate_ratios(train_ratio, val_ratio, test_ratio)
    video_ids = sorted({row["video_id"].strip() for row in rows})
    if len(video_ids) < 3:
        raise ValueError("Cần ít nhất 3 video để tạo train/val/test")
    random.Random(seed).shuffle(video_ids)
    ratios = (train_ratio, val_ratio, test_ratio)
    raw_counts = [len(video_ids) * ratio for ratio in ratios]
    counts = [math.floor(count) for count in raw_counts]
    order = sorted(range(3), key=lambda i: raw_counts[i] - counts[i], reverse=True)
    for i in order[:len(video_ids) - sum(counts)]:
        counts[i] += 1
    for i in range(3):
        if counts[i] == 0:
            donor = max(range(3), key=counts.__getitem__)
            counts[donor] -= 1
            counts[i] += 1
    val_start = counts[0]
    test_start = counts[0] + counts[1]
    return _split_manual_video(rows, video_ids[val_start:test_start], video_ids[test_start:])


def _validate_assignment(rows: list[dict[str, str]], allow_same_session: bool = False) -> None:
    expected_splits = {"train", "val", "test"}
    actual_splits = {row["split"] for row in rows}
    missing_splits = expected_splits - actual_splits
    if missing_splits:
        raise ValueError(f"Split bị rỗng: {sorted(missing_splits)}")

    group_splits: dict[tuple[str, str], set[str]] = {}
    video_splits: dict[str, set[str]] = {}
    for row in rows:
        group_splits.setdefault(_group_key(row), set()).add(row["split"])
        video_splits.setdefault(row["video_id"].strip(), set()).add(row["split"])
    leaking_groups = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    leaking_videos = sorted(video for video, splits in video_splits.items() if len(splits) > 1)
    if leaking_groups and not allow_same_session:
        raise ValueError(f"Rò rỉ person/session giữa các split: {leaking_groups}")
    if leaking_videos:
        raise ValueError(f"Rò rỉ video giữa các split: {leaking_videos}")


def _print_summary(rows: list[dict[str, str]], actions=None) -> None:
    print("\n  === Tóm tắt phân chia ===")
    all_actions = set(actions) if actions is not None else {row["action_name"] for row in rows}
    for split_name in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split_name]
        counts = Counter(row["action_name"] for row in split_rows)
        groups = {_group_key(row) for row in split_rows}
        videos = {row["video_id"] for row in split_rows}
        print(f"  {split_name:5s}: {len(videos)} video, {len(split_rows)} đoạn, {len(groups)} group | {dict(counts)}")
        missing_actions = sorted(all_actions - set(counts))
        if missing_actions:
            print(f"    WARNING thiếu nhãn: {missing_actions}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chia annotation thành train/val/test theo person/session"
    )
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ACTION_ANNOTATIONS)
    parser.add_argument("--output", type=Path, help="CSV sau khi chia; bỏ trống để cập nhật file input")
    parser.add_argument("--config", type=Path, help="Kiểm tra nhãn theo action config trước khi chia")
    parser.add_argument(
        "--mode",
        choices=("group-ratio", "video-ratio", "ratio", "manual-video"),
        default="group-ratio",
        help="group-ratio là chế độ khuyến nghị để tránh data leakage",
    )
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-videos", nargs="*", default=[])
    parser.add_argument("--test-videos", nargs="*", default=[])
    parser.add_argument("--allow-same-session", action="store_true", help="Chỉ dùng cho pilot chia theo video trong cùng session")
    args = parser.parse_args()

    if not args.annotations.is_file():
        raise SystemExit(f"Không tìm thấy annotation: {args.annotations}")
    try:
        rows = _read_csv(args.annotations)
        if not rows:
            raise ValueError("File annotation rỗng")
        config = load_action_model_config(args.config) if args.config else None
        if config:
            unknown = sorted({row["action_name"] for row in rows} - set(config.actions))
            if unknown:
                raise ValueError(f"Nhãn không thuộc config: {unknown}; cần annotation lại")
        if args.mode == "group-ratio":
            if args.allow_same_session:
                raise ValueError("group-ratio không dùng --allow-same-session")
            rows = _split_group_ratio(rows, args.train, args.val, args.test, args.seed)
        elif args.mode in ("video-ratio", "ratio"):
            if not args.allow_same_session:
                raise ValueError("Pilot video-ratio cần --allow-same-session; dùng group-ratio cho split theo session")
            rows = _split_video_ratio(rows, args.train, args.val, args.test, args.seed)
        else:
            rows = _split_manual_video(rows, args.val_videos, args.test_videos)
        _validate_assignment(rows, allow_same_session=args.allow_same_session)
    except ValueError as error:
        raise SystemExit(f"Không thể chia dataset: {error}") from error

    if args.allow_same_session:
        print("PILOT: các tập có thể cùng người/session; kết quả không đo khả năng tổng quát sang session mới.")
    _print_summary(rows, config.actions if config else None)
    output = args.output or args.annotations
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output, rows)
    print(f"\nĐã ghi cột split: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

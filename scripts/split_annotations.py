"""split_annotations.py – Phân chia annotations.csv thành train/val/test.

Script này đọc file annotations.csv (được tạo bởi annotate_actions.py)
và tự động gán cột 'split' theo chiến lược Leave-One-Video-Out hoặc
tỷ lệ cố định. Với dữ liệu nhỏ (4 video), khuyên dùng leave-one-out.

Cách dùng
---------
  # Chế độ mặc định: leave-one-video-out (mỗi video luân phiên làm val/test)
  python scripts/split_annotations.py

  # Chế độ tỷ lệ cố định (70/15/15) theo người hoặc theo video
  python scripts/split_annotations.py --mode ratio --train 0.7 --val 0.15 --test 0.15

  # Chỉ định rõ video nào làm val và test
  python scripts/split_annotations.py --mode manual \\
      --val-videos per1_01_correct_20260917_164921_002 \\
      --test-videos per1_01_correct_20260917_165027_004
"""

from __future__ import annotations

import argparse
import csv
import sys
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

COLUMNS = [
    "video_id", "person_id", "session_id", "split",
    "start_time_s", "end_time_s", "action_name",
]


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _split_leave_one_out(rows: list[dict]) -> list[dict]:
    """Với N video, luân phiên: 1 video làm val, 1 làm test, còn lại làm train.
    Phù hợp nhất khi có ≤ 6 video."""
    video_ids = list(dict.fromkeys(row["video_id"] for row in rows))  # giữ thứ tự
    n = len(video_ids)
    if n < 2:
        print("  WARN: Chỉ có 1 video duy nhất — gán toàn bộ là train.")
        for row in rows:
            row["split"] = "train"
        return rows
    if n == 2:
        val_vid = video_ids[-1]
        test_vid = video_ids[-1]
    else:
        val_vid = video_ids[-2]
        test_vid = video_ids[-1]

    print(f"  Leave-one-out: val={val_vid}, test={test_vid}")
    print(f"  Train: {[v for v in video_ids if v not in {val_vid, test_vid}]}")

    assignment = {vid: "train" for vid in video_ids}
    assignment[val_vid] = "val"
    assignment[test_vid] = "test"

    for row in rows:
        row["split"] = assignment[row["video_id"]]
    return rows


def _split_ratio(rows: list[dict], train_r: float, val_r: float, test_r: float) -> list[dict]:
    """Chia video theo tỷ lệ. Luôn chia theo đơn vị video (không theo từng dòng)
    để tránh data leakage."""
    import math
    video_ids = list(dict.fromkeys(row["video_id"] for row in rows))
    n = len(video_ids)
    n_train = max(1, math.floor(n * train_r))
    n_val = max(0, math.floor(n * val_r))
    n_test = max(0, n - n_train - n_val)

    split_map = {}
    for i, vid in enumerate(video_ids):
        if i < n_train:
            split_map[vid] = "train"
        elif i < n_train + n_val:
            split_map[vid] = "val"
        else:
            split_map[vid] = "test"

    print("  Phân chia theo tỷ lệ:")
    for vid, split in split_map.items():
        print(f"    {split:5s}  {vid}")

    for row in rows:
        row["split"] = split_map[row["video_id"]]
    return rows


def _split_manual(rows: list[dict], val_videos: list[str], test_videos: list[str]) -> list[dict]:
    """Gán split theo danh sách video_id chỉ định tay."""
    val_set = set(val_videos)
    test_set = set(test_videos)
    for row in rows:
        vid = row["video_id"]
        if vid in test_set:
            row["split"] = "test"
        elif vid in val_set:
            row["split"] = "val"
        else:
            row["split"] = "train"
    return rows


def _print_summary(rows: list[dict]) -> None:
    from collections import Counter
    split_counts = Counter(row["split"] for row in rows)
    action_counts: dict[str, Counter] = {}
    for row in rows:
        action_counts.setdefault(row["split"], Counter())[row["action_name"]] += 1

    print("\n  === Tóm tắt phân chia ===")
    for split in ("train", "val", "test"):
        count = split_counts.get(split, 0)
        actions = action_counts.get(split, Counter())
        print(f"  {split:5s}: {count} annotation | {dict(actions)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gán cột split (train/val/test) vào annotations.csv"
    )
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ACTION_ANNOTATIONS,
                        help="File annotations.csv cần cập nhật")
    parser.add_argument("--mode", choices=["leave-one-out", "ratio", "manual"],
                        default="leave-one-out",
                        help="Chiến lược phân chia (mặc định: leave-one-out)")
    parser.add_argument("--train", type=float, default=0.70, help="Tỷ lệ train (chỉ dùng với --mode ratio)")
    parser.add_argument("--val",   type=float, default=0.15, help="Tỷ lệ val   (chỉ dùng với --mode ratio)")
    parser.add_argument("--test",  type=float, default=0.15, help="Tỷ lệ test  (chỉ dùng với --mode ratio)")
    parser.add_argument("--val-videos",  nargs="*", default=[],
                        help="Danh sách video_id làm val (chỉ dùng với --mode manual)")
    parser.add_argument("--test-videos", nargs="*", default=[],
                        help="Danh sách video_id làm test (chỉ dùng với --mode manual)")
    args = parser.parse_args()

    if not args.annotations.exists():
        raise SystemExit(f"Không tìm thấy {args.annotations}. Hãy chạy annotate_actions.py trước.")

    rows = _read_csv(args.annotations)
    if not rows:
        raise SystemExit("File annotations.csv rỗng.")

    print(f"  Đọc {len(rows)} dòng annotation từ {args.annotations}")

    if args.mode == "leave-one-out":
        rows = _split_leave_one_out(rows)
    elif args.mode == "ratio":
        rows = _split_ratio(rows, args.train, args.val, args.test)
    else:
        rows = _split_manual(rows, args.val_videos, args.test_videos)

    _print_summary(rows)
    _write_csv(args.annotations, rows)
    print(f"\n  Đã cập nhật cột 'split' trong {args.annotations}")
    print(f"\n  Bước tiếp theo:")
    print(f"    python scripts/extract_spatial_features.py")
    print(f"    python scripts/train_action_model.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

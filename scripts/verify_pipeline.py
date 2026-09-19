"""verify_pipeline.py – Kiểm tra nhanh toàn bộ pipeline trước khi train.

Chạy script này sau khi đã có annotations.csv và features/*.npy để xác nhận:
  1. annotations.csv hợp lệ và có đủ các split
  2. Tất cả video_id trong annotation đều có file feature .npy tương ứng
  3. Kích thước feature đúng với config
  4. Mỗi split có ít nhất 1 mẫu
  5. Phân phối nhãn trong mỗi split (để phát hiện mất cân bằng nghiêm trọng)

Cách dùng
---------
  python scripts/verify_pipeline.py
  python scripts/verify_pipeline.py --config configs/action_earbud_config.json \\
      --annotations data/earbud_actions/annotations.csv \\
      --features-dir data/earbud_actions/features
"""

from __future__ import annotations

import argparse
import csv
import json
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

from assembly.action_config import load_action_model_config
from assembly.paths import (
    DEFAULT_ACTION_ANNOTATIONS,
    DEFAULT_ACTION_CONFIG,
    DEFAULT_FEATURE_CACHE,
)


def _read_annotations(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra pipeline trước khi train")
    parser.add_argument("--config", type=Path, default=DEFAULT_ACTION_CONFIG)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ACTION_ANNOTATIONS)
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURE_CACHE)
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    # --- 1. Config ---
    print("[1/5] Kiểm tra config...")
    try:
        config = load_action_model_config(args.config)
        print(f"  ✓ Config OK: {len(config.actions)} nhãn = {list(config.actions)}")
        print(f"  ✓ Backbone: {config.spatial.backbone}, embedding_dim={config.spatial.embedding_dim}")
    except Exception as e:
        errors.append(f"Config lỗi: {e}")
        print(f"  ✗ {e}")

    # --- 2. Annotations file ---
    print("[2/5] Kiểm tra annotations.csv...")
    if not args.annotations.exists():
        errors.append(f"Không tìm thấy {args.annotations}")
        print(f"  ✗ File không tồn tại: {args.annotations}")
    else:
        rows = _read_annotations(args.annotations)
        if not rows:
            errors.append("annotations.csv rỗng")
        else:
            print(f"  ✓ Tổng cộng {len(rows)} dòng annotation")
            split_counts = Counter(row["split"] for row in rows)
            print(f"  ✓ Phân chia: {dict(split_counts)}")
            if "train" not in split_counts:
                errors.append("Không có dòng nào có split='train'")
            if "val" not in split_counts:
                warnings.append("Không có split='val' — sẽ không thể theo dõi overfitting khi train")

            # Kiểm tra action_name hợp lệ
            unknown_actions = set()
            for row in rows:
                if "actions" in dir(config) and row["action_name"] not in config.actions:
                    unknown_actions.add(row["action_name"])
            if unknown_actions:
                errors.append(f"action_name không có trong config: {unknown_actions}")
            else:
                print(f"  ✓ Tất cả action_name hợp lệ")

    # --- 3. Feature files ---
    print("[3/5] Kiểm tra feature files...")
    if not args.features_dir.exists():
        errors.append(f"Thư mục features không tồn tại: {args.features_dir}")
        print(f"  ✗ Chưa chạy extract_spatial_features.py")
    else:
        npy_files = list(args.features_dir.glob("*.npy"))
        print(f"  Tìm thấy {len(npy_files)} file .npy trong {args.features_dir}")

        if rows:
            video_ids = {row["video_id"] for row in rows}
            missing = []
            for vid in sorted(video_ids):
                npy_path = args.features_dir / f"{vid}.npy"
                json_path = args.features_dir / f"{vid}.json"
                if not npy_path.exists():
                    missing.append(f"{vid}.npy")
                elif not json_path.exists():
                    missing.append(f"{vid}.json (metadata)")
                else:
                    try:
                        import numpy as np
                        arr = np.load(npy_path)
                        meta = json.loads(json_path.read_text(encoding="utf-8"))
                        expected_dim = config.spatial.embedding_dim
                        if arr.ndim != 2:
                            errors.append(f"{vid}.npy có ndim={arr.ndim}, phải là 2")
                        elif arr.shape[1] != expected_dim:
                            errors.append(f"{vid}.npy dim={arr.shape[1]}, config yêu cầu {expected_dim}")
                        else:
                            print(f"  ✓ {vid}: shape={arr.shape}, fps={meta.get('sample_fps')}")
                    except Exception as e:
                        errors.append(f"Lỗi đọc {vid}.npy: {e}")
            if missing:
                errors.append(f"Thiếu feature cho các video_id: {missing}")
                print(f"  ✗ Thiếu: {missing}")

    # --- 4. Window coverage ---
    print("[4/5] Kiểm tra độ phủ temporal window...")
    if rows and not errors:
        try:
            import numpy as np
            seq_len = config.temporal.sequence_length
            sample_fps = config.spatial.sample_fps
            total_windows = 0
            for row in rows:
                npy_path = args.features_dir / f"{row['video_id']}.npy"
                if not npy_path.exists():
                    continue
                meta_path = args.features_dir / f"{row['video_id']}.json"
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                fps = float(meta.get("sample_fps", sample_fps))
                start_f = int(float(row["start_time_s"]) * fps)
                end_f = int(float(row["end_time_s"]) * fps)
                n_frames = max(0, end_f - start_f)
                n_windows = max(1, n_frames - seq_len + 1)
                total_windows += n_windows
            print(f"  ✓ Ước tính tổng training windows: ~{total_windows} (sequence_length={seq_len})")
            if total_windows < 50:
                warnings.append(f"Chỉ có ~{total_windows} windows — rất ít, dễ overfit. Cân nhắc augmentation.")
        except Exception as e:
            warnings.append(f"Không thể tính window coverage: {e}")

    # --- 5. Label distribution ---
    print("[5/5] Phân phối nhãn theo split...")
    if rows:
        for split_name in ("train", "val", "test"):
            split_rows = [r for r in rows if r["split"] == split_name]
            if not split_rows:
                continue
            counts = Counter(r["action_name"] for r in split_rows)
            print(f"  {split_name}: {dict(counts)}")
            if len(counts) < 2:
                warnings.append(f"Split '{split_name}' chỉ có 1 nhãn — không thể phân loại")

    # --- Kết quả ---
    print("\n" + "=" * 60)
    if errors:
        print(f"  ✗ {len(errors)} LỖI cần sửa trước khi train:")
        for e in errors:
            print(f"    • {e}")
    else:
        print("  ✓ Không có lỗi nghiêm trọng!")

    if warnings:
        print(f"\n  ⚠  {len(warnings)} cảnh báo:")
        for w in warnings:
            print(f"    • {w}")

    if not errors:
        print("\n  Bước tiếp theo:")
        print("    python scripts/train_action_model.py")

    print("=" * 60)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

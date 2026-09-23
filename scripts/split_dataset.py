from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
import yaml

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "datasets" / "earbud_geometry"
DEFAULT_RAW_IMAGES = DEFAULT_DATASET / "raw"
DEFAULT_CAMERA_CONFIG = ROOT / "configs" / "camera_earbud_config.json"

CLASS_NAMES = {
    0: "open_case",
    1: "close_case",
    2: "earbud",
    3: "empty_left",
    4: "empty_right",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def split_dataset(
    images_dir: Path,
    labels_dir: Path | None,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    copy_files: bool = True,
    create_empty_labels: bool = False,
) -> dict[str, int]:
    """Tự động chia tập dữ liệu thành train, val, test cho YOLO."""
    if not images_dir.exists():
        raise FileNotFoundError(f"Thư mục ảnh không tồn tại: {images_dir}")

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-4:
        raise ValueError(f"Tổng tỷ lệ phải bằng 1.0 (hiện tại: {total_ratio})")

    # Lấy danh sách ảnh
    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if not image_paths:
        raise RuntimeError(f"Không tìm thấy file ảnh nào trong: {images_dir}")

    random.seed(seed)
    shuffled_images = image_paths.copy()
    random.shuffle(shuffled_images)

    total = len(shuffled_images)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    # n_test lấy phần còn lại để không sót ảnh nào
    splits: dict[str, list[Path]] = {
        "train": shuffled_images[:n_train],
        "val": shuffled_images[n_train : n_train + n_val],
        "test": shuffled_images[n_train + n_val :],
    }

    # Tạo các thư mục đích
    for split_name in ("train", "val", "test"):
        (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    found_labels_count = 0
    empty_labels_count = 0

    for split_name, imgs in splits.items():
        counts[split_name] = len(imgs)
        for img_path in imgs:
            dest_img = output_dir / "images" / split_name / img_path.name
            if copy_files:
                shutil.copy2(img_path, dest_img)
            else:
                shutil.move(img_path, dest_img)

            # Tìm file label .txt tương ứng
            label_stem = img_path.stem
            source_label: Path | None = None

            # 1. Tìm trong labels_dir nếu có
            if labels_dir and (labels_dir / f"{label_stem}.txt").exists():
                source_label = labels_dir / f"{label_stem}.txt"
            # 2. Tìm cùng thư mục với ảnh
            elif (img_path.parent / f"{label_stem}.txt").exists():
                source_label = img_path.parent / f"{label_stem}.txt"
            # 3. Tìm trong output_dir / labels / raw nếu có
            elif (output_dir / "labels_raw" / f"{label_stem}.txt").exists():
                source_label = output_dir / "labels_raw" / f"{label_stem}.txt"

            dest_label = output_dir / "labels" / split_name / f"{label_stem}.txt"
            if source_label and source_label.exists():
                shutil.copy2(source_label, dest_label)
                found_labels_count += 1
            elif create_empty_labels:
                # Tạo file nhãn rỗng (YOLO chấp nhận file nhãn rỗng coi như background / negative sample)
                dest_label.touch(exist_ok=True)
                empty_labels_count += 1

    print("=" * 60)
    print("HOÀN THÀNH CHIA DATASET:")
    print(f" - Tổng số ảnh: {total}")
    print(f" - Train: {counts['train']} ảnh ({train_ratio*100:.0f}%) -> {output_dir / 'images' / 'train'}")
    print(f" - Val:   {counts['val']} ảnh ({val_ratio*100:.0f}%) -> {output_dir / 'images' / 'val'}")
    print(f" - Test:  {counts['test']} ảnh ({test_ratio*100:.0f}%) -> {output_dir / 'images' / 'test'}")
    print(f" - Số nhãn có sẵn: {found_labels_count}")
    print(f" - Số nhãn rỗng (placeholder) được tạo: {empty_labels_count}")
    print("=" * 60)

    if empty_labels_count > 0:
        print("LƯU Ý:")
        print("  Các file nhãn .txt được tạo tự động hiện đang rỗng.")
        print("  Nếu train ngay lúc này, YOLO sẽ hiểu toàn bộ ảnh là nền (background, không có vật thể).")
        print("  Để nhận diện được tai nghe và hộp sạc, bạn cần dùng công cụ gán nhãn (Label Studio, CVAT, Roboflow...)")
        print("  để khoanh bounding box và ghi tọa độ vào các file .txt này.")
        print("=" * 60)

    return counts


def resolve_class_names(yaml_path: Path, camera_config: Path | None = None) -> dict[int, str]:
    """Preserve existing label IDs; only new datasets use the geometry schema."""
    existing = None
    if yaml_path.exists():
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig"))
        names = raw.get("names") if isinstance(raw, dict) else None
        if isinstance(names, list):
            existing = dict(enumerate(names))
        elif isinstance(names, dict):
            existing = {int(key): value for key, value in names.items()}
        if not existing or set(existing) != set(range(len(existing))):
            raise ValueError("data.yaml phải có names với ID liên tiếp từ 0; không tự gán lại ID.")

    selected = None
    if camera_config is not None:
        raw = json.loads(camera_config.read_text(encoding="utf-8-sig"))
        selected = {i: item["label"] for i, item in enumerate(raw["classes"])}
        if existing is not None and existing != selected:
            raise ValueError("Class ID/names trong data.yaml khác --camera-config; giữ nguyên dataset.")
    return existing if existing is not None else (selected if selected is not None else CLASS_NAMES.copy())


def update_data_yaml(
    yaml_path: Path, dataset_dir: Path, class_names: dict[int, str] | None = None,
) -> None:
    """Tạo hoặc cập nhật file data.yaml với đường dẫn chuẩn."""
    config = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": class_names if class_names is not None else resolve_class_names(yaml_path),
    }

    # Khi data.yaml nằm ngay trong dataset root, bỏ `path` để Ultralytics
    # tự resolve theo thư mục chứa YAML. Cách này portable giữa các máy.
    if yaml_path.parent.resolve() != dataset_dir.resolve():
        try:
            rel_path = dataset_dir.resolve().relative_to(ROOT.resolve())
            config["path"] = rel_path.as_posix()
        except ValueError:
            config["path"] = dataset_dir.resolve().as_posix()

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

    print(f"Đã cập nhật file cấu hình YAML: {yaml_path}")
    print(f"Nội dung file {yaml_path.name}:")
    print("-" * 40)
    with yaml_path.open("r", encoding="utf-8") as f:
        print(f.read().strip())
    print("-" * 40)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tự động chia ảnh & nhãn thành train/val/test và cập nhật data.yaml")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_RAW_IMAGES, help="Thư mục chứa ảnh thô")
    parser.add_argument("--labels-dir", type=Path, default=None, help="Thư mục chứa file label .txt (nếu có)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATASET, help="Thư mục gốc của dataset")
    parser.add_argument("--yaml-path", type=Path, default=None, help="Mặc định: <output-dir>/data.yaml")
    parser.add_argument("--camera-config", type=Path, default=None,
                        help="Class ID theo thứ tự classes; nếu YAML đã có thì phải khớp, không đổi nhãn")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Tỷ lệ tập train (mặc định 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Tỷ lệ tập val (mặc định 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Tỷ lệ tập test (mặc định 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--move", action="store_true", help="Di chuyển file thay vì sao chép (copy)")
    parser.add_argument(
        "--create-empty-labels",
        action="store_true",
        help="Tạo nhãn rỗng cho ảnh nền thật sự; không dùng cho ảnh có linh kiện chưa gán nhãn",
    )
    parser.add_argument(
        "--no-empty-labels",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    print(
        "CẢNH BÁO: lệnh này chia ảnh ngẫu nhiên và chỉ phù hợp để smoke test. "
        "Khi đánh giá mô hình thật, hãy chia theo person/session để tránh rò rỉ dữ liệu."
    )

    yaml_path = args.yaml_path or args.output_dir / "data.yaml"
    try:
        class_names = resolve_class_names(yaml_path, args.camera_config)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    split_dataset(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        copy_files=not args.move,
        create_empty_labels=args.create_empty_labels and not args.no_empty_labels,
    )

    update_data_yaml(yaml_path, args.output_dir, class_names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

"""Prepare a Roboflow COCO export and train an Ultralytics detector on Kaggle.

This file is intentionally self-contained: it can be uploaded to a Kaggle
Notebook without installing this repository as a Python package.
"""

import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ANNOTATION_FILE = "_annotations.coco.json"
GEOMETRY_V2_CLASSES = (
    "open_case",
    "close_case",
    "left_earbud",
    "right_earbud",
    "empty_left",
    "empty_right",
)
SPLIT_ALIASES = {
    "train": ("train", "training"),
    "val": ("valid", "val", "validation"),
    "test": ("test", "testing"),
}


@dataclass(frozen=True)
class Taxonomy:
    category_to_class: dict[int, int]
    class_names: tuple[str, ...]
    boxes_per_class: dict[str, int]


def _configure_utf8_console() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"COCO JSON không phải object: {path}")
    return payload


def find_split_dirs(dataset_root: Path) -> dict[str, Path]:
    """Return canonical split names mapped to Roboflow split directories."""
    children = {
        child.name.casefold(): child
        for child in dataset_root.iterdir()
        if child.is_dir()
    }
    found: dict[str, Path] = {}
    for canonical, aliases in SPLIT_ALIASES.items():
        for alias in aliases:
            candidate = children.get(alias.casefold())
            if candidate is not None and (candidate / ANNOTATION_FILE).is_file():
                found[canonical] = candidate
                break
    return found


def is_coco_dataset_root(path: Path) -> bool:
    try:
        splits = find_split_dirs(path)
    except (OSError, ValueError):
        return False
    return "train" in splits and "val" in splits


def discover_dataset_roots(input_root: Path) -> list[Path]:
    if not input_root.is_dir():
        return []
    candidates = {
        annotation.parent.parent.resolve()
        for annotation in input_root.rglob(ANNOTATION_FILE)
        if annotation.parent.name.casefold() in {
            alias.casefold()
            for aliases in SPLIT_ALIASES.values()
            for alias in aliases
        }
    }
    return sorted((path for path in candidates if is_coco_dataset_root(path)), key=str)


def resolve_dataset_root(dataset_root: Path | None, input_root: Path) -> Path:
    if dataset_root is not None:
        resolved = dataset_root.expanduser().resolve()
        if not is_coco_dataset_root(resolved):
            raise ValueError(
                f"{resolved} không có train + valid/val, mỗi split cần {ANNOTATION_FILE}."
            )
        return resolved

    candidates = discover_dataset_roots(input_root)
    if not candidates:
        raise ValueError(
            f"Không tìm thấy COCO export trong {input_root}. "
            "Hãy Add Data rồi truyền --dataset-root /kaggle/input/<slug>/<folder>."
        )
    if len(candidates) > 1:
        listing = "\n".join(f"  - {path}" for path in candidates)
        raise ValueError(
            "Tìm thấy nhiều COCO dataset; hãy chọn đúng một dataset bằng --dataset-root:\n"
            f"{listing}"
        )
    return candidates[0]


def inspect_taxonomy(split_dirs: dict[str, Path]) -> Taxonomy:
    category_names: dict[int, str] = {}
    referenced = Counter()
    referenced_by_split: dict[str, Counter[int]] = {}

    for split_name, split_dir in split_dirs.items():
        payload = _load_json(split_dir / ANNOTATION_FILE)
        for category in payload.get("categories", []):
            category_id = int(category["id"])
            category_name = str(category["name"]).strip()
            previous = category_names.get(category_id)
            if previous is not None and previous != category_name:
                raise ValueError(
                    f"category_id={category_id} đổi tên giữa các split: "
                    f"{previous!r} != {category_name!r}"
                )
            category_names[category_id] = category_name

        counts = Counter(int(item["category_id"]) for item in payload.get("annotations", []))
        referenced.update(counts)
        referenced_by_split[split_name] = counts

    unknown = sorted(set(referenced) - set(category_names))
    if unknown:
        raise ValueError(f"Annotation tham chiếu category_id không tồn tại: {unknown}")
    if not referenced:
        raise ValueError("Dataset không có bounding box nào.")

    used_category_ids = sorted(referenced)
    category_to_class = {
        category_id: class_id
        for class_id, category_id in enumerate(used_category_ids)
    }
    class_names = tuple(category_names[category_id] for category_id in used_category_ids)
    if len(set(class_names)) != len(class_names):
        raise ValueError(f"Tên class COCO bị trùng: {class_names}")

    missing_in_train = [
        category_names[category_id]
        for category_id in used_category_ids
        if referenced_by_split.get("train", Counter())[category_id] == 0
    ]
    if missing_in_train:
        raise ValueError(
            "Các class sau không có box trong train: " + ", ".join(missing_in_train)
        )

    boxes_per_class = {
        category_names[category_id]: referenced[category_id]
        for category_id in used_category_ids
    }
    return Taxonomy(category_to_class, class_names, boxes_per_class)


def _source_image(split_dir: Path, file_name: str) -> Path:
    # Roboflow normally stores images beside _annotations.coco.json. Some
    # exporters use an images/ subdirectory, so support both without allowing
    # paths from JSON to escape the split directory.
    safe_name = Path(str(file_name).replace("\\", "/")).name
    candidates = (split_dir / safe_name, split_dir / "images" / safe_name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Không tìm thấy ảnh {file_name!r} trong {split_dir}")


def _yolo_bbox(annotation: dict[str, Any], width: float, height: float) -> tuple[float, ...] | None:
    bbox = annotation.get("bbox", [])
    if len(bbox) != 4 or width <= 0 or height <= 0:
        return None
    x, y, box_width, box_height = (float(value) for value in bbox)
    x1 = max(0.0, min(width, x))
    y1 = max(0.0, min(height, y))
    x2 = max(0.0, min(width, x + box_width))
    y2 = max(0.0, min(height, y + box_height))
    if x2 <= x1 or y2 <= y1:
        return None
    return (
        ((x1 + x2) / 2.0) / width,
        ((y1 + y2) / 2.0) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def _write_data_yaml(output_root: Path, split_dirs: dict[str, Path], names: tuple[str, ...]) -> Path:
    lines = [
        f"path: {output_root.as_posix()}",
        "train: train/images",
        "val: val/images",
    ]
    if "test" in split_dirs:
        lines.append("test: test/images")
    lines.extend(("", f"nc: {len(names)}", "names:"))
    lines.extend(f"  {index}: {json.dumps(name, ensure_ascii=False)}" for index, name in enumerate(names))
    yaml_path = output_root / "data.yaml"
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def convert_coco_to_yolo(
    dataset_root: Path,
    output_root: Path,
    *,
    clean: bool = True,
) -> tuple[Path, dict[str, Any]]:
    split_dirs = find_split_dirs(dataset_root)
    if "train" not in split_dirs or "val" not in split_dirs:
        raise ValueError("Dataset cần ít nhất train và valid/val.")
    taxonomy = inspect_taxonomy(split_dirs)

    if clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "source": str(dataset_root),
        "class_names": list(taxonomy.class_names),
        "boxes_per_class": taxonomy.boxes_per_class,
        "splits": {},
        "skipped_invalid_boxes": 0,
    }

    for split_name, split_dir in split_dirs.items():
        payload = _load_json(split_dir / ANNOTATION_FILE)
        image_rows = payload.get("images", [])
        image_by_id = {int(item["id"]): item for item in image_rows}
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in payload.get("annotations", []):
            image_id = int(annotation["image_id"])
            if image_id not in image_by_id:
                raise ValueError(
                    f"{split_name}: annotation dùng image_id={image_id} không tồn tại"
                )
            annotations_by_image[image_id].append(annotation)

        image_output = output_root / split_name / "images"
        label_output = output_root / split_name / "labels"
        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)
        used_names: set[str] = set()
        written_boxes = 0

        for image in image_rows:
            image_id = int(image["id"])
            source = _source_image(split_dir, str(image["file_name"]))
            destination_name = source.name
            label_name = f"{source.stem}.txt"
            collision_key = destination_name.casefold()
            label_collision_key = label_name.casefold()
            if collision_key in used_names or label_collision_key in used_names:
                raise ValueError(f"{split_name}: tên ảnh/label bị trùng: {source.name}")
            used_names.update((collision_key, label_collision_key))
            shutil.copy2(source, image_output / destination_name)

            width = float(image.get("width", 0))
            height = float(image.get("height", 0))
            lines: list[str] = []
            for annotation in annotations_by_image.get(image_id, []):
                category_id = int(annotation["category_id"])
                class_id = taxonomy.category_to_class[category_id]
                normalized = _yolo_bbox(annotation, width, height)
                if normalized is None:
                    report["skipped_invalid_boxes"] += 1
                    continue
                lines.append(
                    f"{class_id} " + " ".join(f"{value:.8f}" for value in normalized)
                )
                written_boxes += 1
            (label_output / label_name).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )

        report["splits"][split_name] = {
            "images": len(image_rows),
            "boxes": written_boxes,
        }

    yaml_path = _write_data_yaml(output_root, split_dirs, taxonomy.class_names)
    report_path = output_root / "dataset_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return yaml_path, report


def print_report(report: dict[str, Any]) -> None:
    print("\n========== COCO -> YOLO REPORT ==========")
    print("Classes:")
    for index, name in enumerate(report["class_names"]):
        print(f"  {index}: {name} ({report['boxes_per_class'][name]} boxes)")
    print("Splits:")
    for split, counts in report["splits"].items():
        print(f"  {split}: {counts['images']} images, {counts['boxes']} boxes")
    print(f"Invalid boxes skipped: {report['skipped_invalid_boxes']}")
    print("==========================================\n")


def check_project_compatibility(class_names: list[str], require_geometry_v2: bool) -> None:
    actual = set(class_names)
    required = set(GEOMETRY_V2_CLASSES)
    if actual == required and len(class_names) == len(GEOMETRY_V2_CLASSES):
        print("[OK] Dataset đúng schema geometry v2 của hệ thống hiện tại.")
        return

    message = (
        "Dataset chưa phải geometry v2. Model vẫn train được để detect baseline, "
        "nhưng chưa thể tự xác nhận đủ các bước open_case -> hai tai -> close_case.\n"
        f"  Hiện có: {class_names}\n"
        f"  Geometry v2 cần: {list(GEOMETRY_V2_CLASSES)}"
    )
    if require_geometry_v2:
        raise ValueError(message)
    print(f"[WARNING] {message}")


def train(args: argparse.Namespace, yaml_path: Path, report: dict[str, Any]) -> Path:
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Thiếu ultralytics/PyTorch. Trên Kaggle chạy trước: "
            "!pip install -q -U ultralytics"
        ) from error

    if args.device == "auto":
        device: str | int = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[INFO] Torch CUDA available: {torch.cuda.is_available()}")
    print(f"[INFO] Training device: {device}")
    if device == "cpu":
        print("[WARNING] Kaggle chưa bật GPU. Settings -> Accelerator -> GPU rồi chạy lại.")

    project_dir = args.work_dir / "training"
    model = YOLO(args.model)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=str(project_dir),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        cache="disk",
        fliplr=0.0,
        flipud=0.0,
        plots=True,
        save=True,
        save_period=10,
        verbose=True,
    )

    run_dir = project_dir / args.name
    best_path = run_dir / "weights" / "best.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"Train xong nhưng không tìm thấy {best_path}")

    best_model = YOLO(str(best_path))
    evaluation_split = "test" if "test" in report["splits"] else "val"
    metrics = best_model.val(
        data=str(yaml_path),
        split=evaluation_split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        plots=True,
        project=str(project_dir),
        name=f"{args.name}_{evaluation_split}",
    )
    print(f"[RESULT] Evaluation split: {evaluation_split}")
    print(f"[RESULT] mAP50: {metrics.box.map50:.4f}")
    print(f"[RESULT] mAP50-95: {metrics.box.map:.4f}")
    print(f"[RESULT] Precision: {metrics.box.mp:.4f}")
    print(f"[RESULT] Recall: {metrics.box.mr:.4f}")

    portable_best = args.work_dir / "best_earbud_detector.pt"
    shutil.copy2(best_path, portable_best)
    shutil.copy2(yaml_path, run_dir / "data_used.yaml")
    report_source = yaml_path.parent / "dataset_report.json"
    shutil.copy2(report_source, run_dir / "dataset_report.json")
    archive_base = args.work_dir / "earbud_training_results"
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=run_dir))
    print(f"[DONE] Best checkpoint: {portable_best}")
    print(f"[DONE] Full results: {archive_path}")
    return portable_best


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Roboflow COCO detection data and train YOLO on Kaggle"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Thư mục chứa train/valid/test; tự tìm trong --input-root nếu bỏ trống",
    )
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--work-dir", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--name", default="earbud_coco_detector")
    parser.add_argument(
        "--require-geometry-v2",
        action="store_true",
        help="Dừng nếu dataset không đúng 6 lớp geometry trái/phải của quy trình mới",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Chỉ convert/validate COCO, chưa train",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_utf8_console()
    args = build_parser().parse_args(argv)
    args.input_root = args.input_root.expanduser().resolve()
    args.work_dir = args.work_dir.expanduser().resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    try:
        dataset_root = resolve_dataset_root(args.dataset_root, args.input_root)
        print(f"[INFO] COCO dataset: {dataset_root}")
        prepared_root = args.work_dir / "earbud_yolo_prepared"
        yaml_path, report = convert_coco_to_yolo(dataset_root, prepared_root)
        print_report(report)
        check_project_compatibility(report["class_names"], args.require_geometry_v2)
        print(f"[INFO] YOLO data config: {yaml_path}")
        if args.prepare_only:
            print("[DONE] Prepare-only hoàn tất; chưa chạy training.")
            return 0
        train(args, yaml_path, report)
        return 0
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

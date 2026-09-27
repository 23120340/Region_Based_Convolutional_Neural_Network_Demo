from __future__ import annotations

"""Merge the seven-class RNN dataset with the extra left/right earbud set.

Both Roboflow exports contain YOLO segmentation polygons. This project trains
a detector, so every polygon is converted to its enclosing YOLO bounding box.
The extra dataset is added to train only; the independent validation/test
splits from the primary dataset remain untouched.
"""

import argparse
import ast
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIMARY = ROOT / "RNN"
DEFAULT_EXTRA = DEFAULT_PRIMARY / "LR_Earbud"
DEFAULT_OUTPUT = ROOT / "datasets" / "earbud_rnn_merged"

OUTPUT_NAMES = (
    "open_case",
    "close_case",
    "left_earbud",
    "right_earbud",
    "empty_left",
    "empty_right",
    "hand",
)

ALIASES = {
    "case_open": "open_case",
    "open_case": "open_case",
    "case_closed": "close_case",
    "closed_case": "close_case",
    "close_case": "close_case",
    "earbud_left": "left_earbud",
    "left_earbud": "left_earbud",
    "earbud_right": "right_earbud",
    "right_earbud": "right_earbud",
    "empty_slot_left": "empty_left",
    "empty_left": "empty_left",
    "empty_slot_right": "empty_right",
    "empty_right": "empty_right",
    "hand": "hand",
}


def _normalise_name(name: str) -> str:
    return "_".join(name.strip().casefold().replace("-", "_").split())


def read_names(data_yaml: Path) -> list[str]:
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("names:"):
            raw = line.split(":", 1)[1].strip()
            names = ast.literal_eval(raw)
            if not isinstance(names, list) or not all(isinstance(item, str) for item in names):
                break
            return names
    raise ValueError(f"Không đọc được danh sách names dạng list trong {data_yaml}")


def build_class_map(names: list[str], *, source: Path) -> dict[int, int]:
    output_index = {name: index for index, name in enumerate(OUTPUT_NAMES)}
    mapping: dict[int, int] = {}
    for source_id, source_name in enumerate(names):
        canonical = ALIASES.get(_normalise_name(source_name))
        if canonical is None:
            raise ValueError(f"Class chưa có ánh xạ trong {source}: {source_name!r}")
        mapping[source_id] = output_index[canonical]
    return mapping


def label_row_to_bbox(row: str, class_map: dict[int, int]) -> str:
    fields = row.strip().split()
    if not fields:
        raise ValueError("Dòng nhãn rỗng")
    source_id = int(fields[0])
    if source_id not in class_map:
        raise ValueError(f"class_id không có trong data.yaml: {source_id}")
    values = [float(value) for value in fields[1:]]

    if len(values) == 4:
        center_x, center_y, width, height = values
    elif len(values) >= 6 and len(values) % 2 == 0:
        xs = values[0::2]
        ys = values[1::2]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
    else:
        raise ValueError(f"Cần bbox 4 giá trị hoặc polygon x/y, nhận {len(values)} giá trị")

    numbers = (center_x, center_y, width, height)
    if not all(0.0 <= value <= 1.0 for value in numbers):
        raise ValueError(f"Tọa độ ngoài [0, 1]: {numbers}")
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"Box không có diện tích: {numbers}")
    target_id = class_map[source_id]
    return f"{target_id} " + " ".join(f"{value:.8f}" for value in numbers)


def _copy_or_link(source: Path, destination: Path, mode: str) -> str:
    if mode in {"auto", "hardlink"}:
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            if mode == "hardlink":
                raise
    shutil.copy2(source, destination)
    return "copy"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ingest_split(
    source_root: Path,
    source_split: str,
    target_root: Path,
    target_split: str,
    class_map: dict[int, int],
    *,
    prefix: str,
    image_mode: str,
    skip_image_hashes: set[str] | None = None,
) -> dict[str, object]:
    image_dir = source_root / source_split / "images"
    label_dir = source_root / source_split / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Thiếu images/labels: {source_root / source_split}")

    target_images = target_root / target_split / "images"
    target_labels = target_root / target_split / "labels"
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    class_counts: Counter[int] = Counter()
    empty_labels = 0
    converted_polygons = 0
    invalid_rows = 0
    invalid_examples: list[str] = []
    link_modes: Counter[str] = Counter()
    images = sorted(path for path in image_dir.iterdir() if path.is_file())
    written_images = 0
    duplicates_skipped = 0
    for image in images:
        if skip_image_hashes is not None and _file_hash(image) in skip_image_hashes:
            duplicates_skipped += 1
            continue
        source_label = label_dir / f"{image.stem}.txt"
        if not source_label.is_file():
            raise FileNotFoundError(f"Ảnh thiếu label: {image}")
        destination_name = f"{prefix}{image.name}"
        destination_label = target_labels / f"{Path(destination_name).stem}.txt"
        if (target_images / destination_name).exists() or destination_label.exists():
            raise FileExistsError(f"Tên output bị trùng: {destination_name}")
        link_mode = _copy_or_link(image, target_images / destination_name, image_mode)
        link_modes[link_mode] += 1
        written_images += 1

        output_rows: list[str] = []
        for line_number, row in enumerate(
            source_label.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not row.strip():
                continue
            if len(row.split()) > 5:
                converted_polygons += 1
            try:
                converted = label_row_to_bbox(row, class_map)
            except (TypeError, ValueError) as error:
                invalid_rows += 1
                if len(invalid_examples) < 20:
                    invalid_examples.append(f"{source_label}:{line_number}: {error}")
                continue
            output_rows.append(converted)
            class_counts[int(converted.split()[0])] += 1
        if not output_rows:
            empty_labels += 1
        destination_label.write_text(
            "\n".join(output_rows) + ("\n" if output_rows else ""), encoding="utf-8"
        )

    return {
        "source_images": len(images),
        "images": written_images,
        "duplicates_skipped": duplicates_skipped,
        "boxes": sum(class_counts.values()),
        "empty_labels": empty_labels,
        "converted_polygons": converted_polygons,
        "invalid_rows_skipped": invalid_rows,
        "invalid_examples": invalid_examples,
        "class_counts": {
            OUTPUT_NAMES[class_id]: count
            for class_id, count in sorted(class_counts.items())
        },
        "image_storage": dict(link_modes),
    }


def merge_datasets(
    primary: Path,
    extra: Path,
    output: Path,
    *,
    image_mode: str = "auto",
) -> dict[str, object]:
    primary = primary.resolve()
    extra = extra.resolve()
    output = output.resolve()
    if output in {primary, extra} or primary in output.parents or extra in output.parents:
        raise ValueError("Output không được nằm trong dataset nguồn")
    primary_map = build_class_map(read_names(primary / "data.yaml"), source=primary)
    extra_map = build_class_map(read_names(extra / "data.yaml"), source=extra)

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    report: dict[str, object] = {
        "primary": str(primary),
        "extra": str(extra),
        "class_names": list(OUTPUT_NAMES),
        "splits": {},
    }
    splits: dict[str, object] = report["splits"]  # type: ignore[assignment]
    for source_split, target_split in (("train", "train"), ("valid", "valid"), ("test", "test")):
        splits[f"primary/{source_split}"] = ingest_split(
            primary,
            source_split,
            output,
            target_split,
            primary_map,
            prefix="main_",
            image_mode=image_mode,
        )
    primary_hashes = {
        _file_hash(image)
        for split in ("train", "valid", "test")
        for image in (primary / split / "images").iterdir()
        if image.is_file()
    }
    splits["extra/train"] = ingest_split(
        extra,
        "train",
        output,
        "train",
        extra_map,
        prefix="lr_extra_",
        image_mode=image_mode,
        skip_image_hashes=primary_hashes,
    )

    yaml_lines = [
        "path: .",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        f"nc: {len(OUTPUT_NAMES)}",
        "names: " + json.dumps(list(OUTPUT_NAMES), ensure_ascii=False),
    ]
    (output / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    (output / "merge_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _configure_utf8_console() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    _configure_utf8_console()
    parser = argparse.ArgumentParser(description="Merge RNN left/right earbud datasets")
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--extra", type=Path, default=DEFAULT_EXTRA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-mode", choices=("auto", "hardlink", "copy"), default="auto")
    args = parser.parse_args()
    try:
        report = merge_datasets(
            args.primary, args.extra, args.output, image_mode=args.image_mode
        )
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2

    print(f"[DONE] Dataset đã ghép: {args.output.resolve()}")
    for source, summary in report["splits"].items():
        print(
            f"  {source}: images={summary['images']}/{summary['source_images']}, "
            f"duplicates={summary['duplicates_skipped']}, boxes={summary['boxes']}, "
            f"polygons->bbox={summary['converted_polygons']}"
        )
    extra_summary = report["splits"]["extra/train"]
    if extra_summary["duplicates_skipped"] == extra_summary["source_images"]:
        print("[INFO] Toàn bộ LR_Earbud đã có sẵn trong primary/train; không nhân đôi ảnh.")
    else:
        print("[INFO] Ảnh LR_Earbud mới chỉ được thêm vào train; valid/test giữ độc lập.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

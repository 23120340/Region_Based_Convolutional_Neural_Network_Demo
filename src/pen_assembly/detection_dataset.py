from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class DetectionDatasetReport:
    dataset_root: Path
    split_images: dict[str, int]
    split_labels: dict[str, int]
    missing_labels: tuple[Path, ...]
    empty_labels: tuple[Path, ...]
    invalid_labels: tuple[str, ...]
    boxes_per_class: dict[int, int]
    class_names: dict[int, str]

    @property
    def total_images(self) -> int:
        return sum(self.split_images.values())

    @property
    def total_boxes(self) -> int:
        return sum(self.boxes_per_class.values())

    @property
    def is_trainable(self) -> bool:
        return self.total_images > 0 and self.total_boxes > 0 and not self.invalid_labels


def _resolve_dataset_root(data_yaml: Path, configured_path: str) -> Path:
    configured = Path(configured_path)
    if configured.is_absolute():
        return configured
    if configured_path.strip() in {"", ".", "./"}:
        return data_yaml.parent.resolve()
    from_working_directory = (Path.cwd() / configured).resolve()
    if from_working_directory.exists():
        return from_working_directory
    return (data_yaml.parent / configured).resolve()


def inspect_detection_dataset(data_yaml: str | Path) -> DetectionDatasetReport:
    yaml_path = Path(data_yaml).resolve()
    with yaml_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    names_raw = config.get("names", {})
    if isinstance(names_raw, list):
        class_names = {index: str(name) for index, name in enumerate(names_raw)}
    else:
        class_names = {int(index): str(name) for index, name in names_raw.items()}
    dataset_root = _resolve_dataset_root(yaml_path, str(config.get("path", yaml_path.parent)))

    split_images: dict[str, int] = {}
    split_labels: dict[str, int] = {}
    missing_labels: list[Path] = []
    empty_labels: list[Path] = []
    invalid_labels: list[str] = []
    boxes_per_class: Counter[int] = Counter()

    for split in ("train", "val", "test"):
        image_setting = config.get(split)
        if image_setting is None:
            split_images[split] = 0
            split_labels[split] = 0
            continue
        image_dir = Path(image_setting)
        if not image_dir.is_absolute():
            image_dir = dataset_root / image_dir
        label_dir = dataset_root / "labels" / split
        images = sorted(
            path for path in image_dir.glob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
        ) if image_dir.exists() else []
        split_images[split] = len(images)
        split_labels[split] = 0

        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                missing_labels.append(label_path)
                continue
            split_labels[split] += 1
            lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                empty_labels.append(label_path)
                continue
            for line_number, line in enumerate(lines, start=1):
                fields = line.split()
                location = f"{label_path}:{line_number}"
                if len(fields) != 5:
                    invalid_labels.append(f"{location}: cần 5 giá trị, nhận {len(fields)}")
                    continue
                try:
                    class_id = int(fields[0])
                    x_center, y_center, width, height = (float(value) for value in fields[1:])
                except ValueError:
                    invalid_labels.append(f"{location}: giá trị không phải số hợp lệ")
                    continue
                if class_id not in class_names:
                    invalid_labels.append(f"{location}: class_id {class_id} không có trong data.yaml")
                    continue
                if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)):
                    invalid_labels.append(f"{location}: tọa độ phải nằm trong [0, 1]")
                    continue
                if width <= 0.0 or height <= 0.0:
                    invalid_labels.append(f"{location}: width và height phải > 0")
                    continue
                boxes_per_class[class_id] += 1

    return DetectionDatasetReport(
        dataset_root=dataset_root,
        split_images=split_images,
        split_labels=split_labels,
        missing_labels=tuple(missing_labels),
        empty_labels=tuple(empty_labels),
        invalid_labels=tuple(invalid_labels),
        boxes_per_class=dict(boxes_per_class),
        class_names=class_names,
    )


def format_detection_report(report: DetectionDatasetReport) -> str:
    lines = [
        f"Dataset root: {report.dataset_root}",
        f"Images: {report.split_images}",
        f"Labels: {report.split_labels}",
        f"Missing labels: {len(report.missing_labels)}",
        f"Empty labels/background images: {len(report.empty_labels)}",
        f"Invalid label rows: {len(report.invalid_labels)}",
        f"Total bounding boxes: {report.total_boxes}",
        "Boxes per class:",
    ]
    for class_id, class_name in report.class_names.items():
        lines.append(f"  {class_id} {class_name}: {report.boxes_per_class.get(class_id, 0)}")
    if report.invalid_labels:
        lines.append("Invalid label examples:")
        lines.extend(f"  - {issue}" for issue in report.invalid_labels[:10])
    lines.append(f"Trainable: {'YES' if report.is_trainable else 'NO'}")
    return "\n".join(lines)

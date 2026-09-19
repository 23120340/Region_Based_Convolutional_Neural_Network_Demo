from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "datasets" / "pen_parts" / "raw" / "person01" / "session02"
DEFAULT_OUTPUT = ROOT / "artifacts" / "detection_pilot" / "person01_session02"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageCandidate:
    path: Path
    sharpness: float
    brightness: float
    perceptual_hash: int
    quality: float


def _configure_utf8_console() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def select_diverse_candidates(
    candidates: list[ImageCandidate],
    count: int,
) -> list[ImageCandidate]:
    """Chọn ảnh rõ trước, sau đó ưu tiên ảnh khác nhau về nội dung."""
    if count <= 0 or not candidates:
        return []
    remaining = sorted(candidates, key=lambda item: (-item.quality, item.path.name))
    selected = [remaining.pop(0)]
    while remaining and len(selected) < count:
        best = max(
            remaining,
            key=lambda item: (
                0.75
                * min(
                    _hamming_distance(item.perceptual_hash, chosen.perceptual_hash)
                    for chosen in selected
                )
                / 64.0
                + 0.25 * item.quality,
                item.quality,
                item.path.name,
            ),
        )
        selected.append(best)
        remaining.remove(best)
    return selected


def _analyse_image(path: Path, cv2) -> ImageCandidate | None:
    image = cv2.imread(str(path))
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    exposure_score = max(0.0, 1.0 - abs(brightness - 127.5) / 127.5)
    sharpness_score = sharpness / (sharpness + 10.0)
    quality = 0.7 * sharpness_score + 0.3 * exposure_score

    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    comparisons = resized[:, 1:] > resized[:, :-1]
    perceptual_hash = 0
    for value in comparisons.flatten():
        perceptual_hash = (perceptual_hash << 1) | int(value)
    return ImageCandidate(path, sharpness, brightness, perceptual_hash, quality)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    _configure_utf8_console()
    try:
        import cv2
    except ImportError as error:
        raise SystemExit("Thiếu OpenCV. Hãy cài requirements-camera.txt") from error

    parser = argparse.ArgumentParser(
        description="Chọn bộ ảnh pilot rõ và đa dạng để người dùng kiểm tra rồi upload lên CVAT"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=None,
        help="Ngưỡng Laplacian; mặc định tự lấy percentile 40 của session",
    )
    parser.add_argument("--min-brightness", type=float, default=30.0)
    parser.add_argument("--max-brightness", type=float, default=225.0)
    args = parser.parse_args()

    if args.count < 1:
        raise SystemExit("--count phải lớn hơn 0")
    if not args.source.is_dir():
        raise SystemExit(f"Không tìm thấy thư mục ảnh: {args.source}")
    images_output = args.output / "images"
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(
            f"Thư mục kết quả không rỗng: {args.output}. "
            "Hãy chọn --output khác để không ghi đè kết quả cũ."
        )

    image_paths = sorted(
        path for path in args.source.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise SystemExit(f"Không tìm thấy ảnh trong: {args.source}")

    analysed: list[ImageCandidate] = []
    unreadable: list[Path] = []
    for path in image_paths:
        candidate = _analyse_image(path, cv2)
        if candidate is None:
            unreadable.append(path)
        else:
            analysed.append(candidate)

    if not analysed:
        raise SystemExit("Không có ảnh nào đọc được để phân tích")
    if args.min_sharpness is None:
        ordered_sharpness = sorted(item.sharpness for item in analysed)
        percentile_index = int((len(ordered_sharpness) - 1) * 0.40)
        sharpness_threshold = ordered_sharpness[percentile_index]
    else:
        sharpness_threshold = args.min_sharpness

    eligible = [
        item
        for item in analysed
        if item.sharpness >= sharpness_threshold
        and args.min_brightness <= item.brightness <= args.max_brightness
    ]
    selected = select_diverse_candidates(eligible, min(args.count, len(eligible)))
    selected_paths = {item.path for item in selected}

    images_output.mkdir(parents=True, exist_ok=False)
    for item in selected:
        shutil.copy2(item.path, images_output / item.path.name)

    selection_rows = [
        {
            "rank": rank,
            "filename": item.path.name,
            "source": item.path.as_posix(),
            "sharpness": f"{item.sharpness:.2f}",
            "brightness": f"{item.brightness:.2f}",
            "quality": f"{item.quality:.4f}",
            "manual_review": "",
            "notes": "",
        }
        for rank, item in enumerate(selected, start=1)
    ]
    _write_csv(
        args.output / "annotation_manifest.csv",
        [
            "rank",
            "filename",
            "source",
            "sharpness",
            "brightness",
            "quality",
            "manual_review",
            "notes",
        ],
        selection_rows,
    )

    quality_rows = []
    for item in analysed:
        reasons = []
        if item.sharpness < sharpness_threshold:
            reasons.append("blur")
        if item.brightness < args.min_brightness:
            reasons.append("dark")
        if item.brightness > args.max_brightness:
            reasons.append("bright")
        quality_rows.append(
            {
                "filename": item.path.name,
                "selected": "yes" if item.path in selected_paths else "no",
                "sharpness": f"{item.sharpness:.2f}",
                "brightness": f"{item.brightness:.2f}",
                "quality": f"{item.quality:.4f}",
                "quality_issue": ",".join(reasons),
            }
        )
    for path in unreadable:
        quality_rows.append(
            {
                "filename": path.name,
                "selected": "no",
                "sharpness": "",
                "brightness": "",
                "quality": "",
                "quality_issue": "unreadable",
            }
        )
    _write_csv(
        args.output / "quality_report.csv",
        ["filename", "selected", "sharpness", "brightness", "quality", "quality_issue"],
        quality_rows,
    )

    print(f"Đã đọc: {len(image_paths)} ảnh")
    print(f"Ngưỡng độ nét đã dùng: {sharpness_threshold:.2f}")
    print(f"Đạt ngưỡng chất lượng tự động: {len(eligible)} ảnh")
    print(f"Đã chọn và sao chép: {len(selected)} ảnh -> {images_output}")
    print(f"Danh sách gắn nhãn: {args.output / 'annotation_manifest.csv'}")
    print(f"Báo cáo chất lượng: {args.output / 'quality_report.csv'}")
    print("BẮT BUỘC: mở xem 30 ảnh, loại ảnh sai rồi mới upload lên CVAT.")
    return 0 if len(selected) == args.count else 2


if __name__ == "__main__":
    raise SystemExit(main())

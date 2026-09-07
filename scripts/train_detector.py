from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
DEFAULT_DATA = ROOT / "datasets" / "pen_parts" / "data.yaml"

from pen_assembly.detection_dataset import format_detection_report, inspect_detection_dataset


def _configure_utf8_console() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    _configure_utf8_console()
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("Thiếu ultralytics. Hãy cài requirements-camera.txt") from error

    parser = argparse.ArgumentParser(description="Fine-tune a closed-set pen component detector")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    report = inspect_detection_dataset(args.data)
    print(format_detection_report(report))
    if not report.is_trainable:
        raise SystemExit(
            "Dataset chưa thể train: cần ít nhất một bounding box hợp lệ và không có dòng nhãn lỗi."
        )

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "artifacts" / "training"),
        name="pen_parts_detector",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "datasets" / "pen_parts" / "data.yaml"


def main() -> int:
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


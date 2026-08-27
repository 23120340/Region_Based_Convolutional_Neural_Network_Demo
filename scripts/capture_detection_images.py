from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "datasets" / "pen_parts" / "raw"


def main() -> int:
    try:
        import cv2
    except ImportError as error:
        raise SystemExit("Thiếu OpenCV. Hãy cài requirements-camera.txt") from error

    parser = argparse.ArgumentParser(description="Capture raw camera images for bounding-box annotation")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    capture = cv2.VideoCapture(args.camera, backend)
    if not capture.isOpened():
        raise SystemExit(f"Không mở được camera {args.camera}")

    saved = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"SPACE save | Q quit | saved={saved}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 255, 80), 2)
            cv2.imshow("Capture Pen Dataset", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == 32:
                path = args.output / f"pen_{time.strftime('%Y%m%d_%H%M%S')}_{saved:04d}.jpg"
                cv2.imwrite(str(path), frame)
                saved += 1
                print(path)
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


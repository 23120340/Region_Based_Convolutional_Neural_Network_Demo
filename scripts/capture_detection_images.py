from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "datasets" / "pen_parts" / "raw"


def _configure_utf8_console() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    _configure_utf8_console()
    try:
        import cv2
    except ImportError as error:
        raise SystemExit("Thiếu OpenCV. Hãy cài requirements-camera.txt") from error

    parser = argparse.ArgumentParser(description="Capture raw camera images for bounding-box annotation")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-mirror", action="store_true", help="Không lật ngang hình camera")
    args = parser.parse_args()
    if args.width < 1 or args.height < 1:
        raise SystemExit("--width và --height phải lớn hơn 0")
    args.output.mkdir(parents=True, exist_ok=True)

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    capture = cv2.VideoCapture(args.camera, backend)
    if not capture.isOpened():
        raise SystemExit(f"Không mở được camera {args.camera}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    saved = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            clean_frame = frame if args.no_mirror else cv2.flip(frame, 1)
            preview = clean_frame.copy()
            actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cv2.putText(
                preview,
                f"SPACE save | Q quit | saved={saved} | {actual_width}x{actual_height}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (30, 255, 80),
                2,
            )
            cv2.imshow("Capture Pen Dataset", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == 32:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                path = args.output / f"pen_{timestamp}_{saved:04d}.jpg"
                if not cv2.imwrite(str(path), clean_frame):
                    raise SystemExit(f"Không lưu được ảnh: {path}")
                saved += 1
                print(path)
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

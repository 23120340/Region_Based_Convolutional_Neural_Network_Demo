from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assembly.detection_dataset import format_detection_report, inspect_detection_dataset


def _configure_utf8_console() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    _configure_utf8_console()
    parser = argparse.ArgumentParser(description="Validate a YOLO earbud dataset before training")
    parser.add_argument("--data", type=Path, default=ROOT / "datasets" / "earbud_geometry" / "data.yaml")
    args = parser.parse_args()
    report = inspect_detection_dataset(args.data)
    print(format_detection_report(report))
    if not report.is_trainable:
        print("\nBLOCKED: Hãy gán bounding box hợp lệ trước khi train YOLO.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

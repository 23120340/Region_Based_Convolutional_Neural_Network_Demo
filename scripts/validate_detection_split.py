from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assembly.dataset_split import validate_detection_capture_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Reject frame/video/session leakage in YOLO splits")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="CSV columns: image_id,video_id,session_id,split")
    args = parser.parse_args()
    try:
        counts = validate_detection_capture_manifest(args.manifest)
    except ValueError as error:
        print(f"BLOCKED: {error}")
        return 2
    print(f"PASS: session/video split is isolated; images={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

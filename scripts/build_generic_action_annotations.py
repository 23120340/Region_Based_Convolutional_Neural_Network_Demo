"""Create a non-destructive 4-label LSTM annotation file."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assembly.holdout import load_holdout_registry

MAPPING = {
    "insert_first_earbud": "insert_earbud",
    "insert_second_earbud": "insert_earbud",
    "insert_earbud": "insert_earbud",
}
ALLOWED = {"idle", "open_case", "insert_earbud", "close_case"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert LSTM labels to generic 4-action schema")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path,
                        default=ROOT / "configs/holdout_regression.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite {args.output}; choose a new path or pass --overwrite")
    with args.input.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if "video_id" not in fieldnames or "action_name" not in fieldnames:
        raise SystemExit("Annotation CSV must contain video_id and action_name")
    registry = load_holdout_registry(args.holdout_manifest)
    try:
        registry.reject_video_ids(row["video_id"] for row in rows)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    for row in rows:
        row["action_name"] = MAPPING.get(row["action_name"], row["action_name"])
    unknown = sorted({row["action_name"] for row in rows} - ALLOWED)
    if unknown:
        raise SystemExit(f"Unsupported actions for generic model: {unknown}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} generic-action rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

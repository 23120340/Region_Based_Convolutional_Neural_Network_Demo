from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pen_assembly.config import load_config
from pen_assembly.fsm import ConfigurableAssemblyTracker
from pen_assembly.paths import DEFAULT_CONFIG
from pen_assembly.scenarios import SCENARIOS


def _configure_utf8_console() -> None:
    """Keep Vietnamese output readable on legacy Windows console code pages."""

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    _configure_utf8_console()
    parser = argparse.ArgumentParser(description="Mô phỏng quy trình lắp bút không cần camera")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="correct")
    args = parser.parse_args()

    config = load_config(DEFAULT_CONFIG)
    tracker = ConfigurableAssemblyTracker(config)
    has_violation = False
    print(f"Kịch bản: {args.scenario}")
    for action in SCENARIOS[args.scenario]:
        outcome = tracker.process(action)
        has_violation = has_violation or outcome.type == "VIOLATION"
        print(f"[{outcome.type:9}] {action:16} {outcome.previous_state} -> {outcome.state}")
        print(f"            {outcome.message}")
    print(f"Kết quả: {'CẢNH BÁO' if has_violation else 'HOÀN TẤT' if tracker.is_complete else 'CHƯA HOÀN TẤT'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

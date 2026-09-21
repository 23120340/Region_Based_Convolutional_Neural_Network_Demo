"""Backward-compatible entry point for the earbud project profile."""

from pathlib import Path

from run_hybrid import main


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    raise SystemExit(main(ROOT / "configs" / "projects" / "earbud.json"))

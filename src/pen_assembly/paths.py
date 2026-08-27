from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "pen_fsm_config.json"
DEFAULT_EVENT_LOG = PROJECT_ROOT / "artifacts" / "events.jsonl"


from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "pen_fsm_config.json"
DEFAULT_ACTION_CONFIG = PROJECT_ROOT / "configs" / "action_model_config.json"
DEFAULT_EVENT_LOG = PROJECT_ROOT / "artifacts" / "events.jsonl"
DEFAULT_ACTION_VIDEOS = PROJECT_ROOT / "data" / "pen_actions" / "raw_videos"
DEFAULT_ACTION_ANNOTATIONS = PROJECT_ROOT / "data" / "pen_actions" / "annotations.csv"
DEFAULT_FEATURE_CACHE = PROJECT_ROOT / "data" / "pen_actions" / "features"

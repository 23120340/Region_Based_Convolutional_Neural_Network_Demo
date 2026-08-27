import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pen_assembly.config import load_config
from pen_assembly.paths import DEFAULT_CONFIG


class ConfigTests(unittest.TestCase):
    def test_project_config_is_valid(self) -> None:
        config = load_config(DEFAULT_CONFIG)
        self.assertEqual(config.initial_state, "S0_IDLE")
        self.assertEqual(len(config.workflow), 5)

    def test_rejects_unknown_transition_target(self) -> None:
        raw = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        raw["states"]["S0_IDLE"]["allowed"]["pick_barrel"] = "DOES_NOT_EXIST"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "state lạ"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()


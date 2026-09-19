import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assembly.config import load_config
from assembly.fsm import ConfigurableAssemblyTracker
from assembly.model_contract import Prediction
from assembly.monitor import AssemblyMonitor, JsonlEventLogger
from assembly.paths import DEFAULT_CONFIG
from assembly.smoother import TemporalDebouncer


class MonitorTests(unittest.TestCase):
    def test_stable_prediction_is_logged_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            config = load_config(DEFAULT_CONFIG)
            monitor = AssemblyMonitor(
                ConfigurableAssemblyTracker(config),
                TemporalDebouncer(idle_actions=config.idle_actions),
                JsonlEventLogger(path),
            )
            outcomes = [
                monitor.submit_prediction(Prediction("pick_case", 0.95))
                for _ in range(8)
            ]
            self.assertEqual(sum(item is not None for item in outcomes), 1)
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["type"], "PASS")


if __name__ == "__main__":
    unittest.main()



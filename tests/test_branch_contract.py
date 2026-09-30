"""Guard the generic action model while preserving the detailed FSM workflow."""
import csv
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from assembly.action_config import load_action_model_config
from assembly.config import load_config
from assembly.project_config import load_project_config
from assembly.temporal_config import load_temporal_fusion_config


class BranchContractTests(unittest.TestCase):
    def test_main_action_configs_keep_five_separate_labels(self):
        expected = ("idle", "open_case", "insert_first_earbud", "insert_second_earbud", "close_case")
        for name in ("action_earbud_pilot_config.json", "action_earbud_v2_config.json"):
            self.assertEqual(load_action_model_config(ROOT / "configs" / name).actions, expected)

    def test_annotations_keep_first_second_labels_and_pilot_video_splits(self):
        for name in ("annotations_v2.csv", "annotations_v2_pilot_split.csv"):
            with (ROOT / "data/earbud_actions" / name).open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            labels = {row["action_name"] for row in rows}
            self.assertIn("insert_first_earbud", labels)
            self.assertIn("insert_second_earbud", labels)
            self.assertNotIn("insert_earbud", labels)
            if "pilot_split" in name:
                assignments = {}
                for row in rows:
                    assignments.setdefault(row["video_id"], set()).add(row["split"])
                self.assertTrue(all(len(splits) == 1 for splits in assignments.values()))
                self.assertEqual({row["split"] for row in rows}, {"train", "val", "test"})

    def test_profile_and_fsm_separate_model_labels_from_workflow(self):
        profile = load_project_config(ROOT / "configs/projects/earbud_v2.json", ROOT)
        self.assertEqual(profile.action_config, ROOT / "configs/action_earbud_generic_config.json")
        self.assertEqual(profile.action_model, ROOT / "artifacts/action_model_insert_earbud/best.pt")
        self.assertEqual(
            load_action_model_config(profile.action_config).actions,
            ("idle", "open_case", "insert_earbud", "close_case"),
        )
        fsm = load_config(profile.fsm_config)
        self.assertEqual([step.action for step in fsm.workflow],
                         ["open_case", "insert_first_earbud", "insert_second_earbud", "close_case"])
        self.assertIsNotNone(profile.temporal_config)
        temporal = load_temporal_fusion_config(profile.temporal_config)
        self.assertEqual(temporal.confidence.action_min, 0.5)
        self.assertEqual(temporal.confidence.occupied_earbud_min, 0.5)


if __name__ == "__main__":
    unittest.main()

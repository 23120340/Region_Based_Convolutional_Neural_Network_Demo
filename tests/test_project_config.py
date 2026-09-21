import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assembly.earbud_fusion import EarbudFusionEngine
from assembly.project_config import build_fusion_engine, load_project_config


class ProjectConfigTests(unittest.TestCase):
    def test_earbud_profile_builds_fusion_engine(self) -> None:
        profile = load_project_config(ROOT / "configs" / "projects" / "earbud.json", ROOT)
        self.assertEqual(profile.name, "earbud")
        self.assertEqual(
            profile.camera_config,
            (ROOT / "configs" / "camera_earbud_hybrid_config.json").resolve(),
        )
        engine = build_fusion_engine(profile.fusion, stable_frames=2)
        self.assertIsInstance(engine, EarbudFusionEngine)
        self.assertEqual(engine.stable_frames, 2)

    def test_profile_rejects_path_outside_project(self) -> None:
        raw = json.loads((ROOT / "configs" / "projects" / "earbud.json").read_text(encoding="utf-8"))
        raw["camera_config"] = "../outside.json"
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "invalid-project.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "project root"):
                load_project_config(path, ROOT)

    def test_relative_profile_falls_back_to_project_root(self) -> None:
        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                profile = load_project_config("configs/projects/earbud.json", ROOT)
            finally:
                os.chdir(previous_cwd)
        self.assertEqual(profile.name, "earbud")
        self.assertEqual(
            profile.action_model,
            (ROOT / "artifacts" / "action_model" / "best.pt").resolve(),
        )


if __name__ == "__main__":
    unittest.main()

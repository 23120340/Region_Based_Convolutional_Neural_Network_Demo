import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "split_dataset.py"
SPEC = importlib.util.spec_from_file_location("split_dataset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SplitDatasetConfigTests(unittest.TestCase):
    def test_new_dataset_uses_camera_class_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera = root / "camera.json"
            camera.write_text(
                json.dumps({"classes": [{"label": "open_case"}, {"label": "earbud"}]}),
                encoding="utf-8",
            )
            names = MODULE.resolve_class_names(root / "data.yaml", camera)
        self.assertEqual(names, {0: "open_case", 1: "earbud"})

    def test_existing_yaml_cannot_be_silently_relabelled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            yaml_path = root / "data.yaml"
            yaml_path.write_text("names: [Case, Earbud]\n", encoding="utf-8")
            camera = root / "camera.json"
            camera.write_text(
                json.dumps({"classes": [{"label": "open_case"}, {"label": "earbud"}]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "khác"):
                MODULE.resolve_class_names(yaml_path, camera)


if __name__ == "__main__":
    unittest.main()

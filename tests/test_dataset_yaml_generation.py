import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "merge_and_finetune_earbud.py"
SPEC = importlib.util.spec_from_file_location("merge_and_finetune_earbud", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DatasetYamlGenerationTests(unittest.TestCase):
    def test_generated_yaml_does_not_embed_machine_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "earbud_merged"
            root.mkdir()
            yaml_path = MODULE.write_data_yaml(root)
            contents = yaml_path.read_text(encoding="utf-8")

        self.assertNotIn("path:", contents)
        self.assertIn("train: train/images", contents)
        self.assertNotIn(str(root), contents)


if __name__ == "__main__":
    unittest.main()

import json
import hashlib
import shutil
import contextlib
import io
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_earbud_coco_kaggle as kaggle_train


class KaggleCocoTrainingTests(unittest.TestCase):
    def _make_split(self, root: Path, split: str) -> None:
        split_dir = root / split
        split_dir.mkdir(parents=True)
        (split_dir / f"{split}.jpg").write_bytes(f"{root.name}-{split}".encode())
        payload = {
            "images": [
                {
                    "id": 10,
                    "file_name": f"{split}.jpg",
                    "width": 100,
                    "height": 50,
                }
            ],
            "categories": [
                {"id": 0, "name": "earbud-detect"},
                {"id": 1, "name": "Earphone_Case"},
                {"id": 3, "name": "Left_Earbud"},
            ],
            "annotations": [
                {"id": 1, "image_id": 10, "category_id": 1, "bbox": [0, 0, 50, 25]},
                {"id": 2, "image_id": 10, "category_id": 3, "bbox": [90, 40, 20, 20]},
            ],
        }
        (split_dir / kaggle_train.ANNOTATION_FILE).write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def test_conversion_ignores_unused_root_category_and_remaps_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "earbud.coco"
            for split in ("train", "valid", "test"):
                self._make_split(source, split)
            output = base / "prepared"

            yaml_path, report = kaggle_train.convert_coco_to_yolo(source, output)

            self.assertEqual(report["class_names"], ["Earphone_Case", "Left_Earbud"])
            self.assertEqual(report["splits"]["train"], {"images": 1, "boxes": 2})
            label_lines = (output / "train/labels/train.txt").read_text().splitlines()
            self.assertEqual(label_lines[0], "0 0.25000000 0.25000000 0.50000000 0.50000000")
            self.assertEqual(label_lines[1], "1 0.95000000 0.90000000 0.10000000 0.20000000")
            yaml_text = yaml_path.read_text(encoding="utf-8")
            self.assertIn('0: "Earphone_Case"', yaml_text)
            self.assertIn('1: "Left_Earbud"', yaml_text)
            self.assertNotIn("earbud-detect", yaml_text)

    def test_discovery_requires_explicit_choice_for_multiple_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            input_root = Path(directory)
            for dataset_name in ("one", "two"):
                source = input_root / dataset_name
                self._make_split(source, "train")
                self._make_split(source, "valid")
            with self.assertRaisesRegex(ValueError, "nhiều COCO dataset"):
                kaggle_train.resolve_dataset_root(None, input_root)

    def test_geometry_guard_rejects_legacy_classes(self):
        with self.assertRaisesRegex(ValueError, "chưa phải geometry v2"):
            kaggle_train.check_project_compatibility(
                ["Earphone_Case", "Empty_Slot", "Left_Earbud", "Right_Earbud"],
                require_geometry_v2=True,
            )

    def test_kaggle_notebook_is_valid_and_code_cells_compile(self):
        notebook_path = ROOT / "Kaggle_Training_Earbud.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        sources = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        self.assertEqual(
            sum("r'/kaggle/input/CHANGE_ME/earbud-detect.coco'" in source for source in sources),
            1,
        )
        for index, source in enumerate(sources, start=1):
            if source.lstrip().startswith("# CELL 2"):
                continue  # %pip is valid IPython syntax, not plain Python syntax.
            compile(source, f"notebook-cell-{index}", "exec")

    def test_kaggle_notebook_merges_two_coco_roots_by_class_name(self):
        notebook = json.loads(
            (ROOT / "Kaggle_Training_Earbud.ipynb").read_text(encoding="utf-8")
        )
        converter = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
            and "# CELL 4" in "".join(cell.get("source", []))
        )
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            sources = [base / "source_one", base / "source_two"]
            for source in sources:
                for split in ("train", "valid", "test"):
                    self._make_split(source, split)
            prepared = (base / "prepared").as_posix()
            converter = converter.replace(
                "/kaggle/working/earbud_yolo_prepared", prepared
            )
            namespace = {
                "Counter": Counter,
                "DATASET_ROOTS": sources,
                "Path": Path,
                "STRICT_GEOMETRY_V2": False,
                "defaultdict": defaultdict,
                "hashlib": hashlib,
                "json": json,
                "shutil": shutil,
            }
            with contextlib.redirect_stdout(io.StringIO()):
                exec(converter, namespace)

            report = namespace["report"]
            self.assertEqual(report["class_names"], ["Earphone_Case", "Left_Earbud"])
            self.assertEqual(report["splits"]["train"]["images"], 2)
            self.assertEqual(report["splits"]["val"]["images"], 2)
            self.assertEqual(report["splits"]["test"]["images"], 2)
            self.assertEqual(report["skipped_duplicate_images"], 0)


if __name__ == "__main__":
    unittest.main()

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import merge_rnn_earbud_datasets as merger


class MergeRnnEarbudDatasetTests(unittest.TestCase):
    def test_lr_kaggle_notebook_is_valid_and_code_cells_compile(self):
        notebook = json.loads(
            (ROOT / "Kaggle_Training_Earbud_LR.ipynb").read_text(encoding="utf-8")
        )
        self.assertEqual(notebook["nbformat"], 4)
        sources = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        self.assertEqual(sum("DATASET_ROOT = r'" in source for source in sources), 1)
        for index, source in enumerate(sources, start=1):
            if source.lstrip().startswith("# CELL 2"):
                continue
            compile(source, f"lr-notebook-cell-{index}", "exec")

    def test_polygon_is_converted_to_enclosing_bbox_and_class_is_remapped(self):
        mapping = {0: 2}
        output = merger.label_row_to_bbox(
            "0 0.10 0.20 0.30 0.20 0.30 0.60 0.10 0.60", mapping
        )
        self.assertEqual(output, "2 0.20000000 0.40000000 0.20000000 0.40000000")

    def test_degenerate_polygon_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "không có diện tích"):
            merger.label_row_to_bbox(
                "0 0.20 0.10 0.20 0.30 0.20 0.50", {0: 2}
            )

    def test_known_aliases_map_to_canonical_seven_classes(self):
        mapping = merger.build_class_map(
            [
                "Case_Closed",
                "Case_Open",
                "Earbud_Left",
                "Earbud_Right",
                "Empty_Slot_Left",
                "Empty_Slot_Right",
                "Hand",
            ],
            source=Path("primary"),
        )
        self.assertEqual(mapping, {0: 1, 1: 0, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6})
        extra_mapping = merger.build_class_map(
            ["Earbud_Left", "Right-Earbud"], source=Path("extra")
        )
        self.assertEqual(extra_mapping, {0: 2, 1: 3})

    def test_extra_data_is_added_to_train_only(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            primary = base / "primary"
            extra = base / "extra"
            output = base / "output"
            (primary / "data.yaml").parent.mkdir(parents=True)
            (primary / "data.yaml").write_text(
                "names: ['Case_Open', 'Case_Closed', 'Earbud_Left', "
                "'Earbud_Right', 'Empty_Slot_Left', 'Empty_Slot_Right', 'Hand']\n",
                encoding="utf-8",
            )
            (extra / "data.yaml").parent.mkdir(parents=True)
            (extra / "data.yaml").write_text(
                "names: ['Earbud_Left', 'Right-Earbud']\n", encoding="utf-8"
            )

            for split in ("train", "valid", "test"):
                image_dir = primary / split / "images"
                label_dir = primary / split / "labels"
                image_dir.mkdir(parents=True)
                label_dir.mkdir(parents=True)
                (image_dir / f"{split}.jpg").write_bytes(b"image")
                (label_dir / f"{split}.txt").write_text(
                    "2 0.5 0.5 0.2 0.2\n", encoding="utf-8"
                )
            (extra / "train/images").mkdir(parents=True)
            (extra / "train/labels").mkdir(parents=True)
            (extra / "train/images/extra.jpg").write_bytes(b"extra-image")
            (extra / "train/labels/extra.txt").write_text(
                "1 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )

            report = merger.merge_datasets(primary, extra, output, image_mode="copy")

            self.assertEqual(report["splits"]["extra/train"]["images"], 1)
            self.assertEqual(len(list((output / "train/images").iterdir())), 2)
            self.assertEqual(len(list((output / "valid/images").iterdir())), 1)
            self.assertEqual(len(list((output / "test/images").iterdir())), 1)
            self.assertEqual(
                (output / "train/labels/lr_extra_extra.txt").read_text().split()[0],
                "3",
            )

    def test_extra_image_already_in_primary_is_not_duplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            primary = base / "primary"
            extra = base / "extra"
            output = base / "output"
            (primary / "data.yaml").parent.mkdir(parents=True)
            (primary / "data.yaml").write_text(
                "names: ['Case_Open', 'Case_Closed', 'Earbud_Left', "
                "'Earbud_Right', 'Empty_Slot_Left', 'Empty_Slot_Right', 'Hand']\n",
                encoding="utf-8",
            )
            (extra / "data.yaml").parent.mkdir(parents=True)
            (extra / "data.yaml").write_text(
                "names: ['Earbud_Left', 'Right-Earbud']\n", encoding="utf-8"
            )
            for split in ("train", "valid", "test"):
                image_dir = primary / split / "images"
                label_dir = primary / split / "labels"
                image_dir.mkdir(parents=True)
                label_dir.mkdir(parents=True)
                content = b"same-image" if split == "train" else split.encode()
                (image_dir / f"{split}.jpg").write_bytes(content)
                (label_dir / f"{split}.txt").write_text(
                    "2 0.5 0.5 0.2 0.2\n", encoding="utf-8"
                )
            (extra / "train/images").mkdir(parents=True)
            (extra / "train/labels").mkdir(parents=True)
            (extra / "train/images/copy.jpg").write_bytes(b"same-image")
            (extra / "train/labels/copy.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )

            report = merger.merge_datasets(primary, extra, output, image_mode="copy")

            self.assertEqual(report["splits"]["extra/train"]["images"], 0)
            self.assertEqual(report["splits"]["extra/train"]["duplicates_skipped"], 1)
            self.assertEqual(len(list((output / "train/images").iterdir())), 1)


if __name__ == "__main__":
    unittest.main()

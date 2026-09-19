import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assembly.detection_dataset import inspect_detection_dataset


class DetectionDatasetTests(unittest.TestCase):
    def make_dataset(
        self,
        label: str,
        *,
        include_val: bool = True,
    ) -> tuple[tempfile.TemporaryDirectory, Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        for split in ("train", "val") if include_val else ("train",):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
            (root / "images" / split / f"{split}_sample.jpg").write_bytes(b"placeholder")
            (root / "labels" / split / f"{split}_sample.txt").write_text(label, encoding="utf-8")
        yaml_path = root / "data.yaml"
        yaml_path.write_text(
            "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: barrel\n",
            encoding="utf-8",
        )
        return temporary, yaml_path

    def test_valid_box_is_trainable(self) -> None:
        temporary, yaml_path = self.make_dataset("0 0.5 0.5 0.4 0.2\n")
        self.addCleanup(temporary.cleanup)
        report = inspect_detection_dataset(yaml_path)
        self.assertTrue(report.is_trainable)
        self.assertEqual(report.total_boxes, 2)

    def test_all_empty_labels_are_not_trainable(self) -> None:
        temporary, yaml_path = self.make_dataset("")
        self.addCleanup(temporary.cleanup)
        report = inspect_detection_dataset(yaml_path)
        self.assertFalse(report.is_trainable)
        self.assertEqual(len(report.empty_labels), 2)

    def test_invalid_coordinate_is_reported(self) -> None:
        temporary, yaml_path = self.make_dataset("0 1.2 0.5 0.4 0.2\n")
        self.addCleanup(temporary.cleanup)
        report = inspect_detection_dataset(yaml_path)
        self.assertFalse(report.is_trainable)
        self.assertEqual(len(report.invalid_labels), 2)

    def test_validation_split_is_required(self) -> None:
        temporary, yaml_path = self.make_dataset(
            "0 0.5 0.5 0.4 0.2\n",
            include_val=False,
        )
        self.addCleanup(temporary.cleanup)
        report = inspect_detection_dataset(yaml_path)
        self.assertFalse(report.is_trainable)
        self.assertIn("split val chưa có ảnh", report.training_issues)

    def test_every_configured_class_requires_a_box(self) -> None:
        temporary, yaml_path = self.make_dataset("0 0.5 0.5 0.4 0.2\n")
        self.addCleanup(temporary.cleanup)
        with yaml_path.open("a", encoding="utf-8") as file:
            file.write("  1: spring\n")
        report = inspect_detection_dataset(yaml_path)
        self.assertFalse(report.is_trainable)
        self.assertEqual(report.missing_classes, (1,))


if __name__ == "__main__":
    unittest.main()


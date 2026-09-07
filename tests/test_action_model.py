import importlib
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pen_assembly.action_config import load_action_model_config
from pen_assembly.models import PenAssemblyActionNet


class ActionModelTests(unittest.TestCase):
    def test_legacy_model_import_points_to_package_model(self) -> None:
        legacy = importlib.import_module("models.pen_action_net")
        self.assertIs(legacy.PenAssemblyActionNet, PenAssemblyActionNet)

    def test_project_action_config_is_valid(self) -> None:
        config = load_action_model_config(ROOT / "configs" / "action_model_config.json")
        self.assertEqual(config.spatial.embedding_dim, 768)
        self.assertEqual(config.temporal.sequence_length, 16)
        self.assertEqual(len(config.actions), 6)

    def test_bilstm_uses_final_forward_and_backward_hidden_states(self) -> None:
        torch.manual_seed(7)
        model = PenAssemblyActionNet(
            input_dim=8,
            hidden_dim=4,
            num_layers=2,
            num_classes=3,
            dropout=0.0,
            bidirectional=True,
            head_dim=6,
        ).eval()
        features = torch.randn(2, 5, 8)
        with torch.no_grad():
            _, (hidden, _) = model.lstm(features)
            expected_context = torch.cat((hidden[-2], hidden[-1]), dim=1)
            actual_context = model.temporal_context(features)
            logits = model(features)
        self.assertTrue(torch.allclose(actual_context, expected_context))
        self.assertEqual(tuple(logits.shape), (2, 3))

    def test_rejects_invalid_input_rank(self) -> None:
        model = PenAssemblyActionNet(8, 4, 1, 3, bidirectional=False)
        with self.assertRaisesRegex(ValueError, "shape"):
            model(torch.randn(5, 8))


if __name__ == "__main__":
    unittest.main()

import importlib
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assembly.action_config import load_action_model_config
from assembly.models import AssemblyActionNet
from assembly.models.vit_lstm_recognizer import ViTLstmActionRecognizer


class ActionModelTests(unittest.TestCase):
    def test_project_action_config_is_valid(self) -> None:
        config = load_action_model_config(ROOT / "configs" / "action_earbud_config.json")
        self.assertEqual(config.spatial.embedding_dim, 768)
        self.assertEqual(config.temporal.sequence_length, 16)
        self.assertEqual(len(config.actions), 4)

        v2 = load_action_model_config(ROOT / "configs" / "action_earbud_v2_config.json")
        self.assertEqual(v2.spatial.backbone, "facebook/dinov2-small")
        self.assertEqual(v2.spatial.embedding_dim, 384)
        self.assertEqual(
            v2.actions,
            (
                "idle",
                "open_case",
                "insert_first_earbud",
                "insert_second_earbud",
                "close_case",
                "remove_earbud",
            ),
        )

    def test_bilstm_uses_final_forward_and_backward_hidden_states(self) -> None:
        torch.manual_seed(7)
        model = AssemblyActionNet(
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
        model = AssemblyActionNet(8, 4, 1, 3, bidirectional=False)
        with self.assertRaisesRegex(ValueError, "shape"):
            model(torch.randn(5, 8))

    def test_cached_embeddings_produce_prediction_without_encoder(self) -> None:
        recognizer = ViTLstmActionRecognizer.__new__(ViTLstmActionRecognizer)
        recognizer.config = load_action_model_config(ROOT / "configs" / "action_earbud_config.json")
        recognizer.device = torch.device("cpu")
        temporal = recognizer.config.temporal
        recognizer.model = AssemblyActionNet(
            input_dim=recognizer.config.spatial.embedding_dim,
            hidden_dim=temporal.hidden_dim,
            num_layers=temporal.num_layers,
            num_classes=len(recognizer.config.actions),
            dropout=0.0,
            bidirectional=temporal.bidirectional,
            head_dim=temporal.head_dim,
        ).eval()
        embeddings = torch.randn(
            recognizer.config.temporal.sequence_length,
            recognizer.config.spatial.embedding_dim,
        )
        prediction = recognizer.predict_embeddings(embeddings)
        self.assertIn(prediction.action, recognizer.config.actions)
        self.assertGreaterEqual(prediction.confidence, 0.0)
        self.assertLessEqual(prediction.confidence, 1.0)

    def test_cached_embeddings_validate_feature_dimension(self) -> None:
        recognizer = ViTLstmActionRecognizer.__new__(ViTLstmActionRecognizer)
        recognizer.config = load_action_model_config(ROOT / "configs" / "action_earbud_config.json")
        recognizer.device = torch.device("cpu")
        embeddings = torch.randn(recognizer.config.temporal.sequence_length, 10)
        with self.assertRaisesRegex(ValueError, "Embedding"):
            recognizer.predict_embeddings(embeddings)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from ..action_config import load_action_model_config
from ..model_contract import Prediction
from .action_net import PenAssemblyActionNet
from .spatial_encoder import ViTSpatialEncoder


class ViTLstmActionRecognizer:
    """Runtime implementation of ActionRecognizer for a trained checkpoint."""

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: str | None = None,
        local_files_only: bool = True,
    ) -> None:
        self.config = load_action_model_config(config_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder = ViTSpatialEncoder(
            self.config.spatial.backbone,
            str(self.device),
            freeze=True,
            local_files_only=local_files_only,
        )
        if self.encoder.embedding_dim != self.config.spatial.embedding_dim:
            raise ValueError("embedding_dim của backbone không khớp action_model_config")
        temporal = self.config.temporal
        self.model = PenAssemblyActionNet(
            input_dim=self.config.spatial.embedding_dim,
            hidden_dim=temporal.hidden_dim,
            num_layers=temporal.num_layers,
            num_classes=len(self.config.actions),
            dropout=temporal.dropout,
            bidirectional=temporal.bidirectional,
            head_dim=temporal.head_dim,
        ).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        checkpoint_actions = tuple(checkpoint.get("actions", ()))
        if checkpoint_actions and checkpoint_actions != self.config.actions:
            raise ValueError("Danh sách action trong checkpoint không khớp action config")
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def encode_frame(self, frame: Any) -> torch.Tensor:
        """Encode only the newest RGB frame for a realtime embedding cache."""

        return self.encoder.encode_images([frame], batch_size=1)[0]

    @torch.inference_mode()
    def predict_embeddings(self, embeddings: Sequence[torch.Tensor] | torch.Tensor) -> Prediction:
        """Classify an already encoded temporal window without rerunning ViT."""

        required = self.config.temporal.sequence_length
        if isinstance(embeddings, torch.Tensor):
            features = embeddings
        else:
            features = torch.stack(tuple(embeddings)) if embeddings else torch.empty(0)
        if len(features) < required:
            raise ValueError(f"Cần ít nhất {required} embeddings, hiện có {len(features)}")
        if features.ndim != 2 or features.shape[-1] != self.config.spatial.embedding_dim:
            raise ValueError(
                "Embedding phải có shape "
                f"(N, {self.config.spatial.embedding_dim}), nhận {tuple(features.shape)}"
            )
        logits = self.model(features[-required:].unsqueeze(0).to(self.device))
        probabilities = torch.softmax(logits, dim=-1)[0]
        confidence, class_id = torch.max(probabilities, dim=0)
        return Prediction(self.config.actions[int(class_id.item())], float(confidence.item()))

    @torch.inference_mode()
    def predict(self, frames: Sequence[Any]) -> Prediction:
        required = self.config.temporal.sequence_length
        if len(frames) < required:
            raise ValueError(f"Cần ít nhất {required} frames, hiện có {len(frames)}")
        embeddings = self.encoder.encode_images(frames[-required:])
        return self.predict_embeddings(embeddings)

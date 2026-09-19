from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


class ViTSpatialEncoder:
    """Frozen Hugging Face ViT feature extractor returning CLS embeddings."""

    def __init__(self, model_name: str, device: str | None = None, freeze: bool = True) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as error:
            raise RuntimeError("Thiếu transformers; hãy cài requirements-ml.txt") from error

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.embedding_dim = int(self.backbone.config.hidden_size)
        if freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

    @torch.inference_mode()
    def encode_images(self, images: Sequence[Any], batch_size: int = 16) -> torch.Tensor:
        """Encode RGB NumPy/PIL images and return a CPU tensor shaped ``(N, D)``."""

        if not images:
            return torch.empty((0, self.embedding_dim), dtype=torch.float32)
        batches: list[torch.Tensor] = []
        for start in range(0, len(images), batch_size):
            inputs = self.processor(images=list(images[start : start + batch_size]), return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            outputs = self.backbone(pixel_values=pixel_values)
            batches.append(outputs.last_hidden_state[:, 0, :].detach().cpu())
        return torch.cat(batches, dim=0)

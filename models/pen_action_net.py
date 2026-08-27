"""Optional Hybrid ViT-LSTM classifier for phase 2.

This module is deliberately not imported by the dependency-free MVP. It becomes
usable after real labelled features are available and PyTorch is installed.
"""

from __future__ import annotations

import torch
from torch import nn


class PenAssemblyActionNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 6,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Classify tensors shaped ``(batch, time, embedding_dim)``."""

        sequence, _ = self.lstm(features)
        return self.head(sequence[:, -1, :])


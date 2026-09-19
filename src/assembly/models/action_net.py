from __future__ import annotations

import torch
from torch import nn


class PenAssemblyActionNet(nn.Module):
    """Configurable LSTM/BiLSTM classifier for cached spatial embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.3,
        bidirectional: bool = True,
        head_dim: int = 128,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        context_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(context_dim, head_dim),
            nn.LayerNorm(head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, num_classes),
        )

    def temporal_context(self, features: torch.Tensor) -> torch.Tensor:
        """Return final recurrent context for the complete input window."""

        _, (hidden, _) = self.lstm(features)
        if self.bidirectional:
            # The last layer occupies the last two entries: forward, backward.
            return torch.cat((hidden[-2], hidden[-1]), dim=1)
        return hidden[-1]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Classify embeddings shaped ``(batch, time, input_dim)``."""

        if features.ndim != 3:
            raise ValueError("features phải có shape (batch, time, embedding_dim)")
        return self.head(self.temporal_context(features))

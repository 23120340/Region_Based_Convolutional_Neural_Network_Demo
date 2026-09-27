from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpatialConfig:
    backbone: str
    embedding_dim: int
    sample_fps: float
    freeze: bool


@dataclass(frozen=True)
class TemporalConfig:
    sequence_length: int
    window_stride: int
    hidden_dim: int
    num_layers: int
    bidirectional: bool
    head_dim: int
    dropout: float


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    seed: int


@dataclass(frozen=True)
class InferenceConfig:
    min_confidence: float = 0.5


@dataclass(frozen=True)
class ActionModelConfig:
    spatial: SpatialConfig
    temporal: TemporalConfig
    training: TrainingConfig
    actions: tuple[str, ...]
    inference: InferenceConfig

    @property
    def action_to_id(self) -> dict[str, int]:
        return {action: index for index, action in enumerate(self.actions)}

    def accepts_confidence(self, confidence: float) -> bool:
        """The runtime accepts only predictions strictly above the configured threshold."""

        return confidence > self.inference.min_confidence


def load_action_model_config(path: str | Path) -> ActionModelConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if raw.get("schema_version") != 1:
        raise ValueError("action_model_config schema_version phải bằng 1")

    spatial = SpatialConfig(**raw["spatial"])
    temporal = TemporalConfig(**raw["temporal"])
    training = TrainingConfig(**raw["training"])
    actions = tuple(str(action) for action in raw["actions"])
    inference = InferenceConfig(**raw.get("inference", {}))

    if spatial.embedding_dim < 1:
        raise ValueError("embedding_dim phải > 0")
    if spatial.sample_fps <= 0:
        raise ValueError("sample_fps phải > 0")
    if temporal.sequence_length < 1 or temporal.window_stride < 1:
        raise ValueError("sequence_length và window_stride phải > 0")
    if temporal.num_layers < 1 or temporal.hidden_dim < 1 or temporal.head_dim < 1:
        raise ValueError("cấu hình temporal phải > 0")
    if not 0.0 <= temporal.dropout < 1.0:
        raise ValueError("dropout phải nằm trong [0, 1)")
    if len(actions) < 2 or len(set(actions)) != len(actions):
        raise ValueError("actions phải có ít nhất hai nhãn và không được trùng")
    if not 0.0 <= inference.min_confidence <= 1.0:
        raise ValueError("inference.min_confidence phải nằm trong [0, 1]")

    return ActionModelConfig(spatial, temporal, training, actions, inference)

"""Reusable core for camera-based assembly monitoring projects."""

from .config import AssemblyConfig, load_config
from .earbud_fusion import EarbudFusionEngine, EarbudFusionEvent, EarbudSceneState
from .fsm import ConfigurableAssemblyTracker, FsmOutcome
from .model_contract import Prediction
from .project_config import FusionConfig, ProjectConfig, build_fusion_engine, load_project_config
from .smoother import StablePrediction, TemporalDebouncer

__all__ = [
    "AssemblyConfig",
    "ConfigurableAssemblyTracker",
    "EarbudFusionEngine",
    "EarbudFusionEvent",
    "EarbudSceneState",
    "FsmOutcome",
    "FusionConfig",
    "Prediction",
    "ProjectConfig",
    "StablePrediction",
    "TemporalDebouncer",
    "build_fusion_engine",
    "load_config",
    "load_project_config",
]

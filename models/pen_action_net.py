"""Backward-compatible import; implementation now lives inside the package."""

try:
    from pen_assembly.models.action_net import PenAssemblyActionNet
except ModuleNotFoundError as error:
    if error.name != "pen_assembly":
        raise
    from src.pen_assembly.models.action_net import PenAssemblyActionNet

__all__ = ["PenAssemblyActionNet"]

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FusionConfig:
    factory: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    display_name: str
    camera_config: Path
    fsm_config: Path
    action_config: Path
    action_model: Path
    event_log: Path
    fusion: FusionConfig


def _project_path(project_root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"project config thiếu đường dẫn {field!r}")
    path = (project_root / value).resolve()
    try:
        path.relative_to(project_root.resolve())
    except ValueError as error:
        raise ValueError(f"{field!r} phải nằm trong project root") from error
    return path


def load_project_config(path: str | Path, project_root: str | Path) -> ProjectConfig:
    root = Path(project_root).resolve()
    profile_path = Path(path).expanduser()
    if not profile_path.is_absolute() and not profile_path.is_file():
        profile_path = root / profile_path
    profile_path = profile_path.resolve()
    with profile_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if raw.get("schema_version") != 1:
        raise ValueError("project config schema_version phải bằng 1")

    name = str(raw.get("name", "")).strip()
    if not name:
        raise ValueError("project config phải có name")
    display_name = str(raw.get("display_name", name)).strip() or name

    fusion_raw = raw.get("fusion")
    if not isinstance(fusion_raw, dict):
        raise ValueError("project config phải có fusion")
    factory = str(fusion_raw.get("factory", "")).strip()
    if ":" not in factory:
        raise ValueError("fusion.factory phải có dạng 'package.module:ClassName'")
    parameters = fusion_raw.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("fusion.parameters phải là object JSON")

    return ProjectConfig(
        name=name,
        display_name=display_name,
        camera_config=_project_path(root, raw.get("camera_config"), "camera_config"),
        fsm_config=_project_path(root, raw.get("fsm_config"), "fsm_config"),
        action_config=_project_path(root, raw.get("action_config"), "action_config"),
        action_model=_project_path(root, raw.get("action_model"), "action_model"),
        event_log=_project_path(root, raw.get("event_log"), "event_log"),
        fusion=FusionConfig(factory=factory, parameters=dict(parameters)),
    )


def build_fusion_engine(config: FusionConfig, **overrides: Any):
    module_name, class_name = config.factory.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, class_name, None)
    if factory is None or not callable(factory):
        raise ValueError(f"Không tìm thấy fusion factory: {config.factory}")
    parameters = {**config.parameters, **{key: value for key, value in overrides.items() if value is not None}}
    engine = factory(**parameters)
    for member in ("reset", "update", "status_text"):
        if not hasattr(engine, member):
            raise TypeError(f"Fusion Engine {config.factory} thiếu {member!r}")
    return engine

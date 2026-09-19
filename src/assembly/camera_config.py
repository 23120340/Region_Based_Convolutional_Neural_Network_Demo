from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .vision import NormalizedZone


@dataclass(frozen=True)
class VisionClass:
    label: str
    prompt: str
    action: str | None
    color_bgr: tuple[int, int, int]


@dataclass(frozen=True)
class CameraConfig:
    model: str
    confidence: float
    image_size: int
    infer_every_n_frames: int
    dwell_frames: int
    work_zone: NormalizedZone
    classes: tuple[VisionClass, ...]

    @property
    def prompts(self) -> list[str]:
        return [item.prompt for item in self.classes]

    @property
    def prompt_map(self) -> dict[str, VisionClass]:
        return {item.prompt: item for item in self.classes}

    @property
    def label_map(self) -> dict[str, VisionClass]:
        return {item.label: item for item in self.classes}

    @property
    def action_map(self) -> dict[str, str]:
        return {item.label: item.action for item in self.classes if item.action is not None}


def load_camera_config(path: str | Path) -> CameraConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if raw.get("schema_version") != 1:
        raise ValueError("camera_config schema_version phải bằng 1")

    classes: list[VisionClass] = []
    labels: set[str] = set()
    prompts: set[str] = set()
    for item in raw.get("classes", []):
        label = str(item["label"])
        prompt = str(item["prompt"])
        if label in labels or prompt in prompts:
            raise ValueError("label và prompt trong camera_config không được trùng")
        color = tuple(int(channel) for channel in item.get("color_bgr", [0, 255, 0]))
        if len(color) != 3 or not all(0 <= channel <= 255 for channel in color):
            raise ValueError(f"color_bgr không hợp lệ cho {label}")
        labels.add(label)
        prompts.add(prompt)
        classes.append(VisionClass(label, prompt, item.get("action"), color))
    if not classes:
        raise ValueError("camera_config phải có ít nhất một class")

    zone_values = raw.get("work_zone_normalized", [])
    if len(zone_values) != 4:
        raise ValueError("work_zone_normalized phải có bốn giá trị")

    return CameraConfig(
        model=str(raw.get("model", "yolov8s-worldv2.pt")),
        confidence=float(raw.get("confidence", 0.12)),
        image_size=int(raw.get("image_size", 640)),
        infer_every_n_frames=max(1, int(raw.get("infer_every_n_frames", 2))),
        dwell_frames=max(1, int(raw.get("dwell_frames", 4))),
        work_zone=NormalizedZone(*(float(value) for value in zone_values)),
        classes=tuple(classes),
    )

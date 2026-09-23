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
class EarbudGeometryConfig:
    open_case_label: str
    closed_case_label: str
    earbud_labels: tuple[str, ...]
    empty_slot_labels: tuple[str, ...]
    earbud_slot_pairs: tuple[tuple[str, str], ...] = ()
    containment_threshold: float = 0.6


@dataclass(frozen=True)
class CameraConfig:
    model: str
    confidence: float
    image_size: int
    infer_every_n_frames: int
    dwell_frames: int
    work_zone: NormalizedZone
    classes: tuple[VisionClass, ...]
    capture_width: int = 1280
    capture_height: int = 720
    capture_fps: int = 30
    capture_buffer_size: int = 1
    half_precision: bool = True
    max_detections: int = 50
    earbud_geometry: EarbudGeometryConfig | None = None

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

    geometry_raw = raw.get("earbud_geometry")
    geometry: EarbudGeometryConfig | None = None
    if geometry_raw is not None:
        _labels = labels
        open_case_label = str(geometry_raw.get("open_case_label", ""))
        closed_case_label = str(geometry_raw.get("closed_case_label", ""))
        earbud_labels = tuple(str(item) for item in geometry_raw.get("earbud_labels", []))
        empty_slot_labels = tuple(
            str(item) for item in geometry_raw.get("empty_slot_labels", [])
        )
        pair_values = geometry_raw.get("earbud_slot_pairs", {})
        if not isinstance(pair_values, dict):
            raise ValueError("earbud_slot_pairs phải là object earbud_label -> empty_slot_label")
        earbud_slot_pairs = tuple(
            (str(earbud_label), str(slot_label))
            for earbud_label, slot_label in pair_values.items()
        )
        referenced = {open_case_label, closed_case_label, *earbud_labels, *empty_slot_labels}
        unknown = referenced - _labels
        if unknown:
            raise ValueError(f"earbud_geometry dùng label không tồn tại: {sorted(unknown)}")
        if not open_case_label or not closed_case_label:
            raise ValueError("earbud_geometry phải khai báo nhãn hộp mở và hộp đóng")
        if not earbud_labels or len(empty_slot_labels) != 2:
            raise ValueError("earbud_geometry cần nhãn earbud và đúng 2 nhãn khe trống")
        invalid_pairs = [
            (earbud_label, slot_label)
            for earbud_label, slot_label in earbud_slot_pairs
            if earbud_label not in earbud_labels or slot_label not in empty_slot_labels
        ]
        if invalid_pairs:
            raise ValueError(
                "earbud_slot_pairs tham chiếu nhãn ngoài earbud_labels/empty_slot_labels: "
                f"{invalid_pairs}"
            )
        if len({slot for _, slot in earbud_slot_pairs}) != len(earbud_slot_pairs):
            raise ValueError("Mỗi empty slot chỉ được ghép với một earbud label")
        threshold = float(geometry_raw.get("containment_threshold", 0.6))
        if not 0.0 < threshold <= 1.0:
            raise ValueError("containment_threshold phải nằm trong (0, 1]")
        geometry = EarbudGeometryConfig(
            open_case_label=open_case_label,
            closed_case_label=closed_case_label,
            earbud_labels=earbud_labels,
            empty_slot_labels=empty_slot_labels,
            earbud_slot_pairs=earbud_slot_pairs,
            containment_threshold=threshold,
        )

    return CameraConfig(
        model=str(raw.get("model", "yolov8s-worldv2.pt")),
        confidence=float(raw.get("confidence", 0.12)),
        image_size=int(raw.get("image_size", 640)),
        infer_every_n_frames=max(1, int(raw.get("infer_every_n_frames", 2))),
        capture_width=max(1, int(raw.get("capture_width", 1280))),
        capture_height=max(1, int(raw.get("capture_height", 720))),
        capture_fps=max(1, int(raw.get("capture_fps", 30))),
        capture_buffer_size=max(1, int(raw.get("capture_buffer_size", 1))),
        half_precision=bool(raw.get("half_precision", True)),
        max_detections=max(1, int(raw.get("max_detections", 50))),
        earbud_geometry=geometry,
        dwell_frames=max(1, int(raw.get("dwell_frames", 4))),
        work_zone=NormalizedZone(*(float(value) for value in zone_values)),
        classes=tuple(classes),
    )

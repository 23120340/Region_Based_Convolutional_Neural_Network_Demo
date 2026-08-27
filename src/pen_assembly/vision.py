from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Detection:
    label: str
    prompt: str
    confidence: float
    box_xyxy: tuple[int, int, int, int]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box_xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass(frozen=True)
class NormalizedZone:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError("Tọa độ ROI chuẩn hóa phải nằm trong [0, 1]")
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError("ROI phải có diện tích dương")

    def to_pixels(self, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        return (
            int(self.x1 * frame_width),
            int(self.y1 * frame_height),
            int(self.x2 * frame_width),
            int(self.y2 * frame_height),
        )

    def contains(self, point: tuple[float, float], frame_width: int, frame_height: int) -> bool:
        x1, y1, x2, y2 = self.to_pixels(frame_width, frame_height)
        x, y = point
        return x1 <= x <= x2 and y1 <= y <= y2


@dataclass(frozen=True)
class ComponentSuggestion:
    action: str
    label: str
    confidence: float
    dwell_count: int


class ComponentDwellGate:
    """Suggest a workflow action after its component dwells in the work zone.

    Detection is evidence that a component is present, not proof that assembly is
    complete. The camera app therefore asks for confirmation unless explicitly
    launched with its experimental auto-advance option.
    """

    def __init__(self, label_to_action: dict[str, str], dwell_frames: int = 4) -> None:
        if dwell_frames < 1:
            raise ValueError("dwell_frames phải >= 1")
        self.label_to_action = dict(label_to_action)
        self.dwell_frames = dwell_frames
        self._candidate: tuple[str, str] | None = None
        self._count = 0
        self._emitted = False
        self._suggestion: ComponentSuggestion | None = None

    @property
    def current_suggestion(self) -> ComponentSuggestion | None:
        return self._suggestion

    def reset(self) -> None:
        self._candidate = None
        self._count = 0
        self._emitted = False
        self._suggestion = None

    def update(
        self,
        detections: Iterable[Detection],
        expected_actions: Iterable[str],
        zone: NormalizedZone,
        frame_width: int,
        frame_height: int,
    ) -> ComponentSuggestion | None:
        expected = set(expected_actions)
        candidates = [
            detection
            for detection in detections
            if self.label_to_action.get(detection.label) in expected
            and zone.contains(detection.center, frame_width, frame_height)
        ]
        if not candidates:
            self.reset()
            return None

        detection = max(candidates, key=lambda item: item.confidence)
        action = self.label_to_action[detection.label]
        key = (action, detection.label)
        if key != self._candidate:
            self._candidate = key
            self._count = 1
            self._emitted = False
            self._suggestion = None
        else:
            self._count += 1

        if self._count < self.dwell_frames or self._emitted:
            return None
        self._emitted = True
        self._suggestion = ComponentSuggestion(action, detection.label, detection.confidence, self._count)
        return self._suggestion

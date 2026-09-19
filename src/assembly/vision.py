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

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box_xyxy
        return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))

    def contains_point(self, point: tuple[float, float]) -> bool:
        x1, y1, x2, y2 = self.box_xyxy
        x, y = point
        return x1 <= x <= x2 and y1 <= y <= y2

    def intersection_area(self, other: Detection) -> float:
        x1 = max(self.box_xyxy[0], other.box_xyxy[0])
        y1 = max(self.box_xyxy[1], other.box_xyxy[1])
        x2 = min(self.box_xyxy[2], other.box_xyxy[2])
        y2 = min(self.box_xyxy[3], other.box_xyxy[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        return float((x2 - x1) * (y2 - y1))

    def overlap_ratio_with(self, other: Detection) -> float:
        min_area = min(self.area, other.area)
        if min_area <= 0.0:
            return 0.0
        return self.intersection_area(other) / min_area

    def is_inside(self, container: Detection, threshold: float = 0.5) -> bool:
        """Check if self is mostly inside container (e.g. Earbud inside Case or Empty_Slot)."""
        if self.area <= 0.0:
            return False
        return (self.intersection_area(container) / self.area) >= threshold


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
    is_expected: bool


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
        self._counts: dict[str, int] = {}
        self._latched_labels: set[str] = set()
        self._suggestion: ComponentSuggestion | None = None

    @property
    def current_suggestion(self) -> ComponentSuggestion | None:
        return self._suggestion

    def reset(self) -> None:
        self._counts.clear()
        self._latched_labels.clear()
        self._suggestion = None

    def acknowledge(self) -> None:
        """Clear the current UI suggestion without re-emitting a held component."""

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
        best_by_label: dict[str, Detection] = {}
        for detection in detections:
            if detection.label not in self.label_to_action:
                continue
            if not zone.contains(detection.center, frame_width, frame_height):
                continue
            previous = best_by_label.get(detection.label)
            if previous is None or detection.confidence > previous.confidence:
                best_by_label[detection.label] = detection

        active_labels = set(best_by_label)
        for missing_label in set(self._counts) - active_labels:
            self._counts.pop(missing_label, None)
            self._latched_labels.discard(missing_label)
            if self._suggestion is not None and self._suggestion.label == missing_label:
                self._suggestion = None

        newly_stable: list[Detection] = []
        for label, detection in best_by_label.items():
            self._counts[label] = self._counts.get(label, 0) + 1
            if self._counts[label] >= self.dwell_frames and label not in self._latched_labels:
                newly_stable.append(detection)

        if not newly_stable:
            return None

        # Perception reports what it actually sees. Expectedness is metadata for
        # the UI; only the FSM is allowed to decide PASS versus VIOLATION.
        detection = max(newly_stable, key=lambda item: item.confidence)
        action = self.label_to_action[detection.label]
        self._latched_labels.add(detection.label)
        self._suggestion = ComponentSuggestion(
            action=action,
            label=detection.label,
            confidence=detection.confidence,
            dwell_count=self._counts[detection.label],
            is_expected=action in expected,
        )
        return self._suggestion

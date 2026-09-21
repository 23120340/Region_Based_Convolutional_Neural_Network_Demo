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

    def acknowledge(self, rearm: bool = False) -> None:
        """Clear the current UI suggestion without re-emitting a held component."""

        if rearm and self._suggestion is not None:
            self._latched_labels.discard(self._suggestion.label)
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


@dataclass(frozen=True)
class EarbudGeometryStatus:
    """Snapshot of the geometric evidence used by the earbud workflow."""

    case_state: str
    empty_slots: int
    earbuds_inside: int
    message: str
    is_error: bool = False


class EarbudAssemblyGate:
    """Turn spatial detections into strict two-earbud assembly actions.

    A detector only describes the current image. This gate derives workflow
    events from that image and deliberately refuses to emit either insert event
    unless the required Earbud boxes are geometrically inside the open case.
    """

    def __init__(
        self,
        open_case_label: str,
        closed_case_label: str,
        earbud_labels: Iterable[str],
        empty_slot_labels: Iterable[str],
        dwell_frames: int = 3,
        containment_threshold: float = 0.6,
    ) -> None:
        if dwell_frames < 1:
            raise ValueError("dwell_frames phải >= 1")
        if not 0.0 < containment_threshold <= 1.0:
            raise ValueError("containment_threshold phải nằm trong (0, 1]")
        self.open_case_label = open_case_label
        self.closed_case_label = closed_case_label
        self.earbud_labels = frozenset(earbud_labels)
        self.empty_slot_labels = frozenset(empty_slot_labels)
        self.dwell_frames = dwell_frames
        self.containment_threshold = containment_threshold
        self._candidate_action: str | None = None
        self._candidate_count = 0
        self._latched_actions: set[str] = set()
        self._blocked_action_until_change: str | None = None
        self._suggestion: ComponentSuggestion | None = None
        self._status = EarbudGeometryStatus(
            case_state="missing",
            empty_slots=0,
            earbuds_inside=0,
            message="Chưa thấy hộp sạc trong WORK ZONE.",
        )

    @property
    def current_suggestion(self) -> ComponentSuggestion | None:
        return self._suggestion

    @property
    def status(self) -> EarbudGeometryStatus:
        return self._status

    def reset(self) -> None:
        self._candidate_action = None
        self._candidate_count = 0
        self._latched_actions.clear()
        self._blocked_action_until_change = None
        self._suggestion = None
        self._status = EarbudGeometryStatus(
            case_state="missing",
            empty_slots=0,
            earbuds_inside=0,
            message="Chưa thấy hộp sạc trong WORK ZONE.",
        )

    def acknowledge(self, rearm: bool = False) -> None:
        if rearm and self._suggestion is not None:
            action = self._suggestion.action
            self._latched_actions.discard(action)
            if action == "remove_earbud_to_one":
                self._latched_actions.discard("insert_earbud_2")
                self._latched_actions.discard("close_case")
            elif action == "remove_earbud_to_zero":
                self._latched_actions.discard("insert_earbud_1")
                self._latched_actions.discard("insert_earbud_2")
                self._latched_actions.discard("close_case")
            self._blocked_action_until_change = action
        self._suggestion = None

    @staticmethod
    def _deduplicate(detections: Iterable[Detection]) -> list[Detection]:
        kept: list[Detection] = []
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            if all(detection.overlap_ratio_with(item) < 0.65 for item in kept):
                kept.append(detection)
        return kept

    def _inside(self, detections: Iterable[Detection], case: Detection) -> list[Detection]:
        return self._deduplicate(
            detection
            for detection in detections
            if detection.is_inside(case, self.containment_threshold)
        )

    def update(
        self,
        detections: Iterable[Detection],
        expected_actions: Iterable[str],
        zone: NormalizedZone,
        frame_width: int,
        frame_height: int,
    ) -> ComponentSuggestion | None:
        detection_list = list(detections)
        expected = set(expected_actions)
        cases = [
            item
            for item in detection_list
            if item.label in {self.open_case_label, self.closed_case_label}
            and zone.contains(item.center, frame_width, frame_height)
        ]
        if not cases:
            self._set_no_candidate(
                EarbudGeometryStatus(
                    "missing", 0, 0, "Không thấy hộp sạc trong WORK ZONE.", bool(expected)
                )
            )
            return None

        case = max(cases, key=lambda item: item.confidence)
        if case.label == self.closed_case_label:
            is_early = "close_case" not in expected
            message = (
                "LỖI: Hộp đã đóng khi chưa lắp đủ hai tai nghe."
                if is_early
                else "Đã thấy hộp đóng; sẵn sàng xác nhận hoàn tất."
            )
            self._status = EarbudGeometryStatus("closed", 0, 0, message, is_early)
            return self._advance_candidate("close_case", case.label, case.confidence, expected)

        slot_candidates: list[Detection] = []
        # empty_left/empty_right each count at most once, even if the detector
        # produces duplicate boxes for a slot.
        for label in self.empty_slot_labels:
            matching = [item for item in detection_list if item.label == label]
            inside = self._inside(matching, case)
            if inside:
                slot_candidates.append(max(inside, key=lambda item: item.confidence))
        earbud_candidates = self._inside(
            (item for item in detection_list if item.label in self.earbud_labels),
            case,
        )
        empty_slots = len(slot_candidates)
        earbuds_inside = len(earbud_candidates)

        action: str | None = None
        label = case.label
        confidence = case.confidence
        # Positive evidence of a newly visible empty slot means the physical
        # scene regressed after an insertion was already accepted. Emit a
        # dedicated violation action instead of misclassifying it as open_case
        # or a repeated insertion. The dwell filter below still has to confirm
        # this scene for multiple inference results.
        if (
            "insert_earbud_2" in expected
            and empty_slots == 2
            and earbuds_inside == 0
        ):
            action = "remove_earbud_to_zero"
            label = slot_candidates[0].label
            confidence = min(case.confidence, *(item.confidence for item in slot_candidates))
            message = "LỖI: Tai nghe đã lắp bị tháo ra; hai khe trống xuất hiện trở lại."
        elif (
            "close_case" in expected
            and empty_slots == 1
            and earbuds_inside >= 1
        ):
            action = "remove_earbud_to_one"
            label = slot_candidates[0].label
            confidence = min(
                case.confidence,
                slot_candidates[0].confidence,
                earbud_candidates[0].confidence,
            )
            message = "LỖI: Vừa tháo một tai nghe ra; một khe trống đã xuất hiện trở lại."
        elif (
            "close_case" in expected
            and empty_slots == 2
            and earbuds_inside == 0
        ):
            action = "remove_earbud_to_zero"
            label = slot_candidates[0].label
            confidence = min(case.confidence, *(item.confidence for item in slot_candidates))
            message = "LỖI: Cả hai tai nghe đã bị tháo ra; hai khe trống xuất hiện trở lại."
        elif empty_slots == 2 and earbuds_inside == 0:
            action = "open_case"
            message = "Hộp mở hợp lệ: thấy đủ 2 khe trống."
        elif empty_slots == 1 and earbuds_inside >= 1:
            action = "insert_earbud_1"
            label = earbud_candidates[0].label
            confidence = min(case.confidence, earbud_candidates[0].confidence)
            message = "Đã xác minh tai thứ nhất nằm trong hộp; còn 1 khe trống."
        elif empty_slots == 0 and earbuds_inside >= 2:
            action = "insert_earbud_2"
            label = earbud_candidates[0].label
            confidence = min(case.confidence, *(item.confidence for item in earbud_candidates[:2]))
            message = "Đã xác minh đủ 2 tai nằm trong hộp; không còn khe trống."
        elif any(item.label in self.earbud_labels for item in detection_list) and earbuds_inside == 0:
            message = "LỖI: Có tai nghe nhưng khung Earbud chưa nằm bên trong khung open_case."
        elif empty_slots == 1:
            message = "LỖI: Còn 1 khe trống nhưng chưa xác minh được tai thứ nhất ở trong hộp."
        elif empty_slots == 0:
            message = f"LỖI: Không thấy khe trống nhưng mới xác minh {earbuds_inside}/2 tai trong hộp."
        else:
            message = f"Đang kiểm tra: {earbuds_inside}/2 tai trong hộp, {empty_slots}/2 khe trống."

        is_error = action is None or action not in expected
        self._status = EarbudGeometryStatus(
            "open", empty_slots, earbuds_inside, message, is_error
        )
        if action is None:
            self._set_no_candidate(self._status)
            return None
        return self._advance_candidate(action, label, confidence, expected)

    def _set_no_candidate(self, status: EarbudGeometryStatus) -> None:
        self._status = status
        self._candidate_action = None
        self._candidate_count = 0
        self._blocked_action_until_change = None
        self._suggestion = None

    def _advance_candidate(
        self,
        action: str,
        label: str,
        confidence: float,
        expected_actions: set[str],
    ) -> ComponentSuggestion | None:
        if (
            self._blocked_action_until_change is not None
            and action != self._blocked_action_until_change
        ):
            self._blocked_action_until_change = None
        if action != self._candidate_action:
            self._candidate_action = action
            self._candidate_count = 1
        else:
            self._candidate_count += 1

        if (
            self._candidate_count < self.dwell_frames
            or action in self._latched_actions
            or action == self._blocked_action_until_change
        ):
            return None
        self._latched_actions.add(action)
        self._suggestion = ComponentSuggestion(
            action=action,
            label=label,
            confidence=confidence,
            dwell_count=self._candidate_count,
            is_expected=action in expected_actions,
        )
        return self._suggestion

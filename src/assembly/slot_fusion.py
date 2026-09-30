"""Elapsed-time evidence fusion for the left/right earbud workflow."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time
from typing import Iterable

from .earbud_fusion import EarbudFusionEngine, EarbudFusionEvent, EarbudSceneState, _normalise_label
from .fsm import ObservationState
from .model_contract import Prediction
from .temporal_config import TemporalFusionConfig, default_temporal_fusion_config
from .temporal_evidence import TemporalEvidenceAccumulator
from .vision import Detection


@dataclass(frozen=True)
class SlotView:
    label: str
    status: str
    coverage: float
    evidence_seconds: float
    required_seconds: float
    confirmed: bool
    box: tuple[int, int, int, int] | None
    confidence: float = 0.0
    coverage_available: bool = True
    empty_confidence: float | None = None

    @property
    def stable_count(self) -> int:
        """Compatibility-only UI value; decisions never use frame counts."""
        return round(self.evidence_seconds * 10)


class PairedEarbudFusionEngine(EarbudFusionEngine):
    """Combine YOLO geometry and one generic ``insert_earbud`` LSTM action."""

    def __init__(
        self, *, temporal_config: TemporalFusionConfig | None = None,
        test_start_open: bool | None = None, require_closed_case: bool = True,
        min_detection_confidence: float | None = None,
        min_action_confidence: float | None = None, **kwargs,
    ) -> None:
        self.temporal_config = temporal_config or default_temporal_fusion_config()
        if test_start_open is not None:
            self.temporal_config = replace(self.temporal_config, test_start_open=test_start_open)
        confidence = self.temporal_config.confidence
        self.min_detection_confidence = (
            confidence.occupied_earbud_min if min_detection_confidence is None
            else float(min_detection_confidence)
        )
        self.require_closed_case = require_closed_case
        kwargs.pop("stable_frames", None)
        super().__init__(
            stable_frames=1,
            min_action_confidence=(confidence.action_min if min_action_confidence is None
                                   else min_action_confidence),
            **kwargs,
        )
        if self.required_earbuds != 2 or len(self.earbud_slot_pairs) != 2:
            raise ValueError("Paired fusion requires exactly two earbud/slot pairs")
        if len(set(self.earbud_slot_pairs.values())) != 2:
            raise ValueError("Each physical earbud must map to a different slot")

    def reset(self) -> None:
        super().reset()
        max_gap = getattr(self, "temporal_config", default_temporal_fusion_config()).duration.max_gap_s
        self._evidence = TemporalEvidenceAccumulator(max_gap_s=max_gap)
        self._anchors: dict[str, tuple[float, float, float, float]] = {}
        self._empty_evidence: dict[str, deque] = {}
        self._confirmed_slots: set[str] = set()
        self._recovery_slots: set[str] = set()
        self._views: tuple[SlotView, ...] = ()
        self._tracked_case: tuple[int, int, int, int] | None = None
        self._wrong_latch: tuple[str, ...] = ()
        self._removal_armed: set[str] = set()
        self._action_times: dict[str, float] = {}
        self._close_latched = False
        self._baseline_latched = False
        self._observation_state = ObservationState.WAIT_FOR_OPEN
        self._instruction = "Đưa hộp vào vùng quan sát và mở nắp để bắt đầu."

    @property
    def confirmed_insertions(self) -> int:
        return len(self._confirmed_slots)

    @property
    def slot_views(self) -> tuple[SlotView, ...]:
        return tuple(replace(view, confirmed=view.label in self._confirmed_slots) for view in self._views)

    @property
    def observation_state(self) -> ObservationState:
        return self._observation_state

    @property
    def instruction(self) -> str:
        return self._instruction

    @property
    def calibration_progress(self) -> dict[str, int]:
        return {s: sum(item is not None for item in self._empty_evidence.get(s, ()))
                for s in self.slot_earbud_pairs}

    @property
    def status_text(self) -> str:
        details = " | ".join(f"{view.label}:{view.status}" for view in self.slot_views)
        return f"{self._observation_state.value} confirmed={self.confirmed_insertions}/2 | {details}"

    def _action_is(self, prediction: Prediction | None, action: str) -> bool:
        return bool(prediction and prediction.action == action
                    and prediction.confidence >= self.min_action_confidence)

    @staticmethod
    def _relative(slot: Detection, case: Detection) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = case.box_xyxy
        a, b, c, d = slot.box_xyxy
        return ((a-x1)/(x2-x1), (b-y1)/(y2-y1), (c-x1)/(x2-x1), (d-y1)/(y2-y1))

    def _slot_box(self, label: str, case: Detection) -> Detection:
        x1, y1, x2, y2 = case.box_xyxy
        a, b, c, d = self._anchors[label]
        box = (round(x1+a*(x2-x1)), round(y1+b*(y2-y1)),
               round(x1+c*(x2-x1)), round(y1+d*(y2-y1)))
        return Detection(label, label, 1.0, box)

    @staticmethod
    def _coverage(earbud: Detection, slot: Detection) -> float:
        return earbud.intersection_area(slot) / slot.area if slot.area else 0.0

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        intersection = max(0, x2-x1) * max(0, y2-y1)
        area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
        area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
        union = area_a + area_b - intersection
        return intersection / union if union else 0.0

    def _same_case(self, box: tuple[int, int, int, int]) -> bool:
        if self._tracked_case is None:
            return True
        if self._iou(self._tracked_case, box) >= self.temporal_config.tracking.case_iou_min:
            return True
        old = self._tracked_case
        old_center = ((old[0]+old[2])/2, (old[1]+old[3])/2)
        center = ((box[0]+box[2])/2, (box[1]+box[3])/2)
        diagonal = max(1.0, math.hypot(old[2]-old[0], old[3]-old[1]))
        return math.dist(old_center, center) / diagonal <= self.temporal_config.tracking.case_center_shift_ratio

    @staticmethod
    def _sides(labels: Iterable[str]) -> str:
        return ", ".join("trái" if "left" in label else "phải" for label in sorted(labels))

    def _remember_action(self, prediction: Prediction | None, now: float) -> None:
        if prediction and prediction.confidence >= self.min_action_confidence:
            self._action_times[prediction.action] = now

    def _recent_action(self, action: str, now: float) -> bool:
        timestamp = self._action_times.get(action)
        return timestamp is not None and now-timestamp <= self.temporal_config.duration.action_latch_s

    def _event(self, action: str, reason: str, prediction: Prediction | None = None) -> EarbudFusionEvent:
        assert self._last_scene is not None
        confidence = self._last_scene.confidence
        if prediction is not None:
            confidence = min(confidence, prediction.confidence)
        return EarbudFusionEvent(action, confidence, reason, self._last_scene)

    def _unknown_views(self, labels: tuple[str, ...]) -> None:
        self._views = tuple(
            SlotView(
                label=label,
                status="unknown",
                coverage=0.0,
                evidence_seconds=0.0,
                required_seconds=0.0,
                confirmed=label in self._confirmed_slots,
                box=None,
                coverage_available=False,
            )
            for label in labels
        )

    def update(
        self, detections: Iterable[Detection], action_prediction: Prediction | None = None,
        *, timestamp_s: float | None = None,
    ) -> EarbudFusionEvent | None:
        now = time.perf_counter() if timestamp_s is None else float(timestamp_s)
        self._remember_action(action_prediction, now)
        cfg, duration = self.temporal_config, self.temporal_config.duration
        labels = tuple(self.slot_earbud_pairs)
        items = [item for item in detections if item.area > 0]
        cases = [item for item in items if _normalise_label(item.label) in self.CASE_LABELS
                 and item.confidence >= cfg.confidence.case_min]
        case = max(cases, key=lambda item: item.confidence, default=None)
        if case is None or not self._same_case(case.box_xyxy):
            self._observation_state = ObservationState.UNKNOWN
            self._unknown_views(labels)
            self._last_scene = EarbudSceneState(False, False, 0, 0, self.confirmed_insertions, 0.0)
            self._instruction = "UNKNOWN: không thấy đúng hộp đang theo dõi hoặc vùng thao tác đang bị che."
            return None
        self._tracked_case = case.box_xyxy
        closed = _normalise_label(case.label) in self.CLOSED_CASE_LABELS
        case_key = "case:closed" if closed else "case:open"
        case_duration = self._evidence.observe(case_key, True, now).duration_s
        self._evidence.observe("case:open" if closed else "case:closed", False, now)

        if closed:
            self._observation_state = (ObservationState.WAIT_FOR_OPEN if not self._case_registered
                                       else ObservationState.INSUFFICIENT_EVIDENCE)
            self._unknown_views(labels)
            self._last_scene = EarbudSceneState(True, True, 0, 0, self.confirmed_insertions, case.confidence)
            if not self._case_registered:
                self._instruction = "WAIT_FOR_OPEN: hộp đang đóng; hãy mở nắp để bắt đầu (không phải vi phạm)."
                return None
            if (case_duration >= duration.close_s and self.confirmed_insertions == 2
                    and self._recent_action("close_case", now) and not self._close_latched):
                self._close_latched = True
                self._instruction = "Hoàn tất. Nhấn R để bắt đầu hộp mới."
                return self._event("close_case", "Hộp đóng ổn định và LSTM xác nhận close_case.", action_prediction)
            self._instruction = "Chưa đủ bằng chứng đóng hợp lệ; mở hộp và kiểm tra đủ hai tai."
            return None
        if case_duration >= duration.open_case_s:
            self._close_latched = False

        earbuds = [item for item in items if _normalise_label(item.label) in self.earbud_slot_pairs
                   and item.confidence >= cfg.confidence.occupied_earbud_min
                   and item.is_inside(case, self.containment_threshold)]
        slots: dict[str, Detection | None] = {
            label: max((item for item in items if _normalise_label(item.label) == label
                        and item.confidence > cfg.confidence.empty_slot_min
                        and item.is_inside(case, self.containment_threshold)),
                       key=lambda item: item.confidence, default=None)
            for label in labels
        }
        boxes = {label: self._slot_box(label, case) if label in self._anchors else slots[label]
                 for label in labels}
        assignments: dict[str, list[Detection]] = {label: [] for label in labels}
        ambiguous: set[str] = set()
        for earbud in earbuds:
            scores = sorted(((self._coverage(earbud, box), label) for label, box in boxes.items()
                             if box is not None), reverse=True)
            if not scores or scores[0][0] < self.slot_overlap_threshold:
                continue
            if len(scores) > 1 and scores[1][0] >= self.slot_overlap_threshold and (
                    scores[0][0]-scores[1][0] < cfg.tracking.assignment_ambiguity_margin):
                ambiguous.update(label for _, label in scores)
            else:
                assignments[scores[0][1]].append(earbud)

        stable_empty: set[str] = set()
        stable_occupied: set[str] = set()
        stable_wrong: set[str] = set()
        views: list[SlotView] = []
        for label in labels:
            box = boxes[label]
            candidates = assignments[label]
            # Show geometric overlap even below the decision threshold. Decision
            # state still depends on the strict assignment in ``candidates``.
            coverage = max((self._coverage(item, box) for item in earbuds), default=0.0) if box else 0.0
            coverage_available = bool(box is not None and earbuds)
            confidence = max((item.confidence for item in candidates), default=(slots[label].confidence if slots[label] else 0.0))
            if label in ambiguous or box is None:
                status = "unknown"
            elif candidates:
                identities = {_normalise_label(item.label) for item in candidates}
                expected = self.slot_earbud_pairs[label]
                if identities == {expected}:
                    status = "occupied"
                elif max(item.confidence for item in candidates) >= cfg.confidence.wrong_side_earbud_min:
                    status = "wrong_side"
                else:
                    status = "unknown"
            elif slots[label] is not None:
                obscured = any(self._coverage(item, box) > 0.1 for item in earbuds) if box else False
                status = "unknown" if obscured else "empty"
            else:
                status = "unknown"

            required = (duration.insert_s if status == "occupied" else
                        duration.wrong_side_s if status == "wrong_side" else
                        duration.removal_s if status == "empty" and label in self._confirmed_slots else
                        duration.open_case_s if status == "empty" else duration.unknown_arm_s)
            snapshot = self._evidence.observe(f"slot:{label}:{status}", True, now)
            for other in ("unknown", "empty", "occupied", "wrong_side"):
                if other != status:
                    self._evidence.observe(f"slot:{label}:{other}", False, now)
            if status == "empty" and snapshot.duration_s >= required:
                stable_empty.add(label)
                if label not in self._confirmed_slots and slots[label] is not None:
                    self._anchors[label] = self._relative(slots[label], case)
            elif status == "occupied" and snapshot.duration_s >= duration.insert_s and label in self._anchors:
                stable_occupied.add(label)
            elif status == "wrong_side" and snapshot.duration_s >= duration.wrong_side_s:
                stable_wrong.add(label)
            if status == "unknown" and snapshot.duration_s >= duration.unknown_arm_s:
                self._removal_armed.add(label)
            elif status == "occupied":
                self._removal_armed.discard(label)
            views.append(SlotView(
                label=label,
                status=status,
                coverage=coverage,
                evidence_seconds=snapshot.duration_s,
                required_seconds=required,
                confirmed=label in self._confirmed_slots,
                box=box.box_xyxy if box is not None else None,
                confidence=confidence,
                coverage_available=coverage_available,
                empty_confidence=slots[label].confidence if slots[label] is not None else None,
            ))
        self._views = tuple(views)
        confidence_values = [case.confidence] + [item.confidence for item in earbuds]
        self._last_scene = EarbudSceneState(
            True, False, len(stable_occupied), len(stable_empty), self.confirmed_insertions,
            min(confidence_values),
            next((_normalise_label(item.label) for label in sorted(stable_wrong)
                  for item in assignments[label]
                  if _normalise_label(item.label) != self.slot_earbud_pairs[label]), None),
            next(iter(stable_wrong)) if stable_wrong else None,
        )

        statuses = {view.status for view in views}
        self._observation_state = (ObservationState.UNKNOWN if "unknown" in statuses or ambiguous
                                   else ObservationState.INSUFFICIENT_EVIDENCE)

        if not self._case_registered:
            self._instruction = "Chờ hai khe trống rõ và ổn định trước khi xác nhận hộp đã mở."
            if len(stable_empty) == 2:
                if cfg.test_start_open and not self._baseline_latched:
                    self._baseline_latched = True
                    self._case_registered = True
                    self._observation_state = ObservationState.READY
                    self._instruction = "Test mode: đã lấy hộp mở làm mốc; chưa ghi nhận hành động open_case."
                    return self._event("initialize_open_case", self._instruction)
                if not self.require_case_action or self._recent_action(self.case_action, now):
                    self._case_registered = True
                    self._action_times.pop(self.case_action, None)
                    self._observation_state = ObservationState.READY
                    self._instruction = "Hộp sẵn sàng. Lắp tai trái hoặc phải vào đúng khe."
                    return self._event(self.case_action, "Hai khe trống rõ, ổn định; LSTM xác nhận open_case.", action_prediction)
            return None

        removal = {
            label for label in self._confirmed_slots
            if label in stable_empty and not assignments[label]
        }
        if removal:
            previous = self.confirmed_insertions
            self._confirmed_slots.difference_update(removal)
            self._recovery_slots.update(removal)
            self._removal_armed.difference_update(removal)
            self._observation_state = ObservationState.READY
            action = "remove_earbud_to_one" if self.confirmed_insertions else "remove_earbud_to_zero"
            self._instruction = f"Đã xác nhận tháo tai {self._sides(removal)}; hãy lắp lại đúng khe."
            return self._event(action, f"Occupancy giảm {previous}->{self.confirmed_insertions} sau {duration.removal_s:g}s thấy khe trống rõ và không có bbox tai nghe.")

        signature = tuple(sorted(stable_wrong))
        if stable_wrong:
            self._observation_state = ObservationState.READY
            self._instruction = f"Sai bên tại khe {self._sides(stable_wrong)}. Tháo ra và lắp đúng khe."
            if signature != self._wrong_latch:
                self._wrong_latch = signature
                return self._event("wrong_earbud_side", self._instruction)
            return None
        if not any(view.status == "wrong_side" for view in views):
            self._wrong_latch = ()

        new_slots = stable_occupied - self._confirmed_slots
        if new_slots and self.confirmed_insertions < 2 and self._recent_action("insert_earbud", now):
            label = sorted(new_slots)[0]
            expected = "insert_first_earbud" if self.confirmed_insertions == 0 else "insert_second_earbud"
            self._confirmed_slots.add(label)
            self._action_times.pop("insert_earbud", None)
            self._recovery_slots.discard(label)
            self._observation_state = ObservationState.READY
            self._instruction = ("Đã đủ hai tai. Hãy đóng nắp hộp." if self.confirmed_insertions == 2
                                 else "Đã xác nhận một tai. Lắp tai còn lại vào đúng khe.")
            return self._event(expected, f"YOLO xác nhận tai {self._sides([label])} đúng khe trong {duration.insert_s:g}s; LSTM xác nhận insert_earbud.", action_prediction)

        self._instruction = (
            f"Lắp lại tai {self._sides(self._recovery_slots)} vào đúng khe."
            if self._recovery_slots else
            "Đã đủ hai tai. Hãy đóng nắp hộp." if self.confirmed_insertions == 2 else
            "Đã thấy tai đúng khe nhưng chờ LSTM insert_earbud." if new_slots else
            f"Chờ bằng chứng ổn định để lắp tai thứ {self.confirmed_insertions + 1}."
        )
        return None

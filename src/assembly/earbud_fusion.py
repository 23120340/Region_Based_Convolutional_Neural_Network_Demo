from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .model_contract import Prediction
from .vision import Detection


def _normalise_label(label: str) -> str:
    return label.strip().casefold().replace("-", "_").replace(" ", "_")


@dataclass(frozen=True)
class EarbudSceneState:
    case_present: bool
    case_closed: bool
    earbuds_inside_case: int
    empty_slots_inside_case: int
    occupancy_estimate: int
    confidence: float


@dataclass(frozen=True)
class EarbudFusionEvent:
    action: str
    confidence: float
    reason: str
    scene: EarbudSceneState


class EarbudFusionEngine:
    """Fuse YOLO geometry with temporal action evidence for a two-earbud SOP.

    The detector answers what and where. The action model answers what motion is
    occurring. This class only emits an insertion step when both sources agree,
    preventing an earbud merely lying near the case from advancing the FSM.
    """

    CASE_LABELS = frozenset(
        {
            "case",
            "earphone_case",
            "case_open",
            "case_closed",
            "open_case",
            "close_case",
        }
    )
    CLOSED_CASE_LABELS = frozenset({"case_closed", "close_case"})
    EARBUD_LABELS = frozenset(
        {
            "earbud",
            "left_earbud",
            "right_earbud",
            "earbud_left",
            "earbud_right",
        }
    )
    EMPTY_SLOT_LABELS = frozenset(
        {
            "empty_slot",
            "empty_slot_left",
            "empty_slot_right",
            "left_empty_slot",
            "right_empty_slot",
            "empty_left",
            "empty_right",
        }
    )

    def __init__(
        self,
        *,
        required_earbuds: int = 2,
        stable_frames: int = 3,
        containment_threshold: float = 0.5,
        duplicate_overlap_threshold: float = 0.7,
        min_action_confidence: float = 0.6,
    ) -> None:
        if required_earbuds < 1:
            raise ValueError("required_earbuds phải >= 1")
        if stable_frames < 1:
            raise ValueError("stable_frames phải >= 1")
        if not 0.0 <= containment_threshold <= 1.0:
            raise ValueError("containment_threshold phải nằm trong [0, 1]")
        if not 0.0 <= duplicate_overlap_threshold <= 1.0:
            raise ValueError("duplicate_overlap_threshold phải nằm trong [0, 1]")
        if not 0.0 <= min_action_confidence <= 1.0:
            raise ValueError("min_action_confidence phải nằm trong [0, 1]")
        self.required_earbuds = required_earbuds
        self.stable_frames = stable_frames
        self.containment_threshold = containment_threshold
        self.duplicate_overlap_threshold = duplicate_overlap_threshold
        self.min_action_confidence = min_action_confidence
        self.reset()

    @property
    def confirmed_insertions(self) -> int:
        return self._confirmed_insertions

    @property
    def last_scene(self) -> EarbudSceneState | None:
        return self._last_scene

    @property
    def status_text(self) -> str:
        scene = self._last_scene
        if scene is None:
            return "waiting for stable YOLO scene"
        return (
            f"inside={scene.earbuds_inside_case} "
            f"empty={scene.empty_slots_inside_case} "
            f"confirmed={self.confirmed_insertions}/{self.required_earbuds}"
        )

    def reset(self) -> None:
        self._candidate_signature: tuple[bool, bool, int, int] | None = None
        self._candidate_count = 0
        self._last_scene: EarbudSceneState | None = None
        self._case_registered = False
        self._confirmed_insertions = 0
        self._max_empty_slots_seen = 0
        self._last_insertion_signature: tuple[bool, bool, int, int] | None = None
        self._close_latched = False

    def _deduplicate(self, detections: Iterable[Detection]) -> list[Detection]:
        kept: list[Detection] = []
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            duplicate = any(
                detection.overlap_ratio_with(other) >= self.duplicate_overlap_threshold
                for other in kept
            )
            if not duplicate:
                kept.append(detection)
        return kept

    def _raw_scene(
        self,
        detections: Iterable[Detection],
    ) -> tuple[EarbudSceneState, tuple[bool, bool, int, int]]:
        items = tuple(detections)
        cases = [item for item in items if _normalise_label(item.label) in self.CASE_LABELS]
        case = max(cases, key=lambda item: item.confidence) if cases else None
        if case is None:
            scene = EarbudSceneState(False, False, 0, 0, 0, 0.0)
            return scene, (False, False, 0, 0)

        case_closed = _normalise_label(case.label) in self.CLOSED_CASE_LABELS
        earbuds = self._deduplicate(
            item
            for item in items
            if _normalise_label(item.label) in self.EARBUD_LABELS
            and item.is_inside(case, self.containment_threshold)
        )
        slots = self._deduplicate(
            item
            for item in items
            if _normalise_label(item.label) in self.EMPTY_SLOT_LABELS
            and item.is_inside(case, self.containment_threshold)
        )
        confidence_values = [case.confidence]
        confidence_values.extend(item.confidence for item in earbuds)
        confidence_values.extend(item.confidence for item in slots)
        scene = EarbudSceneState(
            case_present=True,
            case_closed=case_closed,
            earbuds_inside_case=min(len(earbuds), self.required_earbuds),
            empty_slots_inside_case=min(len(slots), self.required_earbuds),
            occupancy_estimate=min(len(earbuds), self.required_earbuds),
            confidence=sum(confidence_values) / len(confidence_values),
        )
        signature = (
            scene.case_present,
            scene.case_closed,
            scene.earbuds_inside_case,
            scene.empty_slots_inside_case,
        )
        return scene, signature

    def _with_occupancy(self, scene: EarbudSceneState) -> EarbudSceneState:
        self._max_empty_slots_seen = max(
            self._max_empty_slots_seen,
            scene.empty_slots_inside_case,
        )
        slot_based = max(0, self._max_empty_slots_seen - scene.empty_slots_inside_case)
        occupancy = min(
            self.required_earbuds,
            max(scene.earbuds_inside_case, slot_based),
        )
        return EarbudSceneState(
            case_present=scene.case_present,
            case_closed=scene.case_closed,
            earbuds_inside_case=scene.earbuds_inside_case,
            empty_slots_inside_case=scene.empty_slots_inside_case,
            occupancy_estimate=occupancy,
            confidence=scene.confidence,
        )

    def _action_is(self, prediction: Prediction | None, action: str) -> bool:
        return bool(
            prediction is not None
            and prediction.action == action
            and prediction.confidence >= self.min_action_confidence
        )

    def update(
        self,
        detections: Iterable[Detection],
        action_prediction: Prediction | None = None,
    ) -> EarbudFusionEvent | None:
        raw_scene, signature = self._raw_scene(detections)
        if signature == self._candidate_signature:
            self._candidate_count += 1
        else:
            self._candidate_signature = signature
            self._candidate_count = 1

        if not self._action_is(action_prediction, "close_case"):
            self._close_latched = False
        if self._candidate_count < self.stable_frames:
            return None

        scene = self._with_occupancy(raw_scene)
        self._last_scene = scene
        if not scene.case_present:
            self._case_registered = False
            self._confirmed_insertions = 0
            self._max_empty_slots_seen = 0
            self._last_insertion_signature = None
            return None

        if not self._case_registered:
            self._case_registered = True
            return EarbudFusionEvent(
                action="pick_case",
                confidence=scene.confidence,
                reason="Hộp sạc xuất hiện ổn định trong vùng quan sát.",
                scene=scene,
            )

        # Removing an already-confirmed earbud is derived from positive YOLO
        # evidence: one or more empty slots reappear and the stable occupancy
        # drops. No separate "remove" action class is required from ViT-LSTM.
        expected_empty_slots = self.required_earbuds - self._confirmed_insertions
        removal_detected = (
            self._confirmed_insertions > 0
            and scene.empty_slots_inside_case > expected_empty_slots
            and scene.occupancy_estimate < self._confirmed_insertions
        )
        if removal_detected:
            previous_insertions = self._confirmed_insertions
            self._confirmed_insertions = scene.occupancy_estimate
            self._last_insertion_signature = signature
            action = (
                "remove_earbud_to_one"
                if self._confirmed_insertions == 1
                else "remove_earbud_to_zero"
            )
            return EarbudFusionEvent(
                action=action,
                confidence=scene.confidence,
                reason=(
                    f"YOLO xác nhận occupancy giảm từ {previous_insertions} "
                    f"xuống {self._confirmed_insertions} và khe trống xuất hiện trở lại."
                ),
                scene=scene,
            )

        if self._action_is(action_prediction, "insert_earbud"):
            has_new_physical_evidence = scene.occupancy_estimate > self._confirmed_insertions
            new_scene = signature != self._last_insertion_signature
            if has_new_physical_evidence and new_scene:
                self._confirmed_insertions += 1
                self._last_insertion_signature = signature
                ordinal = "thứ nhất" if self._confirmed_insertions == 1 else "thứ hai"
                return EarbudFusionEvent(
                    action=(
                        "insert_first_earbud"
                        if self._confirmed_insertions == 1
                        else "insert_second_earbud"
                    ),
                    confidence=min(scene.confidence, action_prediction.confidence),
                    reason=(
                        f"ViT-LSTM nhận diện insert_earbud và YOLO xác nhận "
                        f"tai nghe {ordinal} đã chiếm khe."
                    ),
                    scene=scene,
                )

        if self._action_is(action_prediction, "close_case") and not self._close_latched:
            self._close_latched = True
            enough = self._confirmed_insertions >= self.required_earbuds
            return EarbudFusionEvent(
                action="close_case",
                confidence=min(scene.confidence, action_prediction.confidence),
                reason=(
                    "ViT-LSTM nhận diện đóng nắp sau khi đủ hai tai nghe."
                    if enough
                    else "Phát hiện đóng nắp khi chưa xác nhận đủ hai tai nghe."
                ),
                scene=scene,
            )
        return None

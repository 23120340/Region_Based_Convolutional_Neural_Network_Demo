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
    wrong_side_earbud_label: str | None = None
    wrong_side_slot_label: str | None = None


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
        slot_overlap_threshold: float = 0.4,
        min_action_confidence: float = 0.6,
        case_action: str = "pick_case",
        require_case_action: bool = False,
        earbud_slot_pairs: dict[str, str] | None = None,
    ) -> None:
        if required_earbuds < 1:
            raise ValueError("required_earbuds phải >= 1")
        if stable_frames < 1:
            raise ValueError("stable_frames phải >= 1")
        if not 0.0 <= containment_threshold <= 1.0:
            raise ValueError("containment_threshold phải nằm trong [0, 1]")
        if not 0.0 <= duplicate_overlap_threshold <= 1.0:
            raise ValueError("duplicate_overlap_threshold phải nằm trong [0, 1]")
        if not 0.0 <= slot_overlap_threshold <= 1.0:
            raise ValueError("slot_overlap_threshold phải nằm trong [0, 1]")
        if not 0.0 <= min_action_confidence <= 1.0:
            raise ValueError("min_action_confidence phải nằm trong [0, 1]")
        if not case_action.strip():
            raise ValueError("case_action không được rỗng")
        self.required_earbuds = required_earbuds
        self.stable_frames = stable_frames
        self.containment_threshold = containment_threshold
        self.duplicate_overlap_threshold = duplicate_overlap_threshold
        self.slot_overlap_threshold = slot_overlap_threshold
        self.min_action_confidence = min_action_confidence
        self.case_action = case_action
        self.require_case_action = require_case_action
        self.earbud_slot_pairs = {
            _normalise_label(earbud_label): _normalise_label(slot_label)
            for earbud_label, slot_label in (earbud_slot_pairs or {}).items()
        }
        # Reverse map: slot -> earbud, for wrong-side detection
        self.slot_earbud_pairs: dict[str, str] = {
            slot: earbud for earbud, slot in self.earbud_slot_pairs.items()
        }
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
        status = (
            f"inside={scene.earbuds_inside_case} "
            f"empty={scene.empty_slots_inside_case} "
            f"confirmed={self.confirmed_insertions}/{self.required_earbuds}"
        )
        if scene.wrong_side_earbud_label is not None:
            status += f" WRONG_SIDE={scene.wrong_side_earbud_label}"
        return status

    def reset(self) -> None:
        self._candidate_signature: tuple[bool, bool, int, int, str | None, str | None] | None = None
        self._candidate_count = 0
        self._last_scene: EarbudSceneState | None = None
        self._case_registered = False
        self._confirmed_insertions = 0
        self._max_empty_slots_seen = 0
        self._last_insertion_signature: (
            tuple[bool, bool, int, int, str | None, str | None] | None
        ) = None
        self._close_latched = False
        self._wrong_side_latched_signature: tuple[bool, bool, int, int, str | None, str | None] | None = None

    def _deduplicate(self, detections: Iterable[Detection]) -> list[Detection]:
        """Remove duplicate detections — only compares within the same label group.

        Earbuds are only deduplicated against other earbuds, empty slots against
        other empty slots, etc.  This prevents ``left_earbud`` from suppressing
        ``right_earbud`` when the two bboxes happen to overlap heavily.
        """
        by_group: dict[str, list[Detection]] = {}
        for detection in detections:
            norm = _normalise_label(detection.label)
            if norm in self.EARBUD_LABELS:
                group = "earbud"
            elif norm in self.EMPTY_SLOT_LABELS:
                group = "slot"
            elif norm in self.CASE_LABELS:
                group = "case"
            else:
                group = norm  # keep unknown labels in their own singleton group
            by_group.setdefault(group, []).append(detection)

        kept: list[Detection] = []
        for group_detections in by_group.values():
            for detection in sorted(group_detections, key=lambda d: d.confidence, reverse=True):
                duplicate = any(
                    detection.overlap_ratio_with(other) >= self.duplicate_overlap_threshold
                    for other in kept
                    if _normalise_label(other.label) in (
                        self.EARBUD_LABELS if _normalise_label(detection.label) in self.EARBUD_LABELS
                        else self.EMPTY_SLOT_LABELS if _normalise_label(detection.label) in self.EMPTY_SLOT_LABELS
                        else self.CASE_LABELS
                    )
                )
                if not duplicate:
                    kept.append(detection)
        return kept

    def _raw_scene(
        self,
        detections: Iterable[Detection],
    ) -> tuple[
        EarbudSceneState,
        tuple[bool, bool, int, int, str | None, str | None],
        list[tuple[Detection, Detection]],  # geo_insertions: (earbud, matched_slot)
    ]:
        items = tuple(detections)
        cases = [item for item in items if _normalise_label(item.label) in self.CASE_LABELS]
        case = max(cases, key=lambda item: item.confidence) if cases else None
        if case is None:
            scene = EarbudSceneState(False, False, 0, 0, 0, 0.0)
            return scene, (False, False, 0, 0, None, None), []

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
        # --- Wrong-side detection ---
        # A wrong-side insertion occurs when an earbud is detected INSIDE the case
        # but its paired empty slot is STILL visible (meaning it hasn't filled its
        # own slot — it must be sitting in the wrong one).
        wrong_side_earbud: Detection | None = None
        wrong_side_slot: Detection | None = None
        for earbud in earbuds:
            expected_slot_label = self.earbud_slot_pairs.get(_normalise_label(earbud.label))
            if expected_slot_label is None:
                continue  # No pairing configured for this earbud type
            # Its own slot is still visible as empty -> earbud is in the wrong slot
            own_slot_still_empty = any(
                _normalise_label(slot.label) == expected_slot_label for slot in slots
            )
            if own_slot_still_empty:
                # Choose the wrong slot (the one it actually overlaps most with)
                other_slots = [
                    slot for slot in slots
                    if _normalise_label(slot.label) != expected_slot_label
                ]
                if other_slots:
                    wrong_side_slot = max(other_slots, key=lambda s: earbud.overlap_ratio_with(s))
                wrong_side_earbud = earbud
                break  # Report only the first wrong-side earbud per frame

        # --- Geometry-based insertion evidence ---
        # Count earbuds that overlap with their CORRECT matched slot sufficiently,
        # OR earbuds whose corresponding empty slot is no longer detected (assumed covered).
        geo_insertions: list[tuple[Detection, Detection | None]] = []  # (earbud, matched_slot or None)
        for earbud in earbuds:
            expected_slot_label = self.earbud_slot_pairs.get(_normalise_label(earbud.label))
            if expected_slot_label is None:
                continue
            
            matched_slot = None
            for slot in slots:
                if _normalise_label(slot.label) == expected_slot_label:
                    matched_slot = slot
                    break
            
            if matched_slot is not None:
                if earbud.overlap_ratio_with(matched_slot) >= self.slot_overlap_threshold:
                    geo_insertions.append((earbud, matched_slot))
            else:
                # The expected slot is missing (overshadowed by the earbud)
                geo_insertions.append((earbud, None))

        scene = EarbudSceneState(
            case_present=True,
            case_closed=case_closed,
            earbuds_inside_case=min(len(earbuds), self.required_earbuds),
            empty_slots_inside_case=min(len(slots), self.required_earbuds),
            occupancy_estimate=min(len(earbuds), self.required_earbuds),
            confidence=sum(confidence_values) / len(confidence_values),
            wrong_side_earbud_label=(
                wrong_side_earbud.label if wrong_side_earbud is not None else None
            ),
            wrong_side_slot_label=(
                wrong_side_slot.label if wrong_side_slot is not None else None
            ),
        )
        signature = (
            scene.case_present,
            scene.case_closed,
            scene.earbuds_inside_case,
            scene.empty_slots_inside_case,
            scene.wrong_side_earbud_label,
            scene.wrong_side_slot_label,
        )
        return scene, signature, geo_insertions

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
            wrong_side_earbud_label=scene.wrong_side_earbud_label,
            wrong_side_slot_label=scene.wrong_side_slot_label,
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
        raw_scene, signature, geo_insertions = self._raw_scene(detections)
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
            self._wrong_side_latched_signature = None
            return None

        if not self._case_registered:
            if self.require_case_action and not self._action_is(
                action_prediction, self.case_action
            ):
                return None
            self._case_registered = True
            return EarbudFusionEvent(
                action=self.case_action,
                confidence=(
                    min(scene.confidence, action_prediction.confidence)
                    if action_prediction is not None
                    else scene.confidence
                ),
                reason=(
                    f"YOLO thấy hộp ổn định và ViT-LSTM nhận diện {self.case_action}."
                    if self.require_case_action
                    else "Hộp sạc xuất hiện ổn định trong vùng quan sát."
                ),
                scene=scene,
            )

        if scene.wrong_side_earbud_label is None:
            self._wrong_side_latched_signature = None
        elif signature != self._wrong_side_latched_signature:
            self._wrong_side_latched_signature = signature
            return EarbudFusionEvent(
                action="wrong_earbud_side",
                confidence=scene.confidence,
                reason=(
                    f"YOLO thấy {scene.wrong_side_earbud_label} trong hộp nhưng "
                    f"{scene.wrong_side_slot_label} vẫn trống; tai nghe đang nằm nhầm khe."
                ),
                scene=scene,
            )

        # --- Removal detection (VIOLATION) ---
        # Triggered when YOLO sees empty slots reappear. No action model class needed.
        # After removal, reset tracking so re-insertion is re-confirmed from scratch.
        expected_empty_slots = self.required_earbuds - self._confirmed_insertions
        removal_detected = (
            self._confirmed_insertions > 0
            and scene.empty_slots_inside_case > expected_empty_slots
        )
        if removal_detected:
            previous_insertions = self._confirmed_insertions
            self._confirmed_insertions = min(
                scene.occupancy_estimate,
                max(0, self.required_earbuds - scene.empty_slots_inside_case)
            )
            # Reset insertion tracking so the next insertion is re-confirmed
            self._last_insertion_signature = None
            self._max_empty_slots_seen = scene.empty_slots_inside_case
            action = (
                "remove_earbud_to_one"
                if self._confirmed_insertions == 1
                else "remove_earbud_to_zero"
            )
            action_supports_removal = self._action_is(action_prediction, "remove_earbud")
            return EarbudFusionEvent(
                action=action,
                confidence=(
                    min(scene.confidence, action_prediction.confidence)
                    if action_supports_removal and action_prediction is not None
                    else scene.confidence
                ),
                reason=(
                    f"[VIOLATION] YOLO xác nhận tai nghe bị tháo: occupancy giảm từ "
                    f"{previous_insertions} xuống {self._confirmed_insertions}; "
                    f"cần lắp lại từ bước insert_{'first' if self._confirmed_insertions == 0 else 'second'}_earbud"
                    + (
                        "; ViT-LSTM đồng thời nhận diện remove_earbud."
                        if action_supports_removal
                        else "."
                    )
                ),
                scene=scene,
            )

        expected_insert_action = (
            "insert_first_earbud"
            if self._confirmed_insertions == 0
            else "insert_second_earbud"
        )
        insertion_action_matches = self._action_is(
            action_prediction, expected_insert_action
        ) or self._action_is(action_prediction, "insert_earbud")

        # Geometry evidence: earbud bbox overlaps with its matched slot bbox
        # This works even if ViT-LSTM hasn't caught up yet (e.g. fast insertion).
        has_geo_evidence = len(geo_insertions) > self._confirmed_insertions
        has_new_physical_evidence = scene.occupancy_estimate > self._confirmed_insertions
        new_scene = signature != self._last_insertion_signature

        # Two paths to confirm an insertion:
        # Path A — action model agrees + YOLO occupancy changed
        # Path B — YOLO geometry (earbud overlapping correct slot) alone
        if new_scene and (has_new_physical_evidence or has_geo_evidence):
            if insertion_action_matches or has_geo_evidence:
                self._confirmed_insertions += 1
                self._last_insertion_signature = signature
                ordinal = "thứ nhất" if self._confirmed_insertions == 1 else "thứ hai"
                insert_action_name = (
                    "insert_first_earbud"
                    if self._confirmed_insertions == 1
                    else "insert_second_earbud"
                )
                if has_geo_evidence and insertion_action_matches and action_prediction is not None:
                    confidence = min(scene.confidence, action_prediction.confidence)
                    reason = (
                        f"YOLO geometry xác nhận tai nghe {ordinal} lấp khe đúng chiều "
                        f"và ViT-LSTM nhận diện {action_prediction.action}."
                    )
                elif has_geo_evidence:
                    confidence = scene.confidence
                    reason = (
                        f"YOLO geometry: bbox tai nghe {ordinal} chồng lấp khe tương ứng "
                        f">= {self.slot_overlap_threshold:.0%}; xác nhận insert mà không cần action model."
                    )
                else:
                    confidence = min(scene.confidence, action_prediction.confidence)  # type: ignore[union-attr]
                    reason = (
                        f"ViT-LSTM nhận diện {action_prediction.action} và YOLO xác nhận "
                        f"tai nghe {ordinal} đã chiếm khe."
                    )
                return EarbudFusionEvent(
                    action=insert_action_name,
                    confidence=confidence,
                    reason=reason,
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

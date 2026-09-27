"""Per-slot evidence for the left/right earbud workflow."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .earbud_fusion import EarbudFusionEngine, EarbudFusionEvent, EarbudSceneState, _normalise_label
from .model_contract import Prediction
from .vision import Detection


@dataclass(frozen=True)
class SlotView:
    label: str
    status: str
    coverage: float
    stable_count: int
    confirmed: bool
    box: tuple[int, int, int, int] | None


class PairedEarbudFusionEngine(EarbudFusionEngine):
    """Require positive slot geometry AND a confident temporal insertion action.

    Remember empty-slot boxes relative to the open case, since empty-slot
    detections legitimately disappear when covered. Missing detections never
    mean insertion or removal. One physical case is monitored until reset.
    """

    def __init__(
        self, *, min_detection_confidence: float = 0.5,
        require_closed_case: bool = True, **kwargs,
    ) -> None:
        if not 0 <= min_detection_confidence <= 1:
            raise ValueError("min_detection_confidence phải nằm trong [0, 1]")
        self.min_detection_confidence = min_detection_confidence
        self.require_closed_case = require_closed_case
        super().__init__(**kwargs)
        if self.required_earbuds != 2 or len(self.earbud_slot_pairs) != 2:
            raise ValueError("Paired fusion yêu cầu hai cặp tai nghe/khe khác nhau")
        if len(set(self.earbud_slot_pairs.values())) != 2:
            raise ValueError("Mỗi tai nghe phải có một khe riêng")
        if self.slot_overlap_threshold <= 0:
            raise ValueError("slot_overlap_threshold phải > 0")

    def reset(self) -> None:
        super().reset()
        self._anchors: dict[str, tuple[float, float, float, float]] = {}
        self._confirmed_slots: set[str] = set()
        self._recovery_slots: set[str] = set()
        self._runs: dict[str, tuple[str, int]] = {}
        self._views: tuple[SlotView, ...] = ()
        self._case_run: tuple[str, int] = ("unknown", 0)
        self._wrong_latch: tuple[str, ...] = ()
        self._order_latch: str | None = None
        self._unsafe_slots: set[str] = set()
        self._instruction = "Mở hộp rỗng, để camera nhìn rõ cả hai khe."

    @property
    def confirmed_insertions(self) -> int:
        return len(self._confirmed_slots)

    @property
    def slot_views(self) -> tuple[SlotView, ...]:
        return tuple(
            SlotView(v.label, v.status, v.coverage, v.stable_count,
                     v.label in self._confirmed_slots, v.box)
            for v in self._views
        )

    @property
    def instruction(self) -> str:
        return self._instruction

    @property
    def status_text(self) -> str:
        return f"confirmed={self.confirmed_insertions}/2 | " + " | ".join(
            f"{v.label}:{v.status}" for v in self.slot_views
        )

    def _action_is(self, prediction: Prediction | None, action: str) -> bool:
        return bool(prediction and prediction.action == action
                    and prediction.confidence > self.min_action_confidence)

    def _advance(self, label: str, status: str) -> int:
        previous, count = self._runs.get(label, ("unknown", 0))
        count = count + 1 if previous == status else 1
        self._runs[label] = status, count
        return count

    @staticmethod
    def _relative(slot: Detection, case: Detection) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = case.box_xyxy
        a, b, c, d = slot.box_xyxy
        return ((a-x1)/(x2-x1), (b-y1)/(y2-y1),
                (c-x1)/(x2-x1), (d-y1)/(y2-y1))

    def _slot_box(self, label: str, case: Detection) -> Detection:
        x1, y1, x2, y2 = case.box_xyxy
        a, b, c, d = self._anchors[label]
        box = (round(x1+a*(x2-x1)), round(y1+b*(y2-y1)),
               round(x1+c*(x2-x1)), round(y1+d*(y2-y1)))
        return Detection(label, label, case.confidence, box)

    @staticmethod
    def _coverage(earbud: Detection, slot: Detection) -> float:
        return earbud.intersection_area(slot) / slot.area if slot.area else 0.0

    @staticmethod
    def _sides(labels: Iterable[str]) -> str:
        return ", ".join("trái" if "left" in label else "phải" for label in sorted(labels))

    def _event(self, action: str, reason: str, prediction: Prediction | None = None):
        assert self._last_scene is not None
        confidence = self._last_scene.confidence
        if prediction is not None:
            confidence = min(confidence, prediction.confidence)
        return EarbudFusionEvent(action, confidence, reason, self._last_scene)

    def update(
        self, detections: Iterable[Detection], action_prediction: Prediction | None = None,
    ) -> EarbudFusionEvent | None:
        items = [d for d in detections
                 if d.area > 0 and d.confidence >= self.min_detection_confidence]
        cases = [d for d in items if _normalise_label(d.label) in self.CASE_LABELS]
        case = max(cases, key=lambda d: d.confidence) if cases else None
        labels = tuple(self.slot_earbud_pairs)
        if case is None:
            self._case_run = "unknown", 0
            self._runs.clear()
            self._views = tuple(SlotView(s, "unknown", 0, 0, s in self._confirmed_slots, None)
                                for s in labels)
            self._last_scene = EarbudSceneState(False, False, 0, 0,
                                               self.confirmed_insertions, 0)
            self._instruction = "Chưa thấy hộp: đưa lại vào vùng quan sát; R nếu đổi hộp mới."
            return None

        closed = _normalise_label(case.label) in self.CLOSED_CASE_LABELS
        case_status = "closed" if closed else "open"
        last_case, case_count = self._case_run
        self._case_run = case_status, case_count + 1 if last_case == case_status else 1
        case_stable = self._case_run[1] >= self.stable_frames
        if not closed and case_stable:
            self._close_latched = False
        earbuds = [d for d in items if _normalise_label(d.label) in self.earbud_slot_pairs
                   and d.is_inside(case, self.containment_threshold)]
        slots = {
            s: max((d for d in items if _normalise_label(d.label) == s
                    and d.is_inside(case, self.containment_threshold)),
                   key=lambda d: d.confidence, default=None)
            for s in labels
        }
        # Tentative boxes can show overlap, but only stable uncovered empty slots
        # become anchors. An already covered slot cannot calibrate itself.
        boxes = {s: self._slot_box(s, case) if s in self._anchors else slots[s] for s in labels}
        assignments: dict[str, list[Detection]] = {s: [] for s in labels}
        ambiguous: set[str] = set()
        for earbud in earbuds:
            scores = sorted(((self._coverage(earbud, box), s) for s, box in boxes.items()
                             if box is not None), reverse=True)
            if not scores or scores[0][0] < self.slot_overlap_threshold:
                continue
            if len(scores) == 2 and scores[1][0] >= self.slot_overlap_threshold and (
                scores[0][0] - scores[1][0] < 0.1
            ):
                ambiguous.update(s for _, s in scores)
            else:
                assignments[scores[0][1]].append(earbud)

        views = []
        stable_empty: set[str] = set()
        stable_occupied: set[str] = set()
        stable_wrong: set[str] = set()
        for s in labels:
            box = boxes[s]
            candidates = assignments[s]
            coverage = max((self._coverage(d, box) for d in candidates), default=0.0) if box else 0.0
            if closed:
                status = "closed"
            elif s in ambiguous:
                status = "unknown"
            elif candidates:
                correct = {_normalise_label(d.label) for d in candidates}
                status = "occupied" if correct == {self.slot_earbud_pairs[s]} else "wrong_side"
            elif slots[s] is not None:
                # Partial overlap is still a hand-off/motion, not removal evidence.
                obscured = box and any(self._coverage(d, box) > 0.1 for d in earbuds)
                status = "unknown" if obscured else "empty"
            else:
                status = "unknown"
            run = self._advance(s, status)
            if run >= self.stable_frames:
                if status == "empty":
                    stable_empty.add(s)
                    # Register/update the physical position only with positive empty evidence.
                    self._anchors[s] = self._relative(slots[s], case)
                    boxes[s] = slots[s]
                    self._unsafe_slots.discard(s)
                elif status == "occupied" and s in self._anchors:
                    stable_occupied.add(s)
                    self._unsafe_slots.discard(s)
                elif status == "wrong_side":
                    stable_wrong.add(s)
                    self._unsafe_slots.add(s)
            views.append(SlotView(s, status, coverage, run, s in self._confirmed_slots,
                                  boxes[s].box_xyxy if boxes[s] else None))
        self._views = tuple(views)
        self._last_scene = EarbudSceneState(
            True, closed, len(stable_occupied), len(stable_empty), len(stable_occupied),
            min([case.confidence] + [d.confidence for d in earbuds]
                + [d.confidence for d in slots.values() if d is not None]),
            next((_normalise_label(d.label) for s in sorted(stable_wrong)
                  for d in assignments[s]
                  if _normalise_label(d.label) != self.slot_earbud_pairs[s]), None),
            next(iter(stable_wrong)) if stable_wrong else None,
        )

        if not self._case_registered:
            self._instruction = "Mở hộp rỗng, thấy rõ hai khe và chờ xác nhận open_case."
            if len(stable_empty) == 2 and (not self.require_case_action or
                                        self._action_is(action_prediction, self.case_action)):
                self._case_registered = True
                self._instruction = "Hộp sẵn sàng. Lắp tai trái hoặc phải vào đúng khe."
                return self._event(self.case_action, "Đã ghi nhớ vị trí hai khe trống; xác nhận mở hộp.",
                                   action_prediction if self.require_case_action else None)
            premature = (stable_occupied and action_prediction and
                         action_prediction.action in {"insert_first_earbud", "insert_second_earbud"})
            premature_close = closed and case_stable and self._action_is(action_prediction, "close_case")
            if (premature or premature_close) and self._action_is(action_prediction, action_prediction.action):
                if self._order_latch != action_prediction.action:
                    self._order_latch = action_prediction.action
                    return self._event(action_prediction.action,
                                       "Chưa xác nhận bước mở hộp rỗng; hãy bắt đầu đúng quy trình.",
                                       action_prediction)
            return None

        removed = self._confirmed_slots & stable_empty
        if removed:
            previous = self.confirmed_insertions
            self._confirmed_slots.difference_update(removed)
            self._recovery_slots.update(removed)
            self._order_latch = None
            self._instruction = f"Lắp lại tai {self._sides(self._recovery_slots)} vào đúng khe; chưa được đóng nắp."
            action = "remove_earbud_to_one" if self.confirmed_insertions else "remove_earbud_to_zero"
            reason = (f"Tháo tai {self._sides(removed)}: số tai đã xác nhận giảm từ "
                      f"{previous} xuống {self.confirmed_insertions}. {self._instruction}")
            support = action_prediction if self._action_is(action_prediction, "remove_earbud") else None
            return self._event(action, reason, support)

        wrong_signature = tuple(sorted(stable_wrong))
        if stable_wrong:
            self._instruction = f"Sai bên tại khe {self._sides(stable_wrong)}. Lấy tai ra và đặt đúng khe."
            if wrong_signature != self._wrong_latch:
                self._wrong_latch = wrong_signature
                return self._event("wrong_earbud_side", self._instruction)
            return None
        if not closed and not self._unsafe_slots:
            self._wrong_latch = ()
        if self._unsafe_slots:
            self._instruction = "Mở hộp và sửa tai nghe nhầm bên trước khi tiếp tục."
            return None

        self._instruction = (
            f"Lắp lại tai {self._sides(self._recovery_slots)} vào đúng khe; chưa được đóng nắp."
            if self._recovery_slots else
            "Đã đủ hai tai nghe. Đóng nắp hộp." if self.confirmed_insertions == 2 else
            f"Lắp tai nghe thứ {self.confirmed_insertions + 1} vào đúng khe."
        )
        new_slots = stable_occupied - self._confirmed_slots
        expected = "insert_first_earbud" if not self._confirmed_slots else "insert_second_earbud"
        if new_slots and self.confirmed_insertions < 2:
            matching = self._action_is(action_prediction, expected) or self._action_is(
                action_prediction, "insert_earbud"
            )
            if matching:
                s = sorted(new_slots)[0]
                self._confirmed_slots.add(s)
                self._recovery_slots.discard(s)
                self._order_latch = None
                self._instruction = (
                    "Đã đủ hai tai nghe. Đóng nắp hộp." if self.confirmed_insertions == 2 else
                    f"Lắp lại tai {self._sides(self._recovery_slots)} vào đúng khe."
                    if self._recovery_slots else "Đã xác nhận một tai. Lắp tai còn lại vào đúng khe."
                )
                return self._event(expected,
                    f"Tai {self._sides([s])} phủ khe đúng bên ổn định; ViT-LSTM xác nhận {expected}.",
                    action_prediction)
            self._instruction = f"Đã thấy tai đúng khe; đang chờ LSTM: {expected} > {self.min_action_confidence:g}."
            if (action_prediction and action_prediction.action.startswith("insert_")
                    and self._action_is(action_prediction, action_prediction.action)
                    and action_prediction.action != self._order_latch):
                self._order_latch = action_prediction.action
                return self._event(action_prediction.action, "Hành động LSTM không đúng thứ tự hiện tại.",
                                   action_prediction)
        else:
            self._order_latch = None

        close_evidence = (closed and case_stable) or not self.require_closed_case
        if close_evidence and self._action_is(action_prediction, "close_case") and not self._close_latched:
            self._close_latched = True
            enough = self.confirmed_insertions == 2 and not self._recovery_slots
            self._instruction = (
                "Hoàn tất. Nhấn R để bắt đầu hộp mới." if enough else
                "Đóng nắp quá sớm. Mở lại hộp và lắp đủ tai nghe còn thiếu."
            )
            return self._event("close_case", self._instruction, action_prediction)
        if closed and self._close_latched and self.confirmed_insertions == 2:
            self._instruction = "Hoàn tất. Nhấn R để bắt đầu hộp mới."
        return None

"""A readable, Unicode dashboard; never modifies images used for inference."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class ConfirmationView:
    kind: str
    title: str
    detail: str
    steps: tuple[tuple[str, str], ...]


def confirmation_view(tracker, *, outcome=None, recent_outcomes=(),
                      prediction=None, prediction_fresh=False, action_threshold=0.5):
    """FSM progress is the source of truth, never the raw model prediction."""
    history = tuple(recent_outcomes)
    last = outcome if outcome is not None and outcome.type in {"PASS", "VIOLATION", "RESET"} else next(
        (item for item in reversed(history) if item.type in {"PASS", "VIOLATION", "RESET"}), None
    )
    past_passes = {item.action for item in history if item.type == "PASS"}
    steps = tuple((step.label,
                   "confirmed" if step.action in tracker.completed_steps else
                   "rework" if step.action in past_passes else
                   "expected" if step.action in tracker.expected_actions else "pending")
                  for step in tracker.config.workflow)
    if last is not None and last.type == "VIOLATION":
        return ConfirmationView("violation", "VI PHẠM — CẦN KHẮC PHỤC", last.message, steps)
    if tracker.is_complete:
        return ConfirmationView("complete", "CHU TRÌNH ĐÃ HOÀN TẤT", "Đã xác nhận đủ các bước. Nhấn R để bắt đầu lượt mới.", steps)
    if last is not None and last.type == "PASS" and last.action in tracker.completed_steps:
        label = tracker.config.actions.get(last.action, last.action)
        return ConfirmationView("confirmed", f"ĐÃ XÁC NHẬN: {label}",
                                f"FSM chấp nhận • {len(tracker.completed_steps)}/{len(steps)} bước hoàn thành", steps)
    if prediction is not None and prediction_fresh and prediction.confidence > action_threshold and prediction.action not in tracker.config.idle_actions:
        label = tracker.config.actions.get(prediction.action, prediction.action)
        return ConfirmationView("waiting", "ĐÃ NHẬN DIỆN — CHƯA XÁC NHẬN", f"LSTM: {label}. Chờ bằng chứng YOLO và FSM.", steps)
    return ConfirmationView("waiting", "ĐANG CHỜ XÁC NHẬN QUY TRÌNH", tracker.instruction, steps)


@lru_cache(maxsize=12)
def _font(size: int):
    fonts = [
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/segoeui.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in fonts:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def slot_coverage_text(slot) -> str:
    """Do not present missing detector evidence as a measured zero overlap."""
    if slot.status == "closed":
        return "Phủ khe: nắp đóng"
    if slot.box is None:
        return "Phủ khe: thiếu bbox khe"
    if not slot.coverage_available:
        return "Phủ khe: thiếu bbox tai"
    return f"Tai phủ khe {slot.coverage:.0%}"


def draw_dashboard(
    frame, *, tracker, fusion, prediction, prediction_fresh: bool,
    action_threshold: float, outcome=None, recent_outcomes=(),
    embedding_count: int = 0, sequence_length: int = 16,
    yolo_ms: float = 0, vit_ms: float = 0, fps: float = 0, mirror: bool = False,
):
    """Render a fixed logical canvas, resized only by the display window."""
    canvas = np.full((900, 1440, 3), (20, 17, 14), dtype=np.uint8)
    height, width = frame.shape[:2]
    scale = min(940 / width, 480 / height)
    shown = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
    h, w = shown.shape[:2]
    x, y = 20 + (940-w)//2, 94 + (480-h)//2
    canvas[y:y+h, x:x+w] = shown
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    white, muted = (235, 240, 250), (159, 177, 196)
    green, amber, red = (73, 218, 147), (255, 197, 85), (255, 104, 117)

    def text(pos, value, size=21, fill=white):
        draw.text(pos, str(value), font=_font(size), fill=fill)

    def wrapped(pos, value, max_width, size=22, fill=white, max_lines=3):
        x0, y0 = pos
        lines, current = [], ""
        for word in str(value).split():
            candidate = (current + " " + word).strip()
            if current and draw.textlength(candidate, font=_font(size)) > max_width:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        for i, line in enumerate(lines[:max_lines]):
            if i == max_lines-1 and len(lines) > max_lines:
                while line and draw.textlength(line + "…", font=_font(size)) > max_width:
                    line = line[:-1]
                line += "…"
            text((x0, y0+i*(size+8)), line, size, fill)

    view = confirmation_view(tracker, outcome=outcome, recent_outcomes=recent_outcomes,
                             prediction=prediction, prediction_fresh=prediction_fresh,
                             action_threshold=action_threshold)
    text((24, 17), "GIÁM SÁT LẮP TAI NGHE", 30)
    text((650, 26), f"YOLO + DINOv2 + BiLSTM  |  Đã lắp {fusion.confirmed_insertions}/2", 22, green)
    text((24, 61), f"{fps:.1f} FPS  •  YOLO {yolo_ms:.0f} ms  •  DINO/LSTM {vit_ms:.0f} ms", 18, muted)
    text((720, 61), f"Mirror: {'ON' if mirror else 'OFF'}", 18, muted)
    draw.rounded_rectangle((980, 94, 1420, 853), radius=14, fill=(26, 35, 49))
    text((1000, 110), "XÁC NHẬN TỪNG BƯỚC", 23)
    step_names = {"confirmed": "ĐÃ XÁC NHẬN", "expected": "ĐANG CHỜ",
                  "pending": "CHƯA THỰC HIỆN", "rework": "CẦN LÀM LẠI"}
    for i, (label, status) in enumerate(view.steps):
        color = green if status == "confirmed" else red if status == "rework" else amber if status == "expected" else muted
        wrapped((1000, 151+i*54), f"{i+1}. {label}", 396, 20, color, 1)
        text((1022, 176+i*54), step_names[status], 14, color)

    statuses = {
        "empty": "TRỐNG", "occupied": "TAI ĐÚNG KHE", "wrong_side": "SAI BÊN",
        "unknown": "CHƯA RÕ", "closed": "NẮP ĐÓNG",
    }
    for i, slot in enumerate(getattr(fusion, "slot_views", ())):
        top = 382 + i*88
        side = "TRÁI" if "left" in slot.label else "PHẢI"
        color = red if slot.status == "wrong_side" else green if slot.confirmed else amber
        text((1000, top), f"KHE {side}: {statuses.get(slot.status, slot.status)}", 20, color)
        stable = min(slot.stable_count, fusion.stable_frames)
        text((1000, top+29), f"{slot_coverage_text(slot)} • ổn định {stable}/{fusion.stable_frames}", 15, muted)
        empty_text = f"Empty {slot.empty_confidence:.0%}" if slot.empty_confidence is not None else "Empty —"
        text((1000, top+53), f"{empty_text} • " + ("Đã xác nhận lắp" if slot.confirmed else "Chưa xác nhận lắp"), 15, color)
    wrapped((1000, 574), f"FSM: {tracker.state}", 395, 17, muted, 1)
    text((1000, 612), "LỊCH SỬ XÁC NHẬN / VI PHẠM", 18)
    history = [item for item in recent_outcomes if item.type in {"PASS", "VIOLATION"}][-3:]
    if not history:
        text((1000, 654), "Chưa có bước nào được xác nhận.", 17, muted)
    for i, item in enumerate(reversed(history)):
        color = green if item.type == "PASS" else red
        label = tracker.config.actions.get(item.action, item.action)
        result = "ĐÃ XÁC NHẬN" if item.type == "PASS" else "VI PHẠM"
        stamp = item.timestamp[11:19] + " UTC"
        text((1000, 651+i*62), f"{stamp} • {result}", 15, color)
        wrapped((1000, 673+i*62), label, 396, 17, white, 1)
    text((1000, 825), "R: xóa lịch sử, bắt đầu lượt mới", 16, muted)

    banner_color = red if view.kind == "violation" else green if view.kind in {"confirmed", "complete"} else amber
    banner_fill = (64, 29, 37) if view.kind == "violation" else (25, 57, 44) if view.kind in {"confirmed", "complete"} else (49, 43, 27)
    draw.rounded_rectangle((20, 588, 960, 682), radius=12, fill=banner_fill)
    wrapped((36, 598), view.title, 906, 26, banner_color, 1)
    wrapped((36, 639), view.detail, 906, 18, white, 1)

    if prediction is None:
        action_text = (f"Đang lấy mẫu DINOv2: {embedding_count}/{sequence_length}" if embedding_count < sequence_length
                       else "LSTM: chờ kết quả mới sau sự kiện (giữ chuỗi đặc trưng liên tục)")
        action_color = amber
    elif not prediction_fresh:
        action_text = f"LSTM: chờ mẫu mới (kết quả cũ: {prediction.action})"
        action_color = amber
    elif prediction.confidence <= action_threshold:
        action_text = f"LSTM: chưa đủ tin cậy • {prediction.action} {prediction.confidence:.1%} ≤ {action_threshold:.0%}"
        action_color = amber
    else:
        action_text = f"LSTM: {prediction.action} • {prediction.confidence:.1%} (không phải xác nhận FSM)"
        action_color = green
    wrapped((24, 698), action_text, 930, 19, action_color, 1)
    draw.rounded_rectangle((20, 736, 960, 853), radius=12, fill=(25, 46, 52))
    text((36, 746), "VIỆC CẦN LÀM / ĐIỀU KIỆN ĐANG CHỜ", 17, muted)
    instruction = getattr(fusion, "instruction", tracker.instruction)
    wrapped((36, 774), instruction, 906, 21, white, 2)
    text((24, 873), "F: toàn màn hình   |   R: reset   |   Q / Esc: thoát", 17, muted)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

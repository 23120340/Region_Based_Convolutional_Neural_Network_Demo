"""A readable, Unicode dashboard; never modifies images used for inference."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Clickable areas in dashboard (canvas) coordinates: x1, y1, x2, y2.
PHONE_BUTTON = (1080, 866, 1420, 897)
PHONE_NEXT_ADDRESS = (1000, 604, 1400, 638)
PHONE_NEW_SESSION = (996, 652, 1130, 694)
PHONE_DISCONNECT = (1140, 652, 1290, 694)
PHONE_CLOSE_PANEL = (1300, 652, 1404, 694)
_PHONE_STATE_TEXT = {
    "idle": "Chưa kết nối", "waiting": "Chờ quét QR", "page_opened": "Đã mở trang, chờ camera",
    "connecting": "Đang kết nối", "connected": "Connected", "disconnected": "Disconnected",
    "expired": "QR hết hạn", "closed": "Disconnected", "error": "Lỗi",
}


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
    if not getattr(slot, "coverage_available", True):
        return "Phủ khe: thiếu bbox tai"
    return f"Tai phủ khe {slot.coverage:.0%}"


def draw_dashboard(
    frame, *, tracker, fusion, prediction, prediction_fresh: bool,
    action_threshold: float, outcome=None, recent_outcomes=(),
    embedding_count: int = 0, sequence_length: int = 16,
    yolo_ms: float = 0, vit_ms: float = 0, fps: float = 0, mirror: bool = False,
    embedding_fps: float = 0, window_duration_s: float = 0,
    phone=None,
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

    confirmation = confirmation_view(tracker, outcome=outcome, recent_outcomes=recent_outcomes,
                                     prediction=prediction, prediction_fresh=prediction_fresh,
                                     action_threshold=action_threshold)
    text((24, 17), "GIÁM SÁT LẮP TAI NGHE", 30)
    text((650, 26), f"YOLO + DINOv2 + BiLSTM  |  Đã lắp {fusion.confirmed_insertions}/2", 22, green)
    text((24, 61), f"UI {fps:.1f} FPS  •  Embedding {embedding_fps:.1f}/10 FPS  •  Window {window_duration_s:.2f}s", 18, muted)
    text((720, 61), f"YOLO {yolo_ms:.0f} ms  •  DINO/LSTM {vit_ms:.0f} ms  •  Mirror: {'ON' if mirror else 'OFF'}", 16, muted)
    draw.rounded_rectangle((980, 94, 1420, 704), radius=14, fill=(26, 35, 49))
    text((1000, 110), "QUY TRÌNH", 23)
    for i, step in enumerate(tracker.config.workflow):
        done = step.action in tracker.completed_steps
        expected = step.action in tracker.expected_actions
        color = green if done else amber if expected else muted
        mark = "+" if done else "→" if expected else "·"
        wrapped((1000, 154+i*48), f"{mark} {i+1}. {step.label}", 396, 20, color, 1)

    statuses = {
        "empty": "TRỐNG", "occupied": "TAI ĐÚNG KHE", "wrong_side": "SAI BÊN",
        "unknown": "CHƯA RÕ", "closed": "NẮP ĐÓNG",
    }
    for i, slot in enumerate(getattr(fusion, "slot_views", ())):
        top = 365 + i*112
        side = "TRÁI" if "left" in slot.label else "PHẢI"
        color = red if slot.status == "wrong_side" else green if slot.confirmed else amber
        text((1000, top), f"KHE {side}: {statuses.get(slot.status, slot.status)}", 21, color)
        text((1000, top+32),
             f"{slot_coverage_text(slot)}  •  Bằng chứng {slot.evidence_seconds:.2f}/{slot.required_seconds:.2f}s",
             17, muted)
        empty_confidence = getattr(slot, "empty_confidence", None)
        empty_text = f"Empty {empty_confidence:.0%}" if empty_confidence is not None else "Empty —"
        text((1000, top+60),
             f"{empty_text} • " + ("Đã xác nhận" if slot.confirmed else "Chưa xác nhận lắp"),
             18, color)
    observation = getattr(tracker, "observation_state", "UNKNOWN")
    observation = getattr(observation, "value", observation)
    wrapped((1000, 611), f"FSM: {tracker.state} • Quan sát: {observation}", 395, 17, muted, 2)
    text((1000, 673), "R: lượt mới / đặt lại", 18, muted)

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
    violation = confirmation.kind == "violation"
    wrapped((36, 768), instruction, 1362, 25, amber if violation else white, 2)
    if outcome is not None:
        wrapped((36, 833), f"Lần gần nhất: {outcome.type} · {outcome.action}",
                1355, 17, red if violation else green, 1)
    text((24, 873), "F: toàn màn hình  |  R: reset  |  P: camera điện thoại  |  Q / Esc: thoát", 17, muted)
    if phone is not None:
        _draw_phone(pil, draw, text, wrapped, phone, (white, muted, green, amber, red))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _inside(rect, x, y) -> bool:
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def phone_hit_test(x: int, y: int, view) -> str | None:
    """Map a click on the dashboard canvas to a phone-camera action."""
    from .phone_camera_ui import CLOSE_PANEL, DISCONNECT, NEW_SESSION, NEXT_ADDRESS, OPEN

    if _inside(PHONE_BUTTON, x, y):
        return OPEN
    if not view.panel_open:
        return None
    status = view.status
    if _inside(PHONE_NEW_SESSION, x, y):
        return NEW_SESSION
    if _inside(PHONE_DISCONNECT, x, y) and status is not None:
        return DISCONNECT
    if _inside(PHONE_CLOSE_PANEL, x, y):
        return CLOSE_PANEL
    if _inside(PHONE_NEXT_ADDRESS, x, y) and status is not None and status.address_count > 1:
        return NEXT_ADDRESS
    return None


def _phone_label(view) -> tuple[str, str]:
    status = view.status
    if status is None:
        return ("Lỗi" if view.error else "Chưa kết nối"), ("red" if view.error else "muted")
    if status.connected:
        return f"Connected · {status.fps:.0f} FPS", "green"
    label = _PHONE_STATE_TEXT.get(status.state, status.state)
    if status.state in {"disconnected", "closed", "expired", "error"}:
        return label, "red"
    return label, ("muted" if status.state == "idle" else "amber")


def _draw_phone(pil, draw, text, wrapped, view, colors) -> None:
    white, muted, green, amber, red = colors
    palette = {"white": white, "muted": muted, "green": green, "amber": amber, "red": red}
    label, color_name = _phone_label(view)
    color = palette[color_name]
    source = {"phone": "điện thoại", "local": "camera PC", "placeholder": "chờ điện thoại"}
    text((890, 61), f"Nguồn: {source.get(view.source_kind, view.source_kind)}  •  ĐT: {label}", 18, color)

    def button(rect, caption, fill, size=18):
        draw.rounded_rectangle(rect, radius=9, fill=fill)
        width = draw.textlength(caption, font=_font(size))
        text((rect[0] + (rect[2] - rect[0] - width) / 2, rect[1] + (rect[3] - rect[1] - size) / 2 - 2),
             caption, size, white)

    button(PHONE_BUTTON, "Kết nối camera điện thoại", (38, 99, 196) if not view.panel_open else (52, 65, 90))
    if not view.panel_open:
        return

    status = view.status
    draw.rounded_rectangle((980, 94, 1420, 704), radius=14, fill=(22, 30, 44), outline=(61, 139, 253), width=2)
    text((1000, 108), "CAMERA ĐIỆN THOẠI", 23)
    text((1000, 140), label, 20, color)
    if view.error:
        wrapped((1000, 180), view.error, 396, 18, red, 8)
    elif status is not None and status.connected:
        text((1000, 200), "ĐÃ KẾT NỐI", 34, green)
        wrapped((1000, 250), f"Truyền: {status.transport}", 396, 18, muted, 1)
        if status.frame_size:
            text((1000, 280), f"Khung hình: {status.frame_size[0]}×{status.frame_size[1]}", 18, muted)
        wrapped((1000, 310), status.device, 396, 15, muted, 3)
    elif view.qr is not None and status is not None and status.url:
        qr = view.qr
        pil.paste(Image.fromarray(cv2.cvtColor(qr, cv2.COLOR_BGR2RGB)), (1060, 172))
        wrapped((1000, 460), status.url.split("#", 1)[0], 400, 14, muted, 2)
        if status.expires_in is not None:
            text((1000, 500), f"QR hết hạn sau {int(status.expires_in // 60)}:{int(status.expires_in % 60):02d}",
                 16, muted)
    elif status is not None and status.url:
        wrapped((1000, 180), "Thiếu thư viện segno để vẽ QR. Mở link sau trên điện thoại:", 396, 17, amber, 2)
        wrapped((1000, 240), status.url, 396, 15, white, 6)
    elif status is not None:
        wrapped((1000, 180), status.message or "Nhấn \"QR mới\" để tạo phiên kết nối.", 396, 19, white, 6)
    if status is not None and not view.error:
        hint = status.message if not status.connected and status.url else ""
        if status.warnings:
            hint = status.warnings[0]
        if hint:
            wrapped((1000, 524), hint, 396, 15, amber if status.warnings else muted, 3)
        if status.address_count > 1 and not status.connected:
            button(PHONE_NEXT_ADDRESS, f"Đổi IP ({status.address}) nếu ĐT không mở được", (52, 65, 90), 15)
    button(PHONE_NEW_SESSION, "QR mới (N)", (38, 99, 196), 17)
    button(PHONE_DISCONNECT, "Ngắt (D)", (179, 48, 61), 17)
    button(PHONE_CLOSE_PANEL, "Đóng (P)", (52, 65, 90), 17)

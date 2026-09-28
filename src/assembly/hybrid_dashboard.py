"""A readable, Unicode dashboard; never modifies images used for inference."""
from __future__ import annotations

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


def draw_dashboard(
    frame, *, tracker, fusion, prediction, prediction_fresh: bool,
    action_threshold: float, outcome=None,
    embedding_count: int = 0, sequence_length: int = 16,
    yolo_ms: float = 0, vit_ms: float = 0, fps: float = 0, mirror: bool = False,
    phone=None,
):
    """Render a fixed logical canvas, resized only by the display window."""
    canvas = np.full((900, 1440, 3), (20, 17, 14), dtype=np.uint8)
    height, width = frame.shape[:2]
    scale = min(940 / width, 610 / height)
    shown = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
    h, w = shown.shape[:2]
    x, y = 20 + (940-w)//2, 94 + (610-h)//2
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
                line += "…"
            text((x0, y0+i*(size+8)), line, size, fill)

    text((24, 17), "GIÁM SÁT LẮP TAI NGHE", 30)
    text((650, 26), f"YOLO + DINOv2 + BiLSTM  |  Đã lắp {fusion.confirmed_insertions}/2", 22, green)
    text((24, 61), f"{fps:.1f} FPS  •  YOLO {yolo_ms:.0f} ms  •  DINO/LSTM {vit_ms:.0f} ms", 18, muted)
    text((720, 61), f"Mirror: {'ON' if mirror else 'OFF'}", 18, muted)
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
    for i, view in enumerate(getattr(fusion, "slot_views", ())):
        top = 365 + i*112
        side = "TRÁI" if "left" in view.label else "PHẢI"
        color = red if view.status == "wrong_side" else green if view.confirmed else amber
        text((1000, top), f"KHE {side}: {statuses.get(view.status, view.status)}", 21, color)
        stable = min(view.stable_count, fusion.stable_frames)
        text((1000, top+32), f"Phủ khe {view.coverage:.0%}  •  Ổn định {stable}/{fusion.stable_frames}", 18, muted)
        text((1000, top+60), "Đã xác nhận" if view.confirmed else "Chưa xác nhận lắp", 18, color)
    wrapped((1000, 611), f"FSM: {tracker.state}", 395, 17, muted, 2)
    text((1000, 673), "R: lượt mới / đặt lại", 18, muted)

    if prediction is None:
        action_text = f"Đang lấy mẫu DINOv2: {embedding_count}/{sequence_length}"
        action_color = amber
    elif not prediction_fresh:
        action_text = f"LSTM: chờ mẫu mới (kết quả cũ: {prediction.action})"
        action_color = amber
    elif prediction.confidence <= action_threshold:
        action_text = f"LSTM: uncertain • {prediction.action} {prediction.confidence:.1%} ≤ {action_threshold:.0%}"
        action_color = amber
    else:
        action_text = f"LSTM: {prediction.action} • {prediction.confidence:.1%}"
        action_color = green
    text((24, 718), action_text, 22, action_color)
    violation = outcome is not None and outcome.type == "VIOLATION"
    draw.rounded_rectangle((20, 758, 1420, 861), radius=12,
                           fill=(64, 29, 37) if violation else (25, 46, 52))
    instruction = getattr(fusion, "instruction", tracker.instruction)
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

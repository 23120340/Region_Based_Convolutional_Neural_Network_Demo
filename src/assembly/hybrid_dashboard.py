"""A readable, Unicode dashboard; never modifies images used for inference."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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
    text((24, 873), "F: toàn màn hình   |   R: reset   |   Q / Esc: thoát", 17, muted)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

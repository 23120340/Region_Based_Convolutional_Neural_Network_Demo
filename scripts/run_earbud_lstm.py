#!/usr/bin/env python3
"""
run_earbud_lstm.py – Demo hệ thống giám sát lắp ráp earbud sử dụng mô hình Action LSTM.
=====================================================================================
Nhận diện luồng video theo thời gian thực (real-time) bằng cách trích xuất frame,
chạy ViT embedding và đưa vào mạng LSTM để dự đoán hành động, sau đó cập nhật FSM.

Cách dùng:
  python scripts/run_earbud_lstm.py
  python scripts/run_earbud_lstm.py --source 0
  python scripts/run_earbud_lstm.py --source data/earbud_actions/raw_videos/per1_01_test.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

# Đảm bảo import được module trong src/
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import torch

from assembly.action_config import load_action_model_config
from assembly.config import load_config
from assembly.fsm import ConfigurableAssemblyTracker, FsmOutcome
from assembly.models.vit_lstm_recognizer import ViTLstmActionRecognizer
from assembly.monitor import AssemblyMonitor, JsonlEventLogger
from assembly.paths import (
    DEFAULT_ACTION_CONFIG,
    DEFAULT_CONFIG,
    DEFAULT_EVENT_LOG,
)
from assembly.smoother import TemporalDebouncer

DEFAULT_LSTM_MODEL = ROOT / "artifacts" / "action_model" / "best.pt"


def _configure_utf8_console() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _put_text(frame, text: str, origin: tuple[int, int], scale: float = 0.58, color=(255, 255, 255)) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def main() -> int:
    _configure_utf8_console()
    parser = argparse.ArgumentParser(description="Chạy Action LSTM Real-time")
    parser.add_argument("--source", default="0", help="Camera index (0) hoặc đường dẫn video")
    parser.add_argument("--model", type=Path, default=DEFAULT_LSTM_MODEL)
    parser.add_argument("--action-config", type=Path, default=DEFAULT_ACTION_CONFIG)
    parser.add_argument("--fsm-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default=None, help="cpu hoặc cuda")
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=None,
        help="Ghi đè FPS lấy mẫu; nên dùng 4-5 trên CPU yếu (mặc định theo action config)",
    )
    parser.add_argument(
        "--inference-stride",
        type=int,
        default=1,
        help="Số embedding mới giữa hai lần chạy BiLSTM",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Cho phép tải backbone từ Hugging Face nếu máy chưa có cache",
    )
    args = parser.parse_args()

    if args.sample_fps is not None and args.sample_fps <= 0:
        parser.error("--sample-fps phải lớn hơn 0")
    if args.inference_stride < 1:
        parser.error("--inference-stride phải lớn hơn 0")

    if not args.model.exists():
        print(f"[ERR] Không tìm thấy model: {args.model}")
        print("      Vui lòng chạy 'python scripts/train_action_model.py' trước.")
        return 1

    print(f"[INFO] Tải cấu hình và khởi tạo FSM...")
    assembly_config = load_config(args.fsm_config)
    tracker = ConfigurableAssemblyTracker(assembly_config)
    debouncer = TemporalDebouncer(
        window_size=3,
        min_votes=2,
        min_confidence=0.50,
        idle_actions=assembly_config.idle_actions
    )
    monitor = AssemblyMonitor(tracker, debouncer, JsonlEventLogger(DEFAULT_EVENT_LOG))

    print(f"[INFO] Khởi tạo mô hình ViT + LSTM từ {args.model.name}...")
    action_config = load_action_model_config(args.action_config)
    recognizer = ViTLstmActionRecognizer(
        config_path=args.action_config,
        checkpoint_path=args.model,
        device=args.device,
        local_files_only=not args.allow_download,
    )

    seq_len = action_config.temporal.sequence_length
    sample_fps = args.sample_fps or action_config.spatial.sample_fps
    sample_interval = 1.0 / sample_fps
    print(f"[INFO] Device: {recognizer.device} | sample_fps={sample_fps:g} | sequence={seq_len}")
    if recognizer.device.type == "cpu" and sample_fps > 5:
        print(
            "[WARN] ViT-Base đang chạy bằng CPU. Nếu hình vẫn chậm, dùng --sample-fps 4; "
            "muốn giữ đúng 10 FPS như lúc train thì cần GPU hoặc backbone nhỏ hơn và train lại."
        )

    # Chuyển đổi source thành int nếu là số (camera index)
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERR] Không thể mở source: {args.source}")
        return 1

    # Nếu là camera index, set độ phân giải
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"[INFO] Đang chạy nhận diện. Nhấn 'Q' hoặc 'ESC' để thoát.")

    # Chỉ encode frame mới một lần. Code cũ encode lại toàn bộ 16 frame sau
    # mỗi lần lấy mẫu, khiến ViT phải làm việc nặng hơn khoảng 16 lần.
    embedding_buffer = deque(maxlen=seq_len)
    next_sample_time = time.perf_counter()
    last_infer_time_ms = 0.0
    current_prediction = None
    outcome: FsmOutcome | None = None
    total_samples = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str): # Nếu là video thì lặp lại hoặc thoát
                print("[INFO] Hết video.")
                break
            continue

        if isinstance(source, int):
            frame = cv2.flip(frame, 1) # Lật gương camera

        height, width = frame.shape[:2]
        now = time.perf_counter()

        # Rút trích frame theo đúng sample_fps
        if now >= next_sample_time:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            start_infer = time.perf_counter()
            embedding_buffer.append(recognizer.encode_frame(rgb_frame))
            total_samples += 1
            next_sample_time = now + sample_interval

            # Khi đủ embedding, BiLSTM chỉ đọc cache; không chạy lại ViT trên
            # 15 frame cũ. inference_stride=1 cho phản hồi nhanh nhất.
            ready_for_lstm = len(embedding_buffer) == seq_len and (
                (total_samples - seq_len) % args.inference_stride == 0
            )
            if ready_for_lstm:
                pred = recognizer.predict_embeddings(list(embedding_buffer))
                current_prediction = pred
                new_outcome = monitor.submit_prediction(pred)
                if new_outcome is not None:
                    outcome = new_outcome
                    print(f"[{outcome.type}] {outcome.action}: {outcome.message}")
                    if outcome.type == "PASS":
                        # Không để frame của bước vừa hoàn tất tiếp tục chi phối
                        # cửa sổ kế tiếp và gây VIOLATION giả sau close_case.
                        embedding_buffer.clear()
                        total_samples = 0
                        current_prediction = None
            last_infer_time_ms = (time.perf_counter() - start_infer) * 1000

        # ----- Vẽ Giao Diện Lên Video -----
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 105), (18, 24, 38), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Trạng thái FSM
        expected = ", ".join(monitor.tracker.expected_actions) or "hoàn thành"
        _put_text(frame, f"State: {monitor.tracker.state}  |  Expected: {expected}", (18, 32), 0.70, (255, 255, 255))

        # Trạng thái Model
        if current_prediction:
            pred_text = f"Action: {current_prediction.action} ({current_prediction.confidence:.2f})"
            _put_text(frame, pred_text, (width - 400, 32), 0.65, (100, 255, 100))

        # Hiển thị thông báo Outcome của hệ thống
        if outcome is not None:
            color = (80, 230, 100) if outcome.type == "PASS" else (70, 70, 255)
            _put_text(frame, f"> {outcome.type}: {outcome.action}", (18, 64), 0.65, color)
        else:
            _put_text(frame, "> Thực hiện hành động tiếp theo trong tầm nhìn camera...", (18, 64), 0.55, (190, 210, 255))

        # Thông tin kỹ thuật
        buffer_status = f"Emb: {len(embedding_buffer)}/{seq_len}"
        _put_text(frame, f"Keys: R (Reset), Q/ESC (Quit) | Infer: {last_infer_time_ms:.0f}ms | {buffer_status}", (18, 92), 0.48, (170, 180, 190))

        # Khung viền chỉ báo FSM state (Xanh nếu PASS, đỏ nếu ERROR, vàng nếu đang tiến hành)
        if outcome and outcome.type == "ERROR":
            cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 4)
        elif outcome and outcome.type == "PASS":
            cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 255, 0), 4)

        cv2.imshow("Earbud Assembly - Real-time Action LSTM", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            outcome = monitor.reset()
            embedding_buffer.clear()
            total_samples = 0
            current_prediction = None
            print("Đã reset trạng thái hệ thống.")

    cap.release()
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    sys.exit(main())

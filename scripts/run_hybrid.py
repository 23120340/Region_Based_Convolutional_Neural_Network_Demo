from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2

from assembly.action_config import load_action_model_config
from assembly.cadence import CumulativeDeadlineScheduler, EmbeddingTelemetry
from assembly.camera_config import load_camera_config
from assembly.camera_input import LOCAL, PHONE_SOURCE, PLACEHOLDER, CameraInputRouter, is_phone_source
from assembly.config import load_config
from assembly.fsm import ConfigurableAssemblyTracker, FsmOutcome
from assembly.hybrid_dashboard import draw_dashboard, phone_hit_test
from assembly.model_contract import Prediction
from assembly.models.vit_lstm_recognizer import ViTLstmActionRecognizer
from assembly.monitor import AssemblyMonitor, JsonlEventLogger
from assembly.phone_camera import PhoneCameraServer
from assembly.phone_camera_ui import PhoneCameraController
from assembly.project_config import build_fusion_engine, load_project_config
from assembly.smoother import TemporalDebouncer
from assembly.temporal_config import load_temporal_fusion_config
from assembly.vision import Detection
from assembly.yolo_world_detector import YoloWorldDetector


DEFAULT_PROJECT = ROOT / "configs" / "projects" / "earbud_v2.json"


def _configure_utf8_console() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _input_path(value: str | Path) -> Path:
    """Resolve CLI inputs from cwd, with project root as a portable fallback."""
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return path.resolve()
    return (ROOT / path).resolve()


def _source(value: str) -> int | str:
    if is_phone_source(value):
        return PHONE_SOURCE
    if value.isdigit():
        return int(value)
    if "://" in value:
        return value
    return str(_input_path(value))


def _torch_device(value: str | None) -> str | None:
    if value is not None and value.isdigit():
        return f"cuda:{value}"
    return value


def _put_text(frame, text: str, origin: tuple[int, int], scale=0.55, color=(255, 255, 255)) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def _draw_detections(frame, detections: list[Detection], camera_config, min_confidence: float = 0.5) -> None:
    label_map = camera_config.label_map
    for detection in detections:
        if detection.confidence <= min_confidence:
            continue
        vision_class = label_map.get(detection.label)
        color = vision_class.color_bgr if vision_class is not None else (0, 255, 0)
        x1, y1, x2, y2 = detection.box_xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        _put_text(
            frame,
            f"{detection.label} {detection.confidence:.2f}",
            (x1, max(18, y1 - 6)),
            0.45,
            color,
        )


def _resolve_detector_model(configured_model: str, override: Path | None) -> str:
    if override is not None:
        candidate = _input_path(override)
        if not candidate.is_file():
            raise SystemExit(f"Không tìm thấy YOLO checkpoint: {candidate}")
        return str(candidate)
    candidate = Path(configured_model).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    elif candidate.parent != Path("."):
        resolved = (ROOT / candidate).resolve()
    else:
        # A bare model name may be a model alias handled by Ultralytics.
        return configured_model
    if not resolved.is_file():
        raise SystemExit(f"Không tìm thấy YOLO checkpoint: {resolved}")
    return str(resolved)


def build_parser(default_project: Path = DEFAULT_PROJECT) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Runtime dùng lại được: YOLO + ViT-BiLSTM + Fusion Engine + FSM"
    )
    parser.add_argument("--project", type=Path, default=default_project, help="Project profile JSON")
    parser.add_argument("--source", default="0",
                        help="Camera index, đường dẫn video hoặc 'phone' (chỉ dùng camera điện thoại qua QR)")
    parser.add_argument("--camera-config", type=Path, default=None)
    parser.add_argument("--fsm-config", type=Path, default=None)
    parser.add_argument("--action-config", type=Path, default=None)
    parser.add_argument("--action-model", type=Path, default=None)
    parser.add_argument("--fusion-config", type=Path, default=None,
                        help="Ngưỡng cadence/confidence/duration cho temporal fusion")
    parser.add_argument("--event-log", type=Path, default=None)
    parser.add_argument("--yolo-model", type=Path, default=None)
    parser.add_argument("--device", default=None, help="cpu, cuda hoặc GPU index")
    parser.add_argument("--yolo-every", type=int, default=None)
    parser.add_argument("--action-confidence", type=float, default=None)
    parser.add_argument("--action-ttl", type=float, default=1.5)
    parser.add_argument("--test-start-open", action="store_true",
                        help="Test-only: lấy hộp đã mở làm baseline, không giả lập PASS open_case")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--window-width", type=int, default=1440)
    parser.add_argument("--window-height", type=int, default=900)
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--headless", action="store_true", help="Đọc video, ghi log, không mở cửa sổ")
    parser.add_argument("--max-frames", type=int, default=None, help=argparse.SUPPRESS)
    phone = parser.add_argument_group("camera điện thoại (nút 'Kết nối camera điện thoại' / phím P)")
    phone.add_argument("--phone-port", type=int, default=8443,
                       help="Cổng HTTPS cho trang camera; bận thì tự thử 9 cổng kế tiếp")
    phone.add_argument("--phone-host", default=None,
                       help="Ép IP/hostname đưa vào QR; mặc định tự dò IP LAN của máy")
    phone.add_argument("--phone-public-url", default=None,
                       help="URL HTTPS công khai (tunnel) khi điện thoại khác mạng với PC")
    phone.add_argument("--phone-transport", choices=["auto", "webrtc", "websocket"], default="auto",
                       help="auto: WebRTC, lỗi thì WebSocket JPEG")
    phone.add_argument("--phone-ice-server", action="append", default=[],
                       help="STUN/TURN URL cho WebRTC khác mạng, ví dụ stun:stun.l.google.com:19302")
    phone.add_argument("--phone-cert", type=Path, default=None, help="Chứng chỉ TLS tin cậy (PEM)")
    phone.add_argument("--phone-key", type=Path, default=None, help="Private key của --phone-cert")
    phone.add_argument("--phone-session-ttl", type=float, default=600.0,
                       help="Số giây QR còn hiệu lực trước khi có điện thoại kết nối")
    return parser


def main(default_project: Path = DEFAULT_PROJECT) -> int:
    _configure_utf8_console()
    parser = build_parser(default_project)
    args = parser.parse_args()

    if args.yolo_every is not None and args.yolo_every < 1:
        parser.error("--yolo-every phải >= 1")
    if args.action_ttl <= 0:
        parser.error("--action-ttl phải > 0")
    if args.window_width < 640 or args.window_height < 480:
        parser.error("Kích thước cửa sổ tối thiểu 640x480")
    if args.action_confidence is not None and not 0 <= args.action_confidence <= 1:
        parser.error("--action-confidence phải nằm trong [0, 1]")
    if not 0 <= args.phone_port <= 65535:
        parser.error("--phone-port phải nằm trong [0, 65535]")
    if args.phone_session_ttl <= 0:
        parser.error("--phone-session-ttl phải > 0")

    project = load_project_config(args.project, ROOT)
    camera_config_path = _input_path(args.camera_config or project.camera_config)
    fsm_config_path = _input_path(args.fsm_config or project.fsm_config)
    action_config_path = _input_path(args.action_config or project.action_config)
    action_model_path = _input_path(args.action_model or project.action_model)
    fusion_config_value = args.fusion_config or project.temporal_config
    if fusion_config_value is None:
        raise SystemExit("Project phải khai báo temporal_config hoặc dùng --fusion-config")
    fusion_config_path = _input_path(fusion_config_value)
    event_log_path = (args.event_log or project.event_log).resolve()
    for label, path in (
        ("camera config", camera_config_path),
        ("FSM config", fsm_config_path),
        ("action config", action_config_path),
        ("action checkpoint", action_model_path),
        ("temporal fusion config", fusion_config_path),
    ):
        if not path.is_file():
            raise SystemExit(f"Không tìm thấy {label}: {path}")

    camera_config = load_camera_config(camera_config_path)
    action_config = load_action_model_config(action_config_path)
    temporal_config = load_temporal_fusion_config(fusion_config_path)
    assembly_config = load_config(fsm_config_path)
    tracker = ConfigurableAssemblyTracker(assembly_config)
    monitor = AssemblyMonitor(
        tracker,
        TemporalDebouncer(idle_actions=assembly_config.idle_actions),
        JsonlEventLogger(event_log_path),
    )
    fusion = build_fusion_engine(
        project.fusion,
        temporal_config=temporal_config,
        test_start_open=True if args.test_start_open else None,
        min_action_confidence=(args.action_confidence if args.action_confidence is not None
                               else temporal_config.confidence.action_min),
    )

    print("[INFO] Đang nạp checkpoint YOLO...", flush=True)
    detector_model_path = _resolve_detector_model(camera_config.model, args.yolo_model)
    print(f"[INFO] YOLO checkpoint: {detector_model_path}", flush=True)
    detector = YoloWorldDetector(
        camera_config,
        model_path=detector_model_path,
        device=args.device,
    )
    required_labels = set(getattr(fusion, "earbud_slot_pairs", {}))
    required_labels.update(getattr(fusion, "earbud_slot_pairs", {}).values())
    if required_labels:
        names = detector.model.names
        values = names.values() if isinstance(names, dict) else names
        actual_labels = {str(n).strip().casefold().replace("-", "_").replace(" ", "_") for n in values}
        missing = (required_labels | {"open_case", "close_case"}) - actual_labels
        if missing:
            raise SystemExit(f"YOLO checkpoint thiếu nhãn cho geometry: {sorted(missing)}")
    print("[INFO] Đang nạp DINOv2 và BiLSTM từ cache/checkpoint...", flush=True)
    print(f"[INFO] LSTM checkpoint: {action_model_path} | actions={list(action_config.actions)}", flush=True)
    recognizer = ViTLstmActionRecognizer(
        config_path=action_config_path,
        checkpoint_path=action_model_path,
        device=_torch_device(args.device),
        local_files_only=not args.allow_download,
    )
    cpu_mode = recognizer.device.type == "cpu"
    sample_fps = temporal_config.scheduler.target_embedding_fps
    yolo_every = args.yolo_every or camera_config.infer_every_n_frames
    if cpu_mode and isinstance(_source(args.source), int):
        print(
            "[WARN] Đang chạy YOLO và ViT bằng CPU; cadence vẫn giữ mục tiêu "
            f"{sample_fps:g} FPS. UI sẽ hiển thị embedding FPS thực đo."
        )

    source = _source(args.source)
    phone_only = source == PHONE_SOURCE
    local_capture = None
    if not phone_only:
        backend = cv2.CAP_DSHOW if isinstance(source, int) and sys.platform == "win32" else cv2.CAP_ANY
        local_capture = cv2.VideoCapture(source, backend)
        if not local_capture.isOpened():
            raise SystemExit(f"Không mở được camera/video: {source!r}")
        if isinstance(source, int):
            local_capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.capture_width)
            local_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.capture_height)
            local_capture.set(cv2.CAP_PROP_BUFFERSIZE, camera_config.capture_buffer_size)
    # Camera input adapter: phone frames replace the local source while the
    # phone streams; the pipeline below reads it like any cv2 capture.
    capture = CameraInputRouter(
        local_capture, placeholder_size=(camera_config.capture_width, camera_config.capture_height))
    phone = PhoneCameraController(capture, lambda: PhoneCameraServer(
        port=args.phone_port,
        host=args.phone_host,
        public_url=args.phone_public_url,
        transport=args.phone_transport,
        cert_file=args.phone_cert,
        key_file=args.phone_key,
        ice_servers=tuple(args.phone_ice_server),
        session_ttl=args.phone_session_ttl,
        target_size=(camera_config.capture_width, camera_config.capture_height),
        target_fps=camera_config.capture_fps,
    ))
    source_generation = capture.generation

    sequence_length = action_config.temporal.sequence_length
    embeddings = deque(maxlen=sequence_length)
    cadence = CumulativeDeadlineScheduler(sample_fps)
    telemetry = EmbeddingTelemetry(
        sequence_length, horizon_s=temporal_config.scheduler.telemetry_horizon_s)
    latest_action: Prediction | None = None
    latest_action_time = 0.0
    detections: list[Detection] = []
    outcome: FsmOutcome | None = None
    recent_outcomes: deque[FsmOutcome] = deque(maxlen=32)
    frame_index = 0
    local_frame_index = 0  # video clock; phone frames must not advance it
    yolo_ms = 0.0
    vit_ms = 0.0
    previous_frame_time = time.perf_counter()
    display_fps = 0.0
    fullscreen = args.fullscreen
    action_threshold = fusion.min_action_confidence
    file_fps = capture.get(cv2.CAP_PROP_FPS) if isinstance(source, str) and not phone_only else 0.0
    video_file = isinstance(source, str) and Path(source).is_file()
    # For replay, sample by video time rather than CPU processing speed.
    if video_file:
        cadence.reset(0.0)
    print(
        f"[INFO] Project={project.name} | device={recognizer.device} | "
        f"sample_fps={sample_fps:g} | YOLO mỗi {yolo_every} frame | action > {action_threshold:g}."
    )

    try:
        if not args.headless:
            cv2.namedWindow(project.display_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(project.display_name, args.window_width, args.window_height)
            if fullscreen:
                cv2.setWindowProperty(project.display_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.setMouseCallback(project.display_name, phone.on_mouse)
        if phone_only:
            phone.open()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if capture.generation != source_generation:
                # Slot anchors and the temporal window belong to one viewpoint.
                source_generation = capture.generation
                outcome = monitor.reset()
                fusion.reset()
                embeddings.clear()
                detections = []
                latest_action = None
                latest_action_time = 0.0
                cadence.reset()
                telemetry.reset()
                print(f"[INFO] Nguồn camera -> {capture.kind}; đã bắt đầu lượt mới.")
            local_live = capture.kind == LOCAL and isinstance(source, int)
            video_clock = video_file and capture.kind == LOCAL
            if capture.kind == LOCAL:
                local_frame_index += 1
            if not args.no_mirror and local_live:
                frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            wall_now = time.perf_counter()
            frame_duration = max(wall_now - previous_frame_time, 1e-6)
            display_fps = 0.9 * display_fps + 0.1 / frame_duration
            previous_frame_time = wall_now
            now = ((local_frame_index - 1) / file_fps if file_fps > 0 else capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                   ) if video_clock else wall_now
            # Nothing to analyse while waiting for the phone (placeholder frame).
            analyse = capture.kind != PLACEHOLDER

            if analyse and cadence.due(now):
                started = time.perf_counter()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                embeddings.append(recognizer.encode_frame(rgb))
                telemetry.mark(now)
                if len(embeddings) == sequence_length:
                    latest_action = recognizer.predict_embeddings(list(embeddings))
                    latest_action_time = now
                vit_ms = (time.perf_counter() - started) * 1000

            if analyse and frame_index % yolo_every == 0:
                started = time.perf_counter()
                detections = detector.predict(frame)
                yolo_ms = (time.perf_counter() - started) * 1000
                zone_detections = [
                    item
                    for item in detections
                    if camera_config.work_zone.contains(item.center, width, height)
                ]
                action_evidence = (
                    latest_action
                    if latest_action is not None
                    and now - latest_action_time <= args.action_ttl
                    and latest_action.confidence > action_threshold
                    else None
                )
                event = fusion.update(zone_detections, action_evidence, timestamp_s=now)
                tracker.set_observation_state(fusion.observation_state)
                if event is not None:
                    outcome = monitor.submit_stable_action(event.action)
                    print(f"[{outcome.type}] {event.action}: {outcome.message} | fusion={event.reason}")

            phone_view = phone.poll(phone_hit_test)
            if args.headless:
                if args.max_frames is not None and frame_index >= args.max_frames:
                    break
                continue
            zone_box = camera_config.work_zone.to_pixels(width, height)
            cv2.rectangle(frame, zone_box[:2], zone_box[2:], (0, 210, 255), 2)
            _draw_detections(frame, detections, camera_config,
                             temporal_config.confidence.display_min)
            for view in getattr(fusion, "slot_views", ()):
                if view.box:
                    cv2.rectangle(frame, view.box[:2], view.box[2:], (255, 210, 80), 1)
                    coverage = f"{view.coverage:.0%}" if view.coverage_available else "N/A"
                    # The reference box may be an anchor, not an empty detection.
                    slot_label = view.label.replace("empty_", "slot_", 1)
                    _put_text(frame, f"{slot_label} cover:{coverage}", (view.box[0], view.box[3]+18), 0.45)
            screen = draw_dashboard(
                frame, tracker=tracker, fusion=fusion, prediction=latest_action,
                prediction_fresh=latest_action is not None and now-latest_action_time <= args.action_ttl,
                action_threshold=action_threshold, outcome=outcome,
                recent_outcomes=tuple(recent_outcomes),
                embedding_count=len(embeddings), sequence_length=sequence_length,
                yolo_ms=yolo_ms, vit_ms=vit_ms, fps=display_fps,
                embedding_fps=telemetry.snapshot.actual_fps,
                window_duration_s=telemetry.snapshot.window_duration_s,
                mirror=not args.no_mirror and local_live, phone=phone_view,
            )
            cv2.imshow(project.display_name, screen)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            phone.handle_key(key)
            if key in (ord("f"), ord("F")):
                fullscreen = not fullscreen
                cv2.setWindowProperty(project.display_name, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                if not fullscreen:
                    cv2.resizeWindow(project.display_name, args.window_width, args.window_height)
            if key in (ord("r"), ord("R")):
                outcome = monitor.reset()
                fusion.reset()
                recent_outcomes.clear()
                embeddings.clear()
                cadence.reset()
                telemetry.reset()
                latest_action = None
                latest_action_time = 0.0
                print("[INFO] Đã reset FSM, Fusion Engine và temporal buffer.")
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
    finally:
        capture.release()
        phone.close()
        if not args.headless:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

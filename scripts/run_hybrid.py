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
from assembly.camera_config import load_camera_config
from assembly.camera_app import discover_camera_indices
from assembly.config import load_config
from assembly.fsm import ConfigurableAssemblyTracker, FsmOutcome
from assembly.hybrid_dashboard import draw_dashboard
from assembly.model_contract import Prediction
from assembly.models.vit_lstm_recognizer import ViTLstmActionRecognizer
from assembly.monitor import AssemblyMonitor, JsonlEventLogger
from assembly.project_config import build_fusion_engine, load_project_config
from assembly.smoother import TemporalDebouncer
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


def _draw_detections(frame, detections: list[Detection], camera_config) -> None:
    label_map = camera_config.label_map
    for detection in detections:
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
    parser.add_argument("--source", default="0", help="Camera index hoặc đường dẫn video")
    parser.add_argument("--list-cameras", action="store_true", help="Liệt kê camera đọc được, không nạp model")
    parser.add_argument("--max-camera-index", type=int, default=5, help="Chỉ số camera cao nhất cần dò")
    parser.add_argument("--camera-config", type=Path, default=None)
    parser.add_argument("--fsm-config", type=Path, default=None)
    parser.add_argument("--action-config", type=Path, default=None)
    parser.add_argument("--action-model", type=Path, default=None)
    parser.add_argument("--event-log", type=Path, default=None)
    parser.add_argument("--yolo-model", type=Path, default=None)
    parser.add_argument("--device", default=None, help="cpu, cuda hoặc GPU index")
    parser.add_argument("--sample-fps", type=float, default=None)
    parser.add_argument("--yolo-every", type=int, default=None)
    parser.add_argument("--stable-frames", type=int, default=None)
    parser.add_argument("--action-confidence", type=float, default=None)
    parser.add_argument("--action-ttl", type=float, default=1.5)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--window-width", type=int, default=1440)
    parser.add_argument("--window-height", type=int, default=900)
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--headless", action="store_true", help="Đọc video, ghi log, không mở cửa sổ")
    parser.add_argument("--max-frames", type=int, default=None, help=argparse.SUPPRESS)
    return parser


def main(default_project: Path = DEFAULT_PROJECT) -> int:
    _configure_utf8_console()
    parser = build_parser(default_project)
    args = parser.parse_args()

    if args.max_camera_index < 0:
        parser.error("--max-camera-index phải >= 0")
    if args.list_cameras:
        indices = discover_camera_indices(cv2, args.max_camera_index)
        print("Camera có thể đọc: " + (", ".join(map(str, indices)) if indices else "không tìm thấy"))
        print("Chọn đúng camera bằng --source N; số N không cố định theo thiết bị.")
        return 0

    if args.sample_fps is not None and args.sample_fps <= 0:
        parser.error("--sample-fps phải > 0")
    if args.yolo_every is not None and args.yolo_every < 1:
        parser.error("--yolo-every phải >= 1")
    if args.stable_frames is not None and args.stable_frames < 1:
        parser.error("--stable-frames phải >= 1")
    if args.action_ttl <= 0:
        parser.error("--action-ttl phải > 0")
    if args.window_width < 640 or args.window_height < 480:
        parser.error("Kích thước cửa sổ tối thiểu 640x480")
    if args.action_confidence is not None and not 0 <= args.action_confidence <= 1:
        parser.error("--action-confidence phải nằm trong [0, 1]")

    project = load_project_config(args.project, ROOT)
    camera_config_path = _input_path(args.camera_config or project.camera_config)
    fsm_config_path = _input_path(args.fsm_config or project.fsm_config)
    action_config_path = _input_path(args.action_config or project.action_config)
    action_model_path = _input_path(args.action_model or project.action_model)
    event_log_path = (args.event_log or project.event_log).resolve()
    for label, path in (
        ("camera config", camera_config_path),
        ("FSM config", fsm_config_path),
        ("action config", action_config_path),
        ("action checkpoint", action_model_path),
    ):
        if not path.is_file():
            raise SystemExit(f"Không tìm thấy {label}: {path}")

    camera_config = load_camera_config(camera_config_path)
    action_config = load_action_model_config(action_config_path)
    assembly_config = load_config(fsm_config_path)
    tracker = ConfigurableAssemblyTracker(assembly_config)
    monitor = AssemblyMonitor(
        tracker,
        TemporalDebouncer(idle_actions=assembly_config.idle_actions),
        JsonlEventLogger(event_log_path),
    )
    fusion = build_fusion_engine(
        project.fusion,
        stable_frames=args.stable_frames,
        min_action_confidence=(args.action_confidence if args.action_confidence is not None
                               else project.fusion.parameters.get(
                                   "min_action_confidence", action_config.inference.min_confidence)),
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
    # Keep the temporal cadence used during training, including on CPU.
    sample_fps = args.sample_fps or action_config.spatial.sample_fps
    # Do not silently reduce the observations used by geometry stability checks.
    yolo_every = args.yolo_every or camera_config.infer_every_n_frames
    if cpu_mode and isinstance(_source(args.source), int):
        print(
            "[WARN] Đang chạy YOLO và ViT bằng CPU; runtime dùng "
            f"sample_fps mục tiêu={sample_fps:g}, yolo_every={yolo_every}. "
            "Tốc độ thực tế có thể thấp hơn nếu CPU không xử lý kịp."
        )

    source = _source(args.source)
    backend = cv2.CAP_DSHOW if isinstance(source, int) and sys.platform == "win32" else cv2.CAP_ANY
    capture = cv2.VideoCapture(source, backend)
    if not capture.isOpened():
        raise SystemExit(f"Không mở được camera/video: {source!r}")
    if isinstance(source, int):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.capture_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.capture_height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, camera_config.capture_buffer_size)

    sequence_length = action_config.temporal.sequence_length
    embeddings = deque(maxlen=sequence_length)
    next_sample_time = time.perf_counter()
    sample_interval = 1.0 / sample_fps
    latest_action: Prediction | None = None
    latest_action_time = 0.0
    detections: list[Detection] = []
    outcome: FsmOutcome | None = None
    recent_outcomes: deque[FsmOutcome] = deque(maxlen=32)
    frame_index = 0
    yolo_ms = 0.0
    vit_ms = 0.0
    previous_frame_time = time.perf_counter()
    display_fps = 0.0
    fullscreen = args.fullscreen
    action_threshold = fusion.min_action_confidence
    file_fps = capture.get(cv2.CAP_PROP_FPS) if isinstance(source, str) else 0.0
    video_file = isinstance(source, str) and Path(source).is_file()
    # For replay, sample by video time rather than CPU processing speed.
    if video_file:
        sample_fps = args.sample_fps or action_config.spatial.sample_fps
        sample_interval = 1.0 / sample_fps
        next_sample_time = 0.0
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
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if not args.no_mirror and isinstance(source, int):
                frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            wall_now = time.perf_counter()
            frame_duration = max(wall_now - previous_frame_time, 1e-6)
            display_fps = 0.9 * display_fps + 0.1 / frame_duration
            previous_frame_time = wall_now
            now = ((frame_index - 1) / file_fps if file_fps > 0 else capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                   ) if video_file else wall_now

            if now + 1e-8 >= next_sample_time:
                started = time.perf_counter()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                embeddings.append(recognizer.encode_frame(rgb))
                if len(embeddings) == sequence_length:
                    latest_action = recognizer.predict_embeddings(list(embeddings))
                    latest_action_time = now
                vit_ms = (time.perf_counter() - started) * 1000
                next_sample_time = now + sample_interval

            if frame_index % yolo_every == 0:
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
                event = fusion.update(zone_detections, action_evidence)
                if event is not None:
                    outcome = monitor.submit_stable_action(event.action)
                    print(f"[{outcome.type}] {event.action}: {outcome.message} | fusion={event.reason}")
                    if outcome.type in {"PASS", "VIOLATION"}:
                        recent_outcomes.append(outcome)
                        # Invalidate the decision, not the rolling spatial context.
                        # Clearing 16 embeddings created blind gaps after each step.
                        latest_action = None
                        latest_action_time = 0.0

            if args.headless:
                if args.max_frames is not None and frame_index >= args.max_frames:
                    break
                continue
            zone_box = camera_config.work_zone.to_pixels(width, height)
            cv2.rectangle(frame, zone_box[:2], zone_box[2:], (0, 210, 255), 2)
            _draw_detections(frame, detections, camera_config)
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
                mirror=not args.no_mirror and isinstance(source, int),
            )
            cv2.imshow(project.display_name, screen)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
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
                latest_action = None
                latest_action_time = 0.0
                print("[INFO] Đã reset FSM, Fusion Engine và temporal buffer.")
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
    finally:
        capture.release()
        if not args.headless:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

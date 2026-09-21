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
from assembly.config import load_config
from assembly.fsm import ConfigurableAssemblyTracker, FsmOutcome
from assembly.model_contract import Prediction
from assembly.models.vit_lstm_recognizer import ViTLstmActionRecognizer
from assembly.monitor import AssemblyMonitor, JsonlEventLogger
from assembly.project_config import build_fusion_engine, load_project_config
from assembly.smoother import TemporalDebouncer
from assembly.vision import Detection
from assembly.yolo_world_detector import YoloWorldDetector


DEFAULT_PROJECT = ROOT / "configs" / "projects" / "earbud.json"


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
    parser.add_argument("--max-frames", type=int, default=None, help=argparse.SUPPRESS)
    return parser


def main(default_project: Path = DEFAULT_PROJECT) -> int:
    _configure_utf8_console()
    parser = build_parser(default_project)
    args = parser.parse_args()

    if args.sample_fps is not None and args.sample_fps <= 0:
        parser.error("--sample-fps phải > 0")
    if args.yolo_every is not None and args.yolo_every < 1:
        parser.error("--yolo-every phải >= 1")
    if args.stable_frames is not None and args.stable_frames < 1:
        parser.error("--stable-frames phải >= 1")
    if args.action_ttl <= 0:
        parser.error("--action-ttl phải > 0")

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
        min_action_confidence=args.action_confidence,
    )

    detector = YoloWorldDetector(
        camera_config,
        model_path=_resolve_detector_model(camera_config.model, args.yolo_model),
        device=args.device,
    )
    recognizer = ViTLstmActionRecognizer(
        config_path=action_config_path,
        checkpoint_path=action_model_path,
        device=_torch_device(args.device),
        local_files_only=not args.allow_download,
    )
    cpu_mode = recognizer.device.type == "cpu"
    sample_fps = args.sample_fps or (4.0 if cpu_mode else action_config.spatial.sample_fps)
    yolo_every = args.yolo_every or (3 if cpu_mode else camera_config.infer_every_n_frames)
    if cpu_mode:
        print(
            "[WARN] Đang chạy YOLO và ViT bằng CPU; runtime dùng "
            f"sample_fps={sample_fps:g}, yolo_every={yolo_every} để giảm lag."
        )

    source = _source(args.source)
    backend = cv2.CAP_DSHOW if isinstance(source, int) and sys.platform == "win32" else cv2.CAP_ANY
    capture = cv2.VideoCapture(source, backend)
    if not capture.isOpened():
        raise SystemExit(f"Không mở được camera/video: {source!r}")
    if isinstance(source, int):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    sequence_length = action_config.temporal.sequence_length
    embeddings = deque(maxlen=sequence_length)
    next_sample_time = time.perf_counter()
    sample_interval = 1.0 / sample_fps
    latest_action: Prediction | None = None
    latest_action_time = 0.0
    detections: list[Detection] = []
    outcome: FsmOutcome | None = None
    frame_index = 0
    yolo_ms = 0.0
    vit_ms = 0.0
    print(
        f"[INFO] Project={project.name} | device={recognizer.device} | "
        f"sample_fps={sample_fps:g} | YOLO mỗi {yolo_every} frame."
    )

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if not args.no_mirror and isinstance(source, int):
                frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            now = time.perf_counter()

            if now >= next_sample_time:
                started = time.perf_counter()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                embeddings.append(recognizer.encode_frame(rgb))
                if len(embeddings) == sequence_length:
                    latest_action = recognizer.predict_embeddings(list(embeddings))
                    latest_action_time = time.perf_counter()
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
                    and time.perf_counter() - latest_action_time <= args.action_ttl
                    else None
                )
                event = fusion.update(zone_detections, action_evidence)
                if event is not None:
                    outcome = monitor.submit_stable_action(event.action)
                    print(f"[{outcome.type}] {event.action}: {outcome.message} | fusion={event.reason}")

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 146), (18, 24, 38), -1)
            cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)
            zone_box = camera_config.work_zone.to_pixels(width, height)
            cv2.rectangle(frame, zone_box[:2], zone_box[2:], (0, 210, 255), 2)
            _draw_detections(frame, detections, camera_config)

            action_text = (
                f"{latest_action.action} {latest_action.confidence:.2f}"
                if latest_action is not None
                else f"warming up {len(embeddings)}/{sequence_length}"
            )
            _put_text(
                frame,
                f"FSM: {monitor.tracker.state} | Expected: {', '.join(monitor.tracker.expected_actions)}",
                (16, 28),
                0.58,
            )
            _put_text(frame, f"Action: {action_text}", (16, 57), 0.55, (100, 255, 100))
            _put_text(frame, f"Fusion: {fusion.status_text}", (16, 86), 0.55, (0, 210, 255))
            result_text = f"{outcome.type}: {outcome.action}" if outcome is not None else "No fused event"
            result_color = (80, 230, 100) if outcome and outcome.type == "PASS" else (190, 210, 255)
            _put_text(frame, f"Result: {result_text}", (16, 115), 0.55, result_color)
            _put_text(
                frame,
                f"YOLO {yolo_ms:.0f}ms | ViT+LSTM {vit_ms:.0f}ms | R reset | Q quit",
                (16, 140),
                0.45,
                (170, 180, 190),
            )

            cv2.imshow(project.display_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("r"), ord("R")):
                outcome = monitor.reset()
                fusion.reset()
                embeddings.clear()
                latest_action = None
                latest_action_time = 0.0
                print("[INFO] Đã reset FSM, Fusion Engine và temporal buffer.")
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

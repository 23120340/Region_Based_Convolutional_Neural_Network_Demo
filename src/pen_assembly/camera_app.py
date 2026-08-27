from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .camera_config import CameraConfig, load_camera_config
from .config import load_config
from .fsm import FsmOutcome
from .monitor import AssemblyMonitor, JsonlEventLogger
from .paths import DEFAULT_CONFIG, DEFAULT_EVENT_LOG, PROJECT_ROOT
from .smoother import TemporalDebouncer
from .vision import ComponentDwellGate, ComponentSuggestion, Detection, NormalizedZone
from .yolo_world_detector import YoloWorldDetector
from .fsm import ConfigurableAssemblyTracker


DEFAULT_CAMERA_CONFIG = PROJECT_ROOT / "configs" / "camera_config.json"
ACTION_KEYS = {
    ord("1"): "pick_barrel",
    ord("2"): "insert_refill",
    ord("3"): "insert_spring",
    ord("4"): "screw_cap",
    ord("5"): "test_click",
}


def _camera_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def _configure_utf8_console() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _put_text(cv2, frame, text: str, origin: tuple[int, int], scale: float = 0.58, color=(255, 255, 255)) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def _draw_detection(cv2, frame, detection: Detection, config: CameraConfig) -> None:
    vision_class = next(item for item in config.classes if item.label == detection.label)
    x1, y1, x2, y2 = detection.box_xyxy
    color = vision_class.color_bgr
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    caption = f"{detection.label} {detection.confidence:.2f}"
    cv2.rectangle(frame, (x1, max(0, y1 - 25)), (x1 + 150, y1), color, -1)
    _put_text(cv2, frame, caption, (x1 + 4, y1 - 6), 0.48, (20, 20, 20))


def _open_capture(cv2, source: int | str):
    if isinstance(source, int) and sys.platform == "win32":
        capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        capture = cv2.VideoCapture(source)
    if isinstance(source, int):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not capture.isOpened():
        raise RuntimeError(f"Không mở được camera/video source {source!r}")
    return capture


def _build_monitor():
    assembly_config = load_config(DEFAULT_CONFIG)
    tracker = ConfigurableAssemblyTracker(assembly_config)
    monitor = AssemblyMonitor(
        tracker,
        TemporalDebouncer(idle_actions=assembly_config.idle_actions),
        JsonlEventLogger(DEFAULT_EVENT_LOG),
    )
    return assembly_config, monitor


def run_camera(
    source: int | str,
    camera_config_path: str | Path = DEFAULT_CAMERA_CONFIG,
    model_path: str | None = None,
    device: str | None = None,
    mirror: bool = True,
    auto_advance: bool = False,
    max_frames: int | None = None,
) -> None:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "Thiếu OpenCV. Chạy: python -m pip install -r requirements-camera.txt"
        ) from error

    camera_config = load_camera_config(camera_config_path)
    assembly_config, monitor = _build_monitor()
    unknown_actions = set(camera_config.action_map.values()) - set(assembly_config.actions)
    if unknown_actions:
        raise ValueError(f"camera_config dùng action không tồn tại: {sorted(unknown_actions)}")
    detector = YoloWorldDetector(camera_config, model_path=model_path, device=device)
    gate = ComponentDwellGate(camera_config.action_map, camera_config.dwell_frames)
    capture = _open_capture(cv2, source)

    detections: list[Detection] = []
    suggestion: ComponentSuggestion | None = None
    outcome: FsmOutcome | None = None
    frame_index = 0
    last_inference_ms = 0.0
    screenshot_dir = PROJECT_ROOT / "artifacts" / "screenshots"

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            frame_index += 1

            if frame_index % camera_config.infer_every_n_frames == 0:
                started = time.perf_counter()
                detections = detector.predict(frame)
                last_inference_ms = (time.perf_counter() - started) * 1000
                emitted = gate.update(
                    detections,
                    monitor.tracker.expected_actions,
                    camera_config.work_zone,
                    width,
                    height,
                )
                if emitted is not None:
                    if auto_advance:
                        outcome = monitor.submit_stable_action(emitted.action)
                        print(f"[{outcome.type}] {outcome.action}: {outcome.message}")
                        suggestion = None
                        gate.reset()
                suggestion = gate.current_suggestion

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 138), (18, 24, 38), -1)
            cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
            zone_box = camera_config.work_zone.to_pixels(width, height)
            cv2.rectangle(frame, zone_box[:2], zone_box[2:], (0, 210, 255), 2)
            _put_text(cv2, frame, "WORK ZONE", (zone_box[0] + 8, zone_box[1] + 24), 0.55, (0, 210, 255))

            for detection in detections:
                _draw_detection(cv2, frame, detection, camera_config)

            expected = ", ".join(monitor.tracker.expected_actions) or "none"
            mode = "AUTO (experimental)" if auto_advance else "ASSISTED"
            _put_text(cv2, frame, f"Pen Assembly Camera | {mode} | inference {last_inference_ms:.0f} ms", (18, 28), 0.62)
            _put_text(cv2, frame, f"State: {monitor.tracker.state} | Expected: {expected}", (18, 58), 0.60)
            if suggestion is not None:
                _put_text(cv2, frame, f"Suggestion: {suggestion.action} - press SPACE to confirm", (18, 88), 0.62, (0, 255, 255))
            elif outcome is not None:
                color = (80, 230, 100) if outcome.type == "PASS" else (70, 70, 255)
                _put_text(cv2, frame, f"{outcome.type}: {outcome.action}", (18, 88), 0.62, color)
            else:
                _put_text(cv2, frame, "Move the expected component into WORK ZONE", (18, 88), 0.60, (190, 210, 255))
            _put_text(cv2, frame, "SPACE confirm | 1-5 manual steps | R reset | S screenshot | Q quit", (18, 120), 0.52, (205, 213, 224))

            cv2.imshow("Pen Assembly - Real-time Camera", frame)
            key = cv2.waitKey(1) & 0xFF
            if max_frames is not None and frame_index >= max_frames:
                break
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                outcome = monitor.reset()
                suggestion = None
                gate.reset()
            elif key == ord("s"):
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                path = screenshot_dir / f"camera_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(path), frame)
                print(f"Saved screenshot: {path}")
            elif key == 32 and suggestion is not None:
                outcome = monitor.submit_stable_action(suggestion.action)
                print(f"[{outcome.type}] {outcome.action}: {outcome.message}")
                suggestion = None
                gate.reset()
            elif key in ACTION_KEYS:
                outcome = monitor.submit_stable_action(ACTION_KEYS[key])
                print(f"[{outcome.type}] {outcome.action}: {outcome.message}")
                suggestion = None
                gate.reset()
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> int:
    _configure_utf8_console()
    parser = argparse.ArgumentParser(description="Zero-shot pen component detection from camera")
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--camera-config", default=str(DEFAULT_CAMERA_CONFIG))
    parser.add_argument("--model", default=None, help="Override model checkpoint")
    parser.add_argument("--device", default=None, help="cpu, 0, 1, ...; default is automatic")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--auto-advance",
        action="store_true",
        help="Experimental: accept a component dwell without pressing Space",
    )
    args = parser.parse_args()
    run_camera(
        source=_camera_source(args.source),
        camera_config_path=args.camera_config,
        model_path=args.model,
        device=args.device,
        mirror=not args.no_mirror,
        auto_advance=args.auto_advance,
        max_frames=args.max_frames,
    )
    return 0

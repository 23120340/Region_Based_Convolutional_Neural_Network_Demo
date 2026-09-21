from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from .camera_config import CameraConfig, load_camera_config
from .config import load_config
from .fsm import FsmOutcome
from .monitor import AssemblyMonitor, JsonlEventLogger
from .paths import DEFAULT_CONFIG, DEFAULT_EVENT_LOG, PROJECT_ROOT
from .smoother import TemporalDebouncer
from .vision import ComponentDwellGate, Detection, EarbudAssemblyGate
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


def _open_capture(cv2, source: int | str, config: CameraConfig):
    if isinstance(source, int) and sys.platform == "win32":
        capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        capture = cv2.VideoCapture(source)
    if isinstance(source, int):
        # These properties are best-effort: unsupported camera backends simply
        # ignore them. A one-frame buffer prevents processing stale frames.
        if sys.platform == "win32":
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_BUFFERSIZE, config.capture_buffer_size)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.capture_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.capture_height)
        capture.set(cv2.CAP_PROP_FPS, config.capture_fps)
    if not capture.isOpened():
        raise RuntimeError(f"Không mở được camera/video source {source!r}")
    return capture


def discover_camera_indices(cv2, max_camera_index: int = 5) -> list[int]:
    """Return camera indices that can provide at least one frame.

    Cameras are probed only on startup/list requests or when the user presses C,
    not during every inference frame.
    """

    if max_camera_index < 0:
        raise ValueError("max_camera_index phải >= 0")
    available: list[int] = []
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    logging_api = getattr(getattr(cv2, "utils", None), "logging", None)
    get_log_level = getattr(cv2, "getLogLevel", None) or getattr(logging_api, "getLogLevel", None)
    set_log_level = getattr(cv2, "setLogLevel", None) or getattr(logging_api, "setLogLevel", None)
    silent_level = getattr(logging_api, "LOG_LEVEL_SILENT", 0)
    previous_log_level = get_log_level() if get_log_level is not None else None
    try:
        if set_log_level is not None:
            set_log_level(silent_level)
        for index in range(max_camera_index + 1):
            capture = cv2.VideoCapture(index, backend)
            try:
                if capture.isOpened():
                    ok, _ = capture.read()
                    if ok:
                        available.append(index)
            finally:
                capture.release()
    finally:
        if set_log_level is not None and previous_log_level is not None:
            set_log_level(previous_log_level)
    return available


def next_camera_index(current: int, available: list[int]) -> int:
    """Select the next available index, wrapping around deterministically."""

    ordered = sorted(set(available))
    if not ordered:
        return current
    if current not in ordered:
        return ordered[0]
    return ordered[(ordered.index(current) + 1) % len(ordered)]


def _build_monitor(fsm_config_path: str | Path = DEFAULT_CONFIG):
    assembly_config = load_config(fsm_config_path)
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
    fsm_config_path: str | Path = DEFAULT_CONFIG,
    model_path: str | None = None,
    device: str | None = None,
    mirror: bool = True,
    auto_advance: bool = False,
    max_frames: int | None = None,
    max_camera_index: int = 5,
) -> None:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "Thiếu OpenCV. Chạy: python -m pip install -r requirements-camera.txt"
        ) from error

    camera_config = load_camera_config(camera_config_path)
    assembly_config, monitor = _build_monitor(fsm_config_path)
    action_keys: dict[int, str] = {
        ord(str(i + 1)): step.action
        for i, step in enumerate(assembly_config.workflow)
        if i < 9
    } or dict(ACTION_KEYS)
    unknown_actions = set(camera_config.action_map.values()) - set(assembly_config.actions)
    if unknown_actions:
        raise ValueError(f"camera_config dùng action không tồn tại: {sorted(unknown_actions)}")

    capture = _open_capture(cv2, source, camera_config)
    detector = YoloWorldDetector(camera_config, model_path=model_path, device=device)
    actual_width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = capture.get(cv2.CAP_PROP_FPS)
    precision = "FP16" if detector.use_half else "FP32"
    print(
        f"[PERF] Camera {actual_width}x{actual_height} @ {actual_fps:.1f} FPS | "
        f"YOLO {camera_config.image_size}px, {precision}, device={detector.device}"
    )
    geometry = camera_config.earbud_geometry
    if geometry is None:
        gate = ComponentDwellGate(camera_config.action_map, camera_config.dwell_frames)
    else:
        gate = EarbudAssemblyGate(
            open_case_label=geometry.open_case_label,
            closed_case_label=geometry.closed_case_label,
            earbud_labels=geometry.earbud_labels,
            empty_slot_labels=geometry.empty_slot_labels,
            dwell_frames=camera_config.dwell_frames,
            containment_threshold=geometry.containment_threshold,
        )
    active_source = source
    local_video_wait_ms = 1
    if isinstance(active_source, str) and Path(active_source).is_file():
        video_fps = capture.get(cv2.CAP_PROP_FPS)
        if video_fps > 0:
            local_video_wait_ms = max(1, round(1000 / video_fps))
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo-inference")
    inference_future: Future[tuple[int, list[Detection], float]] | None = None
    source_generation = 0

    frame_index = 0
    detections: list[Detection] = []
    last_inference_ms = 0.0
    display_fps = 0.0
    fps_started = time.perf_counter()
    fps_frames = 0
    frames_since_submit = camera_config.infer_every_n_frames
    last_geometry_message: str | None = None
    suggestion = None
    outcome: FsmOutcome | None = None
    window_title = f"{assembly_config.project} - Real-time Camera"

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if mirror:
                frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            # Collect a completed prediction without ever blocking the UI loop.
            # Only one job can exist, which bounds memory and naturally drops
            # intermediate frames when inference is slower than the camera.
            if inference_future is not None and inference_future.done():
                result_generation, result_detections, elapsed_ms = inference_future.result()
                inference_future = None
                if result_generation == source_generation:
                    detections = result_detections
                    last_inference_ms = elapsed_ms
                    emitted = gate.update(
                        detections,
                        monitor.tracker.expected_actions,
                        camera_config.work_zone,
                        width,
                        height,
                    )
                    if isinstance(gate, EarbudAssemblyGate):
                        geometry_message = gate.status.message
                        if geometry_message != last_geometry_message:
                            print(f"[GEOMETRY] {geometry_message}")
                            last_geometry_message = geometry_message
                    if emitted is not None and auto_advance:
                        outcome = monitor.submit_stable_action(emitted.action)
                        print(f"[{outcome.type}] {outcome.action}: {outcome.message}")
                        suggestion = None
                        gate.acknowledge(rearm=outcome.type == "VIOLATION")
                    suggestion = gate.current_suggestion

            frames_since_submit += 1
            if (
                inference_future is None
                and frames_since_submit >= camera_config.infer_every_n_frames
            ):
                submitted_frame = frame.copy()
                submitted_generation = source_generation

                def infer() -> tuple[int, list[Detection], float]:
                    started = time.perf_counter()
                    result = detector.predict(submitted_frame)
                    return submitted_generation, result, (time.perf_counter() - started) * 1000

                inference_future = executor.submit(infer)
                frames_since_submit = 0

            fps_frames += 1
            now = time.perf_counter()
            if now - fps_started >= 0.5:
                display_fps = fps_frames / (now - fps_started)
                fps_started = now
                fps_frames = 0

            overlay = frame.copy()
            panel_height = 132 if geometry is not None else 105
            cv2.rectangle(overlay, (0, 0), (width, panel_height), (18, 24, 38), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            zone_box = camera_config.work_zone.to_pixels(width, height)
            cv2.rectangle(frame, zone_box[:2], zone_box[2:], (0, 210, 255), 2)
            _put_text(cv2, frame, "WORK ZONE", (zone_box[0] + 8, zone_box[1] + 24), 0.55, (0, 210, 255))

            for detection in detections:
                _draw_detection(cv2, frame, detection, camera_config)

            expected = ", ".join(monitor.tracker.expected_actions) or "none"
            _put_text(cv2, frame, f"State: {monitor.tracker.state}  |  Expected: {expected}", (18, 32), 0.70, (255, 255, 255))

            if suggestion is not None:
                prefix = "Expected" if suggestion.is_expected else "UNEXPECTED"
                color = (0, 255, 255) if suggestion.is_expected else (70, 70, 255)
                _put_text(cv2, frame, f"> {prefix}: {suggestion.action} (Press SPACE to confirm)", (18, 64), 0.65, color)
            elif outcome is not None:
                color = (80, 230, 100) if outcome.type == "PASS" else (70, 70, 255)
                _put_text(cv2, frame, f"> {outcome.type}: {outcome.action}", (18, 64), 0.65, color)
            else:
                _put_text(cv2, frame, "> Move expected item into WORK ZONE", (18, 64), 0.55, (190, 210, 255))

            step_hint = f"1-{len(action_keys)}" if action_keys else "no"
            infer_fps = 1000.0 / last_inference_ms if last_inference_ms > 0 else 0.0
            _put_text(cv2, frame, f"Keys: SPACE, {step_hint}, R, C, Q  |  Display: {display_fps:.1f} FPS  |  AI: {infer_fps:.1f} FPS ({last_inference_ms:.0f}ms)", (18, 92), 0.48, (170, 180, 190))
            if isinstance(gate, EarbudAssemblyGate):
                status = gate.status
                status_color = (70, 70, 255) if status.is_error else (90, 230, 130)
                _put_text(
                    cv2,
                    frame,
                    f"Geometry: case={status.case_state} | inside={status.earbuds_inside}/2 | empty={status.empty_slots}/2 | {'ERROR' if status.is_error else 'OK'}",
                    (18, 119),
                    0.46,
                    status_color,
                )

            cv2.imshow(window_title, frame)
            # Local video files otherwise run as fast as decoding allows after
            # inference is moved off the display thread.
            key = cv2.waitKey(local_video_wait_ms) & 0xFF
            if max_frames is not None and frame_index >= max_frames:
                break
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                outcome = monitor.reset()
                suggestion = None
                gate.reset()
            elif key in (ord("c"), ord("C")):
                if not isinstance(active_source, int):
                    print("Không thể chuyển camera khi --source là đường dẫn video/stream.")
                    continue
                previous_source = active_source
                capture.release()
                available = discover_camera_indices(cv2, max_camera_index)
                target_source = next_camera_index(previous_source, available)
                try:
                    capture = _open_capture(cv2, target_source, camera_config)
                except RuntimeError as error:
                    print(f"Không chuyển được camera: {error}")
                    capture = _open_capture(cv2, previous_source, camera_config)
                else:
                    active_source = target_source
                    source_generation += 1
                    last_geometry_message = None
                    detections = []
                    suggestion = None
                    outcome = None
                    gate.reset()
                    if target_source == previous_source:
                        print(f"Chỉ tìm thấy camera {active_source}; tiếp tục sử dụng camera hiện tại.")
                    else:
                        print(f"Đã chuyển camera {previous_source} -> {active_source}.")
            elif key == ord("s"):
                screenshot_dir = PROJECT_ROOT / "artifacts" / "screenshots"
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                path = screenshot_dir / f"camera_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(path), frame)
                print(f"Saved screenshot: {path}")
            elif key == 32 and suggestion is not None:
                outcome = monitor.submit_stable_action(suggestion.action)
                print(f"[{outcome.type}] {outcome.action}: {outcome.message}")
                suggestion = None
                gate.acknowledge(rearm=outcome.type == "VIOLATION")
            elif key in action_keys:
                outcome = monitor.submit_stable_action(action_keys[key])
                print(f"[{outcome.type}] {outcome.action}: {outcome.message}")
                suggestion = None
                gate.acknowledge()
    finally:
        capture.release()
        cv2.destroyAllWindows()
        executor.shutdown(wait=True, cancel_futures=True)


def main() -> int:
    _configure_utf8_console()
    parser = argparse.ArgumentParser(description="Zero-shot pen component detection from camera")
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--camera-config", default=str(DEFAULT_CAMERA_CONFIG))
    parser.add_argument("--fsm-config", default=str(DEFAULT_CONFIG), help="FSM workflow configuration path")
    parser.add_argument("--model", default=None, help="Override model checkpoint")
    parser.add_argument("--device", default=None, help="cpu, 0, 1, ...; default is automatic")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--list-cameras", action="store_true", help="List available camera indices and exit")
    parser.add_argument("--max-camera-index", type=int, default=5, help="Highest camera index to probe")
    parser.add_argument("--max-frames", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--auto-advance",
        action="store_true",
        help="Experimental: accept a component dwell without pressing Space",
    )
    args = parser.parse_args()
    if args.list_cameras:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError(
                "Thiếu OpenCV. Chạy: python -m pip install -r requirements-camera.txt"
            ) from error
        available = discover_camera_indices(cv2, args.max_camera_index)
        if available:
            print("Camera khả dụng: " + ", ".join(str(index) for index in available))
        else:
            print("Không tìm thấy camera khả dụng.")
        return 0
    run_camera(
        source=_camera_source(args.source),
        camera_config_path=args.camera_config,
        fsm_config_path=args.fsm_config,
        model_path=args.model,
        device=args.device,
        mirror=not args.no_mirror,
        auto_advance=args.auto_advance,
        max_frames=args.max_frames,
        max_camera_index=args.max_camera_index,
    )
    return 0

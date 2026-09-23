"""Run the earbud FSM simulator, live camera detector or image inference.

Run from the repository root, for example:
  python scripts/run_earbud.py --mode gui
  python scripts/run_earbud.py --mode camera --source 0 --auto-advance
  python scripts/run_earbud.py --list-cameras
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_FSM_CONFIG = ROOT / "configs" / "earbud_fsm_config.json"
DEFAULT_CAMERA_CONFIG = ROOT / "configs" / "camera_earbud_config.json"


def _utf8_console() -> None:
    for name in ("stdout", "stderr"):
        reconfigure = getattr(getattr(sys, name), "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")


def _input_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path.resolve()
    return ROOT / path


def _resolve_model(camera_config: Path, override: str | None) -> Path:
    from assembly.camera_config import load_camera_config

    selected = override or load_camera_config(camera_config).model
    return _input_path(selected)


def _check_model(model_path: Path) -> None:
    if not model_path.is_file():
        raise SystemExit(
            f"Không tìm thấy checkpoint: {model_path}\n"
            "Xem docs/VIEC_BAN_CAN_LAM_GEOMETRY.md để train đúng bộ nhãn. "
            "Dùng --model nếu checkpoint được lưu ở thư mục khác."
        )


def run_gui(fsm_config: Path) -> None:
    try:
        import tkinter as tk
    except ImportError as error:
        raise SystemExit("Python này chưa có tkinter.") from error
    from assembly.config import load_config
    from assembly.gui import AssemblyApp

    root = tk.Tk()
    AssemblyApp(root, load_config(fsm_config))
    root.mainloop()


def run_camera_mode(
    source: int | str,
    model_path: Path,
    camera_config: Path,
    fsm_config: Path,
    auto_advance: bool,
    device: str | None,
    mirror: bool,
    max_frames: int | None = None,
) -> None:
    _check_model(model_path)
    from assembly.camera_app import run_camera

    print(f"[CAM] Model: {model_path}")
    print("[CAM] SPACE: xác nhận | 1-9: mô phỏng bước FSM | R: reset")
    print("[CAM] C: đổi camera | S: chụp overlay | Q/ESC: thoát")
    print("[CAM] Phím số/SPACE là thao tác tay, không dùng để đo độ chính xác.")
    run_camera(
        source=source,
        camera_config_path=camera_config,
        fsm_config_path=fsm_config,
        model_path=str(model_path),
        device=device,
        mirror=mirror,
        auto_advance=auto_advance,
        max_frames=max_frames,
    )


def run_infer(source: str, model_path: Path, conf: float, device: str | None) -> None:
    _check_model(model_path)
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("Hãy cài requirements-camera.txt trước.") from error

    model = YOLO(str(model_path))
    results = model.predict(
        source=source,
        conf=conf,
        device=device,
        save=True,
        project=str(ROOT / "artifacts"),
        name="infer_earbud",
        exist_ok=True,
        verbose=True,
    )
    total_boxes = sum(len(result.boxes) for result in results if result.boxes is not None)
    print(f"[INFER] {len(results)} ảnh/frame, {total_boxes} bounding boxes.")
    print(f"[INFER] Kết quả: {ROOT / 'artifacts' / 'infer_earbud'}")


def list_cameras() -> None:
    try:
        import cv2
    except ImportError as error:
        raise SystemExit("Hãy cài requirements-camera.txt trước.") from error
    from assembly.camera_app import discover_camera_indices

    cameras = discover_camera_indices(cv2)
    print("Camera khả dụng: " + ", ".join(map(str, cameras)) if cameras else "Không tìm thấy camera.")


def _parse_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def main() -> int:
    _utf8_console()
    parser = argparse.ArgumentParser(description="Giám sát lắp ráp tai nghe")
    parser.add_argument("--mode", choices=["gui", "camera", "infer"], default="gui")
    parser.add_argument("--source", default="0", help="Camera index hoặc đường dẫn ảnh/video")
    parser.add_argument("--model", default=None, help="Checkpoint YOLO; mặc định đọc từ --camera-config")
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument("--fsm-config", type=Path, default=DEFAULT_FSM_CONFIG)
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence cho mode infer")
    parser.add_argument("--device", default=None, help="cpu hoặc 0/1; mặc định tự chọn")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--auto-advance", action="store_true")
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.list_cameras:
        list_cameras()
        return 0
    camera_config = _input_path(args.camera_config)
    fsm_config = _input_path(args.fsm_config)
    if args.mode == "gui":
        run_gui(fsm_config)
        return 0

    model_path = _resolve_model(camera_config, args.model)
    if args.mode == "camera":
        run_camera_mode(
            _parse_source(args.source), model_path, camera_config, fsm_config,
            args.auto_advance, args.device, not args.no_mirror, args.max_frames,
        )
    else:
        run_infer(args.source, model_path, args.conf, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

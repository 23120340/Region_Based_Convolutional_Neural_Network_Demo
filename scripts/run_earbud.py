#!/usr/bin/env python3
"""
demo_earbud.py  â€”  Demo há»‡ thá»‘ng giÃ¡m sÃ¡t láº¯p rÃ¡p earbud
â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
CÃ³ 3 cháº¿ Ä‘á»™ cháº¡y:

  1. GUI (máº·c Ä‘á»‹nh)  â€” mÃ´ phá»ng quy trÃ¬nh láº¯p rÃ¡p qua giao diá»‡n tkinter,
                       khÃ´ng cáº§n camera, khÃ´ng cáº§n model.

  2. Camera live     â€” má»Ÿ webcam, cháº¡y model YOLO Ä‘Ã£ finetune Ä‘á»ƒ nháº­n diá»‡n
                       cÃ¡c linh kiá»‡n earbud theo thá»i gian thá»±c.

  3. Infer áº£nh      â€” cháº¡y inference trÃªn 1 áº£nh / thÆ° má»¥c áº£nh rá»“i lÆ°u káº¿t quáº£.

VÃ­ dá»¥:
  python scripts/demo_earbud.py                              # GUI
  python scripts/demo_earbud.py --mode camera                # webcam 0
  python scripts/demo_earbud.py --mode camera --source 1    # webcam 1
  python scripts/demo_earbud.py --mode infer --source <áº£nh/thÆ° má»¥c>
  python scripts/demo_earbud.py --list-cameras               # liá»‡t kÃª webcam
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# â”€â”€â”€ root / src â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# â”€â”€â”€ default paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DEFAULT_FSM_CONFIG    = ROOT / "configs" / "earbud_fsm_config.json"
DEFAULT_CAMERA_CONFIG = ROOT / "configs" / "camera_earbud_config.json"
DEFAULT_MODEL         = ROOT / "artifacts" / "training" / "earbud_merged_detector" / "weights" / "best.pt"

# Fallback náº¿u model má»›i chÆ°a cÃ³
if not DEFAULT_MODEL.exists():
    DEFAULT_MODEL = ROOT / "artifacts" / "training" / "earbud_detector" / "weights" / "best.pt"


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Helpers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _utf8_console() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name)
        fn = getattr(s, "reconfigure", None)
        if fn:
            fn(encoding="utf-8", errors="replace")


def _check_model(model_path: Path) -> None:
    if not model_path.exists():
        print(f"[WARN] KhÃ´ng tÃ¬m tháº¥y checkpoint: {model_path}")
        print("       Cháº¡y script merge_and_finetune_earbud.py trÆ°á»›c Ä‘á»ƒ train model.")
    else:
        size_mb = model_path.stat().st_size / 1024 / 1024
        print(f"[OK]  Model: {model_path}  ({size_mb:.1f} MB)")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Mode 1 â€” GUI simulation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_gui(fsm_config: Path) -> None:
    """Má»Ÿ giao diá»‡n tkinter mÃ´ phá»ng quy trÃ¬nh láº¯p earbud (khÃ´ng cáº§n camera)."""
    print(f"[GUI]  FSM config : {fsm_config}")
    print("[GUI]  Khá»Ÿi Ä‘á»™ng giao diá»‡n mÃ´ phá»ng...")

    try:
        import tkinter as tk  # noqa: F401
    except ImportError:
        raise SystemExit("[ERR] tkinter khÃ´ng kháº£ dá»¥ng trong Python nÃ y.")

    from assembly.config import load_config
    from assembly.fsm import ConfigurableAssemblyTracker, FsmOutcome
    from assembly.monitor import AssemblyMonitor, JsonlEventLogger
    from assembly.paths import DEFAULT_EVENT_LOG
    from assembly.smoother import TemporalDebouncer

    # Patch gui Ä‘á»ƒ dÃ¹ng earbud FSM config thay vÃ¬ pen config
    import assembly.paths as _paths_mod
    _paths_mod.DEFAULT_CONFIG = fsm_config  # type: ignore[attr-defined]

    # Cháº¡y app
    import tkinter as tk
    from assembly.config import load_config as _lc
    from assembly.gui import PenAssemblyApp

    config = _lc(fsm_config)
    root = tk.Tk()
    PenAssemblyApp(root, config)
    root.mainloop()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Mode 2 â€” Camera live
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
    try:
        import cv2  # noqa: F401
    except ImportError:
        raise SystemExit(
            "[ERR] OpenCV chÆ°a cÃ i. Cháº¡y:\n"
            "      pip install -r requirements-camera.txt"
        )
    try:
        from ultralytics import YOLO  # noqa: F401
    except ImportError:
        raise SystemExit(
            "[ERR] ultralytics chÆ°a cÃ i. Cháº¡y:\n"
            "      pip install ultralytics"
        )

    _check_model(model_path)

    print(f"[CAM]  Source       : {source}")
    print(f"[CAM]  Model        : {model_path}")
    print(f"[CAM]  Camera cfg   : {camera_config}")
    print(f"[CAM]  FSM cfg      : {fsm_config}")
    print(f"[CAM]  Auto-advance : {auto_advance}")
    print()
    print("PhÃ­m táº¯t trong cá»­a sá»• camera:")
    print("  SPACE  â€” xÃ¡c nháº­n hÃ nh Ä‘á»™ng hiá»‡n táº¡i")
    print("  1-4    — nhập tay (open_case / tai 1 / tai 2 / close_case)")
    print("  R      â€” Ä‘áº·t láº¡i chu trÃ¬nh")
    print("  S      â€” chá»¥p mÃ n hÃ¬nh")
    print("  C      â€” chuyá»ƒn camera")
    print("  Q/ESC  â€” thoÃ¡t")
    print()

    from assembly.camera_app import run_camera
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


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Mode 3 â€” Static inference on image(s)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_infer(source: str, model_path: Path, conf: float, device: str | None) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "[ERR] ultralytics chÆ°a cÃ i. Cháº¡y:\n"
            "      pip install ultralytics"
        )

    _check_model(model_path)
    out_dir = ROOT / "artifacts" / "infer_earbud"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INF]  Source  : {source}")
    print(f"[INF]  Model   : {model_path}")
    print(f"[INF]  Conf    : {conf}")
    print(f"[INF]  Out dir : {out_dir}")
    print()

    model = YOLO(str(model_path))
    results = model.predict(
        source=source,
        conf=conf,
        device=device or "cpu",
        save=True,
        project=str(out_dir.parent),
        name=out_dir.name,
        exist_ok=True,
        verbose=True,
    )

    total_boxes = sum(len(r.boxes) for r in results)
    print(f"\nâœ“ Inference xong: {len(results)} áº£nh, {total_boxes} bounding boxes")
    print(f"  Káº¿t quáº£ lÆ°u táº¡i: {out_dir}")

    # In summary per class
    classes = model.names
    print("\nâ”€â”€â”€ Káº¿t quáº£ phÃ¡t hiá»‡n â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    for r in results:
        img_name = Path(r.path).name
        if len(r.boxes) == 0:
            print(f"  {img_name}: (khÃ´ng phÃ¡t hiá»‡n gÃ¬)")
            continue
        from collections import Counter
        cls_counts = Counter(int(c) for c in r.boxes.cls)
        summary = ", ".join(f"{classes[k]}: {v}" for k, v in sorted(cls_counts.items()))
        print(f"  {img_name}: {summary}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# List cameras
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def list_cameras() -> None:
    try:
        import cv2
    except ImportError:
        raise SystemExit("[ERR] OpenCV chÆ°a cÃ i. Cháº¡y: pip install opencv-python")
    from assembly.camera_app import discover_camera_indices
    cams = discover_camera_indices(cv2)
    if cams:
        print("Camera kháº£ dá»¥ng: " + ", ".join(str(i) for i in cams))
    else:
        print("KhÃ´ng tÃ¬m tháº¥y camera nÃ o.")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CLI
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _parse_source(val: str) -> int | str:
    return int(val) if val.isdigit() else val


def main() -> int:
    _utf8_console()

    parser = argparse.ArgumentParser(
        description="Demo há»‡ thá»‘ng giÃ¡m sÃ¡t láº¯p rÃ¡p Earbud",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["gui", "camera", "infer"], default="gui",
        help=(
            "gui    â€” giao diá»‡n mÃ´ phá»ng tkinter (máº·c Ä‘á»‹nh)\n"
            "camera â€” webcam/video live vá»›i YOLO\n"
            "infer  â€” inference tÄ©nh trÃªn áº£nh/thÆ° má»¥c"
        ),
    )
    parser.add_argument("--source",  default="0",
                        help="Camera index (0,1,...) hoáº·c Ä‘Æ°á»ng dáº«n video/áº£nh")
    parser.add_argument("--model",   default=str(DEFAULT_MODEL),
                        help=f"ÄÆ°á»ng dáº«n checkpoint YOLO (default: {DEFAULT_MODEL.name})")
    parser.add_argument("--camera-config", default=str(DEFAULT_CAMERA_CONFIG),
                        help="File cáº¥u hÃ¬nh camera JSON")
    parser.add_argument("--fsm-config",    default=str(DEFAULT_FSM_CONFIG),
                        help="File cáº¥u hÃ¬nh FSM/workflow JSON")
    parser.add_argument("--conf",    type=float, default=0.25,
                        help="NgÆ°á»¡ng confidence (dÃ¹ng cho --mode infer, default=0.25)")
    parser.add_argument("--device",  default=None,
                        help="cpu | 0 | 1 | cuda (default: tá»± Ä‘á»™ng)")
    parser.add_argument("--no-mirror", action="store_true",
                        help="Táº¯t láº­t ngang áº£nh camera")
    parser.add_argument("--auto-advance", action="store_true",
                        help="Tá»± Ä‘á»™ng xÃ¡c nháº­n hÃ nh Ä‘á»™ng khi dwell Ä‘á»§ frame")
    parser.add_argument("--list-cameras", action="store_true",
                        help="Liá»‡t kÃª cÃ¡c webcam kháº£ dá»¥ng rá»“i thoÃ¡t")
    parser.add_argument("--max-frames", type=int, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.list_cameras:
        list_cameras()
        return 0

    model_path   = Path(args.model)
    camera_cfg   = Path(args.camera_config)
    fsm_cfg      = Path(args.fsm_config)

    print("â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•")
    print("   Demo Earbud Assembly Monitor")
    print(f"   Mode   : {args.mode.upper()}")
    print(f"   Model  : {model_path.name}")
    print("â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•\n")

    if args.mode == "gui":
        run_gui(fsm_cfg)

    elif args.mode == "camera":
        run_camera_mode(
            source=_parse_source(args.source),
            model_path=model_path,
            camera_config=camera_cfg,
            fsm_config=fsm_cfg,
            auto_advance=args.auto_advance,
            device=args.device,
            mirror=not args.no_mirror,
            max_frames=args.max_frames,
        )

    elif args.mode == "infer":
        run_infer(
            source=args.source,
            model_path=model_path,
            conf=args.conf,
            device=args.device,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

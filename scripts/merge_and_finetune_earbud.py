#!/usr/bin/env python3
"""
merge_and_finetune_earbud.py
────────────────────────────
Gom tất cả datasets trong ./datasets/dataset_earbud thành một pool duy nhất:
  • earbud_parts          – YOLO format, 3 classes (Case / Earbud / Empty_Slot)
  • COCO clone 9-8        – COCO JSON, 4 classes
  • COCO clone 9-10       – COCO JSON, 5 classes (+ Hand)
  • COCO clone 9-14       – COCO JSON, 5 classes (+ Hand)

Unified class schema (6 classes):
  0  Earphone_Case   ← earbud_parts "Case"  | COCO "Earphone_Case"
  1  Earbud          ← earbud_parts "Earbud" (general)
  2  Empty_Slot      ← tất cả "Empty_Slot"
  3  Left_Earbud     ← COCO "Left_Earbud"
  4  Right_Earbud    ← COCO "Right_Earbud"
  5  Hand            ← COCO 9-10 / 9-14 "Hand"

Sau khi merge → shuffle → split 80 / 10 / 10 → train YOLOv2-nano.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

# ─── paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

EARBUD_DIR = ROOT / "datasets" / "dataset_earbud"
OUTPUT_DIR = ROOT / "datasets" / "earbud_merged"

# ─── unified classes ──────────────────────────────────────────────────────────
UNIFIED_CLASSES = [
    "Earphone_Case",   # 0
    "Earbud",          # 1  (general – from earbud_parts only)
    "Empty_Slot",      # 2
    "Left_Earbud",     # 3
    "Right_Earbud",    # 4
    "Hand",            # 5
]

# ─── earbud_parts YOLO class mapping ────────────────────────────────────────
# Original nc=3  names=['Case','Earbud','Empty_Slot']
YOLO_CLASS_MAP: dict[int, int] = {
    0: 0,   # Case        → Earphone_Case
    1: 1,   # Earbud      → Earbud
    2: 2,   # Empty_Slot  → Empty_Slot
}

# ─── COCO category_id → unified index ────────────────────────────────────────
# category_id 0 is the root "earbud-detect" supercategory – skip it
# COCO 9-8  : {1:Earphone_Case, 2:Empty_Slot, 3:Left_Earbud, 4:Right_Earbud}
COCO_9_8_MAP: dict[int, int] = {1: 0, 2: 2, 3: 3, 4: 4}
# COCO 9-10 / 9-14 : {1:Earphone_Case, 2:Empty_Slot, 3:Hand, 4:Left_Earbud, 5:Right_Earbud}
COCO_9_10_MAP: dict[int, int] = {1: 0, 2: 2, 3: 5, 4: 3, 5: 4}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ─── helpers ─────────────────────────────────────────────────────────────────

def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def _copy_with_prefix(src_img: Path, src_lbl: Path,
                       dst_img_dir: Path, dst_lbl_dir: Path,
                       prefix: str) -> None:
    """Copy image + label into the pool, adding a prefix to avoid name clashes."""
    stem   = prefix + src_img.stem
    dst_img = dst_img_dir / (stem + src_img.suffix)
    dst_lbl = dst_lbl_dir / (stem + ".txt")
    if not dst_img.exists():
        shutil.copy2(src_img, dst_img)
    if src_lbl.exists() and not dst_lbl.exists():
        shutil.copy2(src_lbl, dst_lbl)


def _remap_yolo_label(src: Path, dst: Path, class_map: dict[int, int]) -> None:
    """Re-write a YOLO .txt label, remapping class indices."""
    lines_out: list[str] = []
    for line in src.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        old_cls = int(parts[0])
        new_cls = class_map.get(old_cls)
        if new_cls is None:
            continue          # skip unknown class
        lines_out.append(" ".join([str(new_cls)] + parts[1:]))
    dst.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")


def _coco_to_yolo_line(ann: dict, img_w: int, img_h: int, unified_cls: int) -> str:
    """Convert one COCO bbox annotation to a YOLO format line."""
    x, y, w, h = ann["bbox"]           # COCO: x_min, y_min, width, height
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    # Clamp to [0, 1]
    cx, cy, nw, nh = (max(0.0, min(1.0, v)) for v in (cx, cy, nw, nh))
    return f"{unified_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


# ─── source processors ───────────────────────────────────────────────────────

def ingest_yolo_source(
    split_dir: Path,
    pool_img: Path,
    pool_lbl: Path,
    class_map: dict[int, int],
    prefix: str,
    verbose: bool,
) -> int:
    """Pull all images+labels from a YOLO-format split dir into the pool."""
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    count = 0
    for img_path in img_dir.iterdir():
        if not _is_image(img_path):
            continue
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            if verbose:
                print(f"  [WARN] no label for {img_path.name} – skipping")
            continue
        # Remap class indices into the unified schema
        stem_new = prefix + img_path.stem
        dst_lbl  = pool_lbl / (stem_new + ".txt")
        _remap_yolo_label(lbl_path, dst_lbl, class_map)
        dst_img = pool_img / (stem_new + img_path.suffix)
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)
        count += 1
    return count


def ingest_coco_source(
    split_dir: Path,
    pool_img: Path,
    pool_lbl: Path,
    cat_map: dict[int, int],
    prefix: str,
    verbose: bool,
) -> int:
    """Convert a COCO-format split dir to YOLO and add to pool."""
    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        print(f"  [WARN] no _annotations.coco.json in {split_dir} – skipping")
        return 0

    with ann_path.open(encoding="utf-8") as f:
        coco = json.load(f)

    # Build lookup tables
    img_info: dict[int, dict] = {img["id"]: img for img in coco.get("images", [])}
    anns_by_img: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    count = 0
    for img_id, info in img_info.items():
        file_name = info["file_name"]
        src_img   = split_dir / file_name
        if not src_img.exists():
            if verbose:
                print(f"  [WARN] image not found: {src_img} – skipping")
            continue

        img_w = info.get("width", 0)
        img_h = info.get("height", 0)

        # If dimensions missing, try to read from file
        if img_w == 0 or img_h == 0:
            try:
                from PIL import Image as PILImage
                with PILImage.open(src_img) as pil:
                    img_w, img_h = pil.size
            except Exception:
                if verbose:
                    print(f"  [WARN] cannot determine size of {file_name} – skipping")
                continue

        stem_new = prefix + src_img.stem
        dst_img  = pool_img / (stem_new + src_img.suffix)
        dst_lbl  = pool_lbl / (stem_new + ".txt")

        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        yolo_lines: list[str] = []
        for ann in anns_by_img.get(img_id, []):
            unified_cls = cat_map.get(ann["category_id"])
            if unified_cls is None:
                continue
            yolo_lines.append(_coco_to_yolo_line(ann, img_w, img_h, unified_cls))

        dst_lbl.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
        count += 1
    return count


# ─── split ───────────────────────────────────────────────────────────────────

def split_pool(
    pool_img: Path,
    pool_lbl: Path,
    out_root: Path,
    train_r: float,
    val_r: float,
    seed: int,
) -> tuple[int, int, int]:
    """Shuffle pool and write train/valid/test splits."""
    images = sorted(p for p in pool_img.iterdir() if _is_image(p))
    random.seed(seed)
    random.shuffle(images)

    n        = len(images)
    n_train  = int(n * train_r)
    n_val    = int(n * val_r)
    n_test   = n - n_train - n_val

    splits = {
        "train": images[:n_train],
        "valid": images[n_train: n_train + n_val],
        "test":  images[n_train + n_val:],
    }

    for split_name, img_list in splits.items():
        (out_root / split_name / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split_name / "labels").mkdir(parents=True, exist_ok=True)
        for img_path in img_list:
            dst_img = out_root / split_name / "images" / img_path.name
            dst_lbl = out_root / split_name / "labels" / (img_path.stem + ".txt")
            shutil.copy2(img_path, dst_img)
            src_lbl = pool_lbl / (img_path.stem + ".txt")
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)

    return n_train, n_val, n_test


def write_data_yaml(out_root: Path) -> Path:
    yaml_path = out_root / "data.yaml"
    names_str = str(UNIFIED_CLASSES).replace("'", '"')
    yaml_path.write_text(
        # Không ghi đường dẫn tuyệt đối của máy hiện tại. Khi không có `path`,
        # Ultralytics và validator đều resolve các split từ thư mục data.yaml.
        f"train: train/images\n"
        f"val:   valid/images\n"
        f"test:  test/images\n"
        f"\n"
        f"nc: {len(UNIFIED_CLASSES)}\n"
        f"names: {names_str}\n",
        encoding="utf-8",
    )
    return yaml_path


# ─── main ────────────────────────────────────────────────────────────────────

def _configure_utf8_console() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name)
        fn = getattr(s, "reconfigure", None)
        if fn:
            fn(encoding="utf-8", errors="replace")


def main() -> int:
    _configure_utf8_console()

    parser = argparse.ArgumentParser(
        description="Merge earbud datasets and finetune YOLO"
    )
    parser.add_argument("--model",      default="yolo26n.pt")
    parser.add_argument("--epochs",     type=int, default=60)
    parser.add_argument("--imgsz",      type=int, default=640)
    parser.add_argument("--batch",      type=int, default=8)
    parser.add_argument("--device",     default=None)
    parser.add_argument("--train-r",    type=float, default=0.80)
    parser.add_argument("--val-r",      type=float, default=0.10)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--no-train",   action="store_true",
                        help="Chỉ merge dataset, không chạy train")
    parser.add_argument("--rebuild",    action="store_true",
                        help="Xóa output cũ và merge lại từ đầu")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # ── 1. Setup output dirs ──────────────────────────────────────────────────
    if args.rebuild and OUTPUT_DIR.exists():
        print(f"[INFO] Xóa output cũ: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    pool_img = OUTPUT_DIR / "_pool" / "images"
    pool_lbl = OUTPUT_DIR / "_pool" / "labels"
    pool_img.mkdir(parents=True, exist_ok=True)
    pool_lbl.mkdir(parents=True, exist_ok=True)

    total = 0
    print("\n═══ BƯỚC 1: THU THẬP DỮ LIỆU ══════════════════════════════════\n")

    # ── 2. Ingest earbud_parts (YOLO) ────────────────────────────────────────
    yolo_root = EARBUD_DIR / "earbud_parts"
    if yolo_root.exists():
        for sp in ("train", "valid", "test"):
            n = ingest_yolo_source(
                split_dir=yolo_root / sp,
                pool_img=pool_img,
                pool_lbl=pool_lbl,
                class_map=YOLO_CLASS_MAP,
                prefix=f"yolo_{sp}_",
                verbose=args.verbose,
            )
            total += n
            print(f"  earbud_parts/{sp:5s} → {n:3d} ảnh")
    else:
        print(f"  [WARN] Không tìm thấy {yolo_root}")

    # ── 3. Ingest COCO datasets ───────────────────────────────────────────────
    coco_datasets: list[tuple[str, str, dict[int, int]]] = [
        ("9-8",  "earbud detect-Clone on 9-8-2026 from haiyang-re0vv-Left_incar.coco",                                          COCO_9_8_MAP),
        ("9-10", "earbud detect-Clone on 9-10-2026 from traigautchin1983-gmail-com-Case Earbud Empty Slot.coco",                COCO_9_10_MAP),
        ("9-14", "earbud detect-Clone on 9-14-2026 from traigautchin1983-gmail-com-Case Earbud Empty Slot.coco",                COCO_9_10_MAP),
        ("9-29", "9-29 pm.coco",                                                                                                COCO_9_10_MAP),
        ("10-30", "10-30-Auto Label.coco",                                                                                      COCO_9_10_MAP),
    ]

    for tag, folder_name, cat_map in coco_datasets:
        coco_root = EARBUD_DIR / folder_name
        if not coco_root.exists():
            print(f"  [WARN] Không tìm thấy: {folder_name}")
            continue
        for sp in ("train", "valid", "test"):
            split_dir = coco_root / sp
            if not split_dir.exists():
                continue
            n = ingest_coco_source(
                split_dir=split_dir,
                pool_img=pool_img,
                pool_lbl=pool_lbl,
                cat_map=cat_map,
                prefix=f"coco{tag}_{sp}_",
                verbose=args.verbose,
            )
            total += n
            print(f"  COCO {tag}/{sp:5s}    → {n:3d} ảnh")

    print(f"\n  ✓ Tổng pool: {total} ảnh")

    # ── 4. Split pool → train/valid/test ─────────────────────────────────────
    print("\n═══ BƯỚC 2: CHIA LẠI TRAIN/VALID/TEST ══════════════════════════\n")
    n_train, n_val, n_test = split_pool(
        pool_img=pool_img,
        pool_lbl=pool_lbl,
        out_root=OUTPUT_DIR,
        train_r=args.train_r,
        val_r=args.val_r,
        seed=args.seed,
    )
    print(f"  Train : {n_train}")
    print(f"  Valid : {n_val}")
    print(f"  Test  : {n_test}")

    # ── 5. Write data.yaml ────────────────────────────────────────────────────
    yaml_path = write_data_yaml(OUTPUT_DIR)
    print(f"\n  ✓ data.yaml → {yaml_path}")
    print(f"  Classes ({len(UNIFIED_CLASSES)}): {UNIFIED_CLASSES}")

    if args.no_train:
        print("\n[INFO] --no-train: dừng trước bước training.")
        return 0

    # ── 6. Finetune ───────────────────────────────────────────────────────────
    print("\n═══ BƯỚC 3: FINETUNE ════════════════════════════════════════════\n")
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("Thiếu ultralytics. Cài: pip install ultralytics") from e

    model_path = ROOT / args.model
    if not model_path.exists():
        model_path = Path(args.model)   # fallback: dùng trực tiếp tên model

    print(f"  Model  : {model_path}")
    print(f"  Data   : {yaml_path}")
    print(f"  Epochs : {args.epochs}")
    print(f"  Imgsz  : {args.imgsz}")
    print(f"  Batch  : {args.batch}")
    print(f"  Device : {args.device or 'auto'}\n")

    model = YOLO(str(model_path))
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "artifacts" / "training"),
        name="earbud_merged_detector",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

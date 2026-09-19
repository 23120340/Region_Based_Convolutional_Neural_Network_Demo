"""annotate_actions.py – Công cụ gán nhãn hành động theo thời gian từ video.

Cách dùng
---------
  python scripts/annotate_actions.py                    # dùng đường dẫn mặc định
  python scripts/annotate_actions.py --videos-dir data/earbud_actions/raw_videos \\
      --output data/earbud_actions/annotations.csv \\
      --config configs/action_earbud_config.json

Phím tắt khi đang xem video
----------------------------
  SPACE    : tạm dừng / tiếp tục
  s        : bắt đầu đánh dấu (SET START) của một đoạn hành động
  e        : kết thúc đánh dấu (SET END) và chọn nhãn qua terminal
  r        : làm lại frame hiện tại (xem lại vùng vừa đánh dấu)
  b        : quay lại 5 giây
  f        : tua tới 5 giây
  q        : bỏ qua video này và chuyển sang video tiếp theo
  ESC      : thoát chương trình

Cấu trúc file annotations.csv được tạo ra
------------------------------------------
  video_id, person_id, session_id, split, start_time_s, end_time_s, action_name
  (split mặc định "train"; chạy split_annotations.py để tách train/val/test)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Đảm bảo import được module trong src/
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

from assembly.action_config import load_action_model_config
from assembly.paths import (
    DEFAULT_ACTION_ANNOTATIONS,
    DEFAULT_ACTION_CONFIG,
    DEFAULT_ACTION_VIDEOS,
)


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

ANNOTATIONS_COLUMNS = [
    "video_id",
    "person_id",
    "session_id",
    "split",
    "start_time_s",
    "end_time_s",
    "action_name",
]


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------

def _parse_video_meta(video_path: Path) -> dict[str, str]:
    """Cố gắng đọc person_id / session_id từ tên file (vd: per1_01_correct_...).
    Trả về dict với giá trị mặc định nếu không parse được."""
    parts = video_path.stem.split("_")
    person_id = parts[0] if len(parts) >= 1 else "unknown"
    session_id = parts[1] if len(parts) >= 2 else "01"
    return {"person_id": person_id, "session_id": session_id}


def _load_existing_annotations(output_path: Path) -> list[dict]:
    """Đọc annotations đã tồn tại để có thể bỏ qua video đã xong."""
    if not output_path.exists():
        return []
    with output_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _write_rows(output_path: Path, rows: list[dict], write_header: bool) -> None:
    mode = "a" if output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ANNOTATIONS_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _ask_action_label(actions: tuple[str, ...], start: float, end: float) -> str | None:
    """Hỏi người dùng qua terminal để chọn nhãn hành động."""
    print(f"\n  Đoạn từ {start:.2f}s → {end:.2f}s ({end - start:.2f}s)")
    print("  Chọn nhãn (hoặc 'skip' để bỏ qua đoạn này):")
    for i, action in enumerate(actions):
        print(f"    {i}: {action}")
    print("    s: bỏ qua đoạn này")
    while True:
        choice = input("  Nhập số hoặc 's': ").strip().lower()
        if choice == "s":
            return None
        try:
            idx = int(choice)
            if 0 <= idx < len(actions):
                return actions[idx]
            else:
                print(f"  Số phải từ 0 đến {len(actions) - 1}")
        except ValueError:
            print("  Nhập không hợp lệ. Thử lại.")


# ---------------------------------------------------------------------------
# Vòng lặp gán nhãn chính
# ---------------------------------------------------------------------------

def annotate_video(
    video_path: Path,
    actions: tuple[str, ...],
    existing_video_ids: set[str],
    output_path: Path,
    first_write: bool,
) -> int:
    """Mở video trong OpenCV, cho phép người dùng đánh dấu đoạn và gán nhãn.
    Trả về số lượng annotation đã thêm vào."""
    try:
        import cv2
    except ImportError:
        raise SystemExit("Thiếu OpenCV. Hãy cài: pip install opencv-python")

    video_id = video_path.stem
    if video_id in existing_video_ids:
        print(f"  SKIP: {video_id} — đã có annotation")
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  WARN: Không mở được {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f"\n{'='*60}")
    print(f"  Video: {video_path.name}")
    print(f"  FPS={fps:.1f}  Frames={total_frames}  Duration={duration:.2f}s")
    print(f"  Phím: [SPACE]=dừng  [s]=start  [e]=end+gán nhãn  [b]=-5s  [f]=+5s  [q]=bỏ qua  [ESC]=thoát")
    print(f"{'='*60}")

    meta = _parse_video_meta(video_path)
    new_rows: list[dict] = []
    mark_start: float | None = None
    paused = False
    pos_frame = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("  [Hết video — nhấn q để sang video tiếp theo hoặc ESC để thoát]")
                paused = True
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
                ret, frame = cap.read()
                if not ret:
                    break
            pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        pos_sec = pos_frame / fps

        # Overlay thông tin lên frame
        display = frame.copy()
        status_color = (0, 200, 0) if mark_start is None else (0, 0, 255)
        cv2.putText(display, f"{pos_sec:.2f}s / {duration:.2f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        if mark_start is not None:
            cv2.putText(display, f"START={mark_start:.2f}s  [e]=end", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        if paused:
            cv2.putText(display, "PAUSED", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        cv2.imshow(f"Annotate: {video_path.name}", display)
        key = cv2.waitKey(30) & 0xFF  # ~30ms mỗi frame khi play

        if key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            if new_rows:
                _write_rows(output_path, new_rows, write_header=first_write)
                print(f"  Đã lưu {len(new_rows)} annotation trước khi thoát.")
            raise SystemExit(0)

        elif key == ord("q"):
            break

        elif key == ord(" "):
            paused = not paused

        elif key == ord("s"):
            mark_start = pos_sec
            print(f"  [START đặt tại {mark_start:.2f}s]")
            paused = True

        elif key == ord("e"):
            if mark_start is None:
                print("  [WARN] Chưa đặt START. Nhấn 's' trước.")
            else:
                mark_end = pos_sec
                if mark_end <= mark_start:
                    print("  [WARN] END phải sau START. Điều chỉnh vị trí và nhấn 'e' lại.")
                else:
                    # Tạm dừng video để hỏi nhãn qua terminal
                    cv2.destroyWindow(f"Annotate: {video_path.name}")
                    label = _ask_action_label(actions, mark_start, mark_end)
                    if label:
                        new_rows.append({
                            "video_id": video_id,
                            "person_id": meta["person_id"],
                            "session_id": meta["session_id"],
                            "split": "train",  # mặc định; chỉnh sau khi cần
                            "start_time_s": f"{mark_start:.4f}",
                            "end_time_s": f"{mark_end:.4f}",
                            "action_name": label,
                        })
                        print(f"  ✓ Đã thêm: {label} [{mark_start:.2f}s – {mark_end:.2f}s]")
                    else:
                        print("  Bỏ qua đoạn này.")
                    mark_start = None
                    # Mở lại cửa sổ
                    cv2.imshow(f"Annotate: {video_path.name}", display)

        elif key == ord("b"):  # lùi 5 giây
            new_pos = max(0, pos_frame - int(5 * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            pos_frame = new_pos
            paused = True

        elif key == ord("f"):  # tua 5 giây
            new_pos = min(total_frames - 1, pos_frame + int(5 * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            pos_frame = new_pos
            paused = True

    cap.release()
    cv2.destroyAllWindows()

    if new_rows:
        _write_rows(output_path, new_rows, write_header=first_write)
        print(f"  ✓ Đã lưu {len(new_rows)} annotation cho {video_id}")
    else:
        print(f"  (Không có annotation nào được thêm cho {video_id})")

    return len(new_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gán nhãn hành động theo thời gian từ video bằng phím tắt OpenCV"
    )
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_ACTION_VIDEOS,
                        help="Thư mục chứa các video cần gán nhãn (tìm đệ quy)")
    parser.add_argument("--output", type=Path, default=DEFAULT_ACTION_ANNOTATIONS,
                        help="Đường dẫn file annotations.csv sẽ ghi ra")
    parser.add_argument("--config", type=Path, default=DEFAULT_ACTION_CONFIG,
                        help="File action_model_config.json (để lấy danh sách nhãn)")
    parser.add_argument("--resume", action="store_true",
                        help="Nếu bật, bỏ qua video đã có annotation trong file output")
    args = parser.parse_args()

    config = load_action_model_config(args.config)
    actions = config.actions
    print(f"  Nhãn hành động: {list(actions)}")

    videos = sorted(
        path for path in args.videos_dir.rglob("*")
        if path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise SystemExit(f"Không tìm thấy video nào trong {args.videos_dir}")

    print(f"  Tìm thấy {len(videos)} video trong {args.videos_dir}")

    # Đọc annotation đã có để --resume có thể bỏ qua
    existing_annotations = _load_existing_annotations(args.output) if args.resume else []
    existing_video_ids = {row["video_id"] for row in existing_annotations}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Ghi header nếu file chưa tồn tại
    write_header = not args.output.exists()
    if write_header:
        with args.output.open("w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=ANNOTATIONS_COLUMNS).writeheader()
        write_header = False  # Header đã ghi; các lần sau append không cần

    total_added = 0
    for video_path in videos:
        count = annotate_video(
            video_path=video_path,
            actions=actions,
            existing_video_ids=existing_video_ids,
            output_path=args.output,
            first_write=write_header,
        )
        total_added += count

    print(f"\n{'='*60}")
    print(f"  XONG. Tổng số annotation đã thêm: {total_added}")
    print(f"  File annotation: {args.output}")
    print(f"  Bước tiếp theo:")
    print(f"    1. Mở {args.output} và chỉnh cột 'split' (train/val/test) thủ công")
    print(f"    2. Chạy: python scripts/extract_spatial_features.py")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

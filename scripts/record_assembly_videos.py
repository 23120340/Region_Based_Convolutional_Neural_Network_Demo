from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "pen_actions" / "raw_videos"


def _configure_utf8_console() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def main() -> int:
    _configure_utf8_console()
    try:
        import cv2
    except ImportError as error:
        raise SystemExit("Thiếu OpenCV; hãy cài requirements-ml.txt") from error

    parser = argparse.ArgumentParser(description="Record full assembly cycles for temporal action recognition")
    parser.add_argument("--source", default="0")
    parser.add_argument("--person", required=True, help="Ví dụ: person01")
    parser.add_argument("--session", required=True, help="Ví dụ: session01")
    parser.add_argument("--project", default="earbud", choices=["pen", "earbud", "custom"], help="Dự án: pen hoặc earbud")
    parser.add_argument("--scenario", default="correct", help="Kịch bản (vd: correct, wrong_order, missing_earbud, missing_spring...)")
    parser.add_argument("--output", type=Path, default=None, help="Thư mục lưu raw_videos (mặc định theo --project)")
    parser.add_argument("--no-mirror", action="store_true")
    args = parser.parse_args()

    output_root = args.output
    if output_root is None:
        folder_name = "earbud_actions" if args.project == "earbud" else "pen_actions"
        output_root = ROOT / "data" / folder_name / "raw_videos"

    source = _source(args.source)
    backend = cv2.CAP_DSHOW if isinstance(source, int) and sys.platform == "win32" else cv2.CAP_ANY
    capture = cv2.VideoCapture(source, backend)
    if not capture.isOpened():
        raise SystemExit(f"Không mở được camera/video source {source!r}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    fps = source_fps if 5 <= source_fps <= 120 else 30.0

    session_dir = output_root / args.person / args.session
    session_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_root.parent / "recording_log.csv"
    recording = False
    writer = None
    current_path: Path | None = None
    started_at = 0.0
    clip_index = len(list(session_dir.glob("*.mp4"))) + 1

    def finish_clip() -> None:
        nonlocal recording, writer, current_path, started_at, clip_index
        if writer is None or current_path is None:
            return
        writer.release()
        duration = time.monotonic() - started_at
        new_file = not log_path.exists()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            logged_path = current_path.relative_to(ROOT).as_posix()
        except ValueError:
            logged_path = str(current_path.resolve())

        # Sửa lỗi video bị tua nhanh (Auto-correct FPS)
        cap = cv2.VideoCapture(str(current_path))
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_fps_meta = cap.get(cv2.CAP_PROP_FPS)
            current_duration = frame_count / current_fps_meta if current_fps_meta > 0 else 0
            # Nếu thời gian lệch quá 10%, tiến hành ghi đè lại với FPS thực tế
            if current_duration < duration * 0.9:
                real_fps = frame_count / duration
                print(f"-> Đang sửa lỗi tua nhanh (FPS thực {real_fps:.1f})...")
                temp_path = current_path.with_name(current_path.name + ".tmp.mp4")
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fix_writer = cv2.VideoWriter(str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), real_fps, (width, height))
                while True:
                    ok, f = cap.read()
                    if not ok: break
                    fix_writer.write(f)
                fix_writer.release()
                cap.release()
                import shutil
                shutil.move(temp_path, current_path)
            else:
                cap.release()

        with log_path.open("a", encoding="utf-8", newline="") as file:
            csv_writer = csv.DictWriter(
                file,
                fieldnames=["video_id", "person_id", "session_id", "scenario", "path", "duration_s"],
            )
            if new_file:
                csv_writer.writeheader()
            csv_writer.writerow(
                {
                    "video_id": current_path.stem,
                    "person_id": args.person,
                    "session_id": args.session,
                    "scenario": args.scenario,
                    "path": logged_path,
                    "duration_s": f"{duration:.3f}",
                }
            )
        print(f"Đã lưu {current_path} ({duration:.1f}s)")
        recording = False
        writer = None
        current_path = None
        clip_index += 1

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            clean_frame = frame.copy()
            height, width = frame.shape[:2]

            if recording and writer is not None:
                writer.write(clean_frame)
                cv2.circle(frame, (24, 28), 9, (0, 0, 255), -1)
                status = f"REC {time.monotonic() - started_at:.1f}s | {args.scenario}"
                color = (0, 0, 255)
            else:
                status = f"READY | person={args.person} session={args.session} scenario={args.scenario}"
                color = (60, 220, 80)
            cv2.putText(frame, status, (42, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
            cv2.putText(frame, "SPACE start/stop full cycle | Q quit", (18, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Record Assembly Videos", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == 32:
                if recording:
                    finish_clip()
                else:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    video_id = f"{args.person}_{args.session}_{args.scenario}_{timestamp}_{clip_index:03d}"
                    current_path = session_dir / f"{video_id}.mp4"
                    writer = cv2.VideoWriter(
                        str(current_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Không tạo được video {current_path}")
                    started_at = time.monotonic()
                    recording = True
                    print(f"Bắt đầu quay: {current_path}")
    finally:
        if recording:
            finish_clip()
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

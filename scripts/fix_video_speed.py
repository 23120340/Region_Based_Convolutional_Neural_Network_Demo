from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]

def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    log_path = ROOT / "data" / "earbud_actions" / "recording_log.csv"
    if not log_path.exists():
        print(f"Không tìm thấy {log_path}")
        return 1

    # Đọc duration thực tế từ log
    duration_map: dict[str, float] = {}
    with log_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            duration_map[row["video_id"]] = float(row["duration_s"])

    videos_dir = ROOT / "data" / "earbud_actions" / "raw_videos"
    videos = sorted(videos_dir.rglob("*.mp4"))
    
    if not videos:
        print("Không tìm thấy video nào để sửa.")
        return 0

    print(f"Tìm thấy {len(videos)} video. Bắt đầu sửa lỗi tua nhanh...")
    
    for video_path in videos:
        video_id = video_path.stem
        if video_id not in duration_map:
            continue
            
        real_duration = duration_map[video_id]
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Lỗi đọc {video_path.name}")
            continue
            
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_metadata = cap.get(cv2.CAP_PROP_FPS)
        
        # Nếu video đang quá ngắn so với thời gian quay thực (lệch > 10%)
        current_duration = frame_count / fps_metadata if fps_metadata > 0 else 0
        if current_duration >= real_duration * 0.9:
            print(f"Bỏ qua {video_path.name} (hiện {current_duration:.1f}s, gốc {real_duration:.1f}s) - Đã bình thường.")
            cap.release()
            continue

        real_fps = frame_count / real_duration
        print(f"Sửa {video_path.name}: {current_duration:.1f}s -> {real_duration:.1f}s (FPS mới: {real_fps:.2f})")
        
        temp_path = video_path.with_name(video_path.name + ".tmp.mp4")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        writer = cv2.VideoWriter(
            str(temp_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            real_fps,
            (width, height)
        )
        
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            
        cap.release()
        writer.release()
        
        # Ghi đè file cũ
        shutil.move(temp_path, video_path)

    print("Hoàn tất!")
    return 0

if __name__ == "__main__":
    sys.exit(main())

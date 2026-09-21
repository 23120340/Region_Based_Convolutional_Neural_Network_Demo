from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assembly.action_config import load_action_model_config
from assembly.models.spatial_encoder import ViTSpatialEncoder
from assembly.paths import DEFAULT_ACTION_CONFIG, DEFAULT_ACTION_VIDEOS, DEFAULT_FEATURE_CACHE


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _video_id(video_path: Path, videos_dir: Path) -> str:
    del videos_dir  # Kept in the signature for compatibility with earlier calls.
    return video_path.stem


def main() -> int:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise SystemExit("Thiếu OpenCV hoặc NumPy; hãy cài requirements-ml.txt") from error

    parser = argparse.ArgumentParser(description="Extract and cache frozen ViT CLS embeddings from assembly videos")
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_ACTION_VIDEOS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--config", type=Path, default=DEFAULT_ACTION_CONFIG)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_action_model_config(args.config)
    videos = sorted(path for path in args.videos_dir.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise SystemExit(f"Không tìm thấy video trong {args.videos_dir}")
    video_ids = [_video_id(path, args.videos_dir) for path in videos]
    duplicates = sorted({video_id for video_id in video_ids if video_ids.count(video_id) > 1})
    if duplicates:
        raise SystemExit(f"Video ID bị trùng; hãy đổi tên file để duy nhất: {duplicates}")
    encoder = ViTSpatialEncoder(config.spatial.backbone, args.device, config.spatial.freeze)
    if encoder.embedding_dim != config.spatial.embedding_dim:
        raise SystemExit(
            f"Backbone trả dim={encoder.embedding_dim}, config yêu cầu dim={config.spatial.embedding_dim}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for video_path in videos:
        video_id = _video_id(video_path, args.videos_dir)
        feature_path = args.output_dir / f"{video_id}.npy"
        metadata_path = args.output_dir / f"{video_id}.json"
        if feature_path.exists() and metadata_path.exists() and not args.overwrite:
            print(f"SKIP {video_id}: feature đã tồn tại")
            continue

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            print(f"WARN không mở được {video_path}")
            continue
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        sample_fps = min(config.spatial.sample_fps, source_fps)
        sample_interval = source_fps / sample_fps
        next_sample = 0.0
        frame_index = 0
        rgb_frames: list[object] = []
        sampled_indices: list[int] = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index + 1e-9 >= next_sample:
                    rgb_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    sampled_indices.append(frame_index)
                    next_sample += sample_interval
                frame_index += 1
        finally:
            capture.release()
        if not rgb_frames:
            print(f"WARN video không có frame: {video_path}")
            continue

        features = encoder.encode_images(rgb_frames, batch_size=args.batch_size).numpy().astype("float32")
        np.save(feature_path, features)
        metadata = {
            "video_id": video_id,
            "source_path": video_path.relative_to(ROOT).as_posix() if video_path.is_relative_to(ROOT) else str(video_path),
            "source_fps": source_fps,
            "sample_fps": sample_fps,
            "source_frames": frame_index,
            "feature_frames": int(features.shape[0]),
            "embedding_dim": int(features.shape[1]),
            "sampled_source_indices": sampled_indices,
            "backbone": config.spatial.backbone,
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"OK {video_id}: {features.shape} -> {feature_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


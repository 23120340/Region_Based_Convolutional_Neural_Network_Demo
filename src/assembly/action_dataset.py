from __future__ import annotations

import csv
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class WindowReference:
    feature_path: Path
    start_index: int
    end_index: int
    label_id: int
    video_id: str


class CachedActionWindowDataset(Dataset):
    """Create fixed-length temporal windows from cached per-frame embeddings."""

    REQUIRED_COLUMNS = {
        "video_id",
        "split",
        "start_time_s",
        "end_time_s",
        "action_name",
    }

    def __init__(
        self,
        features_dir: str | Path,
        annotations_csv: str | Path,
        split: str,
        action_to_id: dict[str, int],
        sequence_length: int,
        stride: int,
        expected_embedding_dim: int | None = None,
        allow_same_session: bool = False,
    ) -> None:
        self.features_dir = Path(features_dir)
        self.sequence_length = sequence_length
        self.expected_embedding_dim = expected_embedding_dim
        self.references: list[WindowReference] = []
        self._arrays: OrderedDict[Path, np.ndarray] = OrderedDict()
        self._max_cached_videos = 4
        if sequence_length < 1 or stride < 1:
            raise ValueError("sequence_length và stride phải > 0")

        with Path(annotations_csv).open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            missing = self.REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"annotations.csv thiếu cột: {sorted(missing)}")
            fieldnames = set(reader.fieldnames or [])
            all_rows = list(reader)

        allowed_splits = {"train", "val", "test"}
        video_splits: dict[str, set[str]] = {}
        session_splits: dict[tuple[str, str], set[str]] = {}
        has_group_columns = {"person_id", "session_id"}.issubset(fieldnames)
        for row in all_rows:
            row_split = row["split"].strip()
            if row_split not in allowed_splits:
                raise ValueError(f"split không hợp lệ: {row_split!r}")
            video_id = row["video_id"].strip()
            if not video_id:
                raise ValueError("video_id không được để trống")
            video_splits.setdefault(video_id, set()).add(row_split)
            if has_group_columns:
                group = (row["person_id"].strip(), row["session_id"].strip())
                if all(group):
                    session_splits.setdefault(group, set()).add(row_split)

        leaking_videos = sorted(video_id for video_id, splits in video_splits.items() if len(splits) > 1)
        if leaking_videos:
            raise ValueError(f"video_id xuất hiện ở nhiều split: {leaking_videos}")
        if not allow_same_session:
            leaking_sessions = sorted(group for group, splits in session_splits.items() if len(splits) > 1)
            if leaking_sessions:
                raise ValueError(f"person/session xuất hiện ở nhiều split: {leaking_sessions}")

        rows = [row for row in all_rows if row["split"].strip() == split]

        for row in rows:
            action = row["action_name"].strip()
            if action not in action_to_id:
                raise ValueError(f"action_name không tồn tại trong config: {action!r}")
            video_id = row["video_id"].strip()
            feature_path = self.features_dir / f"{video_id}.npy"
            metadata_path = self.features_dir / f"{video_id}.json"
            if not feature_path.exists() or not metadata_path.exists():
                raise FileNotFoundError(f"Thiếu feature/metadata cho video_id={video_id}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            sample_fps = float(metadata["sample_fps"])
            frame_count = int(metadata["feature_frames"])
            start_index = max(0, int(math.floor(float(row["start_time_s"]) * sample_fps)))
            end_index = min(frame_count, int(math.ceil(float(row["end_time_s"]) * sample_fps)))
            if end_index <= start_index:
                raise ValueError(f"Đoạn annotation rỗng cho {video_id}: {row}")

            last_start = max(start_index, end_index - sequence_length)
            starts = list(range(start_index, last_start + 1, stride)) or [start_index]
            if starts[-1] != last_start:
                starts.append(last_start)
            for window_start in starts:
                self.references.append(
                    WindowReference(
                        feature_path=feature_path,
                        start_index=window_start,
                        end_index=min(window_start + sequence_length, end_index),
                        label_id=action_to_id[action],
                        video_id=video_id,
                    )
                )

        if not self.references:
            raise ValueError(f"Không có temporal window nào cho split={split!r}")

    def __len__(self) -> int:
        return len(self.references)

    def _load(self, path: Path) -> np.ndarray:
        if path in self._arrays:
            self._arrays.move_to_end(path)
            return self._arrays[path]
        if path not in self._arrays:
            # Load into memory instead of keeping an mmap handle. On Windows an
            # open mmap prevents dataset folders from being moved or cleaned up.
            array = np.load(path)
            if array.ndim != 2:
                raise ValueError(f"Feature {path} phải có shape (frames, embedding_dim)")
            if self.expected_embedding_dim is not None and array.shape[1] != self.expected_embedding_dim:
                raise ValueError(
                    f"Feature {path} có dim={array.shape[1]}, config yêu cầu {self.expected_embedding_dim}"
                )
            self._arrays[path] = array
            while len(self._arrays) > self._max_cached_videos:
                self._arrays.popitem(last=False)
        return self._arrays[path]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        reference = self.references[index]
        array = self._load(reference.feature_path)
        window = np.asarray(array[reference.start_index : reference.end_index], dtype=np.float32)
        if len(window) < self.sequence_length:
            padding = np.repeat(window[-1:, :], self.sequence_length - len(window), axis=0)
            window = np.concatenate((window, padding), axis=0)
        features = torch.from_numpy(window.copy())
        label = torch.tensor(reference.label_id, dtype=torch.long)
        return features, label

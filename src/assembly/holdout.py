"""Regression holdout registry and leakage guards."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class HoldoutEntry:
    video_id: str
    path: str
    sha256: str
    scenario: str


@dataclass(frozen=True)
class HoldoutRegistry:
    entries: tuple[HoldoutEntry, ...]

    @property
    def video_ids(self) -> frozenset[str]:
        return frozenset(entry.video_id.casefold() for entry in self.entries)

    @property
    def hashes(self) -> frozenset[str]:
        return frozenset(entry.sha256.casefold() for entry in self.entries if entry.sha256)

    def reject_video_ids(self, video_ids: Iterable[str]) -> None:
        found = sorted({str(value).strip().casefold() for value in video_ids} & self.video_ids)
        if found:
            raise ValueError(f"Regression holdout must never enter training data: {found}")

    def reject_paths(self, paths: Iterable[str | Path], *, verify_hashes: bool = True) -> None:
        candidates = [Path(path) for path in paths]
        self.reject_video_ids(path.stem for path in candidates)
        if not verify_hashes or not self.hashes:
            return
        matches = []
        for path in candidates:
            if path.is_file() and _sha256(path).casefold() in self.hashes:
                matches.append(str(path))
        if matches:
            raise ValueError(f"Regression holdout content found under a renamed file: {matches}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_holdout_registry(path: str | Path) -> HoldoutRegistry:
    with Path(path).open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if raw.get("schema_version") != 1 or not isinstance(raw.get("videos"), list):
        raise ValueError("holdout manifest must use schema_version=1 and contain videos[]")
    entries = []
    for index, item in enumerate(raw["videos"]):
        if not isinstance(item, dict):
            raise ValueError(f"videos[{index}] must be an object")
        video_id = str(item.get("video_id", "")).strip()
        sha256 = str(item.get("sha256", "")).strip().casefold()
        if not video_id or (sha256 and (len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256))):
            raise ValueError(f"invalid holdout entry videos[{index}]")
        entries.append(HoldoutEntry(video_id, str(item.get("path", "")), sha256,
                                    str(item.get("scenario", ""))))
    return HoldoutRegistry(tuple(entries))

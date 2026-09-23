"""Sửa timestamp MP4 bị phát nhanh mà không encode lại khung hình.

Mặc định, script so sánh duration trong ``recording_log.csv`` với
duration trong MP4. Video không có log chỉ được sửa khi truyền rõ
``--unlogged-speed-factor``. Trước khi sửa luôn tạo file ``.bak`` cạnh
video; không xóa hoặc encode lại dữ liệu hình.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import struct
import sys
from pathlib import Path
from typing import Iterator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEOS = ROOT / "data" / "earbud_actions" / "raw_videos"
DEFAULT_LOG = ROOT / "data" / "earbud_actions" / "recording_log.csv"
CONTAINER_BOXES = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf"}


def _configure_utf8_console() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _boxes(
    data: bytes | bytearray,
    start: int = 0,
    end: int | None = None,
) -> Iterator[tuple[bytes, int, int]]:
    """Yield ``(type, payload_start, box_end)`` for valid MP4 boxes."""

    limit = len(data) if end is None else end
    position = start
    while position + 8 <= limit:
        size = struct.unpack_from(">I", data, position)[0]
        box_type = bytes(data[position + 4 : position + 8])
        header_size = 8
        if size == 1:
            if position + 16 > limit:
                break
            size = struct.unpack_from(">Q", data, position + 8)[0]
            header_size = 16
        elif size == 0:
            size = limit - position
        if size < header_size or position + size > limit:
            break
        yield box_type, position + header_size, position + size
        position += size


def _walk_boxes(
    data: bytes | bytearray,
    start: int,
    end: int,
) -> Iterator[tuple[bytes, int, int]]:
    for box_type, payload_start, box_end in _boxes(data, start, end):
        yield box_type, payload_start, box_end
        if box_type in CONTAINER_BOXES:
            yield from _walk_boxes(data, payload_start, box_end)


def _scaled(value: int, factor: float, maximum: int) -> int:
    result = round(value * factor)
    if not 0 <= result <= maximum:
        raise ValueError(f"Giá trị MP4 bị tràn sau khi nhân {factor}: {value}")
    return result


def _scale_unsigned(
    data: bytearray,
    offset: int,
    byte_count: int,
    factor: float,
) -> None:
    fmt = ">I" if byte_count == 4 else ">Q"
    maximum = (1 << (byte_count * 8)) - 1
    value = struct.unpack_from(fmt, data, offset)[0]
    struct.pack_into(fmt, data, offset, _scaled(value, factor, maximum))


def _scale_signed_32(data: bytearray, offset: int, factor: float) -> None:
    value = struct.unpack_from(">i", data, offset)[0]
    result = round(value * factor)
    if not -(1 << 31) <= result < (1 << 31):
        raise ValueError("Composition offset bị tràn int32")
    struct.pack_into(">i", data, offset, result)


def _video_timing(path: Path) -> tuple[int, float, float]:
    """Return video sample count, duration and average FPS."""

    data = path.read_bytes()
    moov = next(((start, end) for kind, start, end in _boxes(data) if kind == b"moov"), None)
    if moov is None:
        raise ValueError("MP4 không có box moov")
    for kind, track_start, track_end in _boxes(data, *moov):
        if kind != b"trak":
            continue
        found = {
            box_type: (start, end)
            for box_type, start, end in _walk_boxes(data, track_start, track_end)
            if box_type in {b"hdlr", b"mdhd", b"stts"}
        }
        if not all(key in found for key in (b"hdlr", b"mdhd", b"stts")):
            continue
        handler_start, _ = found[b"hdlr"]
        if bytes(data[handler_start + 8 : handler_start + 12]) != b"vide":
            continue
        media_start, _ = found[b"mdhd"]
        version = data[media_start]
        timescale_offset = media_start + (20 if version == 1 else 12)
        timescale = struct.unpack_from(">I", data, timescale_offset)[0]
        timing_start, _ = found[b"stts"]
        entry_count = struct.unpack_from(">I", data, timing_start + 4)[0]
        samples = 0
        ticks = 0
        entry_offset = timing_start + 8
        for _ in range(entry_count):
            sample_count, sample_delta = struct.unpack_from(">II", data, entry_offset)
            samples += sample_count
            ticks += sample_count * sample_delta
            entry_offset += 8
        duration = ticks / timescale
        return samples, duration, samples / duration
    raise ValueError("MP4 không có video track hợp lệ")


def _retime_mp4(path: Path, factor: float, backup_suffix: str) -> tuple[float, float]:
    """Multiply MP4 timestamps and durations, preserving encoded frames."""

    if factor <= 0:
        raise ValueError("speed factor phải > 0")
    backup_path = path.with_name(path.name + backup_suffix)
    if backup_path.exists():
        raise FileExistsError(
            f"Đã có backup {backup_path.name}; dừng để tránh sửa lặp hai lần"
        )

    _, before_duration, _ = _video_timing(path)
    data = bytearray(path.read_bytes())
    moov = next(((start, end) for kind, start, end in _boxes(data) if kind == b"moov"), None)
    if moov is None:
        raise ValueError("MP4 không có box moov")

    stts_entries = 0
    for box_type, start, _ in _walk_boxes(data, *moov):
        version = data[start] if start < len(data) else 0
        if box_type == b"mvhd":
            _scale_unsigned(data, start + (24 if version == 1 else 16), 8 if version == 1 else 4, factor)
        elif box_type == b"tkhd":
            _scale_unsigned(data, start + (28 if version == 1 else 20), 8 if version == 1 else 4, factor)
        elif box_type == b"mdhd":
            _scale_unsigned(data, start + (24 if version == 1 else 16), 8 if version == 1 else 4, factor)
        elif box_type == b"elst":
            entry_count = struct.unpack_from(">I", data, start + 4)[0]
            entry_offset = start + 8
            entry_size = 20 if version == 1 else 12
            duration_size = 8 if version == 1 else 4
            for _ in range(entry_count):
                _scale_unsigned(data, entry_offset, duration_size, factor)
                entry_offset += entry_size
        elif box_type == b"stts":
            entry_count = struct.unpack_from(">I", data, start + 4)[0]
            entry_offset = start + 8
            for _ in range(entry_count):
                _scale_unsigned(data, entry_offset + 4, 4, factor)
                entry_offset += 8
                stts_entries += 1
        elif box_type == b"ctts":
            entry_count = struct.unpack_from(">I", data, start + 4)[0]
            entry_offset = start + 8
            for _ in range(entry_count):
                if version == 1:
                    _scale_signed_32(data, entry_offset + 4, factor)
                else:
                    _scale_unsigned(data, entry_offset + 4, 4, factor)
                entry_offset += 8

    if stts_entries == 0:
        raise ValueError("MP4 không có sample timing stts")

    shutil.copy2(path, backup_path)
    temp_path = path.with_name(path.name + ".retime.tmp")
    try:
        temp_path.write_bytes(data)
        _, after_duration, _ = _video_timing(temp_path)
        expected = before_duration * factor
        if abs(after_duration - expected) > max(0.01, expected * 0.001):
            raise ValueError(
                f"Duration sau sửa {after_duration:.3f}s không khớp {expected:.3f}s"
            )
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return before_duration, after_duration


def _load_duration_log(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return {
            row["video_id"]: float(row["duration_s"])
            for row in csv.DictReader(file)
        }


def _video_session(path: Path) -> str:
    parts = path.stem.split("_")
    return parts[1] if len(parts) > 1 else path.parent.name


def main() -> int:
    _configure_utf8_console()
    parser = argparse.ArgumentParser(description="Sửa MP4 bị tua nhanh mà không encode lại")
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--session", action="append", help="Chỉ xử lý session, ví dụ: --session 01")
    parser.add_argument(
        "--unlogged-speed-factor",
        type=float,
        default=None,
        help="Video không có recording log: 2.0 nghĩa là tăng duration gấp đôi",
    )
    parser.add_argument("--backup-suffix", default=".before_speed_fix.bak")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    duration_map = _load_duration_log(args.log)
    sessions = set(args.session or [])
    videos = sorted(args.videos_dir.rglob("*.mp4"))
    if sessions:
        videos = [path for path in videos if _video_session(path) in sessions]
    if not videos:
        print("Không tìm thấy video phù hợp.")
        return 0

    fixed = skipped = failed = 0
    for video_path in videos:
        try:
            samples, current_duration, current_fps = _video_timing(video_path)
        except (OSError, ValueError, struct.error) as error:
            print(f"ERROR {video_path.name}: {error}")
            failed += 1
            continue

        logged_duration = duration_map.get(video_path.stem)
        if logged_duration is not None:
            factor = logged_duration / current_duration
            if 0.9 <= current_duration / logged_duration <= 1.1:
                print(
                    f"SKIP {video_path.name}: {current_duration:.3f}s, "
                    "đã khớp recording log"
                )
                skipped += 1
                continue
        elif args.unlogged_speed_factor is not None:
            factor = args.unlogged_speed_factor
        else:
            print(f"SKIP {video_path.name}: không có recording log")
            skipped += 1
            continue

        target_duration = current_duration * factor
        print(
            f"{'WOULD_FIX' if args.dry_run else 'FIX'} {video_path.name}: "
            f"frames={samples}, {current_duration:.3f}s @ {current_fps:.3f} FPS "
            f"-> {target_duration:.3f}s @ {samples / target_duration:.3f} FPS"
        )
        if args.dry_run:
            continue
        try:
            _retime_mp4(video_path, factor, args.backup_suffix)
            fixed += 1
        except (OSError, ValueError, FileExistsError, struct.error) as error:
            print(f"ERROR {video_path.name}: {error}")
            failed += 1

    print(f"Hoàn tất: fixed={fixed}, skipped={skipped}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

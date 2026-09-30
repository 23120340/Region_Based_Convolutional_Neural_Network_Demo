"""Typed configuration for cadence, confidence and temporal fusion thresholds."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class SchedulerThresholds:
    target_embedding_fps: float
    telemetry_horizon_s: float


@dataclass(frozen=True)
class ConfidenceThresholds:
    display_min: float
    case_min: float
    occupied_earbud_min: float
    wrong_side_earbud_min: float
    empty_slot_min: float
    action_min: float


@dataclass(frozen=True)
class DurationThresholds:
    open_case_s: float
    insert_s: float
    wrong_side_s: float
    removal_s: float
    close_s: float
    unknown_arm_s: float
    max_gap_s: float
    action_latch_s: float


@dataclass(frozen=True)
class TrackingThresholds:
    case_iou_min: float
    case_center_shift_ratio: float
    assignment_ambiguity_margin: float


@dataclass(frozen=True)
class TemporalFusionConfig:
    scheduler: SchedulerThresholds
    confidence: ConfidenceThresholds
    duration: DurationThresholds
    tracking: TrackingThresholds
    test_start_open: bool = False


def default_temporal_fusion_config() -> TemporalFusionConfig:
    """Safe defaults matching the tracked production threshold file."""

    return TemporalFusionConfig(
        SchedulerThresholds(10.0, 3.0),
        ConfidenceThresholds(0.5, 0.5, 0.5, 0.58, 0.5, 0.5),
        DurationThresholds(0.3, 0.25, 0.5, 0.8, 0.3, 0.15, 0.35, 1.0),
        TrackingThresholds(0.15, 0.6, 0.1),
        False,
    )


def _section(raw: dict, name: str) -> dict:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"temporal config missing object {name!r}")
    return value


def _number(raw: dict, name: str, *, lower: float = 0.0, upper: float | None = None,
            strict_lower: bool = False) -> float:
    value = raw.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if value < lower or (strict_lower and value <= lower) or (upper is not None and value > upper):
        bracket = ">" if strict_lower else ">="
        raise ValueError(f"{name} must be {bracket} {lower}" + (f" and <= {upper}" if upper is not None else ""))
    return value


def load_temporal_fusion_config(path: str | Path) -> TemporalFusionConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if raw.get("schema_version") != 1:
        raise ValueError("temporal config schema_version must equal 1")
    scheduler = _section(raw, "scheduler")
    confidence = _section(raw, "confidence")
    duration = _section(raw, "duration_seconds")
    tracking = _section(raw, "tracking")
    return TemporalFusionConfig(
        scheduler=SchedulerThresholds(
            _number(scheduler, "target_embedding_fps", strict_lower=True),
            _number(scheduler, "telemetry_horizon_s", strict_lower=True),
        ),
        confidence=ConfidenceThresholds(**{
            key: _number(confidence, key, upper=1.0)
            for key in ("display_min", "case_min", "occupied_earbud_min",
                        "wrong_side_earbud_min", "empty_slot_min", "action_min")
        }),
        duration=DurationThresholds(**{
            key: _number(duration, key, strict_lower=True)
            for key in ("open_case_s", "insert_s", "wrong_side_s", "removal_s", "close_s",
                        "unknown_arm_s", "max_gap_s", "action_latch_s")
        }),
        tracking=TrackingThresholds(
            _number(tracking, "case_iou_min", upper=1.0),
            _number(tracking, "case_center_shift_ratio", strict_lower=True),
            _number(tracking, "assignment_ambiguity_margin", upper=1.0),
        ),
        test_start_open=bool(raw.get("test_start_open", False)),
    )

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Typed FlySeek-Bench record schemas + validation.

All records are plain ``dataclass`` instances with a ``to_dict()`` that returns
JSON-serializable primitives (numpy scalars/arrays are coerced). Validation is
intentionally lightweight (see ``validate_episode`` / ``validate_frame``):

  - required keys exist and are non-None,
  - ``visibility_score`` (and ``occlusion_risk``) are in ``[0, 1]`` if not None,
  - ``target_visible`` / boolean flags are real bools,
  - path fields are non-empty strings.

Poses/velocities are accepted as either ``[x, y, z, yaw]`` lists or
``{"x":..,"y":..,"z":..,"yaw":..}`` dicts — both round-trip through JSON.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

try:  # numpy is a hard dependency of the package, but keep schema importable.
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

Pose = Any   # list[float] | dict[str, float]
Vec = Any    # list[float] | dict[str, float]


class SchemaValidationError(ValueError):
    """Raised when a record fails schema validation."""


# --------------------------------------------------------------------------- #
# JSON coercion                                                               #
# --------------------------------------------------------------------------- #
def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses / numpy / paths to JSON primitives."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if _np is not None:
        if isinstance(obj, _np.ndarray):
            return [to_jsonable(v) for v in obj.tolist()]
        if isinstance(obj, _np.generic):
            return obj.item()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    # Fallback: stringify unknown objects (e.g. Path).
    return str(obj)


# --------------------------------------------------------------------------- #
# Camera config                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class CameraConfig:
    name: str = "front_custom"
    hfov_deg: float = 90.0
    pitch_deg: float = 55.0
    body_forward_m: float = 0.45
    body_down_m: float = 0.25
    width: int | None = None
    height: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


# --------------------------------------------------------------------------- #
# Episode metadata                                                            #
# --------------------------------------------------------------------------- #
REQUIRED_EPISODE_FIELDS: tuple[str, ...] = (
    "episode_id", "scene_id", "difficulty_level", "target_behavior_type",
    "target_class", "instruction", "random_seed", "max_steps", "camera_config",
    "uav_initial_pose", "target_initial_pose", "environment_summary",
    "occluder_summary",
)

# random_seed may legitimately be None (OS-entropy run); these keys are allowed
# to be None even though they are "required to be present".
_EPISODE_NULLABLE: frozenset[str] = frozenset({"random_seed"})


@dataclass
class EpisodeMetadata:
    episode_id: str
    scene_id: str
    difficulty_level: str
    target_behavior_type: str
    target_class: str
    instruction: str
    random_seed: int | None
    max_steps: int
    camera_config: CameraConfig | dict[str, Any]
    uav_initial_pose: Pose
    target_initial_pose: Pose
    environment_summary: dict[str, Any] = field(default_factory=dict)
    occluder_summary: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def validate(self) -> None:
        validate_episode(self.to_dict())


# --------------------------------------------------------------------------- #
# Frame metadata                                                              #
# --------------------------------------------------------------------------- #
REQUIRED_FRAME_FIELDS: tuple[str, ...] = (
    "frame_id", "image_path", "step_index", "timestamp", "uav_pose",
    "target_pose", "uav_velocity", "target_velocity", "target_visible",
    "in_camera_frustum", "line_of_sight_clear", "visibility_score",
    "distance_to_target", "relative_bearing", "occlusion_risk",
    "selected_viewpoint", "collision", "target_behavior_type",
    "difficulty_level",
)

# Keys allowed to be None while still "present". ``in_camera_frustum`` and
# ``line_of_sight_clear`` are nullable because they may be unseparable from the
# overall judgment (see VisibilityEvaluator fallback mode, Prompt 3).
_FRAME_NULLABLE: frozenset[str] = frozenset({
    "visibility_score", "occlusion_risk", "selected_viewpoint",
    "in_camera_frustum", "line_of_sight_clear",
})
# Strict booleans (never None).
_FRAME_BOOL_FIELDS: tuple[str, ...] = ("target_visible", "collision")
# Nullable booleans (bool or None).
_FRAME_NULLABLE_BOOL_FIELDS: tuple[str, ...] = (
    "in_camera_frustum", "line_of_sight_clear",
)
_FRAME_UNIT_FIELDS: tuple[str, ...] = ("visibility_score", "occlusion_risk")


@dataclass
class FrameMetadata:
    frame_id: int
    image_path: str
    step_index: int
    timestamp: float
    uav_pose: Pose
    target_pose: Pose
    uav_velocity: Vec
    target_velocity: Vec
    target_visible: bool
    in_camera_frustum: bool
    line_of_sight_clear: bool
    visibility_score: float | None
    distance_to_target: float
    relative_bearing: float
    occlusion_risk: float | None
    selected_viewpoint: Any | None
    collision: bool
    target_behavior_type: str
    difficulty_level: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def validate(self) -> None:
        validate_frame(self.to_dict())


# --------------------------------------------------------------------------- #
# Instruction / trajectory / metric records                                   #
# --------------------------------------------------------------------------- #
@dataclass
class InstructionRecord:
    episode_id: str
    instruction: str                       # primary / canonical instruction
    instructions: list[str] = field(default_factory=list)  # all kept candidates
    tier: str = ""
    appearance: str = ""
    behavior: dict[str, Any] = field(default_factory=dict)
    backend: str = ""
    rejected: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass
class TrajectoryRecord:
    """Compact per-frame pose/velocity sequence for one episode."""

    episode_id: str
    frames: list[dict[str, Any]] = field(default_factory=list)
    difficulty_level: str = ""
    target_behavior_type: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def add_frame(
        self,
        *,
        step_index: int,
        uav_pose: Pose,
        target_pose: Pose,
        uav_velocity: Vec | None = None,
        target_velocity: Vec | None = None,
        target_visible: bool | None = None,
    ) -> None:
        self.frames.append(to_jsonable({
            "step_index": step_index,
            "uav_pose": uav_pose,
            "target_pose": target_pose,
            "uav_velocity": uav_velocity,
            "target_velocity": target_velocity,
            "target_visible": target_visible,
        }))

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass
class MetricRecord:
    episode_id: str
    track_auc: float
    lost_rate: float
    redetection_time_s: float
    collision_rate: float
    fov_keep_rate: float
    frames: int = 0
    duration_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #
def _as_dict(record: Any) -> dict[str, Any]:
    if isinstance(record, dict):
        return record
    if hasattr(record, "to_dict"):
        return record.to_dict()
    if is_dataclass(record):
        return to_jsonable(record)
    raise SchemaValidationError(f"cannot validate non-mapping record: {type(record)!r}")


def _check_required(data: dict[str, Any], required: tuple[str, ...],
                    nullable: frozenset[str], kind: str) -> None:
    for key in required:
        if key not in data:
            raise SchemaValidationError(f"{kind}: missing required field {key!r}")
        if data[key] is None and key not in nullable:
            raise SchemaValidationError(f"{kind}: field {key!r} must not be None")


def _check_unit_interval(value: Any, name: str, kind: str) -> None:
    if value is None:
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SchemaValidationError(f"{kind}: {name} must be a number, got {type(value)!r}")
    if math.isnan(value) or not (0.0 <= float(value) <= 1.0):
        raise SchemaValidationError(f"{kind}: {name}={value} out of [0, 1]")


def _check_path(value: Any, name: str, kind: str, *, allow_empty: bool = False) -> None:
    if not isinstance(value, str):
        raise SchemaValidationError(f"{kind}: {name} must be a string path, got {type(value)!r}")
    if not allow_empty and not value.strip():
        raise SchemaValidationError(f"{kind}: {name} must be a non-empty string")


def validate_episode(record: Any) -> dict[str, Any]:
    """Validate an episode record (dict or dataclass); returns the dict form."""
    data = _as_dict(record)
    _check_required(data, REQUIRED_EPISODE_FIELDS, _EPISODE_NULLABLE, "episode")
    if not isinstance(data["episode_id"], str) or not data["episode_id"].strip():
        raise SchemaValidationError("episode: episode_id must be a non-empty string")
    if not isinstance(data["max_steps"], int) or isinstance(data["max_steps"], bool):
        raise SchemaValidationError("episode: max_steps must be an int")
    if data["max_steps"] < 0:
        raise SchemaValidationError("episode: max_steps must be >= 0")
    if data["random_seed"] is not None and (
        not isinstance(data["random_seed"], int) or isinstance(data["random_seed"], bool)
    ):
        raise SchemaValidationError("episode: random_seed must be int or None")
    return data


def validate_frame(record: Any) -> dict[str, Any]:
    """Validate a frame record (dict or dataclass); returns the dict form."""
    data = _as_dict(record)
    _check_required(data, REQUIRED_FRAME_FIELDS, _FRAME_NULLABLE, "frame")
    _check_path(data["image_path"], "image_path", "frame")
    for name in _FRAME_UNIT_FIELDS:
        _check_unit_interval(data.get(name), name, "frame")
    for name in _FRAME_BOOL_FIELDS:
        if not isinstance(data.get(name), bool):
            raise SchemaValidationError(
                f"frame: {name} must be a bool, got {type(data.get(name))!r}"
            )
    for name in _FRAME_NULLABLE_BOOL_FIELDS:
        val = data.get(name)
        if val is not None and not isinstance(val, bool):
            raise SchemaValidationError(
                f"frame: {name} must be a bool or None, got {type(val)!r}"
            )
    if not isinstance(data["frame_id"], int) or isinstance(data["frame_id"], bool):
        raise SchemaValidationError("frame: frame_id must be an int")
    return data


__all__ = [
    "Pose", "Vec", "SchemaValidationError",
    "CameraConfig", "EpisodeMetadata", "FrameMetadata",
    "InstructionRecord", "TrajectoryRecord", "MetricRecord",
    "REQUIRED_EPISODE_FIELDS", "REQUIRED_FRAME_FIELDS",
    "validate_episode", "validate_frame", "to_jsonable",
]

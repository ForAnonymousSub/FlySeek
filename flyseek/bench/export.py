# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Serialization helpers for FlySeek-Bench records.

All writers accept either a schema dataclass instance or a plain dict, coerce to
JSON primitives, optionally validate, and write to disk. Parent directories are
created automatically.

  - ``save_metadata_json``     -> ``<dir>/metadata.json``      (one episode)
  - ``append_frame_jsonl``     -> appends one line to ``frames.jsonl``
  - ``save_trajectories_json`` -> ``<dir>/trajectories.json``
  - ``save_instruction_json``  -> ``<dir>/instruction.json``
  - ``save_metrics_json``      -> ``<dir>/metrics.json``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flyseek.bench.schema import (
    SchemaValidationError,
    to_jsonable,
    validate_episode,
    validate_frame,
)


def _coerce(record: Any) -> dict[str, Any]:
    if hasattr(record, "to_dict"):
        return record.to_dict()
    return to_jsonable(record)


def _write_json(path: Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(data), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def save_metadata_json(
    metadata: Any,
    path: str | Path,
    *,
    validate: bool = True,
) -> Path:
    """Write one ``EpisodeMetadata`` (or dict) to ``path`` as JSON."""
    data = _coerce(metadata)
    if validate:
        validate_episode(data)
    return _write_json(Path(path), data)


def append_frame_jsonl(
    frame: Any,
    path: str | Path,
    *,
    validate: bool = True,
) -> Path:
    """Append one ``FrameMetadata`` (or dict) as a line to a JSONL file."""
    data = _coerce(frame)
    if validate:
        validate_frame(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(data), ensure_ascii=False) + "\n")
    return path


def save_trajectories_json(
    trajectory: Any,
    path: str | Path,
) -> Path:
    """Write a ``TrajectoryRecord`` (or dict / list of frames) to JSON."""
    return _write_json(Path(path), _coerce(trajectory))


def save_instruction_json(
    instruction: Any,
    path: str | Path,
) -> Path:
    """Write an ``InstructionRecord`` (or dict) to JSON."""
    data = _coerce(instruction)
    if isinstance(data, dict) and "episode_id" not in data:
        raise SchemaValidationError("instruction: missing required field 'episode_id'")
    return _write_json(Path(path), data)


def save_metrics_json(
    metric: Any,
    path: str | Path,
) -> Path:
    """Write a ``MetricRecord`` (or dict) to JSON."""
    data = _coerce(metric)
    if isinstance(data, dict) and "episode_id" not in data:
        raise SchemaValidationError("metrics: missing required field 'episode_id'")
    return _write_json(Path(path), data)


__all__ = [
    "save_metadata_json",
    "append_frame_jsonl",
    "save_trajectories_json",
    "save_instruction_json",
    "save_metrics_json",
]

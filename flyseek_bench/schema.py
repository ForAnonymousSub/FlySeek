# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""``flyseek_bench.schema`` — re-export of :mod:`flyseek.bench.schema`.

The canonical FlySeek-Bench record schemas live under ``flyseek.bench.schema``
(consistent with the package layout: ``flyseek.eval``, ``flyseek.instruction``,
``flyseek.pipeline``). This thin shim exposes them at the documented
``flyseek_bench.schema`` import path and is safe to import without a simulator.

Exposed record types:
    EpisodeMetadata, FrameMetadata, InstructionRecord, TrajectoryRecord,
    MetricRecord, CameraConfig.

Exposed validation:
    validate_episode, validate_frame, SchemaValidationError,
    REQUIRED_EPISODE_FIELDS, REQUIRED_FRAME_FIELDS, to_jsonable.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow ``import flyseek_bench.schema`` / file-path execution from any cwd by
# making ``flyseek_extend`` importable so ``flyseek.bench`` resolves.
_FLYSEEK_EXTEND = Path(__file__).resolve().parents[1]
if str(_FLYSEEK_EXTEND) not in sys.path:
    sys.path.insert(0, str(_FLYSEEK_EXTEND))

from flyseek.bench.schema import (  # noqa: E402,F401
    CameraConfig,
    EpisodeMetadata,
    FrameMetadata,
    InstructionRecord,
    MetricRecord,
    REQUIRED_EPISODE_FIELDS,
    REQUIRED_FRAME_FIELDS,
    SchemaValidationError,
    TrajectoryRecord,
    to_jsonable,
    validate_episode,
    validate_frame,
)

__all__ = [
    "CameraConfig",
    "EpisodeMetadata",
    "FrameMetadata",
    "InstructionRecord",
    "TrajectoryRecord",
    "MetricRecord",
    "SchemaValidationError",
    "REQUIRED_EPISODE_FIELDS",
    "REQUIRED_FRAME_FIELDS",
    "validate_episode",
    "validate_frame",
    "to_jsonable",
]

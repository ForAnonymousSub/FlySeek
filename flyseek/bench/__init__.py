# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""FlySeek-Bench schema and export layer.

A non-invasive, typed description of the FlySeek-Bench dataset (episode / frame /
instruction / trajectory / metric records) plus JSON/JSONL serialization helpers.

This package does NOT touch the generation pipeline; it only provides schemas and
writers that other code (or tests) may opt into.

Located at ``flyseek.bench`` to stay consistent with the existing package layout
(``flyseek.eval``, ``flyseek.instruction``, ``flyseek.pipeline``).
"""

from flyseek.bench.schema import (
    CameraConfig,
    EpisodeMetadata,
    FrameMetadata,
    InstructionRecord,
    MetricRecord,
    REQUIRED_EPISODE_FIELDS,
    REQUIRED_FRAME_FIELDS,
    SchemaValidationError,
    TrajectoryRecord,
    validate_episode,
    validate_frame,
)
from flyseek.bench.export import (
    append_frame_jsonl,
    save_instruction_json,
    save_metadata_json,
    save_metrics_json,
    save_trajectories_json,
)
from flyseek.bench.visibility import VisibilityEvaluator, evaluate_frame
from flyseek.bench.target_policy import (
    BEHAVIOR_TYPES,
    TargetPolicy,
    generate_target_waypoints,
)
from flyseek.bench.instruction_generator import (
    InstructionGenerator,
    attributes_from_label,
    generate_instruction,
    write_instruction_json,
)
from flyseek.bench.expert_trajectory import (
    ExpertTrajectoryConfig,
    ExpertViewpointPlanner,
    build_expert_trajectory_for_episode,
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
    "save_metadata_json",
    "append_frame_jsonl",
    "save_trajectories_json",
    "save_instruction_json",
    "save_metrics_json",
    "VisibilityEvaluator",
    "evaluate_frame",
    "TargetPolicy",
    "generate_target_waypoints",
    "BEHAVIOR_TYPES",
    "InstructionGenerator",
    "generate_instruction",
    "write_instruction_json",
    "attributes_from_label",
    "ExpertTrajectoryConfig",
    "ExpertViewpointPlanner",
    "build_expert_trajectory_for_episode",
]

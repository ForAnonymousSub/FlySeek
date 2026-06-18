# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""FlySeek expert / reference trajectory generator.

Per the locked scope, the FlySeek-Bench "expert" is the visibility-aware
:class:`~flyseek.expert.adaptive_tracker.AdaptiveTracker` (FSM modes
``track / predict / reacquire / peek / search / hold``). This module gives that
tracker a stable, named entry point so the pipeline, batch runner and any
downstream replay tooling depend on ``flyseek.expert.reference`` rather than on
demo-script internals.

The expert consumes the integrated adversarial ``TargetState`` each tick and
emits the next drone ``DroneState`` (visibility-aware viewpoint planning). The
8-D action it implicitly defines is documented in
:func:`expert_action_labels`.
"""

from __future__ import annotations

from typing import Any

from flyseek.expert.adaptive_tracker import AdaptiveTracker

EXPERT_NAME = "adaptive_fsm"


def build_expert(args: Any, *, occupancy: Any | None = None) -> AdaptiveTracker:
    """Return the configured FlySeek expert tracker.

    ``args`` is the demo / pipeline argument namespace (see
    ``AdaptiveTracker.from_args`` for the consumed fields); ``occupancy`` is an
    optional :class:`PcdOccupancyMap` enabling line-of-sight aware PEEK/SEARCH.
    """
    return AdaptiveTracker.from_args(args, occupancy=occupancy)


def expert_action_labels() -> list[str]:
    """Labels for the 8-D action vector recorded alongside each expert frame.

    Matches ``build_flyseek_meta_record``'s ``action_8d_labels`` so the expert
    trajectory in ``flyseek_meta.jsonl`` is self-describing.
    """
    return [
        "delta_x", "delta_y", "delta_z", "delta_yaw",
        "target_vx", "target_vy", "target_vz", "target_in_fov",
    ]


__all__ = ["EXPERT_NAME", "build_expert", "expert_action_labels"]

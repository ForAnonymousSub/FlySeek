# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""``flyseek_bench.target_policy`` — re-export of :mod:`flyseek.bench.target_policy`.

The canonical adversarial target policies live under
``flyseek.bench.target_policy`` (consistent with the package layout:
``flyseek.eval``, ``flyseek.instruction``, ``flyseek.pipeline``). This thin shim
exposes them at the documented ``flyseek_bench.target_policy`` import path and is
safe to import without a simulator.

The car target supports four adversarial behavior modes — ``direct_escape``,
``sharp_turn``, ``detour_feint``, ``occlusion_seeking`` — at three difficulty
tiers (``easy``/``medium``/``hard``). All motion is deterministic under a seed and
reuses the existing target-movement machinery (``integrate_target`` +
``stabilize_car_state`` + road-graph routing), so no movement code is rewritten.

Interfaces:
    TargetPolicy(config, scene_context, seed)
        .get_next_target_state(t, current_target_state, current_uav_state, history)
    generate_target_waypoints(initial_target_pose, initial_uav_pose,
                              behavior_type, difficulty, seed)   # waypoint envs
    create_target_policy(behavior_type, config, scene_context, seed)  # factory
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow ``import flyseek_bench.target_policy`` / file-path execution from any cwd
# by making ``flyseek_extend`` importable so ``flyseek.bench`` resolves.
_FLYSEEK_EXTEND = Path(__file__).resolve().parents[1]
if str(_FLYSEEK_EXTEND) not in sys.path:
    sys.path.insert(0, str(_FLYSEEK_EXTEND))

from flyseek.bench.target_policy import (  # noqa: E402,F401
    BEHAVIOR_ROUTE_MANEUVER,
    BEHAVIOR_TO_MANEUVER,
    BEHAVIOR_TYPES,
    DIFFICULTY_PRESETS,
    TargetPolicy,
    RouteFollowingTargetPolicy,
    create_target_policy,
    generate_target_waypoints,
)

__all__ = [
    "TargetPolicy",
    "RouteFollowingTargetPolicy",
    "create_target_policy",
    "generate_target_waypoints",
    "BEHAVIOR_TYPES",
    "BEHAVIOR_TO_MANEUVER",
    "BEHAVIOR_ROUTE_MANEUVER",
    "DIFFICULTY_PRESETS",
]

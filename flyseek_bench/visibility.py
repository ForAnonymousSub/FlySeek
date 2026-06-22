# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""``flyseek_bench.visibility`` — re-export of :mod:`flyseek.bench.visibility`.

The canonical ``VisibilityEvaluator`` lives under ``flyseek.bench.visibility``
(consistent with the package layout: ``flyseek.eval``, ``flyseek.instruction``,
``flyseek.pipeline``). This thin shim exposes it at the documented
``flyseek_bench.visibility`` import path and is safe to import without a simulator.

It converts the demo's existing view judgment into the paper-consistent fields:
``in_camera_frustum``, ``line_of_sight_clear``, ``target_visible``,
``visibility_score`` and ``occlusion_risk`` via::

    evaluate_frame(uav_pose, target_pose, camera_config, scene_context,
                   existing_visibility_metadata) -> dict

The evaluator prefers real geometry (pinhole frustum projection + PCD-raycast
LoS) when the inputs allow it, falls back to the recorded ``vis_reason`` when
geometry is unavailable, and finally to the binary recorded judgment (with a
one-time warning) — it never silently invents geometry. ``occlusion_risk`` is
``None`` unless future target positions and a PCD occupancy map are supplied.

This evaluator is already integrated into ``scripts/demo_adversary_chase.py``
(every frame is exported as a standardized ``FrameMetadata`` line in
``frames.jsonl``); this module only provides the documented import alias.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow ``import flyseek_bench.visibility`` / file-path execution from any cwd by
# making ``flyseek_extend`` importable so ``flyseek.bench`` resolves.
_FLYSEEK_EXTEND = Path(__file__).resolve().parents[1]
if str(_FLYSEEK_EXTEND) not in sys.path:
    sys.path.insert(0, str(_FLYSEEK_EXTEND))

from flyseek.bench.visibility import (  # noqa: E402,F401
    VisibilityEvaluator,
    evaluate_frame,
)

__all__ = ["VisibilityEvaluator", "evaluate_frame"]

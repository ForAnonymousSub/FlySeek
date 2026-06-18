# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Adversarial agent factory.

Per SKILL.md §2.2.::

    agent = create_adversarial_agent(
        difficulty="medium",
        config=load_yaml("configs/adversarial_agent.yaml"),
        play_box=PlayBox(x_min=-100, x_max=100, y_min=-100, y_max=100),
        seed=42,
    )
"""

from __future__ import annotations

from typing import Any

import numpy as np

from flyseek.adapters.pcd_occupancy import PcdOccupancyMap

from .base import AdversarialAgent, PlayBox
from .easy import RandomWalkAgent
from .hide_seek import HideSeekCarAgent
from .medium import SCurveEvasionAgent


# When config is a top-level dict with `easy:`, `medium:`, `hard:` sub-keys,
# pick the matching sub-dict. Otherwise treat as a flat config for this
# difficulty.
def _slice_config(config: dict[str, Any] | None, difficulty: str) -> dict[str, Any]:
    if not config:
        return {}
    if difficulty in config and isinstance(config[difficulty], dict):
        return dict(config[difficulty])
    return dict(config)


def create_adversarial_agent(
    difficulty: str,
    config: dict[str, Any] | None = None,
    *,
    play_box: PlayBox | None = None,
    seed: int | None = None,
    occupancy: PcdOccupancyMap | None = None,
) -> AdversarialAgent:
    """Build the right ``AdversarialAgent`` for the requested difficulty.

    Parameters
    ----------
    difficulty
        One of ``"easy"``, ``"medium"``. (``"hard"`` is Phase 2 — requires PCD
        scene_data; not implemented yet.)
    config
        Either the per-difficulty flat dict, or the umbrella dict that nests
        ``easy: {...}``, ``medium: {...}`` blocks.
    play_box
        Optional axis-aligned XY bound; the agent's velocity is reflected at
        the boundary so targets never leave the area.
    seed
        Optional RNG seed (for ``easy`` random walk reproducibility).
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    sub_cfg = _slice_config(config, difficulty)

    difficulty = difficulty.lower().strip()
    if difficulty == "easy":
        return RandomWalkAgent(config=sub_cfg, play_box=play_box, rng=rng)
    if difficulty == "medium":
        return SCurveEvasionAgent(config=sub_cfg, play_box=play_box, rng=rng)
    if difficulty == "hide_seek":
        return HideSeekCarAgent(
            config=sub_cfg, play_box=play_box, rng=rng, occupancy=occupancy,
        )
    if difficulty == "hard":
        raise NotImplementedError(
            "hard difficulty requires PCD scene_data + behavior tree (Phase 2). "
            "See SKILL.md §2.3."
        )
    raise ValueError(
        f"Unknown difficulty '{difficulty}'. "
        "Expected one of: easy, medium, hide_seek, hard."
    )


__all__ = ["create_adversarial_agent"]

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Easy-difficulty adversarial agent: smooth random walk.

Per SKILL.md §2.1:
    easy: 随机游走，速度 1-2 m/s，无主动躲避

A trivial baseline. The target ignores the drone entirely; its heading does
an Ornstein-Uhlenbeck-like random walk that produces a meandering path
without abrupt changes. Useful for sanity checks and for tracker training as
"non-evasive" negative samples.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .base import (
    AdversarialAgent,
    AgentAction,
    DroneState,
    PlayBox,
    TargetState,
    wrap_to_pi,
)


DEFAULTS: dict[str, Any] = {
    "speed_mps": 1.5,                # constant cruise speed
    "heading_noise_rad_per_sqrt_s": 0.4,   # diffusion scale
    "mean_revert_rate": 0.1,         # pulls heading back toward initial (1/s)
}


class RandomWalkAgent(AdversarialAgent):
    """Easy difficulty — heading-noise random walk, no drone awareness."""

    def __init__(self, config: dict | None = None,
                 play_box: PlayBox | None = None,
                 rng: np.random.Generator | None = None) -> None:
        cfg = {**DEFAULTS, **(config or {})}
        super().__init__(config=cfg, play_box=play_box, rng=rng)
        self._anchor_heading: float | None = None
        self._current_heading: float | None = None

    def reset(self, target_state: TargetState) -> None:
        super().reset(target_state)
        self._anchor_heading = target_state.heading
        self._current_heading = target_state.heading

    def _decide(self, drone: DroneState, target: TargetState,
                dt: float) -> AgentAction:
        if self._current_heading is None:
            self._current_heading = target.heading
            self._anchor_heading = target.heading

        # Mean-reverting Ornstein-Uhlenbeck-ish heading walk
        sigma = self.config["heading_noise_rad_per_sqrt_s"]
        k = self.config["mean_revert_rate"]
        anchor = self._anchor_heading if self._anchor_heading is not None else 0.0

        drift = -k * wrap_to_pi(self._current_heading - anchor) * dt
        diffusion = sigma * math.sqrt(dt) * float(self.rng.standard_normal())
        self._current_heading = wrap_to_pi(self._current_heading + drift + diffusion)

        speed = float(self.config["speed_mps"])
        vx = speed * math.cos(self._current_heading)
        vy = speed * math.sin(self._current_heading)

        return AgentAction(
            desired_velocity=np.array([vx, vy, 0.0]),
            desired_heading=self._current_heading,
            behavior_state="cruise",
            decision_log={"mode": "random_walk", "speed_mps": round(speed, 3)},
        )


__all__ = ["RandomWalkAgent", "DEFAULTS"]

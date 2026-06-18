# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Medium-difficulty adversarial agent: S-curve evasion.

Per SKILL.md §2.1:
    medium: 检测到无人机后 S 型规避，横向 ±2m，周期 2s，不利用掩体

Algorithm sketch (no PCD / no scene knowledge — those are "hard"):
    Let r = horizontal distance from target to drone.

    1. **cruise mode** (r > engagement_range_max, e.g. 40 m)
       Drone too far to bother; target keeps current heading at cruise_speed.
       No sway, no panic.

    2. **evade mode** (engagement_range_min ≤ r ≤ engagement_range_max)
       Base escape heading = drone-relative bearing + π (pointing directly away).
       Superimpose a sinusoidal sway:

           desired_heading = escape_heading
                           + sway_amplitude_rad * sin(2π * t_evade / sway_period_s)

       where t_evade is the time spent in evade mode since last transition.
       This produces a left-right S-curve away from the drone.

    3. **panic mode** (r < engagement_range_min, e.g. 5 m)
       Drone too close; bail straight away with no sway.

The actual change-of-heading per step is rate-limited by the *integrator*
(see ``integrate_target``), not by the agent itself. The agent only declares
intent; the integrator enforces kinematics.
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
    bearing_xy,
    horizontal_distance,
    wrap_to_pi,
)


DEFAULTS: dict[str, Any] = {
    "cruise_speed": 2.0,                # m/s, base speed in cruise mode
    "evade_speed": 3.0,                 # m/s, speed when evading
    "panic_speed": 4.0,                 # m/s, max speed when drone too close

    "engagement_range_min": 5.0,        # m, below this → panic
    "engagement_range_max": 40.0,       # m, above this → cruise

    "sway_amplitude_deg": 35.0,         # peak heading sway (±deg) in evade mode
    "sway_period_s": 2.0,               # seconds for one full sway cycle

    "cruise_heading_jitter_deg": 5.0,   # tiny random heading wobble in cruise
    "cruise_persistence_s": 4.0,        # how long to keep a chosen cruise direction
}


class SCurveEvasionAgent(AdversarialAgent):
    """Medium difficulty — adversarial S-curve evasion."""

    def __init__(self, config: dict | None = None,
                 play_box: PlayBox | None = None,
                 rng: np.random.Generator | None = None) -> None:
        cfg = {**DEFAULTS, **(config or {})}
        super().__init__(config=cfg, play_box=play_box, rng=rng)

        self._mode: str = "cruise"
        self._t_in_mode: float = 0.0

        # Cruise mode keeps a randomly-chosen heading for ``cruise_persistence_s``,
        # then picks a new one. This makes "easy parts" of the trajectory not
        # boringly straight.
        self._cruise_heading: float | None = None
        self._cruise_chosen_at: float = -1e9

    # ------------------------------------------------------------------
    def _decide(self, drone: DroneState, target: TargetState,
                dt: float) -> AgentAction:
        r = horizontal_distance(target.position, drone.position)

        # ----- mode transition --------------------------------------------------
        new_mode = self._classify_mode(r)
        if new_mode != self._mode:
            self._mode = new_mode
            self._t_in_mode = 0.0
        else:
            self._t_in_mode += dt

        # ----- per-mode action --------------------------------------------------
        if self._mode == "cruise":
            desired_heading, speed = self._cruise_action(target)
        elif self._mode == "evade":
            desired_heading, speed = self._evade_action(drone, target)
        else:  # "panic"
            desired_heading, speed = self._panic_action(drone, target)

        # ----- materialize velocity --------------------------------------------
        vx = speed * math.cos(desired_heading)
        vy = speed * math.sin(desired_heading)
        vz = 0.0  # ground-locked

        log = {
            "mode": self._mode,
            "t_in_mode_s": round(self._t_in_mode, 3),
            "drone_distance_m": round(r, 3),
            "drone_bearing_rad": round(bearing_xy(target.position, drone.position), 3),
            "desired_heading_rad": round(desired_heading, 3),
            "speed_mps": round(speed, 3),
        }
        return AgentAction(
            desired_velocity=np.array([vx, vy, vz]),
            desired_heading=desired_heading,
            behavior_state=self._mode,
            decision_log=log,
        )

    # ------------------------------------------------------------------
    def _classify_mode(self, r: float) -> str:
        if r < self.config["engagement_range_min"]:
            return "panic"
        if r > self.config["engagement_range_max"]:
            return "cruise"
        return "evade"

    def _cruise_action(self, target: TargetState) -> tuple[float, float]:
        # Pick a new cruise heading every cruise_persistence_s seconds
        persistence = self.config["cruise_persistence_s"]
        if (self._cruise_heading is None
                or (self._time - self._cruise_chosen_at) >= persistence):
            # First time: just use current heading. Later: small random walk.
            base = target.heading if self._cruise_heading is None else self._cruise_heading
            jitter = math.radians(self.config["cruise_heading_jitter_deg"])
            self._cruise_heading = wrap_to_pi(base + self.rng.uniform(-jitter, jitter))
            self._cruise_chosen_at = self._time
        return self._cruise_heading, float(self.config["cruise_speed"])

    def _evade_action(self, drone: DroneState,
                      target: TargetState) -> tuple[float, float]:
        # Direction away from drone (XY plane)
        bearing_to_drone = bearing_xy(target.position, drone.position)
        escape_heading = wrap_to_pi(bearing_to_drone + math.pi)

        # Sinusoidal sway, indexed by time spent in evade mode (smooth transition
        # from cruise — sway starts at 0)
        omega = 2.0 * math.pi / float(self.config["sway_period_s"])
        sway = (math.radians(self.config["sway_amplitude_deg"])
                * math.sin(omega * self._t_in_mode))
        desired_heading = wrap_to_pi(escape_heading + sway)

        return desired_heading, float(self.config["evade_speed"])

    def _panic_action(self, drone: DroneState,
                      target: TargetState) -> tuple[float, float]:
        # Straight away, no sway — fastest exit
        bearing_to_drone = bearing_xy(target.position, drone.position)
        escape_heading = wrap_to_pi(bearing_to_drone + math.pi)
        return escape_heading, float(self.config["panic_speed"])


__all__ = ["SCurveEvasionAgent", "DEFAULTS"]

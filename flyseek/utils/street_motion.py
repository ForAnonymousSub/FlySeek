# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Street-aligned ground motion helpers (PCD BEV, offline numpy)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from flyseek.adapters.pcd_occupancy import PcdOccupancyMap
from flyseek.adversary.base import AgentAction, TargetState, wrap_to_pi


def ray_free_distance_ned(
    occupancy: PcdOccupancyMap,
    pos_ned: np.ndarray,
    heading_rad: float,
    *,
    max_dist_m: float = 40.0,
    step_m: float = 1.5,
    keep_z: float | None = None,
) -> float:
    """How far we can walk along ``heading`` before hitting BEV occupancy."""
    pos = np.asarray(pos_ned, dtype=np.float64).reshape(3)
    z = float(pos[2] if keep_z is None else keep_z)
    dist = 0.0
    while dist < max_dist_m:
        trial = pos.copy()
        trial[0] += math.cos(heading_rad) * step_m
        trial[1] += math.sin(heading_rad) * step_m
        trial[2] = z
        if occupancy.is_bev_occupied_ned(trial):
            break
        pos = trial
        dist += step_m
    return dist


def pick_street_heading(
    occupancy: PcdOccupancyMap,
    pos_ned: np.ndarray,
    rng: np.random.Generator,
    *,
    hint_heading: float | None = None,
    keep_z: float | None = None,
    n_dirs: int = 16,
) -> float:
    """Choose a heading with the longest collision-free ray on the street grid."""
    best_h = hint_heading if hint_heading is not None else 0.0
    best_d = -1.0
    for k in range(n_dirs):
        h = 2.0 * math.pi * k / n_dirs
        d = ray_free_distance_ned(
            occupancy, pos_ned, h, keep_z=keep_z,
        )
        if d > best_d:
            best_d = d
            best_h = h
    if hint_heading is not None and best_d < 3.0:
        # Prefer staying near current street direction if all rays are short.
        best_h = hint_heading
    jitter = float(rng.normal(0.0, 0.12))
    return wrap_to_pi(best_h + jitter)


@dataclass
class StreetMotionHelper:
    """Maintains a wandering street heading with periodic random re-sampling."""

    occupancy: PcdOccupancyMap
    rng: np.random.Generator
    wander_heading: float = 0.0
    next_resample_t: float = 0.0
    resample_interval_s: tuple[float, float] = (2.5, 6.0)
    street_blend: float = 0.4

    def reset(self, target: TargetState) -> None:
        vxy = target.velocity[:2]
        hint = math.atan2(vxy[1], vxy[0]) if np.linalg.norm(vxy) > 0.2 else target.heading
        self.wander_heading = pick_street_heading(
            self.occupancy,
            target.position,
            self.rng,
            hint_heading=hint,
            keep_z=float(target.position[2]),
        )
        self.next_resample_t = float(
            self.rng.uniform(*self.resample_interval_s)
        )

    def update(self, t: float, target: TargetState) -> None:
        if t >= self.next_resample_t:
            self.wander_heading = pick_street_heading(
                self.occupancy,
                target.position,
                self.rng,
                hint_heading=self.wander_heading,
                keep_z=float(target.position[2]),
            )
            self.next_resample_t = t + float(
                self.rng.uniform(*self.resample_interval_s)
            )

    def bias_action(
        self,
        action: AgentAction,
        *,
        min_speed: float = 1.0,
    ) -> AgentAction:
        """Blend agent velocity toward the current street wander direction."""
        blend = float(np.clip(self.street_blend, 0.0, 1.0))
        v = action.desired_velocity.copy()
        speed = float(np.linalg.norm(v[:2]))
        if speed < min_speed:
            speed = min_speed
        street = speed * np.array([
            math.cos(self.wander_heading),
            math.sin(self.wander_heading),
            0.0,
        ])
        mixed = (1.0 - blend) * v + blend * street
        mixed[2] = 0.0
        heading = math.atan2(mixed[1], mixed[0]) if np.linalg.norm(mixed[:2]) > 1e-3 else self.wander_heading
        return AgentAction(
            desired_velocity=mixed,
            desired_heading=heading,
            behavior_state=action.behavior_state,
            decision_log={
                **action.decision_log,
                "street_heading_deg": round(math.degrees(self.wander_heading), 1),
                "street_blend": blend,
            },
        )


def stabilize_car_state(
    prev_pos: np.ndarray,
    state: TargetState,
    occupancy: PcdOccupancyMap | None,
    *,
    keep_z: float | None = None,
    max_turn_rate_rad_s: float = math.radians(45.0),
    dt: float,
) -> TargetState:
    """Keep the car on drivable ground with heading aligned to motion."""
    pos = state.position.copy()
    if keep_z is not None:
        pos[2] = keep_z

    if occupancy is not None:
        pos = occupancy.resolve_bev_move_ned(prev_pos, pos, keep_z=pos[2])
        # When a fixed ground plane is requested (keep_z), DON'T re-snap to the
        # local terrain max — otherwise the car climbs onto the tops of any
        # traversable elevated structure (rooftops). keep_z locks the road level.
        if keep_z is None:
            pos = occupancy.snap_car_to_ground_ned(pos)
        else:
            pos[2] = keep_z

    vel = (pos - prev_pos) / max(dt, 1e-6)
    vel[2] = 0.0
    speed_xy = float(np.linalg.norm(vel[:2]))

    if speed_xy > 0.15:
        desired_h = math.atan2(vel[1], vel[0])
    else:
        desired_h = state.heading

    delta = wrap_to_pi(desired_h - state.heading)
    max_step = max_turn_rate_rad_s * dt
    delta = float(np.clip(delta, -max_step, max_step))
    heading = wrap_to_pi(state.heading + delta)

    return TargetState(
        position=pos,
        velocity=vel,
        heading=heading,
        timestamp=state.timestamp,
    )


__all__ = [
    "StreetMotionHelper",
    "pick_street_heading",
    "ray_free_distance_ned",
    "stabilize_car_state",
]

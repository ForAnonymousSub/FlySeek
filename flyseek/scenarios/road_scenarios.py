# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Route-based target car scenarios for demo/data generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from flyseek.adapters.pcd_occupancy import PcdOccupancyMap
from flyseek.adversary.base import AgentAction, DroneState, TargetState, wrap_to_pi
from flyseek.utils.path_profile import SmoothPath, build_smooth_path
from flyseek.utils.road_graph import RoadRoute, build_route

@dataclass(frozen=True)
class RoadScenarioConfig:
    name: str = "normal_drive"
    route_len_m: float = 260.0
    waypoint_step_m: float = 6.0
    search_radius_m: float = 180.0
    speed_mps: float = 4.0
    high_speed_mps: float = 7.0
    turn_rate_deg_s: float = 70.0
    # Vehicle dynamics for spline + speed profile (Step 2 — "drive like a car").
    a_lat_max_mps2: float = 2.5    # cornering comfort limit
    a_long_max_mps2: float = 1.5   # accel from a stop
    brake_max_mps2: float = 2.5    # comfortable braking
    path_resample_m: float = 0.5   # arclength sample spacing on the spline
    open_road_frac: float = 0.68     # open_then_hide: main-road fraction before alley leg


class RoadScenarioController:
    """Pure-numpy route follower for a car-like target."""

    def __init__(
        self,
        occupancy: PcdOccupancyMap,
        initial_target: TargetState,
        rng: np.random.Generator,
        cfg: RoadScenarioConfig,
        *,
        route: RoadRoute | None = None,
    ) -> None:
        self.occupancy = occupancy
        self.rng = rng
        self.cfg = cfg
        self.keep_z = float(initial_target.position[2])
        if route is not None:
            self.route = route
        else:
            self.route = build_route(
                occupancy,
                initial_target.position,
                rng,
                keep_z=self.keep_z,
                route_len_m=cfg.route_len_m,
                waypoint_step_m=cfg.waypoint_step_m,
                search_radius_m=cfg.search_radius_m,
                maneuver=cfg.name,
                start_at_anchor=True,
                anchor_heading_rad=float(initial_target.heading),
            )
        # Smooth the polyline route into a C¹ spline + speed profile, so the
        # car curves through corners and slows before turns instead of cutting
        # straight-line chords at constant speed.
        self.smooth_path: SmoothPath = self._build_smooth_path(initial_target)
        # State: current arclength on the smooth path (replaces seg/dist).
        self._s = 0.0
        self._stuck_ticks = 0
        # Legacy fields kept for callers (route_progress_frac, _seg_idx).
        self._seg_idx = 0
        self._distance_in_seg = 0.0
        # P1 runtime hide correction (set by RouteFollowingTargetPolicy after reset).
        self._hide_goal: np.ndarray | None = None
        self._hide_vis_cfg: Any | None = None
        self._seg_map: Any | None = None
        # Initial pose comes from the spline at s=0 — but preserve the input
        # target's heading so the first teleport does not flip yaw by 90°.
        # The per-tick rate-limit (turn_rate_deg_s) will smoothly steer toward
        # the spline tangent from the input heading.
        init_xy, _ = self.smooth_path.pose_at(0.0)
        self._pos = np.array([init_xy[0], init_xy[1], self.keep_z],
                             dtype=np.float64)
        self._heading = float(initial_target.heading)
        self._last_state = TargetState(
            position=self._pos.copy(),
            velocity=np.zeros(3),
            heading=self._heading,
            timestamp=float(initial_target.timestamp),
        )

    # ------------------------------------------------------------------ #
    # Smooth path construction                                           #
    # ------------------------------------------------------------------ #
    def _build_smooth_path(self, initial_target: TargetState) -> SmoothPath:
        wps_xy = self.route.waypoints[:, :2]
        # Choose v_cap from the scenario's nominal speed; high-maneuver gets
        # the higher cap so sprints are visible. Spline + dynamics handle the
        # rest (slow into corners, accelerate out).
        if self.cfg.name == "high_maneuver":
            v_cap = max(self.cfg.high_speed_mps, self.cfg.speed_mps)
        elif self.cfg.name == "corner_occlude":
            v_cap = self.cfg.speed_mps * 0.85
        else:
            v_cap = self.cfg.speed_mps
        # v_start ≈ initial XY speed (so we don't crawl when the car is
        # already moving when the scenario starts).
        v_start = float(np.linalg.norm(initial_target.velocity[:2]))
        # v_end = 0 (car parks at end of route — looks like braking to a stop).
        return build_smooth_path(
            wps_xy,
            resample_step_m=self.cfg.path_resample_m,
            v_cap=v_cap,
            a_lat_max=self.cfg.a_lat_max_mps2,
            a_long_max=self.cfg.a_long_max_mps2,
            brake_max=self.cfg.brake_max_mps2,
            v_start=v_start,
            v_end=0.0,
        )

    def configure_hide_runtime(
        self,
        hide_goal: np.ndarray | None,
        hide_vis_cfg: Any | None,
        seg_map: Any | None = None,
    ) -> None:
        """Enable P1 per-tick hide nudge when the chase drone still sees the car."""
        self._hide_goal = (
            np.asarray(hide_goal, dtype=np.float64).reshape(3).copy()
            if hide_goal is not None else None
        )
        self._hide_vis_cfg = hide_vis_cfg
        self._seg_map = seg_map

    def _apply_runtime_hide_nudge(
        self,
        pos: np.ndarray,
        drone: DroneState,
    ) -> np.ndarray:
        from flyseek.utils.hide_visibility import HideVisibilityConfig, target_hidden_from_drone

        if self._hide_vis_cfg is None:
            return pos
        cfg = self._hide_vis_cfg
        if self._seg_map is not None and isinstance(cfg, HideVisibilityConfig):
            from dataclasses import replace
            cfg = replace(
                cfg, seg_building_map=self._seg_map, building_only_los=False,
            )
        cand = np.asarray(pos, dtype=np.float64).reshape(3).copy()
        cand[2] = self.keep_z
        tgt = TargetState(
            position=cand, velocity=np.zeros(3), heading=self._heading,
        )
        hidden, _ = target_hidden_from_drone(
            self.occupancy, drone, tgt, cfg,
        )
        if hidden:
            return pos

        out = cand.copy()
        if self._hide_goal is not None:
            delta = self._hide_goal[:2] - cand[:2]
            dist = float(np.linalg.norm(delta))
            if dist > 0.4:
                step = min(0.75, dist)
                out[0] += delta[0] / dist * step
                out[1] += delta[1] / dist * step
                out[2] = self.keep_z
                if hasattr(self.occupancy, "resolve_bev_move_ned"):
                    out = self.occupancy.resolve_bev_move_ned(
                        cand, out, keep_z=self.keep_z,
                    ).astype(np.float64)
                elif hasattr(self.occupancy, "is_drivable_ned"):
                    if not self.occupancy.is_drivable_ned(out):
                        return pos
        else:
            return pos

        tgt2 = TargetState(
            position=out, velocity=np.zeros(3), heading=self._heading,
        )
        hidden2, _ = target_hidden_from_drone(
            self.occupancy, drone, tgt2, cfg,
        )
        return out if hidden2 else pos

    def initial_state(self) -> TargetState:
        return self._last_state.copy_with()

    def step(
        self,
        drone: DroneState,
        t: float,
        dt: float,
    ) -> tuple[TargetState, AgentAction]:
        prev = self._last_state
        sp = self.smooth_path

        # 0. Snap arclength to the projection of the car's ACTUAL position
        # onto the spline. Without this, ``self._s`` and ``prev.position`` can
        # drift apart whenever ``resolve_bev_move_ned`` clamps a move, and the
        # next tick would compute a fake velocity spike.
        self._s = sp.project_s(prev.position[:2], hint_s=self._s)

        # 1. Advance arclength via the precomputed speed profile.
        speed = sp.speed_at(self._s)
        # Behavior-specific tweak: corner_occlude scenario slows further
        # right before the occluder; high_maneuver bursts above v_cap briefly.
        speed = self._scenario_speed_modifier(speed, t)
        self._s = min(sp.total_length_m, self._s + speed * dt)
        xy, h_des = sp.pose_at(self._s)
        desired_pos = np.array([float(xy[0]), float(xy[1]), self.keep_z],
                               dtype=np.float64)

        # 2. Clamp against PCD obstacles (Step 1 guardrail-tunnel guard).
        if hasattr(self.occupancy, "resolve_bev_move_ned"):
            pos = self.occupancy.resolve_bev_move_ned(
                prev.position, desired_pos, keep_z=self.keep_z
            ).astype(np.float64)
        else:
            pos = desired_pos.astype(np.float64)
        pos[2] = self.keep_z

        # P1: during hide leg, nudge toward verified hide goal if still visible.
        if self.cfg.name == "open_then_hide" and self._hide_vis_cfg is not None:
            seg_frac = self.route_progress_frac
            if seg_frac >= float(self.cfg.open_road_frac):
                pos = self._apply_runtime_hide_nudge(pos, drone)

        # 3. Stuck-detector — if the clamp killed movement for several ticks
        # in a row, jump arclength ahead so we exit the stalled segment.
        desired_move = float(np.linalg.norm(desired_pos[:2] - prev.position[:2]))
        actual_move = float(np.linalg.norm(pos[:2] - prev.position[:2]))
        if desired_move > 1e-3 and actual_move < 0.25 * desired_move:
            self._stuck_ticks += 1
        else:
            self._stuck_ticks = 0
        if self._stuck_ticks >= 6:
            # Tiny nudge — just enough to escape a single-voxel pothole
            # without a visible position jump (0.5 m at 20 Hz ≈ 10 m/s
            # instantaneous, vs the 30 m/s we had with a 1.5 m jump).
            self._s = min(sp.total_length_m,
                          self._s + self.cfg.path_resample_m)
            self._stuck_ticks = 0

        # 4. Heading: derived from spline tangent + bounded rate (steering).
        max_turn = math.radians(self.cfg.turn_rate_deg_s) * dt
        delta = wrap_to_pi(float(h_des) - prev.heading)
        heading = wrap_to_pi(prev.heading + float(np.clip(delta, -max_turn, max_turn)))

        # 5. Update legacy seg_idx (used by _behavior_name + external callers).
        self._seg_idx = self._segment_index_at(self._s)
        # Bookkeeping (unused but kept for compatibility).
        self._pos = pos.copy()
        self._heading = float(heading)

        vel = (pos - prev.position) / max(dt, 1e-6)
        vel[2] = 0.0

        state = TargetState(
            position=pos,
            velocity=vel,
            heading=heading,
            timestamp=t + dt,
        )
        self._last_state = state
        behavior = self._behavior_name(t)
        action = AgentAction(
            desired_velocity=vel,
            desired_heading=heading,
            behavior_state=behavior,
            decision_log={
                "road_scenario": self.cfg.name,
                "route_seg": int(self._seg_idx),
                "speed_mps": round(speed, 2),
                "route_score": round(self.route.score, 2),
            },
        )
        return state, action

    @property
    def route_progress_frac(self) -> float:
        n = max(1, len(self.route.waypoints) - 1)
        return float(self._seg_idx) / n

    def _scenario_speed_modifier(self, base_speed: float, t: float) -> float:
        """Apply scenario-specific multiplier on top of the dynamic profile.

        The profile already encodes physics (curvature, accel, brake); this
        layer adds a slow temporal jitter so the car doesn't look robotic on
        long straights.
        """
        if self.cfg.name == "open_then_hide":
            jitter = 1.0 + 0.04 * math.sin(0.35 * t)
        elif self.cfg.name == "normal_drive":
            jitter = 1.0 + 0.04 * math.sin(0.4 * t)
        elif self.cfg.name == "corner_occlude":
            jitter = 0.9 + 0.1 * math.sin(0.25 * t)
        elif self.cfg.name == "high_maneuver":
            jitter = 0.85 + 0.15 * math.sin(1.3 * t)
        else:
            jitter = 1.0
        return float(base_speed * jitter)

    def _segment_index_at(self, s: float) -> int:
        """Approximate the original waypoint-segment index at arclength ``s``.

        Kept so that ``route_progress_frac`` / ``_behavior_name`` semantics
        stay backwards-compatible with the polyline implementation.
        """
        n_seg = max(1, len(self.route.waypoints) - 1)
        if self.smooth_path.total_length_m <= 1e-6:
            return 0
        frac = s / self.smooth_path.total_length_m
        return int(np.clip(frac * n_seg, 0, n_seg - 1))

    def _behavior_name(self, t: float) -> str:
        if self.cfg.name == "open_then_hide":
            if self.route_progress_frac >= float(self.cfg.open_road_frac):
                return "approach_alley"
            return "open_road"
        if self.cfg.name == "normal_drive":
            return "normal_drive"
        if self.cfg.name == "corner_occlude":
            frac = self._seg_idx / max(1, len(self.route.waypoints) - 1)
            if 0.35 <= frac <= 0.55:
                return "corner_occlude"
            return "normal_drive"
        return "high_maneuver"


__all__ = ["RoadScenarioController", "RoadScenarioConfig"]

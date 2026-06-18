# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Adversarial target policies for the car target (paper §3 adversarial behavior).

Four deterministic behavior modes, all built on top of the *existing* target
movement machinery (no rewrite):

  - ``direct_escape``      : drive away from the UAV along a feasible direction.
  - ``sharp_turn``         : normal cruise with sudden turns at trigger points.
  - ``detour_feint``       : feint one way, then change route to mislead.
  - ``occlusion_seeking``  : prefer directions toward cover / lower visibility.

Each mode emits an :class:`AgentAction` (same contract the adversary agents use),
which is integrated with :func:`flyseek.adversary.base.integrate_target` and then
snapped to drivable ground with
:func:`flyseek.utils.street_motion.stabilize_car_state` — exactly the pipeline
the demo already uses. Feasible-direction selection reuses
``street_motion.pick_street_heading`` / ``ray_free_distance_ned`` and occlusion
selection reuses ``PcdOccupancyMap.find_hide_goal_ned`` / ``los_blocked_ned``.

Determinism: every policy draws only from ``np.random.default_rng(seed)`` which is
re-created on ``reset()``, so a fixed seed reproduces motion exactly.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from flyseek.adversary.base import (
    AgentAction,
    DroneState,
    TargetState,
    bearing_xy,
    horizontal_distance,
    integrate_target,
    wrap_to_pi,
)
from flyseek.scenarios.road_scenarios import RoadScenarioConfig, RoadScenarioController
from flyseek.utils.alley_route import build_alley_hutong_route
from flyseek.utils.annotated_hide_route import build_annotated_hide_route
from flyseek.utils.hide_visibility import HideVisibilityConfig
from flyseek.utils.occlusion_route import (
    BEHAVIOR_ROUTE_MANEUVER,
    build_occlusion_seeking_route,
)
from flyseek.utils.seg_buildings import SegBuildingMap
from flyseek.utils.street_motion import (
    pick_street_heading,
    ray_free_distance_ned,
    stabilize_car_state,
)

BEHAVIOR_TYPES = (
    "direct_escape", "sharp_turn", "detour_feint", "occlusion_seeking",
    "alley_hutong",
)

# Behavior -> road_graph maneuver tag (for waypoint generation that wraps
# build_route). Keeps the offline waypoint path consistent with the live policy.
BEHAVIOR_TO_MANEUVER = dict(BEHAVIOR_ROUTE_MANEUVER)

DIFFICULTY_PRESETS: dict[str, dict[str, float]] = {
    "easy": {
        "speed_mps": 2.0,
        "max_turn_rate_deg_s": 45.0,
        "sharp_turn_interval_s": 14.0,
        "sharp_turn_deg": 40.0,
        "detour_duration_s": 7.0,
        "detour_offset_deg": 50.0,
        "occlusion_weight": 0.3,
        "occlusion_lookahead_m": 12.0,
        "evade_gain": 1.15,
        "engage_range_m": 18.0,
        "route_len_m": 100.0,
    },
    "medium": {
        "speed_mps": 3.0,
        "max_turn_rate_deg_s": 90.0,
        "sharp_turn_interval_s": 9.0,
        "sharp_turn_deg": 75.0,
        "detour_duration_s": 5.0,
        "detour_offset_deg": 65.0,
        "occlusion_weight": 0.6,
        "occlusion_lookahead_m": 16.0,
        "evade_gain": 1.3,
        "engage_range_m": 30.0,
        "route_len_m": 140.0,
    },
    "hard": {
        "speed_mps": 4.0,
        "max_turn_rate_deg_s": 120.0,
        "sharp_turn_interval_s": 5.0,
        "sharp_turn_deg": 110.0,
        "detour_duration_s": 4.0,
        "detour_offset_deg": 80.0,
        "occlusion_weight": 1.0,
        "occlusion_lookahead_m": 22.0,
        "evade_gain": 1.5,
        "engage_range_m": 45.0,
        "route_len_m": 180.0,
    },
}


def _parse_pose(pose: Any) -> tuple[np.ndarray, float | None]:
    if isinstance(pose, (DroneState, TargetState)):
        return pose.position.copy(), float(pose.heading)
    if isinstance(pose, dict):
        if "pos" in pose:
            p = np.asarray(pose["pos"], dtype=np.float64).reshape(3)
        else:
            p = np.array([float(pose["x"]), float(pose["y"]), float(pose["z"])])
        yaw = pose.get("yaw", pose.get("heading"))
        return p, (float(yaw) if yaw is not None else None)
    arr = np.asarray(pose, dtype=np.float64).reshape(-1)
    return arr[:3].copy(), (float(arr[3]) if arr.size >= 4 else None)


class TargetPolicy:
    """Common adversarial target-policy interface.

    ``config`` keys: ``behavior_type`` (one of :data:`BEHAVIOR_TYPES`),
    ``difficulty`` (``easy``/``medium``/``hard``), optional ``dt`` and any
    difficulty-parameter overrides.

    ``scene_context`` keys (all optional): ``occupancy`` (a ``PcdOccupancyMap``),
    ``keep_z`` (fixed NED ground z; else follow local ground), ``drone_eye_agl_m``.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        scene_context: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        config = dict(config or {})
        scene_context = dict(scene_context or {})

        self.behavior_type = str(config.get("behavior_type", "direct_escape"))
        if self.behavior_type not in BEHAVIOR_TYPES:
            raise ValueError(
                f"unknown behavior_type {self.behavior_type!r}; "
                f"expected one of {BEHAVIOR_TYPES}"
            )
        self.difficulty = str(config.get("difficulty", "medium"))
        params = dict(DIFFICULTY_PRESETS.get(self.difficulty, DIFFICULTY_PRESETS["medium"]))
        for key in params:  # allow per-key overrides from config
            if key in config:
                params[key] = config[key]
        self.params = params

        self.occupancy = scene_context.get("occupancy")
        self.keep_z = scene_context.get("keep_z")
        self.drone_eye_agl_m = float(scene_context.get("drone_eye_agl_m", 12.0))
        self.dt = float(config.get("dt", 0.1))
        self.max_speed = float(config.get("max_speed_mps", params["speed_mps"] * 1.6))
        self.max_turn_rate = math.radians(float(params["max_turn_rate_deg_s"]))
        self._n_dirs = 24

        self._seed = seed
        self._dispatch = {
            "direct_escape": self._decide_direct_escape,
            "sharp_turn": self._decide_sharp_turn,
            "detour_feint": self._decide_detour_feint,
            "occlusion_seeking": self._decide_occlusion_seeking,
        }
        self.last_action: AgentAction | None = None
        self.reset_state(initial_heading=0.0)

    # ------------------------------------------------------------------ #
    def reset_state(self, *, initial_heading: float = 0.0) -> None:
        """Re-seed the RNG and clear phase timers (deterministic restart)."""
        self.rng = np.random.default_rng(self._seed)
        self._prev_t: float | None = None
        self._mdir = float(initial_heading)
        self._last_refresh = -1e9
        self._next_sharp_t = self._sharp_interval()
        self._feint_sign = 1.0 if self.rng.random() < 0.5 else -1.0

    def reset(self, target_state: TargetState,
              uav_state: DroneState | None = None) -> None:
        self.reset_state(initial_heading=float(target_state.heading))

    # ------------------------------------------------------------------ #
    def get_next_target_state(
        self,
        t: float,
        current_target_state: TargetState,
        current_uav_state: DroneState,
        history: list[Any] | None = None,
    ) -> TargetState:
        """Advance the target by one step and return the new ``TargetState``."""
        if self._prev_t is not None and t > self._prev_t:
            dt = float(t - self._prev_t)
        else:
            dt = self.dt
        self._prev_t = float(t)

        action = self._dispatch[self.behavior_type](
            t, current_target_state, current_uav_state, history,
        )
        self.last_action = action

        new_state = integrate_target(
            current_target_state, action, dt,
            keep_z=self.keep_z,
            max_speed=self.max_speed,
            max_turn_rate_rad_s=self.max_turn_rate,
        )
        if self.occupancy is not None:
            new_state = stabilize_car_state(
                current_target_state.position, new_state, self.occupancy,
                keep_z=self.keep_z,
                max_turn_rate_rad_s=self.max_turn_rate * 0.5,
                dt=dt,
            )
        return new_state.copy_with(timestamp=float(t))

    # ------------------------------------------------------------------ #
    # Shared helpers                                                      #
    # ------------------------------------------------------------------ #
    def _keep_z(self, target: TargetState) -> float:
        return float(self.keep_z) if self.keep_z is not None else float(target.position[2])

    def _vel(self, heading: float, speed: float) -> np.ndarray:
        return np.array([speed * math.cos(heading), speed * math.sin(heading), 0.0])

    def _feasible_heading(self, target: TargetState, desired: float) -> float:
        """Snap ``desired`` to the nearest drivable street heading (or keep it)."""
        if self.occupancy is None:
            return desired
        try:
            return float(pick_street_heading(
                self.occupancy, target.position, self.rng,
                hint_heading=desired, keep_z=self._keep_z(target),
            ))
        except Exception:
            return desired

    def _sharp_interval(self) -> float:
        base = float(self.params["sharp_turn_interval_s"])
        return base * float(self.rng.uniform(0.8, 1.2))

    # ------------------------------------------------------------------ #
    # Behavior modes                                                      #
    # ------------------------------------------------------------------ #
    def _decide_direct_escape(self, t, target, uav, history) -> AgentAction:
        away = bearing_xy(uav.position, target.position)  # points away from UAV
        heading = self._feasible_heading(target, away)
        r = horizontal_distance(target.position, uav.position)
        speed = float(self.params["speed_mps"])
        if r < float(self.params["engage_range_m"]):
            speed = min(self.max_speed, speed * float(self.params["evade_gain"]))
        self._mdir = heading
        return AgentAction(
            self._vel(heading, speed), heading, "direct_escape",
            {"behavior": "direct_escape", "range_m": round(r, 2),
             "heading_deg": round(math.degrees(heading), 1), "speed_mps": round(speed, 2)},
        )

    def _decide_sharp_turn(self, t, target, uav, history) -> AgentAction:
        behavior = "cruise"
        # Periodically re-snap the cruise heading onto a drivable street so the
        # car keeps following roads between sharp turns.
        if self.occupancy is not None and (t - self._last_refresh) > 2.0:
            self._mdir = self._feasible_heading(target, self._mdir)
            self._last_refresh = t
        if t >= self._next_sharp_t:
            sign = 1.0 if self.rng.random() < 0.5 else -1.0
            turn = sign * math.radians(float(self.params["sharp_turn_deg"]))
            self._mdir = self._feasible_heading(target, wrap_to_pi(self._mdir + turn))
            self._next_sharp_t = t + self._sharp_interval()
            behavior = "sharp_turn"
        heading = self._mdir
        speed = float(self.params["speed_mps"])
        return AgentAction(
            self._vel(heading, speed), heading, behavior,
            {"behavior": behavior, "next_turn_s": round(self._next_sharp_t, 2),
             "heading_deg": round(math.degrees(heading), 1)},
        )

    def _decide_detour_feint(self, t, target, uav, history) -> AgentAction:
        phase_len = max(float(self.params["detour_duration_s"]), 1e-3)
        phase = int(t / phase_len)
        away = bearing_xy(uav.position, target.position)
        offset = math.radians(float(self.params["detour_offset_deg"]))
        if phase % 2 == 0:
            # Feint: veer off the true escape direction to suggest a route.
            desired = away + offset * self._feint_sign
            behavior = "feint"
        else:
            # Commit: swing the other way to a genuinely different route.
            desired = away - offset * self._feint_sign * 0.6
            behavior = "commit"
        heading = self._feasible_heading(target, desired)
        self._mdir = heading
        speed = float(self.params["speed_mps"])
        return AgentAction(
            self._vel(heading, speed), heading, f"detour_{behavior}",
            {"behavior": f"detour_{behavior}", "phase": phase,
             "heading_deg": round(math.degrees(heading), 1)},
        )

    def _decide_occlusion_seeking(self, t, target, uav, history) -> AgentAction:
        keep_z = self._keep_z(target)
        desired = bearing_xy(uav.position, target.position)
        behavior = "occlusion_seeking"
        if self.occupancy is not None:
            goal = None
            try:
                goal = self.occupancy.find_hide_goal_ned(
                    target.position, uav.position, keep_z=keep_z,
                    search_radius_m=float(self.params["occlusion_lookahead_m"]) * 1.5,
                )
            except Exception:
                goal = None
            if goal is not None:
                desired = bearing_xy(target.position, np.asarray(goal, dtype=np.float64))
                behavior = "hide_goal"
            else:
                desired = self._best_occluding_heading(target, uav, keep_z)
        heading = self._feasible_heading(target, desired)
        self._mdir = heading
        speed = float(self.params["speed_mps"]) * 0.9
        return AgentAction(
            self._vel(heading, speed), heading, behavior,
            {"behavior": behavior, "heading_deg": round(math.degrees(heading), 1)},
        )

    def _best_occluding_heading(self, target, uav, keep_z) -> float:
        """Pick a drivable heading whose short lookahead breaks UAV line of sight."""
        la = float(self.params["occlusion_lookahead_m"])
        w = float(self.params["occlusion_weight"])
        away = bearing_xy(uav.position, target.position)
        best_h = away
        best_score = -1e9
        for k in range(self._n_dirs):
            h = -math.pi + 2.0 * math.pi * k / self._n_dirs
            free = ray_free_distance_ned(
                self.occupancy, target.position, h, max_dist_m=la, keep_z=keep_z,
            )
            if free < 2.0:
                continue  # not enough drivable road this way
            look = target.position[:2] + min(free, la) * np.array([
                math.cos(h), math.sin(h)])
            look3 = np.array([look[0], look[1], keep_z], dtype=np.float64)
            try:
                blocked = self.occupancy.los_blocked_ned(
                    uav.position, look3,
                    drone_eye_agl_m=self.drone_eye_agl_m,
                    target_agl_m=max(0.5, -keep_z),
                )
            except Exception:
                blocked = False
            align = math.cos(wrap_to_pi(h - away))
            score = (free / la) + w * (1.0 if blocked else 0.0) + 0.2 * align
            if score > best_score:
                best_score = score
                best_h = h
        return best_h


class RouteFollowingTargetPolicy:
    """Route-following target policy (spline + PCD collision clamp).

    Used for ``occlusion_seeking`` when a PCD occupancy map is available.
    Motion is delegated to :class:`RoadScenarioController` — the same car
    dynamics pipeline as UE demos and ``HideSeekCarAgent`` open-road phase.
    """

    behavior_type = "occlusion_seeking"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        scene_context: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        config = dict(config or {})
        scene_context = dict(scene_context or {})
        self.difficulty = str(config.get("difficulty", "medium"))
        params = dict(DIFFICULTY_PRESETS.get(self.difficulty, DIFFICULTY_PRESETS["medium"]))
        for key in params:
            if key in config:
                params[key] = config[key]
        self.params = params
        self.occupancy = scene_context.get("occupancy")
        if self.occupancy is None:
            raise ValueError("RouteFollowingTargetPolicy requires scene_context['occupancy']")
        self.drone_eye_agl_m = float(scene_context.get("drone_eye_agl_m", 12.0))
        self.follow_distance_m = float(scene_context.get("follow_distance_m", 12.0))
        self.search_radius_m = float(config.get("search_radius_m", 220.0))
        self._occlusion_kw = {
            k: config[k] for k in (
                "min_building_height_m", "min_footprint_cells",
                "hide_search_radius_m", "max_attempts",
                "min_building_occluded_frac", "require_adjacent_building",
                "building_probe_dist_m", "open_road_frac",
                "hide_vis_config", "follow_distance_m", "min_frustum_hidden_frac",
            ) if k in config
        }
        raw_vis = config.get("hide_vis_config")
        if raw_vis is not None:
            self._hide_vis_cfg = (
                raw_vis if isinstance(raw_vis, HideVisibilityConfig)
                else HideVisibilityConfig(**dict(raw_vis))
            )
        else:
            self._hide_vis_cfg = HideVisibilityConfig(
                drone_eye_agl_m=self.drone_eye_agl_m,
                follow_distance_m=self.follow_distance_m,
                min_building_height_m=config.get("min_building_height_m"),
                min_footprint_cells=int(config.get("min_footprint_cells", 9)),
                building_only_los=True,
                use_frustum_projection=True,
            )
        if config.get("route_len_m"):
            self.params["route_len_m"] = float(config["route_len_m"])
        self._seg_map: SegBuildingMap | None = None
        seg_path = config.get("seg_building_jsonl") or scene_context.get("seg_building_jsonl")
        if seg_path:
            self._seg_map = SegBuildingMap.from_jsonl(
                seg_path,
                footprint_radius_m=float(config.get("seg_building_radius_m", 10.0)),
                min_occluder_height_m=float(config.get("seg_building_min_height_m", 8.0)),
            )
            from dataclasses import replace
            self._hide_vis_cfg = replace(
                self._hide_vis_cfg,
                seg_building_map=self._seg_map,
                building_only_los=False,
                occluder_between_required=True,
            )
        self._seed = seed
        self.dt = float(config.get("dt", 0.1))
        self._rng = np.random.default_rng(seed)
        self._ctl: RoadScenarioController | None = None
        self._prev_t: float | None = None
        self.last_action: AgentAction | None = None
        self.route_meta: dict[str, Any] = {}

    def reset(self, target_state: TargetState,
              uav_state: DroneState | None = None) -> None:
        self._rng = np.random.default_rng(self._seed)
        self._prev_t = None
        keep_z = float(target_state.position[2])
        drone_ned = (
            uav_state.position.copy() if uav_state is not None
            else self._estimate_chase_drone_ned(target_state)
        )
        occ_kw = dict(self._occlusion_kw)
        occ_kw.pop("hide_vis_config", None)
        if self._seg_map is not None:
            route, meta = build_annotated_hide_route(
                self.occupancy,
                self._seg_map,
                target_state.position,
                drone_ned,
                self._rng,
                keep_z=keep_z,
                route_len_m=float(self.params["route_len_m"]),
                search_radius_m=self.search_radius_m,
                hide_vis_config=self._hide_vis_cfg,
                follow_distance_m=self.follow_distance_m,
                open_road_frac=float(occ_kw.get("open_road_frac", 0.4)),
                hide_search_radius_m=float(occ_kw.get("hide_search_radius_m", 80.0)),
                max_attempts=int(occ_kw.get("max_attempts", 12)),
            )
            if route is None:
                warnings.warn(
                    "[RouteFollowingTargetPolicy] annotated hide route failed; "
                    "falling back to PCD occlusion route.",
                    stacklevel=2,
                )
                route, meta = build_occlusion_seeking_route(
                    self.occupancy,
                    target_state.position,
                    self._rng,
                    keep_z=keep_z,
                    drone_ned=drone_ned,
                    route_len_m=float(self.params["route_len_m"]),
                    search_radius_m=self.search_radius_m,
                    anchor_heading_rad=float(target_state.heading),
                    drone_eye_agl_m=self.drone_eye_agl_m,
                    hide_vis_config=self._hide_vis_cfg,
                    follow_distance_m=self.follow_distance_m,
                    **occ_kw,
                )
        else:
            route, meta = build_occlusion_seeking_route(
                self.occupancy,
                target_state.position,
                self._rng,
                keep_z=keep_z,
                drone_ned=drone_ned,
                route_len_m=float(self.params["route_len_m"]),
                search_radius_m=self.search_radius_m,
                anchor_heading_rad=float(target_state.heading),
                drone_eye_agl_m=self.drone_eye_agl_m,
                hide_vis_config=self._hide_vis_cfg,
                follow_distance_m=self.follow_distance_m,
                **occ_kw,
            )
        self.route_meta = meta
        hide_goal = meta.get("hide_goal")
        open_frac = float(self._occlusion_kw.get("open_road_frac", 0.45))
        if self._seg_map is not None and meta.get("planner") == "annotated_seg_buildings":
            open_frac = float(meta.get("open_road_frac", open_frac))
        scen_cfg = RoadScenarioConfig(
            name="open_then_hide",
            route_len_m=float(self.params["route_len_m"]),
            speed_mps=float(self.params["speed_mps"]) * 0.9,
            high_speed_mps=float(self.params["speed_mps"]) * 1.2,
            open_road_frac=open_frac,
        )
        self._ctl = RoadScenarioController(
            self.occupancy,
            target_state,
            self._rng,
            scen_cfg,
            route=route,
        )
        self._ctl.configure_hide_runtime(
            hide_goal, self._hide_vis_cfg, seg_map=self._seg_map,
        )
        self.last_action = None

    def _estimate_chase_drone_ned(self, target: TargetState) -> np.ndarray:
        back = float(target.heading) + math.pi
        fd = self.follow_distance_m
        fa = self.drone_eye_agl_m
        return target.position + np.array([
            math.cos(back) * fd,
            math.sin(back) * fd,
            -abs(fa),
        ], dtype=np.float64)

    def get_next_target_state(
        self,
        t: float,
        current_target_state: TargetState,
        current_uav_state: DroneState,
        history: list[Any] | None = None,
    ) -> TargetState:
        if self._ctl is None:
            raise RuntimeError("RouteFollowingTargetPolicy.reset() must be called first")
        if self._prev_t is not None and t > self._prev_t:
            dt = float(t - self._prev_t)
        else:
            dt = self.dt
        self._prev_t = float(t)
        new_state, action = self._ctl.step(current_uav_state, t, dt)
        self.last_action = action
        return new_state.copy_with(timestamp=float(t))


class AlleyHutongTargetPolicy:
    """Route-following policy: drive into a narrow hutong between buildings."""

    behavior_type = "alley_hutong"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        scene_context: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        config = dict(config or {})
        scene_context = dict(scene_context or {})
        self.difficulty = str(config.get("difficulty", "medium"))
        params = dict(DIFFICULTY_PRESETS.get(self.difficulty, DIFFICULTY_PRESETS["medium"]))
        for key in params:
            if key in config:
                params[key] = config[key]
        self.params = params
        self.occupancy = scene_context.get("occupancy")
        if self.occupancy is None:
            raise ValueError("AlleyHutongTargetPolicy requires scene_context['occupancy']")
        seg_path = config.get("seg_building_jsonl") or scene_context.get("seg_building_jsonl")
        if not seg_path:
            raise ValueError(
                "AlleyHutongTargetPolicy requires seg_building_jsonl (annotated buildings)"
            )
        self._seg_map = SegBuildingMap.from_jsonl(
            seg_path,
            footprint_radius_m=float(config.get("seg_building_radius_m", 10.0)),
            min_occluder_height_m=float(config.get("seg_building_min_height_m", 8.0)),
        )
        self.search_radius_m = float(config.get("search_radius_m", 220.0))
        self._alley_kw = {
            k: config[k] for k in (
                "open_approach_m", "min_corridor_width_m", "max_corridor_width_m",
            ) if k in config
        }
        if config.get("route_len_m"):
            self.params["route_len_m"] = float(config["route_len_m"])
        self._seed = seed
        self.dt = float(config.get("dt", 0.1))
        self._rng = np.random.default_rng(seed)
        self._ctl: RoadScenarioController | None = None
        self._prev_t: float | None = None
        self.last_action: AgentAction | None = None
        self.route_meta: dict[str, Any] = {}

    def reset(self, target_state: TargetState,
              uav_state: DroneState | None = None) -> None:
        self._rng = np.random.default_rng(self._seed)
        self._prev_t = None
        keep_z = float(target_state.position[2])
        route, meta = build_alley_hutong_route(
            self.occupancy,
            self._seg_map,
            target_state.position,
            self._rng,
            keep_z=keep_z,
            search_radius_m=self.search_radius_m,
            **self._alley_kw,
        )
        if route is None:
            raise RuntimeError(
                f"alley_hutong route planning failed: {meta.get('error', 'unknown')}"
            )
        self.route_meta = meta
        split_idx = int(meta.get("split_idx", max(0, route.waypoints.shape[0] - 3)))
        open_frac = max(0.15, min(0.85, split_idx / max(route.waypoints.shape[0] - 1, 1)))
        scen_cfg = RoadScenarioConfig(
            name="alley_hutong",
            route_len_m=float(self.params["route_len_m"]),
            speed_mps=float(self.params["speed_mps"]) * 0.85,
            high_speed_mps=float(self.params["speed_mps"]) * 1.1,
            open_road_frac=open_frac,
        )
        self._ctl = RoadScenarioController(
            self.occupancy,
            target_state,
            self._rng,
            scen_cfg,
            route=route,
        )
        self.last_action = None

    def get_next_target_state(
        self,
        t: float,
        current_target_state: TargetState,
        current_uav_state: DroneState,
        history: list[Any] | None = None,
    ) -> TargetState:
        if self._ctl is None:
            raise RuntimeError("AlleyHutongTargetPolicy.reset() must be called first")
        if self._prev_t is not None and t > self._prev_t:
            dt = float(t - self._prev_t)
        else:
            dt = self.dt
        self._prev_t = float(t)
        new_state, action = self._ctl.step(current_uav_state, t, dt)
        self.last_action = action
        return new_state.copy_with(timestamp=float(t))


def create_target_policy(
    behavior_type: str,
    *,
    config: dict[str, Any] | None = None,
    scene_context: dict[str, Any] | None = None,
    seed: int | None = None,
) -> TargetPolicy | RouteFollowingTargetPolicy | AlleyHutongTargetPolicy:
    """Pick reactive vs route-following policy for a FlySeek-Bench behavior."""
    scene_context = dict(scene_context or {})
    occupancy = scene_context.get("occupancy")
    if behavior_type == "alley_hutong" and occupancy is not None:
        return AlleyHutongTargetPolicy(
            config={"behavior_type": behavior_type, **(config or {})},
            scene_context=scene_context,
            seed=seed,
        )
    if behavior_type == "occlusion_seeking" and occupancy is not None:
        return RouteFollowingTargetPolicy(
            config={"behavior_type": behavior_type, **(config or {})},
            scene_context=scene_context,
            seed=seed,
        )
    return TargetPolicy(
        config={"behavior_type": behavior_type, **(config or {})},
        scene_context=scene_context,
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# Offline waypoint generation                                                 #
# --------------------------------------------------------------------------- #
def generate_target_waypoints(
    initial_target_pose: Any,
    initial_uav_pose: Any,
    behavior_type: str,
    difficulty: str = "medium",
    seed: int | None = None,
    *,
    scene_context: dict[str, Any] | None = None,
    n_waypoints: int | None = None,
    step_dt: float = 0.2,
) -> list[list[float]]:
    """Generate a deterministic NED waypoint list for environments that consume
    waypoints rather than per-step control.

    When ``scene_context`` provides an ``occupancy`` map this wraps the existing
    road planner (``road_graph.build_route``) with a behavior-specific maneuver
    tag. Otherwise it rolls out :class:`TargetPolicy` against a *stationary* UAV
    fixed at ``initial_uav_pose`` (documented limitation: offline waypoints cannot
    react to live UAV motion, and ``occlusion_seeking`` cannot truly seek cover
    without a scene — a warning is emitted).
    """
    scene_context = dict(scene_context or {})
    tgt_pos, tgt_yaw = _parse_pose(initial_target_pose)
    uav_pos, uav_yaw = _parse_pose(initial_uav_pose)
    params = DIFFICULTY_PRESETS.get(difficulty, DIFFICULTY_PRESETS["medium"])
    occupancy = scene_context.get("occupancy")

    if (behavior_type == "occlusion_seeking"
            and occupancy is not None
            and hasattr(occupancy, "is_drivable_ned")):
        keep_z = float(scene_context.get("keep_z", tgt_pos[2]))
        uav = np.asarray(uav_pos, dtype=np.float64).reshape(3)
        route, _meta = build_occlusion_seeking_route(
            occupancy, tgt_pos, np.random.default_rng(seed),
            keep_z=keep_z,
            drone_ned=uav,
            route_len_m=float(params["route_len_m"]),
            anchor_heading_rad=(tgt_yaw if tgt_yaw is not None
                                else bearing_xy(uav, tgt_pos)),
        )
        if n_waypoints is None:
            speed = float(params["speed_mps"]) * 0.9
            n_waypoints = max(2, int(params["route_len_m"] / max(speed * step_dt, 1e-3)))
        scen_cfg = RoadScenarioConfig(
            name="open_then_hide",
            route_len_m=float(params["route_len_m"]),
            speed_mps=float(params["speed_mps"]) * 0.9,
        )
        target = TargetState(position=tgt_pos.copy(), velocity=np.zeros(3),
                             heading=(tgt_yaw or 0.0), timestamp=0.0)
        uav_state = DroneState(position=uav.copy(), velocity=np.zeros(3),
                               heading=(uav_yaw or 0.0), timestamp=0.0)
        ctl = RoadScenarioController(
            occupancy, target, np.random.default_rng(seed), scen_cfg, route=route,
        )
        st = ctl.initial_state()
        waypoints = [[float(st.position[0]), float(st.position[1]),
                      float(st.position[2])]]
        for i in range(1, int(n_waypoints)):
            st, _ = ctl.step(uav_state, i * step_dt, step_dt)
            waypoints.append([float(st.position[0]), float(st.position[1]),
                              float(st.position[2])])
        return waypoints

    if occupancy is not None and hasattr(occupancy, "is_drivable_ned"):
        from flyseek.utils.road_graph import build_route
        keep_z = float(scene_context.get("keep_z", tgt_pos[2]))
        route = build_route(
            occupancy, tgt_pos, np.random.default_rng(seed),
            keep_z=keep_z,
            route_len_m=float(params["route_len_m"]),
            maneuver=BEHAVIOR_TO_MANEUVER.get(behavior_type, "normal_drive"),
            start_at_anchor=True,
            anchor_heading_rad=(tgt_yaw if tgt_yaw is not None
                                else bearing_xy(uav_pos, tgt_pos)),
        )
        return [[float(w[0]), float(w[1]), float(w[2])] for w in route.waypoints]

    # ---- geometric fallback: roll out the policy vs. a stationary UAV ----
    if behavior_type == "occlusion_seeking":
        warnings.warn(
            "[generate_target_waypoints] occlusion_seeking without an occupancy "
            "scene cannot truly seek cover; falling back to evasive geometry.",
            stacklevel=2,
        )
    if n_waypoints is None:
        speed = float(params["speed_mps"])
        n_waypoints = max(2, int(params["route_len_m"] / max(speed * step_dt, 1e-3)))

    policy = TargetPolicy(
        config={"behavior_type": behavior_type, "difficulty": difficulty,
                "dt": step_dt},
        scene_context={"keep_z": float(tgt_pos[2])},
        seed=seed,
    )
    target = TargetState(position=tgt_pos.copy(), velocity=np.zeros(3),
                         heading=(tgt_yaw or 0.0), timestamp=0.0)
    uav = DroneState(position=uav_pos.copy(), velocity=np.zeros(3),
                     heading=(uav_yaw or 0.0), timestamp=0.0)
    policy.reset(target)

    waypoints = [[float(target.position[0]), float(target.position[1]),
                  float(target.position[2])]]
    for i in range(1, n_waypoints):
        target = policy.get_next_target_state(i * step_dt, target, uav)
        waypoints.append([float(target.position[0]), float(target.position[1]),
                          float(target.position[2])])
    return waypoints


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

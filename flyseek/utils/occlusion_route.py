# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Occlusion-seeking route builder: open road → alley hide behind buildings.

Uses the PCD occupancy map to (1) plan an ``open_then_hide`` corridor route,
(2) refine the hide leg toward a verified ``find_hide_goal_ned`` point where
the UAV line-of-sight is blocked by *large building* occupancy (not street
lamps / thin poles), and (3) score the route so callers can retry with
different seeds until a usable hide segment exists.

All motion still flows through :class:`RoadScenarioController` (spline +
``resolve_bev_move_ned`` + turn-rate limits) — this module only *plans* waypoints.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from flyseek.adapters.pcd_occupancy import PcdOccupancyMap
from flyseek.utils.hide_visibility import (
    HideVisibilityConfig,
    is_hidden_from_chase_drones,
    sample_chase_drone_poses,
)
from flyseek.utils.road_graph import RoadRoute, build_route

# Must match ``build_route`` open-leg budget for ``open_then_hide``.
OPEN_THEN_HIDE_OPEN_FRAC = 0.68

# Shared behavior → route-maneuver map (UE + AirSim + offline batch).
BEHAVIOR_ROUTE_MANEUVER: dict[str, str] = {
    "direct_escape": "normal_drive",
    "sharp_turn": "high_maneuver",
    "detour_feint": "corner_occlude",
    "occlusion_seeking": "open_then_hide",
}

# Large-building filter defaults (see PcdOccupancyMap.los_blocked_by_building_ned).
DEFAULT_MIN_BUILDING_HEIGHT_M = 18.0
DEFAULT_MIN_FOOTPRINT_CELLS = 9
DEFAULT_HIDE_SEARCH_RADIUS_M = 48.0
DEFAULT_ROUTE_MAX_ATTEMPTS = 16
DEFAULT_MIN_BUILDING_OCCLUDED_FRAC = 0.55
DEFAULT_BUILDING_PROBE_DIST_M = 7.5


def _los_kwargs(
    occupancy: PcdOccupancyMap,
    *,
    drone_eye_agl_m: float,
    min_building_height_m: float | None,
    min_footprint_cells: int,
) -> dict[str, Any]:
    return {
        "drone_eye_agl_m": drone_eye_agl_m,
        "target_agl_m": 1.0,
        "min_building_height_m": (
            min_building_height_m
            if min_building_height_m is not None
            else max(getattr(occupancy.cfg, "min_height_thresh", 6.0),
                     DEFAULT_MIN_BUILDING_HEIGHT_M)
        ),
        "min_footprint_cells": int(min_footprint_cells),
    }


def _headings_for_waypoints(waypoints: np.ndarray) -> np.ndarray:
    wps = np.asarray(waypoints, dtype=np.float64)
    if wps.shape[0] < 2:
        h = 0.0
        return np.array([h], dtype=np.float64)
    headings = []
    for i in range(wps.shape[0] - 1):
        dx = float(wps[i + 1, 0] - wps[i, 0])
        dy = float(wps[i + 1, 1] - wps[i, 1])
        headings.append(math.atan2(dy, dx) if (dx * dx + dy * dy) > 1e-9 else 0.0)
    headings.append(headings[-1])
    return np.asarray(headings, dtype=np.float64)


def _segment_ok(
    occupancy: PcdOccupancyMap,
    a: np.ndarray,
    b: np.ndarray,
    *,
    keep_z: float,
) -> bool:
    b = np.asarray(b, dtype=np.float64).reshape(3).copy()
    b[2] = keep_z
    if not occupancy.is_drivable_ned(b):
        return False
    if hasattr(occupancy, "_segment_drivable"):
        return bool(occupancy._segment_drivable(a, b, keep_z=keep_z))
    return True


def _vis_cfg_from_kw(
    occ_kw: dict[str, Any],
    *,
    drone_eye_agl_m: float,
    min_building_height_m: float | None,
    min_footprint_cells: int,
    follow_distance_m: float = 12.0,
) -> HideVisibilityConfig:
    raw = occ_kw.get("hide_vis_config")
    if isinstance(raw, HideVisibilityConfig):
        return raw
    if isinstance(raw, dict):
        return HideVisibilityConfig(**raw)
    return HideVisibilityConfig(
        drone_eye_agl_m=float(drone_eye_agl_m),
        follow_distance_m=float(occ_kw.get("follow_distance_m", follow_distance_m)),
        min_building_height_m=min_building_height_m,
        min_footprint_cells=int(min_footprint_cells),
        hfov_deg=float(occ_kw.get("hfov_deg", 50.0)),
        max_range_m=float(occ_kw.get("max_range_m", 100.0)),
        building_only_los=True,
        use_frustum_projection=bool(occ_kw.get("use_frustum_projection", True)),
        occluder_between_required=bool(occ_kw.get("require_occluder_between", True)),
        occluder_near_target_m=float(occ_kw.get("occluder_near_target_m", 12.0)),
        chase_drone_samples=int(occ_kw.get("chase_drone_samples", 5)),
        cam_forward_m=float(occ_kw.get("cam_forward_m", 0.45)),
        cam_down_m=float(occ_kw.get("cam_down_m", 0.25)),
        cam_pitch_deg=float(occ_kw.get("cam_pitch_deg", 55.0)),
        cam_width=int(occ_kw.get("cam_width", 256)),
        cam_height=int(occ_kw.get("cam_height", 144)),
    )


def _chase_drones_for_route(
    route: RoadRoute,
    split_idx: int,
    drone_ned: np.ndarray,
    vis_cfg: HideVisibilityConfig,
) -> list:
    return sample_chase_drone_poses(
        route.waypoints,
        split_idx=split_idx,
        cfg=vis_cfg,
        initial_drone_ned=drone_ned,
    )


def analyze_route_occlusion(
    route: RoadRoute,
    occupancy: PcdOccupancyMap,
    drone_ned: np.ndarray,
    *,
    keep_z: float,
    drone_eye_agl_m: float = 12.0,
    target_agl_m: float = 1.0,
    min_building_height_m: float | None = None,
    min_footprint_cells: int = DEFAULT_MIN_FOOTPRINT_CELLS,
    hide_search_radius_m: float = DEFAULT_HIDE_SEARCH_RADIUS_M,
    require_adjacent_building: bool = True,
    building_probe_dist_m: float = DEFAULT_BUILDING_PROBE_DIST_M,
    open_road_frac: float = OPEN_THEN_HIDE_OPEN_FRAC,
    hide_vis_config: HideVisibilityConfig | None = None,
    follow_distance_m: float = 12.0,
) -> dict[str, Any]:
    """Score how well the hide leg breaks UAV LoS behind *large buildings*."""
    los_kw = _los_kwargs(
        occupancy,
        drone_eye_agl_m=drone_eye_agl_m,
        min_building_height_m=min_building_height_m,
        min_footprint_cells=min_footprint_cells,
    )
    occ_kw = {
        "hide_vis_config": hide_vis_config,
        "follow_distance_m": follow_distance_m,
    }
    vis_cfg = hide_vis_config or _vis_cfg_from_kw(
        occ_kw,
        drone_eye_agl_m=drone_eye_agl_m,
        min_building_height_m=los_kw["min_building_height_m"],
        min_footprint_cells=min_footprint_cells,
        follow_distance_m=follow_distance_m,
    )
    wps = np.asarray(route.waypoints, dtype=np.float64)
    n = int(wps.shape[0])
    if n < 3:
        return {
            "occluded_frac": 0.0,
            "building_occluded_frac": 0.0,
            "occluded_count": 0,
            "building_occluded_count": 0,
            "hide_leg_len": 0,
            "split_idx": 0,
            "hide_goal": None,
            "hide_goal_occluded": False,
            "hide_goal_building_occluded": False,
            "mean_alley_width_m": 0.0,
        }

    split = max(1, int(float(open_road_frac) * (n - 1)))
    hide_wps = wps[split:]
    chase_drones = _chase_drones_for_route(route, split, drone_ned, vis_cfg)
    occluded = 0
    building_occluded = 0
    frustum_hidden = 0
    widths: list[float] = []
    for wp in hide_wps:
        wp3 = wp.copy()
        wp3[2] = keep_z
        if occupancy.los_blocked_ned(
            drone_ned, wp3,
            drone_eye_agl_m=drone_eye_agl_m,
            target_agl_m=target_agl_m,
        ):
            occluded += 1
        if occupancy.los_blocked_by_building_ned(drone_ned, wp3, **los_kw):
            building_occluded += 1
        all_hid, _, _ = is_hidden_from_chase_drones(
            occupancy, wp3, chase_drones, vis_cfg, keep_z=keep_z,
        )
        if all_hid:
            frustum_hidden += 1
        if hasattr(occupancy, "_alley_hide_bonus_ned"):
            bonus = occupancy._alley_hide_bonus_ned(wp3, keep_z=keep_z)
            widths.append(max(0.0, 12.0 - bonus / 1.5))

    hide_goal = None
    hide_goal_occluded = False
    hide_goal_building_occluded = False
    hide_goal_frustum_hidden = False
    try:
        hide_goal = occupancy.find_hide_goal_ned(
            wps[split], drone_ned, keep_z=keep_z,
            search_radius_m=float(hide_search_radius_m),
            building_only=True,
            min_building_height_m=los_kw["min_building_height_m"],
            min_footprint_cells=los_kw["min_footprint_cells"],
            require_adjacent_building=require_adjacent_building,
            building_probe_dist_m=building_probe_dist_m,
            hide_vis_config=vis_cfg,
            chase_drone_poses=chase_drones,
            require_occluder_between=vis_cfg.occluder_between_required,
            occluder_near_target_m=vis_cfg.occluder_near_target_m,
        )
        if hide_goal is not None:
            hide_goal_occluded = bool(occupancy.los_blocked_ned(
                drone_ned, hide_goal,
                drone_eye_agl_m=drone_eye_agl_m,
                target_agl_m=target_agl_m,
            ))
            hide_goal_building_occluded = bool(
                occupancy.los_blocked_by_building_ned(drone_ned, hide_goal, **los_kw)
            )
            ghid, _, _ = is_hidden_from_chase_drones(
                occupancy, hide_goal, chase_drones, vis_cfg, keep_z=keep_z,
            )
            hide_goal_frustum_hidden = ghid
        else:
            hide_goal_frustum_hidden = False
    except Exception:
        hide_goal = None
        hide_goal_frustum_hidden = False

    hide_len = max(1, len(hide_wps))
    return {
        "occluded_frac": float(occluded) / hide_len,
        "building_occluded_frac": float(building_occluded) / hide_len,
        "frustum_hidden_frac": float(frustum_hidden) / hide_len,
        "occluded_count": occluded,
        "building_occluded_count": building_occluded,
        "frustum_hidden_count": frustum_hidden,
        "hide_leg_len": hide_len,
        "split_idx": split,
        "hide_goal": (hide_goal.tolist() if hide_goal is not None else None),
        "hide_goal_occluded": hide_goal_occluded,
        "hide_goal_building_occluded": hide_goal_building_occluded,
        "hide_goal_frustum_hidden": hide_goal_frustum_hidden,
        "chase_drone_samples": len(chase_drones),
        "mean_alley_width_m": float(np.mean(widths)) if widths else 0.0,
        "min_building_height_m": los_kw["min_building_height_m"],
        "min_footprint_cells": los_kw["min_footprint_cells"],
    }


def refine_route_hide_leg(
    route: RoadRoute,
    occupancy: PcdOccupancyMap,
    drone_ned: np.ndarray,
    *,
    keep_z: float,
    waypoint_step_m: float = 6.0,
    hide_search_radius_m: float = DEFAULT_HIDE_SEARCH_RADIUS_M,
    min_building_height_m: float | None = None,
    min_footprint_cells: int = DEFAULT_MIN_FOOTPRINT_CELLS,
    require_adjacent_building: bool = True,
    building_probe_dist_m: float = DEFAULT_BUILDING_PROBE_DIST_M,
    drone_eye_agl_m: float = 12.0,
    open_road_frac: float = OPEN_THEN_HIDE_OPEN_FRAC,
    hide_vis_config: HideVisibilityConfig | None = None,
    follow_distance_m: float = 12.0,
) -> RoadRoute:
    """Replace the hide leg with a drivable polyline toward a building hide goal."""
    wps = np.asarray(route.waypoints, dtype=np.float64)
    n = int(wps.shape[0])
    if n < 2:
        return route

    split = max(1, int(float(open_road_frac) * (n - 1)))
    start = wps[split].copy()
    start[2] = keep_z

    los_kw = _los_kwargs(
        occupancy,
        drone_eye_agl_m=drone_eye_agl_m,
        min_building_height_m=min_building_height_m,
        min_footprint_cells=min_footprint_cells,
    )
    vis_cfg = hide_vis_config or _vis_cfg_from_kw(
        {"follow_distance_m": follow_distance_m},
        drone_eye_agl_m=drone_eye_agl_m,
        min_building_height_m=los_kw["min_building_height_m"],
        min_footprint_cells=min_footprint_cells,
        follow_distance_m=follow_distance_m,
    )
    chase_drones = _chase_drones_for_route(route, split, drone_ned, vis_cfg)

    goal = None
    try:
        goal = occupancy.find_hide_goal_ned(
            start, drone_ned, keep_z=keep_z,
            search_radius_m=float(hide_search_radius_m),
            building_only=True,
            min_building_height_m=los_kw["min_building_height_m"],
            min_footprint_cells=los_kw["min_footprint_cells"],
            require_adjacent_building=require_adjacent_building,
            building_probe_dist_m=building_probe_dist_m,
            hide_vis_config=vis_cfg,
            chase_drone_poses=chase_drones,
            require_occluder_between=vis_cfg.occluder_between_required,
            occluder_near_target_m=vis_cfg.occluder_near_target_m,
        )
    except Exception:
        goal = None
    if goal is None:
        return route

    goal = np.asarray(goal, dtype=np.float64).reshape(3).copy()
    goal[2] = keep_z

    tail = [start.copy()]
    cur = start.copy()
    step = max(float(waypoint_step_m), 1.0)
    guard = 0
    while guard < 40:
        guard += 1
        delta = goal[:2] - cur[:2]
        dist = float(np.linalg.norm(delta))
        if dist <= step * 0.55:
            if _segment_ok(occupancy, cur, goal, keep_z=keep_z):
                tail.append(goal.copy())
            break
        direction = delta / max(dist, 1e-9)
        nxt = cur.copy()
        nxt[0] += direction[0] * step
        nxt[1] += direction[1] * step
        nxt[2] = keep_z
        if not _segment_ok(occupancy, cur, nxt, keep_z=keep_z):
            break
        tail.append(nxt.copy())
        cur = nxt

    if len(tail) < 2:
        return route

    prefix = wps[: split + 1]
    if np.linalg.norm(tail[0][:2] - prefix[-1][:2]) < 0.5:
        new_wps = np.vstack([prefix, np.asarray(tail[1:], dtype=np.float64)])
    else:
        new_wps = np.vstack([prefix, np.asarray(tail, dtype=np.float64)])

    return RoadRoute(
        waypoints=new_wps,
        headings=_headings_for_waypoints(new_wps),
        score=float(route.score),
    )


def _route_quality(meta: dict[str, Any]) -> float:
    f_frac = float(meta.get("frustum_hidden_frac", 0.0))
    b_frac = float(meta.get("building_occluded_frac", meta.get("occluded_frac", 0.0)))
    q = f_frac * 140.0 + b_frac * 40.0
    if meta.get("hide_goal_frustum_hidden"):
        q += 50.0
    elif meta.get("hide_goal_building_occluded"):
        q += 30.0
    elif meta.get("hide_goal_occluded"):
        q += 5.0
    if meta.get("hide_goal") is not None:
        q += 10.0
    q += max(0.0, 14.0 - float(meta.get("mean_alley_width_m", 14.0))) * 2.0
    return q


def build_occlusion_seeking_route(
    occupancy: PcdOccupancyMap,
    anchor_ned: np.ndarray,
    rng: np.random.Generator,
    *,
    keep_z: float,
    drone_ned: np.ndarray,
    route_len_m: float = 180.0,
    search_radius_m: float = 220.0,
    waypoint_step_m: float = 6.0,
    anchor_heading_rad: float | None = None,
    drone_eye_agl_m: float = 12.0,
    max_attempts: int = DEFAULT_ROUTE_MAX_ATTEMPTS,
    min_building_occluded_frac: float = DEFAULT_MIN_BUILDING_OCCLUDED_FRAC,
    min_building_height_m: float | None = None,
    min_footprint_cells: int = DEFAULT_MIN_FOOTPRINT_CELLS,
    hide_search_radius_m: float = DEFAULT_HIDE_SEARCH_RADIUS_M,
    require_adjacent_building: bool = True,
    building_probe_dist_m: float = DEFAULT_BUILDING_PROBE_DIST_M,
    open_road_frac: float = OPEN_THEN_HIDE_OPEN_FRAC,
    hide_vis_config: HideVisibilityConfig | None = None,
    follow_distance_m: float = 12.0,
    min_frustum_hidden_frac: float = 0.45,
) -> tuple[RoadRoute, dict[str, Any]]:
    """Build + validate an ``open_then_hide`` route hidden behind large buildings."""
    occ_kw = dict(
        hide_search_radius_m=hide_search_radius_m,
        min_building_height_m=min_building_height_m,
        min_footprint_cells=min_footprint_cells,
        require_adjacent_building=require_adjacent_building,
        building_probe_dist_m=building_probe_dist_m,
        drone_eye_agl_m=drone_eye_agl_m,
        open_road_frac=float(open_road_frac),
        hide_vis_config=hide_vis_config,
        follow_distance_m=follow_distance_m,
    )
    anchor = np.asarray(anchor_ned, dtype=np.float64).reshape(3).copy()
    anchor[2] = keep_z
    drone = np.asarray(drone_ned, dtype=np.float64).reshape(3)

    best_route: RoadRoute | None = None
    best_meta: dict[str, Any] = {}
    best_q = -1e9

    for attempt in range(max(1, int(max_attempts))):
        attempt_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
        route = build_route(
            occupancy,
            anchor,
            attempt_rng,
            keep_z=keep_z,
            route_len_m=float(route_len_m),
            waypoint_step_m=float(waypoint_step_m),
            search_radius_m=float(search_radius_m),
            maneuver="open_then_hide",
            start_at_anchor=True,
            anchor_heading_rad=anchor_heading_rad,
            open_road_frac=float(open_road_frac),
        )
        route = refine_route_hide_leg(
            route, occupancy, drone,
            keep_z=keep_z,
            waypoint_step_m=waypoint_step_m,
            **occ_kw,
        )
        meta = analyze_route_occlusion(
            route, occupancy, drone,
            keep_z=keep_z,
            **occ_kw,
        )
        meta["attempt"] = attempt
        meta["route_waypoints"] = int(route.waypoints.shape[0])
        q = _route_quality(meta)
        if q > best_q:
            best_q = q
            best_route = route
            best_meta = meta
        if (meta.get("frustum_hidden_frac", 0.0) >= min_frustum_hidden_frac
                and meta.get("hide_goal_frustum_hidden")):
            break
        if (meta.get("building_occluded_frac", 0.0) >= min_building_occluded_frac
                and meta.get("hide_goal_building_occluded")):
            break

    assert best_route is not None
    best_meta["quality_score"] = best_q
    spd_est = 3.2  # rough hard-difficulty car speed for planning hints
    est_total_s = float(route_len_m) / max(spd_est, 0.5)
    best_meta["open_road_frac"] = float(open_road_frac)
    best_meta["est_hide_start_s"] = round(est_total_s * float(open_road_frac), 1)
    best_meta["est_route_total_s"] = round(est_total_s, 1)
    best_meta["recommended_duration_s"] = round(max(60.0, est_total_s * 1.05), 0)
    if not best_meta.get("hide_goal_frustum_hidden"):
        print("[warn] occlusion route: hide goal not frustum-hidden from chase "
              "drones; try --min-building-footprint-cells 12, longer --duration, "
              "or another --seed near a building alley.")
    elif not best_meta.get("hide_goal_building_occluded"):
        print("[warn] occlusion route: no building-validated hide goal found; "
              "try --min-building-footprint-cells 12 --hide-search-radius-m 60 "
              "or another --seed / --target near a building alley.")
    return best_route, best_meta


def occlusion_route_kwargs_from_args(args: Any) -> dict[str, Any]:
    """Extract building-hide route knobs from ``demo_adversary_chase`` CLI args."""
    def _get(name: str, default: Any) -> Any:
        return getattr(args, name, default)

    from flyseek.utils.hide_visibility import HideVisibilityConfig

    vis_cfg = HideVisibilityConfig.from_args(args)
    kw: dict[str, Any] = {
        "min_building_height_m": float(_get("min_building_height_m",
                                            DEFAULT_MIN_BUILDING_HEIGHT_M)),
        "min_footprint_cells": int(_get("min_building_footprint_cells",
                                       DEFAULT_MIN_FOOTPRINT_CELLS)),
        "hide_search_radius_m": float(_get("hide_search_radius_m",
                                           DEFAULT_HIDE_SEARCH_RADIUS_M)),
        "search_radius_m": float(_get("route_search_radius_m", 220.0)),
        "max_attempts": int(_get("route_max_attempts", DEFAULT_ROUTE_MAX_ATTEMPTS)),
        "min_building_occluded_frac": float(_get("min_building_occluded_frac",
                                                 DEFAULT_MIN_BUILDING_OCCLUDED_FRAC)),
        "require_adjacent_building": bool(_get("require_adjacent_building", True)),
        "building_probe_dist_m": float(_get("building_probe_dist_m",
                                            DEFAULT_BUILDING_PROBE_DIST_M)),
        "open_road_frac": float(_get("open_road_frac", 0.45)),
        "route_len_m": (float(_get("route_len_m", 0.0)) or None),
        "hide_vis_config": vis_cfg,
        "follow_distance_m": float(_get("follow_distance", 12.0)),
        "min_frustum_hidden_frac": float(_get("min_frustum_hidden_frac", 0.45)),
    }
    if kw["route_len_m"] is None:
        del kw["route_len_m"]
    return kw


__all__ = [
    "BEHAVIOR_ROUTE_MANEUVER",
    "OPEN_THEN_HIDE_OPEN_FRAC",
    "DEFAULT_MIN_BUILDING_HEIGHT_M",
    "DEFAULT_MIN_FOOTPRINT_CELLS",
    "DEFAULT_HIDE_SEARCH_RADIUS_M",
    "DEFAULT_ROUTE_MAX_ATTEMPTS",
    "DEFAULT_MIN_BUILDING_OCCLUDED_FRAC",
    "DEFAULT_BUILDING_PROBE_DIST_M",
    "analyze_route_occlusion",
    "build_occlusion_seeking_route",
    "occlusion_route_kwargs_from_args",
    "refine_route_hide_leg",
]

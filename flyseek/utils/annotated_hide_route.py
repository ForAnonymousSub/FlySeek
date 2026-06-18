# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Hide routes planned against **annotated** seg_map buildings only."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from flyseek.adapters.pcd_occupancy import PcdOccupancyMap
from flyseek.utils.hide_visibility import (
    HideVisibilityConfig,
    is_hidden_from_chase_drones,
    sample_chase_drone_poses,
)
from flyseek.utils.occlusion_route import _headings_for_waypoints, _segment_ok
from flyseek.utils.road_graph import RoadRoute, build_route
from flyseek.utils.seg_buildings import SegBuildingMap


def build_annotated_hide_route(
    occupancy: PcdOccupancyMap,
    seg_map: SegBuildingMap,
    anchor_ned: np.ndarray,
    drone_ned: np.ndarray,
    rng: np.random.Generator,
    *,
    keep_z: float,
    route_len_m: float = 120.0,
    open_road_frac: float = 0.4,
    search_radius_m: float = 180.0,
    waypoint_step_m: float = 6.0,
    hide_search_radius_m: float = 80.0,
    hide_vis_config: HideVisibilityConfig | None = None,
    follow_distance_m: float = 12.0,
    max_attempts: int = 12,
    preset_hide_site: dict[str, Any] | None = None,
) -> tuple[RoadRoute | None, dict[str, Any]]:
    """Plan open-road → hide-behind-annotated-building route.

    Returns ``(route, meta)``. ``route`` is ``None`` when no valid hide site exists.
    """
    vis_cfg = hide_vis_config or HideVisibilityConfig(
        follow_distance_m=follow_distance_m,
        building_only_los=False,
    )
    anchor = np.asarray(anchor_ned, dtype=np.float64).reshape(3).copy()
    anchor[2] = keep_z
    drone = np.asarray(drone_ned, dtype=np.float64).reshape(3)

    open_len = max(30.0, float(route_len_m) * float(open_road_frac))
    best_route: RoadRoute | None = None
    best_meta: dict[str, Any] = {}
    best_q = -1e9

    for attempt in range(max(1, int(max_attempts))):
        attempt_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
        base = build_route(
            occupancy,
            anchor,
            attempt_rng,
            keep_z=keep_z,
            route_len_m=open_len,
            waypoint_step_m=waypoint_step_m,
            search_radius_m=search_radius_m,
            maneuver="normal_drive",
            start_at_anchor=True,
            anchor_heading_rad=None,
        )
        split_idx = max(0, base.waypoints.shape[0] - 1)
        start = np.asarray(base.waypoints[split_idx], dtype=np.float64).copy()
        start[2] = keep_z

        chase = sample_chase_drone_poses(
            base.waypoints, split_idx=split_idx, cfg=vis_cfg,
            initial_drone_ned=drone,
        )
        if preset_hide_site is not None and attempt == 0:
            site = dict(preset_hide_site)
        else:
            site = seg_map.find_best_hide_site(
                occupancy, start, drone,
                keep_z=keep_z,
                search_radius_m=hide_search_radius_m,
                hide_vis_config=vis_cfg,
                chase_drone_poses=chase,
            )
        if site is None:
            continue

        hide = np.asarray(site["hide_goal"], dtype=np.float64).reshape(3)
        hide[2] = keep_z
        tail = [start.copy()]
        cur = start.copy()
        step = max(float(waypoint_step_m), 1.0)
        guard = 0
        while guard < 40:
            guard += 1
            delta = hide[:2] - cur[:2]
            dist = float(np.linalg.norm(delta))
            if dist <= step * 0.55:
                if _segment_ok(occupancy, cur, hide, keep_z=keep_z):
                    tail.append(hide.copy())
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
            continue

        wps = np.vstack([
            np.asarray(base.waypoints, dtype=np.float64),
            np.asarray(tail[1:], dtype=np.float64),
        ])
        route = RoadRoute(
            waypoints=wps,
            headings=_headings_for_waypoints(wps),
            score=float(base.score),
        )
        split = int(base.waypoints.shape[0] - 1)
        hide_wps = wps[split:]
        goal_ok, _, _ = is_hidden_from_chase_drones(
            occupancy, hide, chase, vis_cfg, keep_z=keep_z,
        )
        seg_goal_ok = all(
            seg_map.los_blocked_by_annotated_building_ned(
                d.position, hide, drone_eye_agl_m=vis_cfg.drone_eye_agl_m,
            )
            for d in chase
        )
        frustum_h = 0
        for wp in hide_wps:
            wp3 = wp.copy()
            wp3[2] = keep_z
            ok, _, _ = is_hidden_from_chase_drones(
                occupancy, wp3, chase, vis_cfg, keep_z=keep_z,
            )
            # Override LoS: must also be blocked by annotated building.
            seg_ok = all(
                seg_map.los_blocked_by_annotated_building_ned(
                    d.position, wp3, drone_eye_agl_m=vis_cfg.drone_eye_agl_m,
                )
                for d in chase
            )
            if ok and seg_ok:
                frustum_h += 1
        hide_len = max(1, len(hide_wps))
        meta = dict(site)
        meta.update({
            "attempt": attempt,
            "route_waypoints": int(wps.shape[0]),
            "split_idx": split,
            "hide_leg_len": hide_len,
            "frustum_hidden_frac": frustum_h / hide_len,
            "hide_goal_frustum_hidden": bool(goal_ok and seg_goal_ok),
            "hide_goal_seg_occluded": bool(seg_goal_ok),
            "planner": "annotated_seg_buildings",
            "open_road_frac": float(open_road_frac),
            "est_hide_start_s": round(
                open_len / 3.2 * float(open_road_frac), 1,
            ),
            "est_route_total_s": round(float(route_len_m) / 3.2, 1),
            "recommended_duration_s": round(max(60.0, route_len_m / 3.2 * 1.1), 0),
        })
        q = float(meta["frustum_hidden_frac"]) * 150.0 + site.get("dist_from_anchor_m", 0) * 0.1
        if q > best_q:
            best_q = q
            best_route = route
            best_meta = meta
        if meta.get("hide_goal_frustum_hidden"):
            break

    if best_route is None:
        return None, {
            "planner": "annotated_seg_buildings",
            "error": "no_valid_hide_site",
            "buildings_loaded": len(seg_map),
        }
    best_meta["quality_score"] = best_q
    return best_route, best_meta


def load_seg_map_for_env(repo_root: Path, env_name: str) -> SegBuildingMap:
    """Default path: ``scene_data/seg_map/<env>.jsonl``."""
    p = repo_root / "scene_data" / "seg_map" / f"{env_name}.jsonl"
    return SegBuildingMap.from_jsonl(p)


__all__ = [
    "build_annotated_hide_route",
    "load_seg_map_for_env",
]

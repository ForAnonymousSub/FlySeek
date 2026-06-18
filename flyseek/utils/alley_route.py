# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Plan routes that drive the target car into narrow hutong / alleys between buildings.

Uses annotated ``seg_map`` building pairs to find inter-building gaps, validates
them against PCD drivability + corridor width, then builds:

    anchor → open road → alley entry → midpoint → deep inside the hutong
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from flyseek.adapters.pcd_occupancy import PcdOccupancyMap
from flyseek.utils.occlusion_route import _headings_for_waypoints, _segment_ok
from flyseek.utils.road_graph import (
    RoadRoute,
    _corridor_width,
    _is_free,
    build_route,
    find_major_road_seed,
)
from flyseek.utils.seg_buildings import SegBuilding, SegBuildingMap


@dataclass(frozen=True)
class AlleyCorridor:
    """A validated hutong between two annotated buildings."""

    building_a: SegBuilding
    building_b: SegBuilding
    midpoint_ned: np.ndarray
    axis_heading_rad: float
    gap_width_m: float
    corridor_width_m: float
    depth_m: float
    score: float
    entry_ned: np.ndarray
    deep_ned: np.ndarray


def _is_drivable(occ: PcdOccupancyMap, p: np.ndarray, keep_z: float) -> bool:
    if hasattr(occ, "is_drivable_ned"):
        return bool(occ.is_drivable_ned(p))
    return _is_free(occ, p, keep_z)


def _snap_to_drivable(
    occ: PcdOccupancyMap,
    p: np.ndarray,
    *,
    keep_z: float,
    search_radius_m: float = 48.0,
    step_m: float = 3.0,
) -> np.ndarray | None:
    """Nearest drivable NED point within ``search_radius_m``."""
    p = np.asarray(p, dtype=np.float64).reshape(3).copy()
    p[2] = keep_z
    if _is_drivable(occ, p, keep_z):
        return p
    best: np.ndarray | None = None
    best_d = 1e18
    radii = np.arange(step_m, search_radius_m + 1e-6, step_m)
    for r in radii:
        n = max(12, int(2 * math.pi * r / step_m))
        for k in range(n):
            a = 2.0 * math.pi * k / n
            q = p.copy()
            q[0] += r * math.cos(a)
            q[1] += r * math.sin(a)
            q[2] = keep_z
            if not _is_drivable(occ, q, keep_z):
                continue
            d = float(np.linalg.norm(q[:2] - p[:2]))
            if d < best_d:
                best_d = d
                best = q.copy()
        if best is not None:
            return best
    return best


def _alley_axis_heading(a_xy: np.ndarray, b_xy: np.ndarray) -> float:
    ab = b_xy - a_xy
    if float(np.linalg.norm(ab)) < 1e-6:
        return 0.0
    return math.atan2(ab[1], ab[0]) + math.pi / 2.0


def _walk_along(
    occ: PcdOccupancyMap,
    start: np.ndarray,
    heading: float,
    *,
    keep_z: float,
    max_dist: float,
    step_m: float = 2.0,
    max_corridor_m: float = 18.0,
) -> tuple[np.ndarray, float]:
    cur = np.asarray(start, dtype=np.float64).reshape(3).copy()
    cur[2] = keep_z
    walked = 0.0
    last = cur.copy()
    while walked + step_m <= max_dist:
        nxt = cur.copy()
        nxt[0] += math.cos(heading) * step_m
        nxt[1] += math.sin(heading) * step_m
        nxt[2] = keep_z
        if not _is_drivable(occ, nxt, keep_z):
            break
        w = _corridor_width(occ, nxt, heading, keep_z=keep_z, max_width=20.0)
        if w > max_corridor_m:
            break
        last = nxt.copy()
        cur = nxt
        walked += step_m
    return last, walked


def _find_entry_outside_alley(
    occ: PcdOccupancyMap,
    mid: np.ndarray,
    axis_h: float,
    *,
    keep_z: float,
    search_m: float = 48.0,
    step_m: float = 3.0,
) -> np.ndarray | None:
    """Walk opposite the deep direction until the corridor widens (main road)."""
    mid = np.asarray(mid, dtype=np.float64).reshape(3).copy()
    mid[2] = keep_z
    best = None
    best_w = -1.0
    for sign in (-1.0, 1.0):
        cur = mid.copy()
        for _ in range(int(search_m / step_m)):
            w = _corridor_width(occ, cur, axis_h, keep_z=keep_z, max_width=24.0)
            if w > best_w:
                best_w = w
                best = cur.copy()
            if w >= 14.0:
                return cur.copy()
            nxt = cur.copy()
            nxt[0] -= math.cos(axis_h) * step_m * sign
            nxt[1] -= math.sin(axis_h) * step_m * sign
            nxt[2] = keep_z
            if not _is_drivable(occ, nxt, keep_z):
                break
            cur = nxt
    return best


def _score_alley(
    width: float,
    depth: float,
    dist_anchor: float,
    *,
    max_corridor_width_m: float,
    max_depth_m: float,
) -> float:
    narrow_bonus = max(0.0, max_corridor_width_m - width) * 4.0
    depth_bonus = min(depth, max_depth_m) * 1.5
    return narrow_bonus + depth_bonus - dist_anchor * 0.12


def _build_alley_candidate(
    occ: PcdOccupancyMap,
    a: SegBuilding,
    b: SegBuilding,
    anchor: np.ndarray,
    *,
    keep_z: float,
    min_gap_m: float,
    max_gap_m: float,
    min_corridor_width_m: float,
    max_corridor_width_m: float,
    min_depth_m: float,
    max_depth_m: float,
) -> AlleyCorridor | None:
    d = float(np.linalg.norm(a.xy_ned - b.xy_ned))
    if d < min_gap_m or d > max_gap_m:
        return None
    mid = (a.ned_xyz + b.ned_xyz) * 0.5
    mid[2] = keep_z
    if not _is_drivable(occ, mid, keep_z):
        return None

    axis_h = _alley_axis_heading(a.xy_ned, b.xy_ned)
    width = _corridor_width(occ, mid, axis_h, keep_z=keep_z, max_width=20.0)
    if width < min_corridor_width_m or width > max_corridor_width_m:
        return None

    deep, depth_fwd = _walk_along(
        occ, mid, axis_h, keep_z=keep_z, max_dist=max_depth_m,
    )
    _, depth_back = _walk_along(
        occ, mid, axis_h + math.pi, keep_z=keep_z, max_dist=max_depth_m * 0.5,
    )
    depth = depth_fwd + depth_back
    if depth < min_depth_m:
        return None

    entry = _find_entry_outside_alley(occ, mid, axis_h, keep_z=keep_z)
    if entry is None or not _is_drivable(occ, entry, keep_z):
        return None

    dist_anchor = float(np.linalg.norm(entry[:2] - anchor[:2]))
    score = _score_alley(
        width, depth, dist_anchor,
        max_corridor_width_m=max_corridor_width_m,
        max_depth_m=max_depth_m,
    )
    return AlleyCorridor(
        building_a=a,
        building_b=b,
        midpoint_ned=mid.copy(),
        axis_heading_rad=axis_h,
        gap_width_m=d,
        corridor_width_m=width,
        depth_m=depth,
        score=score,
        entry_ned=entry.copy(),
        deep_ned=deep.copy(),
    )


def find_alley_corridors(
    occ: PcdOccupancyMap,
    seg_map: SegBuildingMap,
    anchor_ned: np.ndarray,
    *,
    keep_z: float,
    min_gap_m: float = 10.0,
    max_gap_m: float = 36.0,
    min_corridor_width_m: float = 3.0,
    max_corridor_width_m: float = 14.0,
    min_depth_m: float = 10.0,
    max_depth_m: float = 55.0,
    search_radius_m: float = 150.0,
) -> list[AlleyCorridor]:
    """Enumerate hutongs between annotated building pairs near ``anchor``."""
    anchor = np.asarray(anchor_ned, dtype=np.float64).reshape(3)
    out: list[AlleyCorridor] = []
    for i, a in enumerate(seg_map.buildings):
        for b in seg_map.buildings[i + 1:]:
            mid = (a.ned_xyz + b.ned_xyz) * 0.5
            if float(np.linalg.norm(mid[:2] - anchor[:2])) > search_radius_m:
                continue
            cand = _build_alley_candidate(
                occ, a, b, anchor, keep_z=keep_z,
                min_gap_m=min_gap_m, max_gap_m=max_gap_m,
                min_corridor_width_m=min_corridor_width_m,
                max_corridor_width_m=max_corridor_width_m,
                min_depth_m=min_depth_m, max_depth_m=max_depth_m,
            )
            if cand is not None:
                out.append(cand)
    out.sort(key=lambda c: -c.score)
    return out


def find_best_alley_scene(
    occ: PcdOccupancyMap,
    seg_map: SegBuildingMap,
    *,
    keep_z: float,
    hint_anchor_ned: np.ndarray | None = None,
    min_corridor_width_m: float = 3.0,
    max_corridor_width_m: float = 12.0,
    min_depth_m: float = 10.0,
) -> tuple[AlleyCorridor | None, np.ndarray | None]:
    """Pick the best hutong globally and a drivable anchor on its entry road."""
    hint = (
        np.asarray(hint_anchor_ned, dtype=np.float64).reshape(3)
        if hint_anchor_ned is not None
        else np.zeros(3, dtype=np.float64)
    )
    best: AlleyCorridor | None = None
    best_anchor: np.ndarray | None = None
    best_q = -1e18

    for i, a in enumerate(seg_map.buildings):
        for b in seg_map.buildings[i + 1:]:
            cand = _build_alley_candidate(
                occ, a, b, hint, keep_z=keep_z,
                min_gap_m=10.0, max_gap_m=36.0,
                min_corridor_width_m=min_corridor_width_m,
                max_corridor_width_m=max_corridor_width_m,
                min_depth_m=min_depth_m, max_depth_m=55.0,
            )
            if cand is None:
                continue
            anchor = _snap_to_drivable(
                occ, cand.entry_ned, keep_z=keep_z, search_radius_m=36.0,
            )
            if anchor is None:
                continue
            # Prefer narrow hutongs with long depth, reachable entry road.
            q = cand.score - 0.08 * float(np.linalg.norm(anchor[:2] - hint[:2]))
            if q > best_q:
                best_q = q
                best = cand
                best_anchor = anchor.copy()

    return best, best_anchor


def _append_drivable_leg(
    occ: PcdOccupancyMap,
    start: np.ndarray,
    goal: np.ndarray,
    *,
    keep_z: float,
    step_m: float = 6.0,
    max_steps: int = 80,
) -> list[np.ndarray]:
    """Stepwise drivable polyline from ``start`` toward ``goal``."""
    out: list[np.ndarray] = []
    cur = np.asarray(start, dtype=np.float64).reshape(3).copy()
    cur[2] = keep_z
    goal = np.asarray(goal, dtype=np.float64).reshape(3).copy()
    goal[2] = keep_z
    guard = 0
    while guard < max_steps:
        guard += 1
        delta = goal[:2] - cur[:2]
        dist = float(np.linalg.norm(delta))
        if dist <= step_m * 0.55:
            if _segment_ok(occ, cur, goal, keep_z=keep_z):
                out.append(goal.copy())
            break
        direction = delta / max(dist, 1e-9)
        nxt = cur.copy()
        nxt[0] += direction[0] * step_m
        nxt[1] += direction[1] * step_m
        nxt[2] = keep_z
        if not _segment_ok(occ, cur, nxt, keep_z=keep_z):
            break
        out.append(nxt.copy())
        cur = nxt
    return out


def _densify_alley_leg(
    occ: PcdOccupancyMap,
    points: list[np.ndarray],
    *,
    keep_z: float,
    step_m: float = 4.0,
) -> list[np.ndarray]:
    """Short-step densification inside the hutong (tight corners)."""
    if not points:
        return []
    out = [np.asarray(points[0], dtype=np.float64).reshape(3).copy()]
    out[0][2] = keep_z
    for tgt in points[1:]:
        leg = _append_drivable_leg(
            occ, out[-1], tgt, keep_z=keep_z, step_m=step_m, max_steps=40,
        )
        out.extend(leg)
    return out


def build_alley_hutong_route(
    occ: PcdOccupancyMap,
    seg_map: SegBuildingMap,
    anchor_ned: np.ndarray,
    rng: np.random.Generator,
    *,
    keep_z: float,
    open_approach_m: float = 45.0,
    waypoint_step_m: float = 6.0,
    search_radius_m: float = 200.0,
    min_corridor_width_m: float = 3.0,
    max_corridor_width_m: float = 12.0,
    preset_alley: AlleyCorridor | None = None,
    max_attempts: int = 24,
) -> tuple[RoadRoute | None, dict[str, Any]]:
    """Build: anchor → open road → alley entry → deep into hutong."""
    anchor = np.asarray(anchor_ned, dtype=np.float64).reshape(3).copy()
    anchor[2] = keep_z
    if not _is_drivable(occ, anchor, keep_z):
        snapped = _snap_to_drivable(occ, anchor, keep_z=keep_z, search_radius_m=60.0)
        if snapped is not None:
            anchor = snapped

    alleys = (
        [preset_alley] if preset_alley is not None
        else find_alley_corridors(
            occ, seg_map, anchor, keep_z=keep_z,
            search_radius_m=search_radius_m,
            min_corridor_width_m=min_corridor_width_m,
            max_corridor_width_m=max_corridor_width_m,
        )
    )
    if not alleys and preset_alley is None:
        alley, suggested = find_best_alley_scene(
            occ, seg_map, keep_z=keep_z,
            hint_anchor_ned=anchor,
            min_corridor_width_m=min_corridor_width_m,
            max_corridor_width_m=max_corridor_width_m,
        )
        if alley is not None:
            alleys = [alley]
            if suggested is not None and float(np.linalg.norm(suggested[:2] - anchor[:2])) < 80.0:
                anchor = suggested

    for attempt, alley in enumerate(alleys[:max(1, max_attempts)]):
        if alley is None:
            continue
        attempt_rng = np.random.default_rng(int(rng.integers(0, 2**31)))

        # Short open-road cruise toward the alley entry (same pattern as hide route).
        open_len = min(open_approach_m, float(np.linalg.norm(alley.entry_ned[:2] - anchor[:2])) * 0.55)
        open_len = max(18.0, open_len)
        base = build_route(
            occ, anchor, attempt_rng,
            keep_z=keep_z,
            route_len_m=open_len,
            waypoint_step_m=waypoint_step_m,
            search_radius_m=min(search_radius_m, open_approach_m + 20.0),
            maneuver="normal_drive",
            start_at_anchor=_is_drivable(occ, anchor, keep_z),
        )
        wps_list: list[np.ndarray] = [
            np.asarray(p, dtype=np.float64).reshape(3).copy()
            for p in base.waypoints
        ]
        for p in wps_list:
            p[2] = keep_z

        start = wps_list[-1].copy()
        approach = _append_drivable_leg(
            occ, start, alley.entry_ned,
            keep_z=keep_z, step_m=waypoint_step_m,
        )
        if not approach:
            # Try snapping start toward a major road seed closer to entry.
            seed, _, _ = find_major_road_seed(
                occ, alley.entry_ned, attempt_rng,
                keep_z=keep_z, search_radius_m=open_approach_m,
            )
            bridge = _append_drivable_leg(
                occ, start, seed, keep_z=keep_z, step_m=waypoint_step_m,
            )
            wps_list.extend(bridge)
            start = wps_list[-1].copy() if wps_list else start
            approach = _append_drivable_leg(
                occ, start, alley.entry_ned,
                keep_z=keep_z, step_m=waypoint_step_m,
            )
        wps_list.extend(approach)

        alley_pts = _densify_alley_leg(
            occ,
            [alley.entry_ned, alley.midpoint_ned, alley.deep_ned],
            keep_z=keep_z,
            step_m=min(4.0, waypoint_step_m),
        )
        if len(alley_pts) < 2:
            continue
        wps_list.extend(alley_pts)

        if len(wps_list) < 4:
            continue

        wps = np.asarray(wps_list, dtype=np.float64)
        split_idx = max(0, len(wps_list) - len(alley_pts))
        total_len = sum(
            float(np.linalg.norm(wps[i + 1][:2] - wps[i][:2]))
            for i in range(len(wps_list) - 1)
        )
        meta = {
            "planner": "alley_hutong",
            "attempt": attempt,
            "alley_building_a": alley.building_a.index,
            "alley_building_b": alley.building_b.index,
            "gap_width_m": round(alley.gap_width_m, 1),
            "corridor_width_m": round(alley.corridor_width_m, 1),
            "alley_depth_m": round(alley.depth_m, 1),
            "split_idx": split_idx,
            "route_waypoints": int(wps.shape[0]),
            "anchor_ned": anchor.tolist(),
            "entry_ned": alley.entry_ned.tolist(),
            "deep_ned": alley.deep_ned.tolist(),
            "est_alley_start_s": round(
                float(np.linalg.norm(wps[split_idx][:2] - wps[0][:2])) / 3.2, 1,
            ),
            "est_route_total_s": round(total_len / 3.2, 1),
        }
        meta["recommended_duration_s"] = round(
            max(50.0, meta["est_route_total_s"] * 1.2), 0,
        )
        return RoadRoute(
            waypoints=wps,
            headings=_headings_for_waypoints(wps),
            score=float(alley.score),
        ), meta

    return None, {"planner": "alley_hutong", "error": "no_alley_found"}


__all__ = [
    "AlleyCorridor",
    "build_alley_hutong_route",
    "find_alley_corridors",
    "find_best_alley_scene",
]

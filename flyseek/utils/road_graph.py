# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Road-corridor route sampling from the PCD BEV map.

This module stays fully offline: it only queries the already-built BEV
occupancy map and returns NED waypoints for teleport rendering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from flyseek.adapters.pcd_occupancy import PcdOccupancyMap
from flyseek.adversary.base import wrap_to_pi


@dataclass
class RoadRoute:
    waypoints: np.ndarray
    headings: np.ndarray
    score: float


def _is_free(occupancy: PcdOccupancyMap, p: np.ndarray, keep_z: float) -> bool:
    q = np.asarray(p, dtype=np.float64).reshape(3).copy()
    q[2] = keep_z
    if hasattr(occupancy, "is_drivable_ned"):
        return occupancy.is_drivable_ned(q)
    return not occupancy.is_bev_occupied_ned(q)


def _segment_lateral_bev_penalty(
    occupancy: PcdOccupancyMap,
    a: np.ndarray,
    b: np.ndarray,
    keep_z: float,
    *,
    car_half_width_m: float = 0.9,
    step_m: float = 1.0,
) -> float:
    """SOFT penalty (count of BEV-occupied cells) found along ±half-width
    of the segment ``a→b``.

    Walls / guardrails / building corners light this up. The penalty is
    additive across midpoint samples so a candidate whose entire segment
    grazes a wall is heavily disfavoured, but one that just clips a corner
    isn't auto-rejected (avoids the build_route deadlock on narrow PCDs).
    """
    if not hasattr(occupancy, "is_bev_free_ned"):
        return 0.0
    a2 = a[:2].astype(np.float64)
    b2 = b[:2].astype(np.float64)
    seg = b2 - a2
    L = float(np.linalg.norm(seg))
    if L < 1e-6:
        return 0.0
    d = seg / L
    perp = np.array([-d[1], d[0]], dtype=np.float64)
    n = max(2, int(math.ceil(L / max(step_m, 0.25))))
    penalty = 0.0
    for i in range(1, n + 1):
        t = i / n
        c = a + t * (b - a)
        c[2] = keep_z
        for sign in (1.0, -1.0):
            q = c.copy()
            q[0] += perp[0] * sign * car_half_width_m
            q[1] += perp[1] * sign * car_half_width_m
            if not occupancy.is_bev_free_ned(q):
                penalty += 1.0
    return penalty


def _edge_safety_penalty(
    occupancy: PcdOccupancyMap,
    p: np.ndarray,
    keep_z: float,
    *,
    safety_margin_m: float = 1.0,
) -> float:
    """SOFT penalty for waypoints adjacent to non-drivable cells.

    Returns a non-negative penalty (subtracted from waypoint score) equal to
    the number of immediate lateral neighbours that are NOT drivable. Used as
    a scoring tie-breaker, NOT a hard reject: on narrow voxel-wide roads (e.g.
    env_airsim_16's 2-4 m strips) every waypoint legitimately fails the strict
    "all 4 neighbours drivable" gate, so a hard reject would deadlock
    ``build_route``. We instead prefer center-lane waypoints when available
    and tolerate edge waypoints when nothing else is reachable.
    """
    if safety_margin_m <= 0.0:
        return 0.0
    penalty = 0.0
    for dx, dy in ((safety_margin_m, 0.0), (-safety_margin_m, 0.0),
                   (0.0, safety_margin_m), (0.0, -safety_margin_m)):
        q = np.asarray(p, dtype=np.float64).reshape(3).copy()
        q[0] += dx
        q[1] += dy
        q[2] = keep_z
        if not _is_free(occupancy, q, keep_z):
            penalty += 1.0
    return penalty


def _ray_free(
    occupancy: PcdOccupancyMap,
    p: np.ndarray,
    heading: float,
    *,
    keep_z: float,
    max_dist: float,
    step: float = 2.0,
) -> float:
    dist = 0.0
    cur = p.copy()
    while dist + step <= max_dist:
        cur = cur.copy()
        cur[0] += math.cos(heading) * step
        cur[1] += math.sin(heading) * step
        cur[2] = keep_z
        if not _is_free(occupancy, cur, keep_z):
            break
        dist += step
    return dist


def _corridor_width(
    occupancy: PcdOccupancyMap,
    p: np.ndarray,
    heading: float,
    *,
    keep_z: float,
    max_width: float = 22.0,
    step: float = 2.0,
) -> float:
    side = heading + math.pi / 2.0
    total = 0.0
    for sign in (-1.0, 1.0):
        cur = p.copy()
        walked = 0.0
        while walked + step <= max_width:
            cur = cur.copy()
            cur[0] += math.cos(side) * step * sign
            cur[1] += math.sin(side) * step * sign
            cur[2] = keep_z
            if not _is_free(occupancy, cur, keep_z):
                break
            walked += step
        total += walked
    return total


def road_score(
    occupancy: PcdOccupancyMap,
    p: np.ndarray,
    heading: float,
    *,
    keep_z: float,
) -> float:
    fwd = _ray_free(occupancy, p, heading, keep_z=keep_z, max_dist=90.0)
    back = _ray_free(occupancy, p, heading + math.pi, keep_z=keep_z, max_dist=35.0)
    width = _corridor_width(occupancy, p, heading, keep_z=keep_z)
    # Large roads should be long and laterally open; narrow alleys score low.
    return fwd * 1.0 + back * 0.25 + width * 2.0


def _best_anchor_heading(
    occupancy: PcdOccupancyMap,
    pos_ned: np.ndarray,
    *,
    keep_z: float,
    hint_heading: float,
    n_dirs: int = 16,
) -> float:
    """Pick the heading at ``pos_ned`` maximising road_score, biased toward
    ``hint_heading`` (which usually comes from find_major_road_seed)."""
    best_h = float(hint_heading)
    best_score = -1e9
    for k in range(n_dirs):
        h = 2.0 * math.pi * k / n_dirs
        s = road_score(occupancy, pos_ned, h, keep_z=keep_z)
        # Bias toward hint heading to keep the route consistent with the
        # surrounding major road's orientation.
        s -= 1.5 * abs(wrap_to_pi(h - hint_heading))
        if s > best_score:
            best_score = s
            best_h = h
    return best_h


def find_major_road_seed(
    occupancy: PcdOccupancyMap,
    anchor_ned: np.ndarray,
    rng: np.random.Generator,
    *,
    keep_z: float,
    search_radius_m: float = 180.0,
    sample_step_m: float = 12.0,
) -> tuple[np.ndarray, float, float]:
    """Find a nearby point that looks like a major street, not a small alley."""
    anchor = np.asarray(anchor_ned, dtype=np.float64).reshape(3).copy()
    anchor[2] = keep_z
    best_p = anchor.copy()
    best_h = 0.0
    best_score = -1e9

    offsets: list[tuple[float, float]] = [(0.0, 0.0)]
    radii = np.arange(sample_step_m, search_radius_m + 1e-6, sample_step_m)
    for r in radii:
        n = max(12, int(2 * math.pi * r / sample_step_m))
        jitter = float(rng.uniform(0.0, 2.0 * math.pi / n))
        for k in range(n):
            a = jitter + 2.0 * math.pi * k / n
            offsets.append((r * math.cos(a), r * math.sin(a)))

    headings = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
    rng.shuffle(offsets)
    for dx, dy in offsets[:900]:
        p = anchor.copy()
        p[0] += dx
        p[1] += dy
        p[2] = keep_z
        if not _is_free(occupancy, p, keep_z):
            continue
        for h in headings:
            score = road_score(occupancy, p, float(h), keep_z=keep_z)
            # Mild preference for not teleporting too far from the actor.
            score -= 0.03 * float(np.linalg.norm(p[:2] - anchor[:2]))
            if score > best_score:
                best_score = score
                best_p = p.copy()
                best_h = float(h)

    best_h = wrap_to_pi(best_h + float(rng.normal(0.0, 0.04)))
    return best_p, best_h, best_score


def build_route(
    occupancy: PcdOccupancyMap,
    anchor_ned: np.ndarray,
    rng: np.random.Generator,
    *,
    keep_z: float,
    route_len_m: float = 260.0,
    waypoint_step_m: float = 6.0,
    search_radius_m: float = 180.0,
    maneuver: str = "normal_drive",
    start_at_anchor: bool = False,
    anchor_heading_rad: float | None = None,
    open_road_frac: float = 0.68,
) -> RoadRoute:
    """Build a route constrained to wide BEV-free corridors.

    When ``start_at_anchor`` is True and ``anchor_ned`` is drivable, the route's
    first waypoint is the anchor itself (so a car teleported to the anchor can
    follow the route from tick 0 without first traversing non-drivable cells).
    """
    anchor = np.asarray(anchor_ned, dtype=np.float64).reshape(3).copy()
    anchor[2] = keep_z
    seed_p, seed_h, seed_score = find_major_road_seed(
        occupancy,
        anchor_ned,
        rng,
        keep_z=keep_z,
        search_radius_m=search_radius_m,
    )
    if start_at_anchor and _is_free(occupancy, anchor, keep_z):
        p = anchor
        if anchor_heading_rad is not None:
            heading = float(anchor_heading_rad)
        else:
            # Use the heading whose forward ray is longest from the anchor,
            # biased toward the major-road seed.
            heading = _best_anchor_heading(
                occupancy, p, keep_z=keep_z,
                hint_heading=seed_h,
            )
    else:
        p = seed_p
        heading = seed_h
    points = [p.copy()]
    headings = [heading]
    travelled = 0.0

    open_budget = route_len_m * float(open_road_frac) if maneuver == "open_then_hide" else 0.0

    # Iteration cap: in pathological PCDs (very narrow / no-candidate corridors)
    # the heading-flip retry path could otherwise spin forever. Allow up to 4×
    # the expected waypoint count to retry-with-rotation, then bail with what
    # we have. 4× is generous (a real route grows by 1 waypoint per iteration).
    max_iters = max(60, 4 * int(route_len_m / max(1.0, waypoint_step_m)))
    iters = 0
    while travelled < route_len_m and len(points) < 180 and iters < max_iters:
        iters += 1
        candidates: list[tuple[float, float, np.ndarray]] = []
        in_alley_leg = maneuver == "open_then_hide" and travelled >= open_budget
        if maneuver in ("high_maneuver", "corner_occlude"):
            deltas = (-math.pi / 2, -math.pi / 3, -math.pi / 6, 0.0,
                      math.pi / 6, math.pi / 3, math.pi / 2)
        elif in_alley_leg:
            # Sharper turns toward narrower free corridors (alleys / hutongs).
            deltas = (-math.pi / 2, -math.pi / 3, -math.pi / 6, 0.0,
                      math.pi / 6, math.pi / 3, math.pi / 2)
        else:
            deltas = (-math.pi / 4, -math.pi / 8, 0.0, math.pi / 8, math.pi / 4)

        for d in deltas:
            h = wrap_to_pi(heading + d + float(rng.normal(0.0, 0.03)))
            step = waypoint_step_m
            q = points[-1].copy()
            q[0] += math.cos(h) * step
            q[1] += math.sin(h) * step
            q[2] = keep_z
            if not _is_free(occupancy, q, keep_z):
                continue
            # Hard gate: the *entire* segment from the previous waypoint to
            # this candidate must be drivable (centerline only). Without
            # this, build_route is happy to place waypoints on isolated
            # drivable pockets surrounded by walls — which the spline+clamp
            # then refuses to traverse, freezing the car.
            if hasattr(occupancy, "_segment_drivable"):
                if not occupancy._segment_drivable(
                    points[-1], q, keep_z=keep_z
                ):
                    continue
            score = road_score(occupancy, q, h, keep_z=keep_z)
            if in_alley_leg:
                width = _corridor_width(occupancy, q, h, keep_z=keep_z)
                # Prefer narrower walkable corridors behind building fronts.
                score += max(0.0, 14.0 - width) * 1.8
                score -= abs(d) * 3.0
            else:
                score -= abs(d) * (10.0 if maneuver != "high_maneuver" else 3.0)
            # Avoid immediate backtracking to keep the route visibly progressing.
            if len(points) > 4:
                score += 0.02 * float(np.linalg.norm(q[:2] - points[0][:2]))
            # Soft edge-safety penalty: prefer interior of the corridor when
            # available, but do NOT hard-reject edge waypoints (the road is
            # often only 1-2 voxels wide).
            score -= 2.5 * _edge_safety_penalty(
                occupancy, q, keep_z, safety_margin_m=1.0,
            )
            # Soft segment-lateral penalty: discourage candidates whose
            # connecting segment grazes a building / guardrail. Heavier weight
            # (3.0×) because hugging a wall along an ENTIRE segment is worse
            # than a single neighbouring cell at one waypoint.
            score -= 3.0 * _segment_lateral_bev_penalty(
                occupancy, points[-1], q, keep_z,
                car_half_width_m=0.9, step_m=1.0,
            )
            candidates.append((score, h, q))

        if not candidates:
            heading = wrap_to_pi(heading + math.pi / 2.0)
            continue

        candidates.sort(key=lambda x: x[0], reverse=True)
        _, heading, p = candidates[0]
        points.append(p.copy())
        headings.append(heading)
        travelled += waypoint_step_m

    return RoadRoute(
        waypoints=np.asarray(points, dtype=np.float64),
        headings=np.asarray(headings, dtype=np.float64),
        score=float(seed_score),
    )


__all__ = ["RoadRoute", "build_route", "find_major_road_seed", "road_score"]

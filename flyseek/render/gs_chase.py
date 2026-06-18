# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""GS-frame UAV-tracks-car geometry (backend-agnostic, no rendering).

Because the aerial 3DGS reconstruction has no usable occupancy road network,
the car follows a hand-traced polyline (config ``car_route.waypoints``) on a
real road in the good region. The UAV trails the car at a fixed horizontal
lag in the good-render altitude band and its camera yaw/pitch are solved so
the car stays centered, using the validated ``gs_camera`` projection.

Everything is in the PCD/GS world frame (== unified metric).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flyseek.render.gs_camera import Intrinsics, bridge_extrinsics, project


@dataclass
class CarState:
    t: float
    pos: np.ndarray          # world (x,y,z)
    heading: float           # yaw of travel direction (rad, atan2(dy,dx))


@dataclass
class UavState:
    t: float
    pos: np.ndarray          # world (x,y,z) = camera center
    yaw_input: float         # deg, as passed to gs_bridge.set_camera_pose
    pitch_deg: float         # deg, as passed (overrides the hardcoded -40)
    car_uv: tuple | None     # car pixel (u,v) or None if out of frame
    car_in_view: bool


def resample_polyline(waypoints, step_m: float):
    """Resample an ordered list of (x,y) waypoints to ~uniform spacing.

    Returns (pts [N,2], headings [N])."""
    wp = np.asarray(waypoints, dtype=np.float64)
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    total = float(seg.sum())
    n = max(2, int(np.ceil(total / step_m)) + 1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.linspace(0.0, total, n)
    xs = np.interp(s, cum, wp[:, 0])
    ys = np.interp(s, cum, wp[:, 1])
    pts = np.stack([xs, ys], axis=1)
    d = np.gradient(pts, axis=0)
    headings = np.arctan2(d[:, 1], d[:, 0])
    return pts, headings


def build_car_track(cfg: dict, geom=None) -> list[CarState]:
    """Car states along the route. If `geom` (SceneGeometry) is given, the car
    z hugs the local ground per-position; else falls back to cfg['ground_z']."""
    route = cfg["car_route"]
    gz_default = float(cfg["ground_z"])
    pts, headings = resample_polyline(route["waypoints"], float(route["waypoint_step_m"]))
    speed = float(route["speed_mps"])
    step = float(route["waypoint_step_m"])
    dt = step / max(speed, 1e-3)
    out = []
    for i, ((x, y), h) in enumerate(zip(pts, headings)):
        z = gz_default
        if geom is not None:
            # prefer a detected parked-car base height (reliable road surface),
            # fall back to the sparse-cloud ground estimate
            za = geom.car_anchor_z(x, y, default=None)
            z = za if za is not None else geom.ground_z(x, y, default=gz_default)
        out.append(CarState(t=i * dt, pos=np.array([x, y, z]), heading=float(h)))
    return out


def build_car_track_policy(cfg: dict, geom=None) -> list[CarState]:
    """env_airsim_16-style adversarial target trajectory in the GS world frame.

    Rolls out :class:`flyseek.bench.target_policy.TargetPolicy` (the same
    evasive behaviours used in the AirSim demos: ``direct_escape`` /
    ``sharp_turn`` / ``detour_feint``) in the unified metric XY plane. The
    occupancy map is *not* used (the aerial 3DGS reconstruction has no usable
    road graph); instead the car is confined to a ``play_box`` corridor on the
    validated street via velocity reflection, and z is snapped to the local
    road surface (detected parked-car anchors -> sparse-cloud ground).

    A virtual chaser (trailing the car at ``follow_distance``) is fed to the
    policy each step so reactive behaviours (escape/feint) respond as they do
    in the closed-loop AirSim chase.
    """
    # lazy import: target_policy pulls flyseek.scenarios (avoid import cycle)
    from flyseek.adversary.base import (DroneState, PlayBox, TargetState,
                                        bearing_xy)
    from flyseek.bench.target_policy import TargetPolicy

    tcfg = cfg["target"]
    gz_default = float(cfg["ground_z"])
    dt = float(tcfg.get("dt", 0.2))
    dur = float(tcfg.get("episode_seconds", 12.0))
    n = max(2, int(round(dur / dt)) + 1)
    sx, sy = tcfg["start"]
    z0 = gz_default
    if geom is not None:
        za = geom.car_anchor_z(sx, sy, default=None)
        z0 = za if za is not None else geom.ground_z(sx, sy, default=gz_default)
    h0 = np.radians(float(tcfg.get("init_heading_deg", -90.0)))

    pb = None
    box = tcfg.get("play_box")
    if box is not None:
        pb = PlayBox(float(box["xmin"]), float(box["xmax"]),
                     float(box["ymin"]), float(box["ymax"]))

    # Tube confinement around the curved road centerline (preferred over the
    # axis-aligned play_box for a non-axis-aligned street). Uses the car_route
    # waypoints as the centerline unless target.road_centerline is given.
    center = None
    half_w = float(tcfg.get("corridor_half_width_m", 3.0))
    cl_wps = tcfg.get("road_centerline") or cfg.get("car_route", {}).get("waypoints")
    if cl_wps is not None and len(cl_wps) >= 2:
        center, _ = resample_polyline(cl_wps, 1.0)  # dense centerline [M,2]

    follow = float(cfg["uav"].get("follow_distance_m", 12.0))
    alt = float(cfg["uav"].get("altitude_z", 2.0))

    policy = TargetPolicy(
        config={"behavior_type": tcfg.get("behavior", "sharp_turn"),
                "difficulty": tcfg.get("difficulty", "medium"),
                "dt": dt, **{k: tcfg[k] for k in
                             ("speed_mps", "max_turn_rate_deg_s",
                              "sharp_turn_interval_s", "sharp_turn_deg")
                             if k in tcfg}},
        scene_context={"keep_z": z0},
        seed=int(tcfg.get("seed", 42)),
    )
    tgt = TargetState(position=np.array([sx, sy, z0]),
                      velocity=np.zeros(3), heading=h0, timestamp=0.0)
    policy.reset(tgt)

    out = [CarState(t=0.0, pos=tgt.position.copy(), heading=float(tgt.heading))]
    for i in range(1, n):
        back = float(tgt.heading) + np.pi
        drone = DroneState(
            position=tgt.position + np.array([np.cos(back) * follow,
                                              np.sin(back) * follow, alt]),
            velocity=np.zeros(3), heading=float(tgt.heading), timestamp=i * dt)
        tgt = policy.get_next_target_state(i * dt, tgt, drone)
        # confine to the renderable street corridor (axis-aligned box)
        if pb is not None and not pb.contains(tgt.position):
            v = pb.reflect_velocity_at_boundary(tgt.position, tgt.velocity)
            clamped = pb.clamp(tgt.position)
            new_h = bearing_xy(tgt.position, tgt.position + v) if np.linalg.norm(v[:2]) > 1e-6 else float(tgt.heading) + np.pi
            tgt = tgt.copy_with(position=clamped, velocity=v, heading=new_h)
            policy._mdir = new_h  # steer the cruise heading back inward
        # tube confinement around the curved road centerline
        if center is not None:
            pxy = tgt.position[:2]
            d2 = np.sum((center - pxy) ** 2, axis=1)
            k = int(np.argmin(d2))
            if d2[k] > half_w * half_w:
                foot = center[k]
                k2 = min(k + 1, len(center) - 1)
                tang = center[k2] - center[max(k - 1, 0)]
                tn = np.linalg.norm(tang)
                tang = tang / tn if tn > 1e-6 else np.array([np.cos(tgt.heading), np.sin(tgt.heading)])
                normal = pxy - foot
                nn = np.linalg.norm(normal)
                normal = normal / nn if nn > 1e-6 else np.zeros(2)
                new_xy = foot + half_w * normal
                # reflect the lateral velocity component, keep longitudinal flow
                v = tgt.velocity.copy()
                v[:2] = v[:2] - 2.0 * np.dot(v[:2], normal) * normal
                # bias heading along the road tangent (forward sense preserved)
                fwd = tang if np.dot(v[:2], tang) >= 0 else -tang
                new_h = float(np.arctan2(fwd[1], fwd[0]))
                p2 = tgt.position.copy(); p2[0], p2[1] = new_xy
                tgt = tgt.copy_with(position=p2, velocity=v, heading=new_h)
                policy._mdir = new_h
        # snap z to local road surface (prefer detected road-car anchor; reject
        # podium/roof ground estimates above road_z_max -> use the road default)
        road_z_max = float(tcfg.get("road_z_max", -13.5))
        z = gz_default
        if geom is not None:
            za = geom.car_anchor_z(tgt.position[0], tgt.position[1], default=None)
            if za is not None:
                z = za
            else:
                zg = geom.ground_z(tgt.position[0], tgt.position[1], default=gz_default)
                z = zg if zg <= road_z_max else gz_default
        p = tgt.position.copy(); p[2] = z
        tgt = tgt.copy_with(position=p)
        out.append(CarState(t=i * dt, pos=p.copy(), heading=float(tgt.heading)))
    return out


def aim_camera_at(cam_xyz, target_xyz, intr: Intrinsics,
                  yaw0_deg: float = 0.0):
    """Solve (yaw_input_deg, pitch_deg) so target projects nearest image center.

    Coarse grid + local refine over the validated bridge projection.
    """
    cam = np.asarray(cam_xyz, float)
    tgt = np.asarray(target_xyz, float).reshape(1, 3)
    cx, cy = intr.cx, intr.cy

    def cost(yaw, pitch):
        R, t = bridge_extrinsics(cam, yaw, pitch)
        uv, z, _ = project(tgt, R, t, intr)
        if z[0] <= 1e-6:
            return 1e18, None
        du, dv = uv[0, 0] - cx, uv[0, 1] - cy
        return du * du + dv * dv, uv[0]

    best = (1e30, 0.0, -45.0, None)
    for yaw in range(0, 360, 6):
        for pitch in range(-88, -9, 4):
            c, uv = cost(float(yaw), float(pitch))
            if c < best[0]:
                best = (c, float(yaw), float(pitch), uv)
    # local refine
    by, bp = best[1], best[2]
    for yaw in np.arange(by - 5, by + 5.01, 1.0):
        for pitch in np.arange(bp - 4, bp + 4.01, 1.0):
            c, uv = cost(float(yaw), float(pitch))
            if c < best[0]:
                best = (c, float(yaw), float(pitch), uv)
    return best[1], best[2], best[3]


def build_uav_track(car: list[CarState], cfg: dict, intr: Intrinsics,
                    geom=None, region: dict | None = None) -> list[UavState]:
    """LOS-aware UAV viewpoint selection.

    For each car state, search candidate drone poses (follow distance, lateral
    offset, altitude) and pick the one that (a) has clear line-of-sight to the
    car (no building between, via geom.los_clear), (b) stays in the good
    region, (c) is smooth vs. the previous pose, and (d) keeps a pleasant
    oblique pitch. Falls back to the simple trail when geom is None.
    """
    uav_cfg = cfg["uav"]
    z0 = float(uav_cfg["altitude_z"])
    lag0 = float(uav_cfg["follow_distance_m"])
    alpha = float(uav_cfg.get("smooth_alpha", 0.4))
    desired_pitch = float(uav_cfg.get("desired_pitch_deg", -45.0))
    lat_max = float(uav_cfg.get("lateral_max_m", 9.0))
    z_band = uav_cfg.get("alt_band", [-2.0, 5.0])

    def in_region(x, y):
        if region is None:
            return True
        return (region["xmin"] <= x <= region["xmax"]
                and region["ymin"] <= y <= region["ymax"])

    # include short follow distances -> steeper, near-overhead poses that clear
    # building occlusion in narrow canyons (the scorer still prefers the desired
    # oblique pitch, but will accept a steeper unoccluded pose over a blocked one)
    follows = ([lag0 * 0.35, lag0 * 0.55, lag0 * 0.7, lag0, lag0 * 1.3]
               if geom is not None else [lag0])
    laterals = [-lat_max, -lat_max / 2, 0.0, lat_max / 2, lat_max] if geom is not None else [0.0]
    alts = [z0 - 2, z0, z0 + 2, z0 + 4] if geom is not None else [z0]
    alts = [a for a in alts if z_band[0] <= a <= z_band[1]] or [z0]

    out: list[UavState] = []
    prev_xy = None
    for cs in car:
        hx, hy = np.cos(cs.heading), np.sin(cs.heading)
        hdir = np.array([hx, hy]); perp = np.array([-hy, hx])
        best = None
        for lag in follows:
            for lat in laterals:
                for za in alts:
                    xy = cs.pos[:2] - lag * hdir + lat * perp
                    if not in_region(xy[0], xy[1]):
                        continue
                    cam = np.array([xy[0], xy[1], za])
                    yaw, pitch, _ = aim_camera_at(cam, cs.pos, intr)
                    if geom is not None:
                        Rc, tc = bridge_extrinsics(cam, yaw, pitch)
                        los = geom.point_visible(cs.pos, Rc, tc, intr)
                    else:
                        los = True
                    score = 0.0
                    score += 1000.0 if los else 0.0
                    score -= abs(pitch - desired_pitch)
                    score -= 0.15 * abs(lag - lag0) + 0.1 * abs(lat)
                    if prev_xy is not None:
                        score -= 0.4 * float(np.linalg.norm(xy - prev_xy))
                    cand = (score, cam, yaw, pitch, los)
                    if best is None or cand[0] > best[0]:
                        best = cand
        _, cam, yaw, pitch, los = best
        # light EMA on xy for smoothness
        if prev_xy is not None:
            sm = alpha * cam[:2] + (1 - alpha) * prev_xy
            cam = np.array([sm[0], sm[1], cam[2]])
            yaw, pitch, _ = aim_camera_at(cam, cs.pos, intr)
            if geom is not None:
                Rc, tc = bridge_extrinsics(cam, yaw, pitch)
                los = geom.point_visible(cs.pos, Rc, tc, intr)
        prev_xy = cam[:2]
        R, t = bridge_extrinsics(cam, yaw, pitch)
        uvp, zz, valid = project(cs.pos.reshape(1, 3), R, t, intr)
        out.append(UavState(
            t=cs.t, pos=cam, yaw_input=yaw, pitch_deg=pitch,
            car_uv=(float(uvp[0, 0]), float(uvp[0, 1])) if bool(valid[0]) else None,
            car_in_view=bool(valid[0]) and los,
        ))
    return out


__all__ = ["CarState", "UavState", "resample_polyline", "build_car_track",
           "build_car_track_policy", "aim_camera_at", "build_uav_track"]

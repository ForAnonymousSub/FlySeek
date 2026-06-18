# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Hide-goal visibility — align route planning with drone camera frustum + LoS.

P0: ``target_hidden_from_drone`` mirrors demo ``target_visible`` (FOV + building LoS).
P0: ``sample_chase_drone_poses`` validates hide points against multiple chase poses.
P1: delegates occluder-between check to ``PcdOccupancyMap.building_occludes_between_ned``.
P2: pinhole frustum via ``project_ned_to_pixel``; BEV preview via ``render_route_bev_preview``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from flyseek.adapters.pcd_occupancy import PcdOccupancyMap
from flyseek.adversary.base import DroneState, TargetState, bearing_xy, horizontal_distance
from flyseek.utils.seg_bbox import project_ned_to_pixel
from flyseek.utils.visibility import visibility_status


@dataclass(frozen=True)
class HideVisibilityConfig:
    """Knobs shared by route planning, runtime correction, and BEV preview."""

    hfov_deg: float = 50.0
    max_range_m: float = 100.0
    drone_eye_agl_m: float = 12.0
    follow_distance_m: float = 12.0
    cam_forward_m: float = 0.45
    cam_down_m: float = 0.25
    cam_pitch_deg: float = 55.0
    cam_width: int = 256
    cam_height: int = 144
    building_only_los: bool = True
    min_building_height_m: float | None = None
    min_footprint_cells: int = 9
    use_frustum_projection: bool = True
    occluder_between_required: bool = True
    occluder_near_target_m: float = 12.0
    chase_drone_samples: int = 5
    seg_building_map: Any | None = None

    @classmethod
    def from_args(cls, args: Any) -> "HideVisibilityConfig":
        def _g(name: str, default: Any) -> Any:
            return getattr(args, name, default)

        return cls(
            hfov_deg=float(_g("camera_hfov_deg", 50.0)),
            max_range_m=float(_g("vis_max_range_m", 100.0)),
            drone_eye_agl_m=float(_g("follow_altitude", 12.0)),
            follow_distance_m=float(_g("follow_distance", 12.0)),
            cam_forward_m=float(_g("camera_body_forward_m", 0.45)),
            cam_down_m=float(_g("camera_body_down_m", 0.25)),
            cam_pitch_deg=float(_g("camera_pitch_deg", 55.0)),
            cam_width=int(_g("camera_width", 256)),
            cam_height=int(_g("camera_height", 144)),
            min_building_height_m=(
                float(_g("min_building_height_m", 18.0))
                if _g("min_building_height_m", None) is not None else None
            ),
            min_footprint_cells=int(_g("min_building_footprint_cells", 9)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _camera_config(cfg: HideVisibilityConfig) -> dict[str, Any]:
    return {
        "hfov_deg": cfg.hfov_deg,
        "width": cfg.cam_width if cfg.use_frustum_projection else None,
        "height": cfg.cam_height if cfg.use_frustum_projection else None,
        "body_forward_m": cfg.cam_forward_m,
        "body_down_m": cfg.cam_down_m,
        "pitch_deg": cfg.cam_pitch_deg,
    }


def _scene_context(
    occupancy: PcdOccupancyMap,
    cfg: HideVisibilityConfig,
) -> dict[str, Any]:
    ctx: dict[str, Any] = {
        "occupancy": occupancy,
        "drone_eye_agl_m": cfg.drone_eye_agl_m,
    }
    if cfg.building_only_los:
        ctx["building_only_los"] = True
        ctx["min_building_height_m"] = cfg.min_building_height_m
        ctx["min_footprint_cells"] = cfg.min_footprint_cells
    return ctx


def target_in_camera_frustum(
    drone: DroneState,
    target: TargetState,
    cfg: HideVisibilityConfig,
) -> bool:
    """P2: pinhole projection when width/height set; else horizontal FOV bearing."""
    r = horizontal_distance(drone.position, target.position)
    if r > cfg.max_range_m:
        return False
    if cfg.use_frustum_projection and cfg.cam_width > 0 and cfg.cam_height > 0:
        uv = project_ned_to_pixel(
            target.position,
            drone.position,
            float(drone.heading),
            width=int(cfg.cam_width),
            height=int(cfg.cam_height),
            hfov_deg=cfg.hfov_deg,
            cam_forward_m=cfg.cam_forward_m,
            cam_down_m=cfg.cam_down_m,
            cam_pitch_deg=cfg.cam_pitch_deg,
        )
        if uv is None:
            return False
        u, v = uv
        return (0 <= u < cfg.cam_width) and (0 <= v < cfg.cam_height)
    bearing = bearing_xy(drone.position, target.position)
    delta = abs(float(bearing - drone.heading))
    delta = (delta + math.pi) % (2 * math.pi) - math.pi
    return abs(delta) <= math.radians(cfg.hfov_deg * 0.5)


def target_hidden_from_drone(
    occupancy: PcdOccupancyMap | None,
    drone: DroneState,
    target: TargetState,
    cfg: HideVisibilityConfig,
) -> tuple[bool, str]:
    """True when the target is NOT visible from the drone (inverse of demo check)."""
    visible, reason = visibility_status(
        occupancy,
        drone,
        target,
        hfov_deg=cfg.hfov_deg,
        max_range_m=cfg.max_range_m,
        drone_eye_agl_m=cfg.drone_eye_agl_m,
        building_only_los=cfg.building_only_los,
        min_building_height_m=cfg.min_building_height_m,
        min_footprint_cells=cfg.min_footprint_cells,
        use_frustum_projection=cfg.use_frustum_projection,
        cam_forward_m=cfg.cam_forward_m,
        cam_down_m=cfg.cam_down_m,
        cam_pitch_deg=cfg.cam_pitch_deg,
        cam_width=cfg.cam_width,
        cam_height=cfg.cam_height,
        seg_building_map=cfg.seg_building_map,
    )
    if visible:
        return False, reason
    return True, reason


def make_chase_drone_at_target(
    target_pos: np.ndarray,
    target_heading: float,
    cfg: HideVisibilityConfig,
) -> DroneState:
    """Nominal chase drone behind ``target`` at follow distance."""
    back = float(target_heading) + math.pi
    fd = cfg.follow_distance_m
    fa = cfg.drone_eye_agl_m
    pos = np.asarray(target_pos, dtype=np.float64).reshape(3).copy()
    drone_pos = pos + np.array([
        math.cos(back) * fd,
        math.sin(back) * fd,
        -abs(fa),
    ], dtype=np.float64)
    yaw = math.atan2(pos[1] - drone_pos[1], pos[0] - drone_pos[0])
    return DroneState(position=drone_pos, velocity=np.zeros(3), heading=float(yaw))


def sample_chase_drone_poses(
    waypoints: np.ndarray,
    *,
    split_idx: int,
    cfg: HideVisibilityConfig,
    initial_drone_ned: np.ndarray | None = None,
) -> list[DroneState]:
    """P0: sample chase drones along the hide leg (+ optional t=0 pose)."""
    wps = np.asarray(waypoints, dtype=np.float64)
    if wps.shape[0] < 2:
        return []

    hide = wps[max(0, int(split_idx)):]
    n = max(1, int(cfg.chase_drone_samples))
    if hide.shape[0] == 1:
        indices = [0]
    else:
        indices = [
            int(round(i))
            for i in np.linspace(0, hide.shape[0] - 1, min(n, hide.shape[0]))
        ]

    poses: list[DroneState] = []
    if initial_drone_ned is not None:
        d0 = np.asarray(initial_drone_ned, dtype=np.float64).reshape(3)
        yaw0 = math.atan2(
            float(wps[0, 1] - d0[1]), float(wps[0, 0] - d0[0]),
        )
        poses.append(DroneState(position=d0.copy(), velocity=np.zeros(3), heading=yaw0))

    full = wps
    for hi in indices:
        wp = hide[hi]
        # Heading from local segment direction on full route.
        g_idx = min(int(split_idx) + hi, full.shape[0] - 2)
        dx = float(full[g_idx + 1, 0] - full[g_idx, 0])
        dy = float(full[g_idx + 1, 1] - full[g_idx, 1])
        heading = math.atan2(dy, dx) if (dx * dx + dy * dy) > 1e-9 else 0.0
        poses.append(make_chase_drone_at_target(wp, heading, cfg))

    # De-duplicate near-identical positions.
    out: list[DroneState] = []
    for p in poses:
        if out and float(np.linalg.norm(p.position[:2] - out[-1].position[:2])) < 2.0:
            continue
        out.append(p)
    return out


def is_hidden_from_chase_drones(
    occupancy: PcdOccupancyMap,
    target_pos: np.ndarray,
    drone_poses: list[DroneState],
    cfg: HideVisibilityConfig,
    *,
    keep_z: float,
) -> tuple[bool, int, int]:
    """Return ``(all_hidden, hidden_count, total)``."""
    tgt = TargetState(
        position=np.asarray(target_pos, dtype=np.float64).reshape(3).copy(),
        velocity=np.zeros(3),
        heading=0.0,
    )
    tgt.position[2] = keep_z
    hidden = 0
    total = len(drone_poses)
    if total == 0:
        return False, 0, 0
    for drone in drone_poses:
        ok, _ = target_hidden_from_drone(occupancy, drone, tgt, cfg)
        if not ok:
            continue
        if not cfg.occluder_between_required:
            hidden += 1
            continue
        between = False
        if cfg.seg_building_map is not None:
            between = cfg.seg_building_map.building_occludes_between_ned(
                drone.position, tgt.position,
                near_target_m=cfg.occluder_near_target_m,
                drone_eye_agl_m=cfg.drone_eye_agl_m,
            )
        elif hasattr(occupancy, "building_occludes_between_ned"):
            between = occupancy.building_occludes_between_ned(
                drone.position, tgt.position,
                drone_eye_agl_m=cfg.drone_eye_agl_m,
                min_building_height_m=cfg.min_building_height_m,
                min_footprint_cells=cfg.min_footprint_cells,
                near_target_m=cfg.occluder_near_target_m,
            )
        if between:
            hidden += 1
    return hidden == total, hidden, total


def render_route_bev_preview(
    occupancy: PcdOccupancyMap,
    route: Any,
    *,
    drone_poses: list[DroneState] | None = None,
    split_idx: int = 0,
    cfg: HideVisibilityConfig | None = None,
    out_path: Path | str | None = None,
    size_px: int = 512,
    margin_m: float = 30.0,
) -> np.ndarray:
    """P2: top-down BEV PNG — buildings, route, hide leg, drone samples, hide status."""
    from PIL import Image, ImageDraw

    wps = np.asarray(route.waypoints, dtype=np.float64)
    xs = wps[:, 0]
    ys = wps[:, 1]
    x0, x1 = float(xs.min() - margin_m), float(xs.max() + margin_m)
    y0, y1 = float(ys.min() - margin_m), float(ys.max() + margin_m)
    span = max(x1 - x0, y1 - y0, 1.0)

    img = Image.new("RGB", (size_px, size_px), (40, 44, 52))
    draw = ImageDraw.Draw(img)

    def to_px(x: float, y: float) -> tuple[int, int]:
        u = int((x - x0) / span * (size_px - 1))
        v = int((1.0 - (y - y0) / span) * (size_px - 1))
        return u, v

    vw = float(getattr(occupancy, "_vw", 1.5))
    bev = getattr(occupancy, "_bev2d", set())
    ox0 = float(getattr(occupancy, "_x0", 0.0))
    oy0 = float(getattr(occupancy, "_y0", 0.0))

    # Building footprints (sampled for speed on large maps).
    step_cells = max(1, int(math.ceil(span / (size_px * 0.5))))
    for ix, iy in bev:
        cx = ox0 + (ix + 0.5) * vw
        cy = oy0 + (iy + 0.5) * vw
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            continue
        if (ix + iy) % step_cells != 0:
            continue
        u, v = to_px(cx, cy)
        draw.rectangle([u, v, u + 1, v + 1], fill=(90, 90, 100))

    # Route polyline: open leg yellow, hide leg cyan.
    pts_open = [to_px(float(p[0]), float(p[1])) for p in wps[: split_idx + 1]]
    pts_hide = [to_px(float(p[0]), float(p[1])) for p in wps[split_idx:]]
    if len(pts_open) >= 2:
        draw.line(pts_open, fill=(220, 180, 60), width=2)
    if len(pts_hide) >= 2:
        draw.line(pts_hide, fill=(60, 200, 220), width=3)

    cfg = cfg or HideVisibilityConfig()
    keep_z = float(wps[0, 2]) if wps.shape[0] else -0.5

    if drone_poses:
        for i, drone in enumerate(drone_poses):
            u, v = to_px(float(drone.position[0]), float(drone.position[1]))
            draw.ellipse([u - 4, v - 4, u + 4, v + 4], fill=(80, 160, 255))
            if i == 0:
                draw.text((u + 5, v - 5), "D0", fill=(200, 220, 255))

    # Hide-leg waypoint visibility dots.
    seg_map = getattr(cfg, "seg_building_map", None) if cfg else None
    if seg_map is not None:
        for bd in seg_map.buildings:
            u, v = to_px(float(bd.ned_xyz[0]), float(bd.ned_xyz[1]))
            if 0 <= u < size_px and 0 <= v < size_px:
                draw.rectangle([u - 2, v - 2, u + 2, v + 2], fill=(200, 90, 70))

    for wp in wps[split_idx:]:
        pos = np.asarray(wp, dtype=np.float64).reshape(3).copy()
        pos[2] = keep_z
        u, v = to_px(float(pos[0]), float(pos[1]))
        color = (120, 120, 120)
        if drone_poses:
            ok, hid, tot = is_hidden_from_chase_drones(
                occupancy, pos, drone_poses, cfg, keep_z=keep_z,
            )
            if ok:
                color = (40, 200, 80)
            elif hid > 0:
                color = (200, 140, 40)
        draw.rectangle([u - 2, v - 2, u + 2, v + 2], fill=color)

    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path))
    return np.asarray(img)


__all__ = [
    "HideVisibilityConfig",
    "is_hidden_from_chase_drones",
    "make_chase_drone_at_target",
    "render_route_bev_preview",
    "sample_chase_drone_poses",
    "target_hidden_from_drone",
    "target_in_camera_frustum",
]

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Target visibility checks (FOV + line-of-sight) — offline, numpy only."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from flyseek.adapters.pcd_occupancy import PcdOccupancyMap
from flyseek.adversary.base import DroneState, TargetState, bearing_xy, horizontal_distance, wrap_to_pi


def target_bearing_in_fov(
    drone_pos: np.ndarray,
    drone_yaw: float,
    target_pos: np.ndarray,
    hfov_deg: float = 50.0,
) -> bool:
    """True if target lies inside horizontal FOV cone (camera yaw ≈ drone yaw)."""
    bearing = bearing_xy(drone_pos, target_pos)
    delta = abs(wrap_to_pi(bearing - drone_yaw))
    return delta <= math.radians(hfov_deg)


def fov_centering_offset_xy(
    drone_pos: np.ndarray,
    drone_yaw: float,
    target_pos: np.ndarray,
    *,
    gain_m_per_rad: float = 10.0,
    max_offset_m: float = 8.0,
) -> np.ndarray:
    """Lateral NED offset so the target moves toward the camera bore-sight."""
    bearing = bearing_xy(drone_pos, target_pos)
    mis = wrap_to_pi(bearing - drone_yaw)
    perp = np.array([-math.sin(bearing), math.cos(bearing)], dtype=np.float64)
    offset = perp * gain_m_per_rad * mis
    n = float(np.linalg.norm(offset))
    if n > max_offset_m:
        offset *= max_offset_m / n
    return offset


def target_visible(
    occupancy: PcdOccupancyMap | None,
    drone: DroneState,
    target: TargetState,
    *,
    hfov_deg: float = 50.0,
    max_range_m: float = 80.0,
    drone_eye_agl_m: float = 12.0,
    building_only_los: bool = False,
    min_building_height_m: float | None = None,
    min_footprint_cells: int = 9,
    use_frustum_projection: bool = False,
    cam_forward_m: float = 0.45,
    cam_down_m: float = 0.25,
    cam_pitch_deg: float = 55.0,
    cam_width: int = 0,
    cam_height: int = 0,
    seg_building_map: Any | None = None,
    include_pcd_occluders: bool = False,
) -> bool:
    """Target is visible when in FOV, in range, and PCD ray is not blocked."""
    return visibility_status(
        occupancy, drone, target,
        hfov_deg=hfov_deg,
        max_range_m=max_range_m,
        drone_eye_agl_m=drone_eye_agl_m,
        building_only_los=building_only_los,
        min_building_height_m=min_building_height_m,
        min_footprint_cells=min_footprint_cells,
        use_frustum_projection=use_frustum_projection,
        cam_forward_m=cam_forward_m,
        cam_down_m=cam_down_m,
        cam_pitch_deg=cam_pitch_deg,
        cam_width=cam_width,
        cam_height=cam_height,
        seg_building_map=seg_building_map,
        include_pcd_occluders=include_pcd_occluders,
    )[0]


def _in_camera_frustum(
    drone: DroneState,
    target: TargetState,
    *,
    hfov_deg: float,
    max_range_m: float,
    use_frustum_projection: bool,
    cam_forward_m: float,
    cam_down_m: float,
    cam_pitch_deg: float,
    cam_width: int,
    cam_height: int,
) -> tuple[bool, str]:
    """Horizontal FOV or pinhole projection (P2)."""
    r = horizontal_distance(drone.position, target.position)
    if r > max_range_m:
        return False, "out_of_range"
    if use_frustum_projection and cam_width > 0 and cam_height > 0:
        from flyseek.utils.seg_bbox import project_ned_to_pixel

        uv = project_ned_to_pixel(
            target.position,
            drone.position,
            float(drone.heading),
            width=int(cam_width),
            height=int(cam_height),
            hfov_deg=hfov_deg,
            cam_forward_m=cam_forward_m,
            cam_down_m=cam_down_m,
            cam_pitch_deg=cam_pitch_deg,
        )
        if uv is None:
            return False, "out_of_fov"
        u, v = uv
        if (0 <= u < cam_width) and (0 <= v < cam_height):
            return True, "ok"
        return False, "out_of_fov"
    if target_bearing_in_fov(
        drone.position, drone.heading, target.position, hfov_deg,
    ):
        return True, "ok"
    return False, "out_of_fov"


def visibility_status(
    occupancy: PcdOccupancyMap | None,
    drone: DroneState,
    target: TargetState,
    *,
    hfov_deg: float = 50.0,
    max_range_m: float = 80.0,
    drone_eye_agl_m: float = 12.0,
    building_only_los: bool = False,
    min_building_height_m: float | None = None,
    min_footprint_cells: int = 9,
    use_frustum_projection: bool = False,
    cam_forward_m: float = 0.45,
    cam_down_m: float = 0.25,
    cam_pitch_deg: float = 55.0,
    cam_width: int = 0,
    cam_height: int = 0,
    seg_building_map: Any | None = None,
    include_pcd_occluders: bool = False,
) -> tuple[bool, str]:
    """Return ``(visible, reason)`` where reason ∈
    {"ok", "out_of_range", "out_of_fov", "los_blocked", "los_blocked_occluder"}.

    Useful for the adaptive tracker, which reacts differently to FOV misalignment
    (just yaw) vs LOS occlusion (need to peek / reposition).

    When ``building_only_los`` is True, only large building footprints block LoS
    (matches occlusion route planning). When ``include_pcd_occluders`` is True,
    *any* tall PCD column (trees / foliage / poles / structures — not just
    annotated buildings) also blocks LoS; such a block is reported with the
    distinct reason ``"los_blocked_occluder"`` so callers can tell a foliage /
    tree occlusion apart from a building occlusion.
    """
    in_fov, fov_reason = _in_camera_frustum(
        drone, target,
        hfov_deg=hfov_deg,
        max_range_m=max_range_m,
        use_frustum_projection=use_frustum_projection,
        cam_forward_m=cam_forward_m,
        cam_down_m=cam_down_m,
        cam_pitch_deg=cam_pitch_deg,
        cam_width=cam_width,
        cam_height=cam_height,
    )
    if not in_fov:
        return False, fov_reason
    if occupancy is None and seg_building_map is None:
        return True, "ok"
    los_kw = dict(
        drone_eye_agl_m=drone_eye_agl_m,
        target_agl_m=max(0.5, -float(target.position[2])),
    )

    # (1) Building / annotated-occluder line of sight.
    building_blocked = False
    if seg_building_map is not None:
        building_blocked = bool(seg_building_map.los_blocked_by_annotated_building_ned(
            drone.position, target.position, **los_kw,
        ))
    elif building_only_los and hasattr(occupancy, "los_blocked_by_building_ned"):
        building_blocked = bool(occupancy.los_blocked_by_building_ned(
            drone.position,
            target.position,
            min_building_height_m=min_building_height_m,
            min_footprint_cells=min_footprint_cells,
            **los_kw,
        ))
    elif (not include_pcd_occluders) and hasattr(occupancy, "los_blocked_ned"):
        # Legacy general-PCD path (kept only when trees aren't requested
        # separately, so the default behaviour is unchanged).
        building_blocked = bool(occupancy.los_blocked_ned(
            drone.position, target.position, **los_kw,
        ))
    if building_blocked:
        return False, "los_blocked"

    # (2) Any other tall PCD occluder (trees / foliage / poles / structures).
    if include_pcd_occluders and occupancy is not None \
            and hasattr(occupancy, "los_blocked_ned"):
        if bool(occupancy.los_blocked_ned(
            drone.position, target.position, **los_kw,
        )):
            return False, "los_blocked_occluder"

    return True, "ok"


def find_clear_vantage_xy(
    occupancy: PcdOccupancyMap | None,
    target_ned: np.ndarray,
    from_pos_ned: np.ndarray,
    *,
    follow_distance: float = 12.0,
    lateral_offsets_m: tuple[float, ...] = (0.0, 6.0, -6.0, 10.0, -10.0, 14.0, -14.0),
    forward_offsets_m: tuple[float, ...] = (-4.0, 0.0, 4.0, 8.0),
    drone_eye_agl_m: float = 12.0,
    target_agl_m: float = 1.0,
    keep_z_ned: float | None = None,
) -> np.ndarray | None:
    """Pick a vantage XY near ``target - back * follow_distance`` with clear LOS.

    Tries small forward/lateral offsets around the nominal chase point; returns
    the closest candidate whose LOS to the target is not blocked. None if all
    candidates are occluded. When ``occupancy`` is None, returns the nominal
    chase point (no PCD to check against).
    """
    target_ned = np.asarray(target_ned, dtype=np.float64).reshape(3)
    from_pos_ned = np.asarray(from_pos_ned, dtype=np.float64).reshape(3)
    bearing = bearing_xy(from_pos_ned, target_ned)
    back = np.array([-math.cos(bearing), -math.sin(bearing)], dtype=np.float64)
    side = np.array([-math.sin(bearing), math.cos(bearing)], dtype=np.float64)
    base_xy = target_ned[:2] + back * float(follow_distance)
    z = float(from_pos_ned[2] if keep_z_ned is None else keep_z_ned)

    best: tuple[float, np.ndarray] | None = None
    for fwd in forward_offsets_m:
        for lat in lateral_offsets_m:
            xy = base_xy + back * float(fwd) + side * float(lat)
            cand = np.array([xy[0], xy[1], z], dtype=np.float64)
            score = abs(float(fwd)) + abs(float(lat)) * 0.6
            if occupancy is not None and occupancy.los_blocked_ned(
                cand, target_ned,
                drone_eye_agl_m=drone_eye_agl_m,
                target_agl_m=target_agl_m,
            ):
                continue
            if best is None or score < best[0]:
                best = (score, cand)
    return None if best is None else best[1]

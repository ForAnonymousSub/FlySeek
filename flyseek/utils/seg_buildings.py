# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Annotated building landmarks from OpenFly ``seg_map/*.jsonl``.

Coordinates in each record's ``filename`` (``X=..Y=..Z=..``) are in the
OpenFly **map** frame — same as PCD / traj_gen. Convert with
``map_to_airsim_ned`` before comparing to AirSim poses.

Only these landmarks are used as occluders when ``use_seg_buildings_only`` is
enabled for hide planning (not generic PCD BEV footprints).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flyseek.adapters.pcd_occupancy import PcdOccupancyMap
from flyseek.adversary.base import DroneState, TargetState
from flyseek.utils.coords import map_to_airsim_ned

_FILENAME_RE = re.compile(
    r"X=([^Y]+)Y=([^Z]+)Z=(.+?)\.png",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SegBuilding:
    """One annotated building landmark."""

    index: int
    type: str
    map_xyz: np.ndarray
    ned_xyz: np.ndarray
    height_map_z: float
    feature: str = ""

    @property
    def xy_ned(self) -> np.ndarray:
        return self.ned_xyz[:2].copy()


def parse_landmark_filename(filename: str) -> np.ndarray | None:
    """Parse ``X=..Y=..Z=..`` map coordinates from a seg_map filename."""
    m = _FILENAME_RE.search(str(filename))
    if not m:
        return None
    return np.array([float(m.group(1)), float(m.group(2)), float(m.group(3))],
                    dtype=np.float64)


def load_seg_buildings(
    jsonl_path: Path | str,
    *,
    building_types: tuple[str, ...] = ("building",),
) -> list[SegBuilding]:
    """Load building landmarks from a seg_map JSONL file."""
    path = Path(jsonl_path)
    if not path.is_file():
        raise FileNotFoundError(f"seg_map not found: {path}")
    out: list[SegBuilding] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if str(rec.get("type", "")).lower() not in building_types:
            continue
        xyz = parse_landmark_filename(rec.get("filename", ""))
        if xyz is None:
            continue
        ned = map_to_airsim_ned(xyz)
        out.append(SegBuilding(
            index=len(out),
            type=str(rec.get("type", "building")),
            map_xyz=xyz,
            ned_xyz=ned,
            height_map_z=float(xyz[2]),
            feature=str(rec.get("feature", "")),
        ))
    return out


@dataclass
class SegBuildingMap:
    """In-memory annotated building set with hide / LoS helpers."""

    buildings: list[SegBuilding]
    footprint_radius_m: float = 10.0
    min_occluder_height_m: float = 8.0
    ground_map_z: float = 0.0

    @classmethod
    def from_jsonl(
        cls,
        jsonl_path: Path | str,
        *,
        footprint_radius_m: float = 10.0,
        min_occluder_height_m: float = 8.0,
        ground_map_z: float = 0.0,
    ) -> "SegBuildingMap":
        return cls(
            buildings=load_seg_buildings(jsonl_path),
            footprint_radius_m=float(footprint_radius_m),
            min_occluder_height_m=float(min_occluder_height_m),
            ground_map_z=float(ground_map_z),
        )

    def __len__(self) -> int:
        return len(self.buildings)

    def nearest_building_ned(
        self,
        pos_ned: np.ndarray,
        *,
        max_dist_m: float = 80.0,
    ) -> tuple[SegBuilding, float] | None:
        p = np.asarray(pos_ned, dtype=np.float64).reshape(3)
        best: tuple[SegBuilding, float] | None = None
        for b in self.buildings:
            d = float(np.linalg.norm(b.xy_ned - p[:2]))
            if d > max_dist_m:
                continue
            if best is None or d < best[1]:
                best = (b, d)
        return best

    def los_blocked_by_annotated_building_ned(
        self,
        observer_ned: np.ndarray,
        target_ned: np.ndarray,
        *,
        target_agl_m: float = 1.0,
        drone_eye_agl_m: float = 12.0,
    ) -> bool:
        """True when the segment intersects an annotated building column."""
        from flyseek.utils.coords import airsim_ned_to_map

        a = airsim_ned_to_map(observer_ned).copy()
        b = airsim_ned_to_map(target_ned).copy()
        a[2] = max(float(a[2]), float(drone_eye_agl_m))
        b[2] = max(float(b[2]), self.ground_map_z + float(target_agl_m))

        seg = b - a
        length = float(np.linalg.norm(seg))
        if length < 0.5:
            return False
        steps = max(6, int(length / 1.5))
        r = float(self.footprint_radius_m)
        min_h = float(self.min_occluder_height_m)

        for i in range(1, steps):
            t = i / steps
            p = a + t * seg
            pz = float(p[2])
            for bd in self.buildings:
                if pz > max(min_h, bd.height_map_z):
                    continue
                if pz < self.ground_map_z + 0.5:
                    continue
                dxy = float(np.linalg.norm(p[:2] - bd.map_xyz[:2]))
                if dxy <= r:
                    return True
        return False

    def building_occludes_between_ned(
        self,
        observer_ned: np.ndarray,
        target_ned: np.ndarray,
        *,
        near_target_m: float = 14.0,
        **kwargs: object,
    ) -> bool:
        """Occluder on ray and within ``near_target_m`` of the target (P1)."""
        from flyseek.utils.coords import airsim_ned_to_map

        if not self.los_blocked_by_annotated_building_ned(
            observer_ned, target_ned, **kwargs,
        ):
            return False
        tgt = airsim_ned_to_map(
            np.asarray(target_ned, dtype=np.float64).reshape(3))
        r = float(self.footprint_radius_m)
        near = float(near_target_m)
        for bd in self.buildings:
            dxy = float(np.linalg.norm(tgt[:2] - bd.map_xyz[:2]))
            if dxy <= r + near:
                return True
        return False

    def candidate_hide_spots_ned(
        self,
        occupancy: PcdOccupancyMap,
        building: SegBuilding,
        *,
        keep_z: float,
        offsets_m: tuple[float, ...] = (8.0, 10.0, 12.0, 14.0),
        n_angles: int = 16,
    ) -> list[np.ndarray]:
        """Drivable positions beside ``building`` suitable for hiding."""
        spots: list[np.ndarray] = []
        base = building.ned_xyz.copy()
        base[2] = keep_z
        for dist in offsets_m:
            for k in range(n_angles):
                ang = 2.0 * math.pi * k / n_angles
                cand = base.copy()
                cand[0] += dist * math.cos(ang)
                cand[1] += dist * math.sin(ang)
                cand[2] = keep_z
                if occupancy.is_drivable_ned(cand):
                    spots.append(cand)
        return spots

    def find_hide_goal_ned(
        self,
        occupancy: PcdOccupancyMap,
        near_ned: np.ndarray,
        drone_ned: np.ndarray,
        *,
        keep_z: float,
        search_radius_m: float = 80.0,
        hide_vis_config: object | None = None,
        chase_drone_poses: list[DroneState] | None = None,
        require_occluder_between: bool = True,
        occluder_near_target_m: float = 14.0,
    ) -> tuple[np.ndarray, SegBuilding] | None:
        """Best hide spot near ``near_ned`` using **only** annotated buildings."""
        from flyseek.utils.hide_visibility import (
            HideVisibilityConfig,
            target_hidden_from_drone,
        )

        near = np.asarray(near_ned, dtype=np.float64).reshape(3)
        vis_cfg = hide_vis_config or HideVisibilityConfig(building_only_los=False)

        drone_list: list[DroneState] = []
        if chase_drone_poses:
            drone_list = list(chase_drone_poses)
        else:
            yaw = math.atan2(
                float(near[1] - drone_ned[1]), float(near[0] - drone_ned[0]),
            )
            drone_list.append(
                DroneState(
                    position=np.asarray(drone_ned, dtype=np.float64).reshape(3),
                    velocity=np.zeros(3),
                    heading=yaw,
                )
            )

        best: tuple[np.ndarray, SegBuilding, float] | None = None
        for bd in self.buildings:
            if float(np.linalg.norm(bd.xy_ned - near[:2])) > search_radius_m:
                continue
            for cand in self.candidate_hide_spots_ned(
                occupancy, bd, keep_z=keep_z,
            ):
                tgt = TargetState(
                    position=cand.copy(), velocity=np.zeros(3), heading=0.0,
                )
                ok_all = True
                for drone in drone_list:
                    vis_ok, _ = target_hidden_from_drone(
                        occupancy, drone, tgt, vis_cfg,
                    )
                    seg_ok = self.los_blocked_by_annotated_building_ned(
                        drone.position, cand,
                        drone_eye_agl_m=vis_cfg.drone_eye_agl_m,
                    )
                    if not (vis_ok and seg_ok):
                        ok_all = False
                        break
                    if require_occluder_between:
                        if not self.building_occludes_between_ned(
                            drone.position, cand,
                            near_target_m=occluder_near_target_m,
                            drone_eye_agl_m=vis_cfg.drone_eye_agl_m,
                        ):
                            ok_all = False
                            break
                if not ok_all:
                    continue
                away = float(np.linalg.norm(cand[:2] - near[:2]))
                score = away + float(np.linalg.norm(cand[:2] - bd.xy_ned)) * 0.2
                if best is None or score > best[2]:
                    best = (cand, bd, score)
        if best is None:
            return None
        return best[0], best[1]

    def find_best_hide_site(
        self,
        occupancy: PcdOccupancyMap,
        anchor_ned: np.ndarray,
        drone_ned: np.ndarray,
        *,
        keep_z: float,
        search_radius_m: float = 120.0,
        hide_vis_config: object | None = None,
        chase_drone_poses: list[DroneState] | None = None,
    ) -> dict | None:
        """Search buildings near ``anchor`` and return the best hide site meta."""
        anchor = np.asarray(anchor_ned, dtype=np.float64).reshape(3)
        result = self.find_hide_goal_ned(
            occupancy, anchor, drone_ned,
            keep_z=keep_z,
            search_radius_m=search_radius_m,
            hide_vis_config=hide_vis_config,
            chase_drone_poses=chase_drone_poses,
        )
        if result is None:
            return None
        hide, bd = result
        return {
            "hide_goal": hide.tolist(),
            "building_index": bd.index,
            "building_map_xyz": bd.map_xyz.tolist(),
            "building_ned_xy": bd.xy_ned.tolist(),
            "building_height_m": bd.height_map_z,
            "dist_from_anchor_m": float(np.linalg.norm(hide[:2] - anchor[:2])),
        }


__all__ = [
    "SegBuilding",
    "SegBuildingMap",
    "load_seg_buildings",
    "parse_landmark_filename",
]

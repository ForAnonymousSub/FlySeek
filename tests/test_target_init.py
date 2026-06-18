# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Tests for target road initialization scoring."""

from __future__ import annotations

import math

import numpy as np

from flyseek.adapters.pcd_occupancy import OccupancyConfig, PcdOccupancyMap
from flyseek.utils.target_init import (
    TargetInitConfig,
    find_valid_init_pose,
    score_init_pose_ned,
)


class _RoadOcc(PcdOccupancyMap):
    """Synthetic east-west road; narrow north strip = guardrail."""

    def __init__(self) -> None:
        cfg = OccupancyConfig(
            voxel_width=2.0,
            dilate_radius=0.0,
            map_bound=(-50, 250, -50, 50, -10, 50),
            map_elevation=0.0,
            min_height_thresh=6.0,
            min_ground_points_per_cell=1,
            car_agl_m=0.35,
        )
        ground_z = {}
        ground_count = {}
        for ix in range(25, 100):
            for iy in range(22, 28):
                ground_z[(ix, iy)] = 0.0
                ground_count[(ix, iy)] = 4
        for ix in range(60, 90):
            for iy in range(38, 42):
                ground_z[(ix, iy)] = 0.5
                ground_count[(ix, iy)] = 2
        super().__init__(cfg, set(), set(), {}, ground_z, ground_count,
                         ground_enabled=True)

    def is_bev_occupied_map(self, pos_map: np.ndarray) -> bool:
        ix, iy, _ = self._map_to_voxel(pos_map)  # noqa: SLF001
        return (ix, iy) in {(80, 10), (80, 40)}

    def local_roof_map_z(self, pos_map: np.ndarray) -> float:
        ix, iy, _ = self._map_to_voxel(pos_map)  # noqa: SLF001
        if 38 <= iy <= 42:
            return 1.0
        return 20.0

    def local_ground_map_z(self, pos_map: np.ndarray) -> float:
        ix, iy, _ = self._map_to_voxel(pos_map)  # noqa: SLF001
        return float(self._ground_z.get((ix, iy), self.cfg.map_elevation))  # noqa: SLF001


def _ned(x: float, y: float) -> np.ndarray:
    return np.array([x, -y, -0.35])


def test_guardrail_scores_lower_than_road():
    occ = _RoadOcc()
    cfg = TargetInitConfig(min_corridor_width_m=10.0, min_vertical_clearance_m=5.5)
    road_h = 0.0
    road_score, road_reason = score_init_pose_ned(
        occ, _ned(120.0, 0.0), road_h, cfg=cfg,
    )
    rail_score, rail_reason = score_init_pose_ned(
        occ, _ned(120.0, 35.0), road_h, cfg=cfg,
    )
    assert road_reason == "ok"
    assert road_score > 30.0
    assert rail_score < road_score
    assert rail_reason != "ok" or rail_score < road_score * 0.5


def test_find_valid_init_near_guardrail_anchor():
    occ = _RoadOcc()
    rng = np.random.default_rng(0)
    anchor = _ned(120.0, 35.0)
    result = find_valid_init_pose(
        occ, anchor, rng, hint_heading=0.0,
        cfg=TargetInitConfig(search_radius_m=40.0, sample_step_m=4.0),
    )
    assert result.ok
    assert result.score > 30.0
    width_ok = abs(result.position_ned[1]) < 3.0 or result.position_ned[0] > 100.0
    assert width_ok or result.score > 40.0

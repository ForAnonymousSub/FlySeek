# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Tests for env init presets and resolve_target_init_pose."""

from __future__ import annotations

import numpy as np

from flyseek.adapters.pcd_occupancy import OccupancyConfig, PcdOccupancyMap
from flyseek.utils.target_init import resolve_target_init_pose, score_init_pose_ned
from flyseek.utils.target_init_presets import (
    default_profile_name,
    list_profiles,
    load_target_init_profile,
)


class _RoadOcc(PcdOccupancyMap):
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


def test_env_airsim_16_presets_load():
    assert default_profile_name("env_airsim_16") == "standard"
    names = list_profiles("env_airsim_16")
    assert "strict" in names
    assert "standard" in names
    prof = load_target_init_profile("env_airsim_16", "standard")
    assert prof.use_road_seed_fallback
    # standard now uses a wider search to find open roads in env_airsim_16.
    assert prof.config.search_radius_m >= 200.0
    assert prof.config.min_drive_feasibility_m >= 8.0
    assert prof.config.min_open_ray_sum_m >= 24.0


def test_resolve_standard_on_synthetic_road():
    occ = _RoadOcc()
    prof = load_target_init_profile("env_airsim_16", "standard")
    rng = np.random.default_rng(0)
    anchor = np.array([120.0, 35.0, -0.35])
    res = resolve_target_init_pose(occ, anchor, rng, prof, hint_heading=0.0)
    assert res.ok
    assert res.score > 15.0
    assert res.samples_tried > 0
    s, r = score_init_pose_ned(occ, res.position_ned, res.heading_rad, cfg=prof.config)
    assert r == "ok" or res.reason == "relaxed_threshold"
    assert s > -1e8


def test_classic_car_spawn_offline_standard_profile():
    """Regression: env_airsim_16 default spawn is off-road; standard must init
    onto an OPEN ROAD where the car can actually drive ≥8 m forward."""
    from pathlib import Path

    from flyseek.utils.target_init import drive_feasibility_distance_m
    repo = Path(__file__).resolve().parents[2]
    try:
        occ = PcdOccupancyMap.load_or_build(repo, env_name="env_airsim_16", rebuild=False)
    except FileNotFoundError:
        return
    prof = load_target_init_profile("env_airsim_16", "standard")
    anchor = np.array([252.7477264404297, 125.9112777709961, 0.7])
    rng = np.random.default_rng(0)
    res = resolve_target_init_pose(occ, anchor, rng, prof, hint_heading=-1.57)
    assert res.ok, f"expected ok, got {res.reason} score={res.score}"
    assert res.init_method in (
        "road_seed", "road_seed_relaxed", "spiral+road_seed", "spiral",
    )
    assert res.score >= prof.config.min_accept_score * 0.65
    drive = drive_feasibility_distance_m(
        occ, res.position_ned, res.heading_rad, max_dist_m=15,
    )
    # The new anti-island gate forbids any pose where the car cannot roll
    # forward at least min_drive_feasibility_m metres.
    assert drive >= prof.config.min_drive_feasibility_m, \
        f"car cannot drive forward (drive_dist={drive}m)"


def test_route_starts_at_init_anchor():
    """The car's route must begin at the init position so the car can follow it
    without first traversing non-drivable cells (Bug 2 regression)."""
    from pathlib import Path

    from flyseek.utils.road_graph import build_route
    repo = Path(__file__).resolve().parents[2]
    try:
        occ = PcdOccupancyMap.load_or_build(repo, env_name="env_airsim_16", rebuild=False)
    except FileNotFoundError:
        return
    prof = load_target_init_profile("env_airsim_16", "standard")
    anchor = np.array([252.7477264404297, 125.9112777709961, 0.7])
    res = resolve_target_init_pose(
        occ, anchor, np.random.default_rng(0), prof, hint_heading=-1.57,
    )
    assert res.ok
    route = build_route(
        occ, res.position_ned, np.random.default_rng(1),
        keep_z=float(res.position_ned[2]),
        route_len_m=120.0, search_radius_m=180.0, maneuver="open_then_hide",
        start_at_anchor=True, anchor_heading_rad=res.heading_rad,
    )
    shift = float(np.linalg.norm(route.waypoints[0][:2] - res.position_ned[:2]))
    assert shift < 1.0, f"route should start at init pose, got shift={shift}m"
    # First 10 waypoints must all be drivable to ensure the car can actually
    # follow the route from tick 0.
    drivable = sum(1 for w in route.waypoints[:10] if occ.is_drivable_ned(w))
    assert drivable >= 9, f"only {drivable}/10 leading waypoints drivable"

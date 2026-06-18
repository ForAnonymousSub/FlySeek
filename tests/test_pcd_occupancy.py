# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Tests for coordinate transforms and PCD occupancy helpers."""

from __future__ import annotations

import numpy as np

from flyseek.adapters.pcd_occupancy import OccupancyConfig, PcdOccupancyMap
from flyseek.utils.coords import airsim_ned_to_map, map_to_airsim_ned


def test_airsim_map_roundtrip():
    ned = np.array([100.0, -50.0, -20.0])
    mp = airsim_ned_to_map(ned)
    assert np.allclose(mp, [100.0, 50.0, 20.0])
    assert np.allclose(map_to_airsim_ned(mp), ned)


def test_bev_blocks_building_footprint():
    cfg = OccupancyConfig(
        voxel_width=1.0,
        dilate_radius=0.0,
        map_bound=(0, 10, 0, 10, 0, 10),
        map_elevation=0.0,
        min_height_thresh=1.0,
        min_drone_clearance=2.0,
    )
    occ3d = {(5, 5, 3)}
    bev2d = {(5, 5)}
    roof = {(5, 5): 5.0}
    occ = PcdOccupancyMap(cfg, occ3d, bev2d, roof, {(1, 1): 0.0}, {(1, 1): 5})

    free = np.array([1.5, -1.5, 0.5])
    blocked = np.array([5.5, -5.5, 0.5])
    assert occ.is_drivable_ned(free)
    assert not occ.is_drivable_ned(blocked)

    moved = occ.resolve_bev_move_ned(free, blocked, keep_z=0.5)
    assert not occ.is_bev_occupied_ned(moved)
    assert np.linalg.norm(moved[:2] - free[:2]) < 5.0


def test_drone_lifted_above_roof():
    cfg = OccupancyConfig(
        voxel_width=1.0,
        dilate_radius=0.0,
        map_bound=(0, 10, 0, 10, 0, 20),
        min_drone_clearance=3.0,
    )
    occ3d = {(5, 5, 8), (5, 5, 9)}
    bev2d = {(5, 5)}
    roof = {(5, 5): 10.0}
    occ = PcdOccupancyMap(cfg, occ3d, bev2d, roof, {(1, 1): 0.0}, {(1, 1): 5})

    prev = np.array([5.5, -5.5, -5.0])
    proposed = np.array([5.5, -5.5, -5.0])
    fixed = occ.resolve_drone_ned(prev, proposed)
    assert -fixed[2] >= 13.0 - 0.1


def test_los_skips_road_surface_voxels():
    """Regression: PCD road surface voxels under the target must not be
    treated as occluders. Also, the drone's actual altitude (higher than the
    ``drone_eye_agl_m`` floor) should be respected so a climbing drone wins LOS.
    """
    cfg = OccupancyConfig(
        voxel_width=1.0,
        dilate_radius=0.0,
        map_bound=(0, 50, 0, 20, 0, 30),
        min_drone_clearance=3.0,
    )
    # Road surface stacked from z=0 to z=3 (mimics real PCD ground stratum).
    occ3d = set()
    for x in range(10, 30):
        for z in range(0, 4):
            occ3d.add((x, 10, z))
    # A 4 m tall guardrail at x=18 going up to z=8.
    for z in range(0, 8):
        occ3d.add((18, 10, z))
    bev2d = {(18, 10)}
    roof_map = {(x, 10): 4.0 for x in range(10, 30)}
    roof_map[(18, 10)] = 8.0
    ground_map = {(x, 10): 0.0 for x in range(10, 30)}
    elev = {(x, 10): 4 for x in range(10, 30)}
    elev[(18, 10)] = 8
    occ = PcdOccupancyMap(cfg, occ3d, bev2d, roof_map, ground_map, elev)

    # Drone NED z=-15 (alt 15m), target NED z=-1 (alt 1m).
    drone = np.array([12.5, -10.5, -15.0])
    target = np.array([28.5, -10.5, -1.0])
    # With drone_eye_agl_m=12 default (and the new fix using max(actual, floor))
    # the ray should NOT be blocked by the road-surface voxel column under
    # the target. The guardrail at x=18 (top z=8) does NOT extend up to the
    # ray's altitude (~15m) so LOS is clear.
    assert not occ.los_blocked_ned(drone, target, drone_eye_agl_m=12.0,
                                   target_agl_m=1.0)

    # A low-flying drone at z=-5 (alt 5m) cannot see over the 8 m guardrail.
    low_drone = np.array([12.5, -10.5, -5.0])
    assert occ.los_blocked_ned(low_drone, target, drone_eye_agl_m=5.0,
                               target_agl_m=1.0)


def test_building_los_rejects_street_lamp_footprint():
    """A thin 1-cell pole must not count as a building hide occluder."""
    cfg = OccupancyConfig(
        voxel_width=1.0,
        dilate_radius=0.0,
        map_bound=(0, 40, 0, 20, 0, 40),
        map_elevation=0.0,
        min_height_thresh=6.0,
    )
    # Street lamp: one BEV cell, ~10 m tall (would pass generic los_blocked).
    pole_occ = {(15, 10, z) for z in range(0, 10)}
    pole_bev = {(15, 10)}
    pole_roof = {(15, 10): 10.0}
    pole_ground = {(15, 10): 0.0}
    pole = PcdOccupancyMap(
        cfg, pole_occ, pole_bev, pole_roof, pole_ground, {(15, 10): 10},
    )
    drone = np.array([10.5, -10.5, -15.0])
    target = np.array([20.5, -10.5, -1.0])
    assert pole.los_blocked_ned(drone, target)
    assert not pole.los_blocked_by_building_ned(
        drone, target, min_building_height_m=8.0, min_footprint_cells=4,
    )

    # Wide building: 5×3 footprint, 20 m tall — must count.
    bld_occ = set()
    for x in range(14, 19):
        for y in range(9, 12):
            for z in range(0, 20):
                bld_occ.add((x, y, z))
    bld_bev = {(x, y) for x in range(14, 19) for y in range(9, 12)}
    bld_roof = {(x, y): 20.0 for x in range(14, 19) for y in range(9, 12)}
    bld_ground = {(x, y): 0.0 for x in range(14, 19) for y in range(9, 12)}
    bld = PcdOccupancyMap(
        cfg, bld_occ, bld_bev, bld_roof, bld_ground,
        {(x, y): 20 for x in range(14, 19) for y in range(9, 12)},
    )
    assert bld.los_blocked_by_building_ned(
        drone, target, min_building_height_m=12.0, min_footprint_cells=4,
    )

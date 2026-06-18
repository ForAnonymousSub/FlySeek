# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Ground / drivable constraints for vehicles."""

from __future__ import annotations

import numpy as np

from flyseek.adapters.pcd_occupancy import OccupancyConfig, PcdOccupancyMap


def test_water_cell_not_drivable_without_ground():
    cfg = OccupancyConfig(
        voxel_width=1.0,
        dilate_radius=0.0,
        map_bound=(0, 20, 0, 20, 0, 20),
        map_elevation=0.0,
        min_height_thresh=2.0,
        min_ground_points_per_cell=2,
    )
    occ = PcdOccupancyMap(cfg, set(), set(), {}, {}, {}, ground_enabled=True)
    water = np.array([3.0, -3.0, 0.0])
    assert not occ.is_drivable_ned(water)


def test_snap_car_never_returns_nan():
    cfg = OccupancyConfig(car_agl_m=0.35)
    occ = PcdOccupancyMap(cfg, set(), set(), {}, {(5, 5): float("nan")}, {(5, 5): 5})
    snapped = occ.snap_car_to_ground_ned(np.array([5.5, -5.5, 99.0]))
    assert np.isfinite(snapped).all()


def test_legacy_cache_without_ground_falls_back_to_bev_only():
    cfg = OccupancyConfig()
    occ = PcdOccupancyMap(cfg, set(), set(), {})
    assert not occ.has_ground_layer
    assert occ.is_drivable_ned(np.array([1.0, -1.0, 0.0]))


def test_ground_cell_is_drivable():
    cfg = OccupancyConfig(
        voxel_width=1.0,
        dilate_radius=0.0,
        map_bound=(0, 20, 0, 20, 0, 20),
        min_ground_points_per_cell=1,
        car_agl_m=0.35,
    )
    occ = PcdOccupancyMap(
        cfg, set(), set(), {},
        {(5, 5): 0.2},
        {(5, 5): 4},
        ground_enabled=True,
    )
    pos = np.array([5.5, -5.5, 0.0])
    assert occ.is_drivable_ned(pos)
    snapped = occ.snap_car_to_ground_ned(pos)
    assert abs(snapped[2] - (-0.55)) < 0.01

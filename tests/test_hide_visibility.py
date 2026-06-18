# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Tests for frustum-aware hide visibility (P0–P2)."""

from __future__ import annotations

import math

import numpy as np

from flyseek.adversary.base import DroneState, TargetState
from flyseek.utils.hide_visibility import (
    HideVisibilityConfig,
    is_hidden_from_chase_drones,
    sample_chase_drone_poses,
    target_hidden_from_drone,
    target_in_camera_frustum,
)
from flyseek.utils.visibility import visibility_status


class _BoxOcc:
    """Drone at y=-10 looks north; wall at y=15 blocks tall targets."""

    cfg = type("Cfg", (), {"map_elevation": 0.0})()

    def local_ground_map_z(self, pos_map):
        return 0.0

    def los_blocked_ned(self, observer, target, **kwargs):
        return float(target[1]) > 14.0 and float(observer[1]) < 0.0

    def los_blocked_by_building_ned(self, observer, target, **kwargs):
        return self.los_blocked_ned(observer, target, **kwargs)

    def first_building_occluder_on_ray_ned(self, observer, target, **kwargs):
        if self.los_blocked_by_building_ned(observer, target, **kwargs):
            t = 0.5
            mid = observer + t * (target - observer)
            return t, np.array([mid[0], 15.0, 10.0])
        return None

    def building_occludes_between_ned(self, observer, target, **kwargs):
        hit = self.first_building_occluder_on_ray_ned(observer, target, **kwargs)
        if hit is None:
            return False
        t, _ = hit
        return 0.08 <= t <= 0.92


def _cfg(**kw):
    return HideVisibilityConfig(
        hfov_deg=90.0,
        max_range_m=200.0,
        drone_eye_agl_m=14.0,
        use_frustum_projection=False,
        building_only_los=True,
        occluder_between_required=True,
        **kw,
    )


def test_target_hidden_when_building_blocks_los():
    occ = _BoxOcc()
    drone = DroneState(
        position=np.array([0.0, -10.0, -14.0]), velocity=np.zeros(3), heading=math.pi / 2,
    )
    open_tgt = TargetState(position=np.array([0.0, 5.0, -0.5]), velocity=np.zeros(3))
    hid_tgt = TargetState(position=np.array([0.0, 20.0, -0.5]), velocity=np.zeros(3))
    cfg = _cfg()
    assert target_hidden_from_drone(occ, drone, open_tgt, cfg)[0] is False
    assert target_hidden_from_drone(occ, drone, hid_tgt, cfg)[0] is True


def test_visibility_status_building_only_los():
    occ = _BoxOcc()
    drone = DroneState(
        position=np.array([0.0, -10.0, -14.0]), velocity=np.zeros(3), heading=math.pi / 2,
    )
    tgt = TargetState(position=np.array([0.0, 20.0, -0.5]), velocity=np.zeros(3))
    vis, reason = visibility_status(
        occ, drone, tgt, hfov_deg=90.0, building_only_los=True,
    )
    assert vis is False
    assert reason == "los_blocked"


def test_sample_chase_drone_poses_along_hide_leg():
    wps = np.array([
        [0.0, 0.0, -0.5],
        [20.0, 0.0, -0.5],
        [40.0, 0.0, -0.5],
        [60.0, 10.0, -0.5],
    ])
    cfg = HideVisibilityConfig(follow_distance_m=10.0, chase_drone_samples=3)
    poses = sample_chase_drone_poses(wps, split_idx=1, cfg=cfg)
    assert len(poses) >= 2


def test_is_hidden_from_all_chase_drones():
    occ = _BoxOcc()
    cfg = _cfg()
    drones = [
        DroneState(
            position=np.array([0.0, -10.0, -14.0]), velocity=np.zeros(3),
            heading=math.pi / 2,
        ),
    ]
    hidden_pos = np.array([0.0, 20.0, -0.5])
    open_pos = np.array([0.0, 5.0, -0.5])
    ok_h, _, _ = is_hidden_from_chase_drones(
        occ, hidden_pos, drones, cfg, keep_z=-0.5,
    )
    ok_o, _, _ = is_hidden_from_chase_drones(
        occ, open_pos, drones, cfg, keep_z=-0.5,
    )
    assert ok_h is True
    assert ok_o is False


def test_frustum_projection_out_of_frame():
    occ = _BoxOcc()
    drone = DroneState(
        position=np.array([0.0, 0.0, -14.0]), velocity=np.zeros(3), heading=0.0,
    )
    # Target far to the side — outside narrow HFOV.
    tgt = TargetState(position=np.array([0.0, 40.0, -0.5]), velocity=np.zeros(3))
    cfg = HideVisibilityConfig(
        hfov_deg=20.0, use_frustum_projection=False, max_range_m=200.0,
    )
    assert target_in_camera_frustum(drone, tgt, cfg) is False

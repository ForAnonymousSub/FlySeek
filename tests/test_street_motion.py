# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Offline tests for street-aligned motion."""

from __future__ import annotations

import math

import numpy as np

from flyseek.adversary import AgentAction, TargetState
from flyseek.utils.street_motion import (
    StreetMotionHelper,
    pick_street_heading,
    stabilize_car_state,
)


class _FakeOcc:
    def is_bev_occupied_ned(self, pos_ned: np.ndarray) -> bool:
        ix = int(round(float(pos_ned[0])))
        iy = int(round(float(pos_ned[1])))
        return (ix, iy) == (5, 5)

    def is_drivable_ned(self, pos_ned: np.ndarray) -> bool:
        return not self.is_bev_occupied_ned(pos_ned)

    def snap_car_to_ground_ned(self, pos_ned: np.ndarray) -> np.ndarray:
        p = np.asarray(pos_ned, dtype=np.float64).reshape(3).copy()
        p[2] = -0.35
        return p

    def resolve_bev_move_ned(self, prev_ned, proposed_ned, *, keep_z=None):
        prop = np.asarray(proposed_ned, dtype=np.float64).reshape(3).copy()
        if not self.is_drivable_ned(prop):
            return np.asarray(prev_ned, dtype=np.float64).reshape(3).copy()
        return self.snap_car_to_ground_ned(prop)


def test_pick_street_heading_avoids_blocked_cell():
    occ = _FakeOcc()
    pos = np.array([0.0, 0.0, 0.0])
    rng = np.random.default_rng(0)
    h = pick_street_heading(occ, pos, rng, hint_heading=0.0, keep_z=0.0)
    trial = pos + np.array([math.cos(h), math.sin(h), 0.0]) * 2.0
    assert not occ.is_bev_occupied_ned(trial)


def test_street_helper_bias_keeps_horizontal_velocity():
    occ = _FakeOcc()
    rng = np.random.default_rng(1)
    helper = StreetMotionHelper(occupancy=occ, rng=rng, street_blend=0.5)
    target = TargetState(position=np.zeros(3), velocity=np.zeros(3))
    helper.reset(target)
    action = AgentAction(
        desired_velocity=np.array([3.0, 0.0, 5.0]),
        desired_heading=0.0,
        behavior_state="evade",
    )
    out = helper.bias_action(action)
    assert abs(float(out.desired_velocity[2])) < 1e-9


def test_stabilize_car_locks_height_and_heading():
    occ = _FakeOcc()
    prev = np.array([0.0, 0.0, 0.0])
    state = TargetState(
        position=np.array([2.0, 0.0, 0.5]),
        velocity=np.array([4.0, 0.0, 0.0]),
        heading=math.pi / 2,
    )
    out = stabilize_car_state(prev, state, occ, keep_z=None, dt=0.1)
    assert abs(float(out.position[2]) - (-0.35)) < 1e-9
    assert float(out.velocity[2]) == 0.0
    assert float(np.linalg.norm(out.velocity[:2])) >= 0.0

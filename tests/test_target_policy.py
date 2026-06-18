# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Tests for the adversarial target policies (deterministic, sim-free)."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from flyseek.adversary.base import DroneState, TargetState, horizontal_distance
from flyseek.bench.target_policy import (
    BEHAVIOR_TYPES,
    TargetPolicy,
    generate_target_waypoints,
)


def _states():
    target = TargetState(position=np.array([5.0, 0.0, -0.3]),
                         velocity=np.zeros(3), heading=0.0)
    uav = DroneState(position=np.array([0.0, 0.0, -18.0]),
                     velocity=np.zeros(3), heading=0.0)
    return target, uav


def _rollout(behavior, difficulty="medium", seed=0, steps=120, dt=0.2):
    pol = TargetPolicy(
        config={"behavior_type": behavior, "difficulty": difficulty, "dt": dt},
        scene_context={},  # no occupancy -> pure geometric
        seed=seed,
    )
    target, uav = _states()
    pol.reset(target)
    states = [target]
    modes = []
    for i in range(1, steps):
        target = pol.get_next_target_state(i * dt, target, uav)
        states.append(target)
        modes.append(pol.last_action.behavior_state)
    return states, modes


def test_unknown_behavior_rejected():
    with pytest.raises(ValueError):
        TargetPolicy(config={"behavior_type": "teleport"}, seed=0)


def test_determinism_same_seed():
    a, _ = _rollout("sharp_turn", seed=7)
    b, _ = _rollout("sharp_turn", seed=7)
    for sa, sb in zip(a, b):
        assert np.allclose(sa.position, sb.position)
        assert math.isclose(sa.heading, sb.heading)


def test_direct_escape_increases_distance():
    states, _ = _rollout("direct_escape", steps=60)
    _, uav = _states()
    d0 = horizontal_distance(states[0].position, uav.position)
    d1 = horizontal_distance(states[-1].position, uav.position)
    assert d1 > d0 + 5.0


def test_sharp_turn_fires_turns():
    _, modes = _rollout("sharp_turn", difficulty="hard", steps=200, dt=0.2)
    assert "sharp_turn" in modes  # at least one trigger fired


def test_sharp_turn_has_larger_heading_variance_than_escape():
    esc, _ = _rollout("direct_escape", steps=200)
    sharp, _ = _rollout("sharp_turn", difficulty="hard", steps=200)
    esc_h = np.array([s.heading for s in esc])
    sharp_h = np.array([s.heading for s in sharp])
    assert np.std(sharp_h) > np.std(esc_h)


def test_detour_feint_alternates_phases():
    _, modes = _rollout("detour_feint", steps=200, dt=0.2)
    assert any("feint" in m for m in modes)
    assert any("commit" in m for m in modes)


def test_occlusion_seeking_without_scene_warns_but_moves():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        states, modes = _rollout("occlusion_seeking", steps=40)
    moved = horizontal_distance(states[0].position, states[-1].position)
    assert moved > 1.0
    assert all(isinstance(m, str) for m in modes)


def test_difficulty_speed_ordering():
    # Hard target should cover more ground than easy over the same horizon.
    easy, _ = _rollout("direct_escape", difficulty="easy", steps=80)
    hard, _ = _rollout("direct_escape", difficulty="hard", steps=80)
    d_easy = horizontal_distance(easy[0].position, easy[-1].position)
    d_hard = horizontal_distance(hard[0].position, hard[-1].position)
    assert d_hard > d_easy


def test_generate_waypoints_fallback_deterministic():
    kw = dict(initial_target_pose=[5, 0, -0.3], initial_uav_pose=[0, 0, -18, 0],
              behavior_type="direct_escape", difficulty="medium", seed=3)
    wp1 = generate_target_waypoints(**kw)
    wp2 = generate_target_waypoints(**kw)
    assert wp1 == wp2
    assert len(wp1) >= 2
    assert all(len(p) == 3 for p in wp1)


def test_generate_waypoints_occlusion_warns_without_scene():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        wp = generate_target_waypoints(
            [5, 0, -0.3], [0, 0, -18, 0], "occlusion_seeking", "easy", seed=1,
        )
    assert len(wp) >= 2
    assert any("occlusion_seeking" in str(x.message) for x in w)


def test_all_behavior_types_runnable():
    for b in BEHAVIOR_TYPES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            states, _ = _rollout(b, steps=30)
        assert len(states) == 30

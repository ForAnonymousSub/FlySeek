# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Offline tests for hide-and-seek agent and tracking drone controller."""

from __future__ import annotations

import math

import numpy as np
import pytest

from flyseek.adversary import (
    DroneState,
    HideSeekCarAgent,
    TargetState,
    create_adversarial_agent,
    horizontal_distance,
)
from flyseek.adversary.hide_seek import DEFAULTS as HIDE_DEFAULTS
from flyseek.expert.tracking_drone import TrackingDroneController, TrackerConfig
from flyseek.utils.visibility import target_bearing_in_fov, target_visible


def _drone(x: float, y: float, z: float = -12.0) -> DroneState:
    return DroneState(position=np.array([x, y, z]), velocity=np.zeros(3), heading=0.0)


def _target(x: float, y: float, z: float = 0.0) -> TargetState:
    return TargetState(position=np.array([x, y, z]), velocity=np.zeros(3), heading=0.0)


class _Args:
    follow_distance = 12.0
    follow_altitude = 12.0
    drone_smoothing = 3.0
    camera_hfov_deg = 50.0
    lost_after_s = 0.5
    search_orbit_radius = 14.0
    no_collision = True


def test_factory_hide_seek():
    agent = create_adversarial_agent("hide_seek", seed=0)
    assert isinstance(agent, HideSeekCarAgent)


def test_hide_seek_open_road_phase_with_fake_occ():
    class FakeOcc:
        cfg = type("Cfg", (), {
            "map_elevation": 0.0,
            "min_height_thresh": 6.0,
            "car_agl_m": 0.35,
        })()

        def is_bev_occupied_ned(self, pos_ned):
            x, y = float(pos_ned[0]), float(pos_ned[1])
            if abs(y) <= 12.0 and -300.0 <= x <= 300.0:
                return False
            return True

        def has_ground_support_map(self, pos_map):
            return True

        def is_drivable_ned(self, pos_ned):
            return not self.is_bev_occupied_ned(pos_ned)

        def local_ground_map_z(self, _pos_map):
            return 0.0

        def snap_car_to_ground_ned(self, pos_ned):
            p = np.asarray(pos_ned, dtype=np.float64).reshape(3).copy()
            p[2] = -0.35
            return p

        def los_blocked_ned(self, *_a, **_k):
            return False

        def find_hide_goal_ned(self, target_ned, drone_ned, **kw):
            return target_ned + np.array([8.0, 0.0, 0.0])

    cfg = {
        **HIDE_DEFAULTS,
        "open_road_duration_s": 0.5,
        "open_road_min_route_frac": 0.1,
        "hide_trigger_range_m": 200.0,
    }
    agent = HideSeekCarAgent(config=cfg, occupancy=FakeOcc(), rng=np.random.default_rng(0))
    target = _target(0.0, 0.0)
    drone = _drone(5.0, -20.0)
    agent.reset(target)
    assert agent._phase == "open_road"
    modes = []
    for i in range(40):
        action = agent.step(drone, target, 0.1)
        modes.append(action.behavior_state)
        target = target.copy_with(
            position=target.position + action.desired_velocity * 0.1,
            timestamp=target.timestamp + 0.1,
        )
    assert "open_road" in modes
    assert any(m in ("goto_hide", "hiding") for m in modes)


def test_hide_seek_transitions_to_goto_hide_without_pcd():
    cfg = {**HIDE_DEFAULTS, "evade_before_hide_s": 2.0, "hide_trigger_range_m": 100.0}
    agent = HideSeekCarAgent(config=cfg, occupancy=None)
    target = _target(0.0, 0.0)
    drone = _drone(10.0, 0.0)
    agent.reset(target)
    dt = 0.2
    last_mode = "evade"
    for _ in range(int(2.0 / dt) + 5):
        action = agent.step(drone, target, dt)
        last_mode = action.behavior_state
        target = target.copy_with(
            position=target.position + action.desired_velocity * dt,
            timestamp=target.timestamp + dt,
        )
    assert last_mode in ("goto_hide", "hiding", "peek_reemerge")


def test_hide_seek_goto_hide_stuck_timeout_enters_hiding():
    cfg = {
        **HIDE_DEFAULTS,
        "evade_before_hide_s": 0.0,
        "hide_trigger_range_m": 100.0,
        "hide_stuck_timeout_s": 0.3,
    }
    agent = HideSeekCarAgent(config=cfg, occupancy=None)
    target = _target(0.0, 0.0)
    drone = _drone(10.0, 0.0)
    agent.reset(target)

    modes = []
    for _ in range(10):
        action = agent.step(drone, target, 0.1)
        modes.append(action.behavior_state)
        # Do not integrate target position: simulates PCD collision keeping the
        # car pinned while the agent is trying to reach hide_goal.
        target = target.copy_with(timestamp=target.timestamp + 0.1)

    assert "hiding" in modes


def test_tracking_drone_enters_search_when_not_visible():
    class FakeOcc:
        cfg = type("Cfg", (), {"map_elevation": 0.0, "min_drone_clearance": 8.0})()

        def los_blocked_ned(self, *a, **k):
            return True

        def local_ground_map_z(self, _pos_map):
            return 0.0

        def local_roof_map_z_window(self, _pos_map, *, range_m=2.0):
            return 0.0

        def min_safe_map_z(self, _probe_map):
            return 8.0

        def resolve_drone_ned(self, _prev, proposed):
            return proposed

    tracker = TrackingDroneController(
        _Args(), occupancy=FakeOcc(), cfg=TrackerConfig(lost_after_s=0.3)
    )
    drone = _drone(0.0, -15.0)
    target = _target(30.0, 0.0)
    tracker.reset(drone, target)
    dt = 0.2
    modes = []
    for i in range(20):
        t = i * dt
        target = target.copy_with(timestamp=t)
        drone = DroneState(
            position=drone.position.copy(),
            velocity=drone.velocity.copy(),
            heading=drone.heading,
            timestamp=t,
        )
        drone, log = tracker.step(drone, target, dt)
        modes.append(log["tracker_mode"])
    assert "search" in modes


def test_target_bearing_in_fov():
    drone_pos = np.array([0.0, 0.0, -10.0])
    assert target_bearing_in_fov(drone_pos, 0.0, np.array([10.0, 0.0, 0.0]), 50.0)
    assert not target_bearing_in_fov(drone_pos, 0.0, np.array([0.0, 10.0, 0.0]), 10.0)


def test_target_visible_open_sky():
    drone = _drone(0.0, -10.0)
    target = _target(20.0, 0.0)
    assert target_visible(None, drone, target)

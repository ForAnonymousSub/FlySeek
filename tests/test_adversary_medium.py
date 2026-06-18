# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Offline tests for the medium S-curve evasion agent.

Pure numpy, no AirSim/UE. Runs in CI / stdlib test runner.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from flyseek.adversary import (
    AgentAction,
    DroneState,
    PlayBox,
    SCurveEvasionAgent,
    TargetState,
    bearing_xy,
    create_adversarial_agent,
    horizontal_distance,
    integrate_target,
    wrap_to_pi,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _drone(x: float, y: float, z: float = -10.0) -> DroneState:
    return DroneState(position=np.array([x, y, z]),
                      velocity=np.zeros(3), heading=0.0)


def _target(x: float, y: float, z: float = 0.0,
            heading: float = 0.0) -> TargetState:
    return TargetState(position=np.array([x, y, z]),
                       velocity=np.zeros(3), heading=heading)


# --------------------------------------------------------------------------- #
# 1. unit conversions / wrap_to_pi                                            #
# --------------------------------------------------------------------------- #
def test_wrap_to_pi_round_trip():
    for a in [-2 * math.pi, -math.pi, 0.0, math.pi, 2 * math.pi, 7.0]:
        w = wrap_to_pi(a)
        assert -math.pi <= w <= math.pi + 1e-9
        # Equivalence modulo 2π
        diff = (a - w) % (2 * math.pi)
        assert abs(diff) < 1e-6 or abs(diff - 2 * math.pi) < 1e-6


def test_bearing_xy_cardinals():
    origin = np.array([0.0, 0.0, 0.0])
    # +X (north)
    assert abs(bearing_xy(origin, np.array([1.0, 0.0, 0.0])) - 0.0) < 1e-9
    # +Y (east)
    assert abs(bearing_xy(origin, np.array([0.0, 1.0, 0.0])) - math.pi / 2) < 1e-9
    # -X (south)
    assert abs(abs(bearing_xy(origin, np.array([-1.0, 0.0, 0.0]))) - math.pi) < 1e-9


def test_horizontal_distance_ignores_z():
    a = np.array([0.0, 0.0, -100.0])
    b = np.array([3.0, 4.0, 50.0])
    assert abs(horizontal_distance(a, b) - 5.0) < 1e-9


# --------------------------------------------------------------------------- #
# 2. mode classification                                                      #
# --------------------------------------------------------------------------- #
def test_medium_mode_cruise_when_drone_far():
    agent = SCurveEvasionAgent()
    target = _target(0, 0, heading=0.0)
    agent.reset(target)
    # Drone 100 m away (well outside engagement_range_max=40)
    drone = _drone(100, 0)
    action = agent.step(drone, target, dt=0.1)
    assert action.behavior_state == "cruise"
    assert abs(np.linalg.norm(action.desired_velocity[:2]) - 2.0) < 1e-6


def test_medium_mode_evade_when_drone_mid_range():
    agent = SCurveEvasionAgent()
    target = _target(0, 0)
    agent.reset(target)
    # Drone 20 m away in +X — within engagement window
    drone = _drone(20, 0)
    action = agent.step(drone, target, dt=0.1)
    assert action.behavior_state == "evade"
    # Should be roughly running away (−X direction) at first tick (sin(0)=0 sway)
    assert action.desired_velocity[0] < 0  # going -X
    assert abs(action.desired_velocity[1]) < 1e-6  # no sway yet


def test_medium_mode_panic_when_drone_too_close():
    agent = SCurveEvasionAgent()
    target = _target(0, 0)
    agent.reset(target)
    drone = _drone(3, 0)  # 3 m → below engagement_range_min=5
    action = agent.step(drone, target, dt=0.1)
    assert action.behavior_state == "panic"
    # Panic = no sway, straight away
    assert action.desired_velocity[0] < 0
    assert abs(action.desired_velocity[1]) < 1e-6


# --------------------------------------------------------------------------- #
# 3. sway introduces lateral motion in evade mode                             #
# --------------------------------------------------------------------------- #
def test_medium_evade_produces_lateral_sway():
    """After half a sway period, the heading should have swung off-axis."""
    agent = SCurveEvasionAgent(config={
        "engagement_range_max": 40.0,
        "sway_amplitude_deg": 30.0,
        "sway_period_s": 2.0,
    })
    target = _target(0, 0)
    agent.reset(target)
    drone = _drone(20, 0)

    # First tick: sway = sin(0) = 0 → heading ≈ π (running -X)
    a0 = agent.step(drone, target, dt=0.001)
    assert abs(wrap_to_pi(a0.desired_heading - math.pi)) < 1e-3

    # Advance ~0.5s in evade mode (quarter sway period) → max positive sway
    for _ in range(500):
        agent.step(drone, target, dt=0.001)
    a_quarter = agent.step(drone, target, dt=0.001)
    sway_angle = wrap_to_pi(a_quarter.desired_heading - math.pi)
    # Should be ≈ +30° (max positive sway)
    assert abs(sway_angle - math.radians(30)) < math.radians(2)


# --------------------------------------------------------------------------- #
# 4. integrator respects max_turn_rate and keep_z                             #
# --------------------------------------------------------------------------- #
def test_integrator_keep_z_locks_altitude():
    target = _target(0, 0, z=1.0)
    action = AgentAction(
        desired_velocity=np.array([1.0, 0.0, 5.0]),  # nonzero vz on purpose
        desired_heading=0.0,
    )
    new = integrate_target(target, action, dt=1.0, keep_z=1.0)
    assert abs(new.position[2] - 1.0) < 1e-9


def test_integrator_max_turn_rate_caps_heading_change():
    target = _target(0, 0, heading=0.0)
    action = AgentAction(
        desired_velocity=np.array([1.0, 0.0, 0.0]),
        # Pick an unambiguous direction (+π/2 east, not ±π south which would
        # be on the wrap boundary).
        desired_heading=math.pi / 2,
    )
    new = integrate_target(target, action, dt=0.1,
                           max_turn_rate_rad_s=math.pi / 2)  # 90 deg/s
    # 0.1 s * 90 deg/s = 9 deg, positive direction
    assert abs(new.heading - math.radians(9.0)) < 1e-3


def test_integrator_max_speed_caps_velocity():
    target = _target(0, 0)
    action = AgentAction(
        desired_velocity=np.array([10.0, 0.0, 0.0]),
        desired_heading=0.0,
    )
    new = integrate_target(target, action, dt=1.0, max_speed=5.0)
    assert abs(np.linalg.norm(new.velocity[:2]) - 5.0) < 1e-6
    assert abs(new.position[0] - 5.0) < 1e-6


# --------------------------------------------------------------------------- #
# 5. PlayBox boundary reflection                                              #
# --------------------------------------------------------------------------- #
def test_play_box_reflects_velocity_at_boundary():
    box = PlayBox(x_min=-10, x_max=10, y_min=-10, y_max=10)
    agent = SCurveEvasionAgent(play_box=box)
    # Place target at +X boundary; drone behind it → escape direction is +X further
    # (which would push out of bounds). The agent base class should reflect vx.
    target = _target(10.0, 0.0)
    agent.reset(target)
    drone = _drone(-5.0, 0.0)  # drone in -X, evade direction = +X = out of bounds
    action = agent.step(drone, target, dt=0.1)
    # Velocity X component should be reflected to ≤ 0 (escape would push past +10)
    assert action.desired_velocity[0] <= 1e-6


# --------------------------------------------------------------------------- #
# 6. Long-horizon closed-loop simulation: distance grows then stabilizes      #
# --------------------------------------------------------------------------- #
def test_medium_evasion_creates_distance_from_static_drone():
    """If the drone sits still, the evading car should put distance between them."""
    agent = SCurveEvasionAgent()
    target = _target(0, 0)
    agent.reset(target)
    drone = _drone(10, 0)  # drone 10 m east, static

    dt = 0.1
    initial_dist = horizontal_distance(target.position, drone.position)
    for _ in range(200):  # 20 seconds
        action = agent.step(drone, target, dt=dt)
        target = integrate_target(target, action, dt=dt,
                                  keep_z=0.0,
                                  max_speed=6.0,
                                  max_turn_rate_rad_s=math.pi)
    final_dist = horizontal_distance(target.position, drone.position)
    assert final_dist > initial_dist + 10  # ran away by at least 10 m


# --------------------------------------------------------------------------- #
# 7. Factory function                                                         #
# --------------------------------------------------------------------------- #
def test_factory_returns_correct_class():
    a_easy = create_adversarial_agent("easy", seed=42)
    a_med = create_adversarial_agent("medium")
    assert a_easy.__class__.__name__ == "RandomWalkAgent"
    assert a_med.__class__.__name__ == "SCurveEvasionAgent"


def test_factory_rejects_hard_with_message():
    with pytest.raises(NotImplementedError, match="hard difficulty"):
        create_adversarial_agent("hard")


def test_factory_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown difficulty"):
        create_adversarial_agent("nightmare")


def test_factory_slices_umbrella_config():
    cfg = {
        "easy":   {"speed_mps": 99.0},
        "medium": {"cruise_speed": 7.0},
    }
    med = create_adversarial_agent("medium", config=cfg)
    assert med.config["cruise_speed"] == 7.0
    eas = create_adversarial_agent("easy", config=cfg)
    assert eas.config["speed_mps"] == 99.0

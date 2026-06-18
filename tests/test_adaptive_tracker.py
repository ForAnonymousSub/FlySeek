# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""State-machine tests for AdaptiveTracker (offline numpy)."""

from __future__ import annotations

import math

import numpy as np

from flyseek.adversary import DroneState, TargetState
from flyseek.expert.adaptive_tracker import AdaptiveTracker, AdaptiveTrackerConfig


class _NoOcc:
    """Minimal occupancy stub: nothing occluded, no roof, flat ground."""

    cfg = type("Cfg", (), {"map_elevation": 0.0, "min_drone_clearance": 8.0})()

    def los_blocked_ned(self, *_a, **_k) -> bool:
        return False

    def local_ground_map_z(self, _pos_map) -> float:
        return 0.0

    def local_roof_map_z_window(self, _pos_map, *, range_m=2.0) -> float:
        return 12.0

    def resolve_drone_ned(self, _prev, proposed):
        return np.asarray(proposed, dtype=np.float64).reshape(3)


class _LosBlocked(_NoOcc):
    """Same as _NoOcc but always LOS-blocked — triggers PEEK."""

    def los_blocked_ned(self, drone_ned, target_ned, **_k):
        # Only allow LOS from y > 5: peek side-step recovers visibility.
        return float(drone_ned[1]) < 5.0


def _drone_at(x: float, y: float, z: float = -12.0, yaw: float = 0.0) -> DroneState:
    return DroneState(
        position=np.array([x, y, z]),
        velocity=np.zeros(3),
        heading=yaw,
        timestamp=0.0,
    )


def _target_at(
    x: float,
    y: float,
    vx: float = 0.0,
    vy: float = 0.0,
    heading: float = 0.0,
    t: float = 0.0,
) -> TargetState:
    return TargetState(
        position=np.array([x, y, -0.35]),
        velocity=np.array([vx, vy, 0.0]),
        heading=heading,
        timestamp=t,
    )


def _config_short_horizons() -> AdaptiveTrackerConfig:
    return AdaptiveTrackerConfig(
        follow_distance=10.0,
        follow_altitude=12.0,
        max_range_m=80.0,
        predict_after_s=0.2,
        peek_after_s=0.4,
        reacquire_after_s=0.8,
        search_after_s=1.6,
        hold_speed_thresh=0.3,
        hold_dwell_s=0.6,
        hold_resume_speed=1.0,
        drone_smoothing=4.0,
        yaw_gain=3.0,
        motion_dir_tau_s=0.3,
    )


def _step_n(tracker: AdaptiveTracker, drone: DroneState, target_fn, n: int, dt: float):
    modes = []
    for i in range(n):
        target = target_fn(i)
        drone, log = tracker.step(drone, target, dt)
        modes.append(log["tracker_mode"])
    return drone, modes


def test_track_when_visible_moving_target():
    tracker = AdaptiveTracker(
        cfg=_config_short_horizons(),
        occupancy=_NoOcc(),
    )
    target0 = _target_at(20.0, 0.0, vx=3.0, vy=0.0)
    drone = _drone_at(8.0, 0.0, yaw=math.radians(0.0))
    tracker.reset(drone, target0)

    def target_fn(i):
        t = (i + 1) * 0.1
        return _target_at(20.0 + 3.0 * t, 0.0, vx=3.0, vy=0.0, t=t)

    drone, modes = _step_n(tracker, drone, target_fn, n=10, dt=0.1)
    assert all(m == "track" for m in modes), f"unexpected modes: {modes}"


def test_predict_then_reacquire_then_search_on_long_loss():
    cfg = _config_short_horizons()
    tracker = AdaptiveTracker(cfg=cfg, occupancy=_NoOcc())
    target0 = _target_at(20.0, 0.0, vx=4.0, vy=0.0)
    drone = _drone_at(8.0, 0.0, yaw=0.0)
    tracker.reset(drone, target0)
    _, _ = tracker.step(drone, target0, 0.1)

    # Now teleport target far behind drone — out of FOV / range — to lose vis.
    modes = []
    for i in range(20):
        t = 0.1 + (i + 1) * 0.1
        target = _target_at(200.0, 200.0, vx=4.0, vy=0.0, t=t)
        drone, log = tracker.step(drone, target, 0.1)
        modes.append(log["tracker_mode"])

    assert "predict" in modes, modes
    assert "reacquire" in modes, modes
    assert "search" in modes, modes
    # Modes should monotonically degrade (predict → reacquire → search).
    first_predict = modes.index("predict")
    first_reacq = modes.index("reacquire")
    first_search = modes.index("search")
    assert first_predict <= first_reacq <= first_search


def test_peek_engages_on_los_block():
    cfg = _config_short_horizons()
    tracker = AdaptiveTracker(cfg=cfg, occupancy=_LosBlocked())
    target0 = _target_at(20.0, 0.0, vx=2.0, vy=0.0)
    drone = _drone_at(8.0, -5.0, yaw=0.0)  # y<5 => LOS blocked
    tracker.reset(drone, target0)
    seen_peek = False
    for i in range(15):
        t = (i + 1) * 0.1
        target = _target_at(20.0 + 2.0 * t, 0.0, vx=2.0, vy=0.0, t=t)
        drone, log = tracker.step(drone, target, 0.1)
        if log["tracker_mode"] == "peek":
            seen_peek = True
            break
    assert seen_peek, "expected PEEK after LOS-blocked dwell"


def test_hold_after_target_stops_visible():
    cfg = _config_short_horizons()
    cfg.hold_dwell_s = 0.5
    tracker = AdaptiveTracker(cfg=cfg, occupancy=_NoOcc())
    target0 = _target_at(12.0, 0.0, vx=0.0, vy=0.0)
    drone = _drone_at(2.0, 0.0, yaw=0.0)
    tracker.reset(drone, target0)

    def target_fn(i):
        t = (i + 1) * 0.1
        return _target_at(12.0, 0.0, vx=0.0, vy=0.0, t=t)

    drone, modes = _step_n(tracker, drone, target_fn, n=12, dt=0.1)
    assert "hold" in modes, modes
    assert modes[0] == "track"


def test_search_anchor_follows_predicted_target():
    """Regression: SEARCH must orbit a *moving* predicted anchor, not stale pos."""
    cfg = _config_short_horizons()
    cfg.search_after_s = 0.3
    cfg.predict_after_s = 0.1
    cfg.reacquire_after_s = 0.15
    tracker = AdaptiveTracker(cfg=cfg, occupancy=_NoOcc())
    target0 = _target_at(20.0, 0.0, vx=5.0, vy=0.0)
    drone = _drone_at(8.0, 0.0, yaw=0.0)
    tracker.reset(drone, target0)
    _, _ = tracker.step(drone, target0, 0.1)

    last_pred = None
    for i in range(15):
        t = 0.1 + (i + 1) * 0.1
        # Target keeps moving but is out of FOV/range so it stays "lost".
        target = _target_at(300.0 + 5.0 * t, 0.0, vx=5.0, vy=0.0, t=t)
        drone, log = tracker.step(drone, target, 0.1)
        if log["tracker_mode"] == "search":
            cur = log["predicted_xy"][0]
            if last_pred is not None:
                assert cur > last_pred - 0.5, \
                    f"predicted x should grow with last_seen_vel; got {last_pred}→{cur}"
            last_pred = cur
    assert last_pred is not None, "expected SEARCH to engage"

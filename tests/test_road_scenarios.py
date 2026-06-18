# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Offline tests for road route scenarios and UAV predict mode."""

from __future__ import annotations

import numpy as np

from flyseek.adversary import DroneState, TargetState
from flyseek.expert.tracking_drone import TrackerConfig, TrackingDroneController
from flyseek.scenarios import RoadScenarioConfig, RoadScenarioController
from flyseek.utils.road_graph import build_route, find_major_road_seed


class FakeRoadOcc:
    cfg = type("Cfg", (), {
        "map_elevation": 0.0,
        "min_drone_clearance": 8.0,
    })()

    def local_ground_map_z(self, _pos_map):
        return 0.0

    def is_bev_occupied_ned(self, pos_ned: np.ndarray) -> bool:
        return not self.is_drivable_ned(pos_ned)

    def is_drivable_ned(self, pos_ned: np.ndarray) -> bool:
        x, y = float(pos_ned[0]), float(pos_ned[1])
        if abs(y) <= 12.0 and -300.0 <= x <= 300.0:
            return True
        if abs(x - 90.0) <= 12.0 and -140.0 <= y <= 140.0:
            return True
        return False

    def has_ground_support_map(self, _pos_map) -> bool:
        return True

    def local_ground_map_z(self, _pos_map) -> float:
        return 0.0

    def snap_car_to_ground_ned(self, pos_ned: np.ndarray) -> np.ndarray:
        p = np.asarray(pos_ned, dtype=np.float64).reshape(3).copy()
        p[2] = -0.35
        return p

    def resolve_bev_move_ned(self, _prev, proposed, *, keep_z=None):
        return self.snap_car_to_ground_ned(proposed)

    def local_roof_map_z_window(self, _pos_map, *, range_m=2.0):
        return 12.0

    def min_safe_map_z(self, _probe_map):
        return 12.0

    def resolve_drone_ned(self, _prev, proposed):
        return np.asarray(proposed, dtype=np.float64).reshape(3)

    def los_blocked_ned(self, *_a, **_k):
        return True


class Args:
    follow_distance = 12.0
    follow_altitude = 20.0
    drone_smoothing = 3.0
    camera_hfov_deg = 50.0
    lost_after_s = 1.0
    predict_after_s = 0.1
    search_orbit_radius = 14.0
    no_collision = True
    altitude_smooth_tau = 3.0
    max_climb_mps = 1.5
    max_drop_mps = 2.0


def test_major_road_seed_stays_in_corridor():
    occ = FakeRoadOcc()
    rng = np.random.default_rng(0)
    seed, heading, score = find_major_road_seed(
        occ, np.array([0.0, 80.0, 0.0]), rng, keep_z=0.0
    )
    assert not occ.is_bev_occupied_ned(seed)
    assert score > 30.0
    assert abs(np.sin(heading)) < 0.5 or abs(np.cos(heading)) < 0.5


def test_road_scenario_moves_on_free_route():
    occ = FakeRoadOcc()
    rng = np.random.default_rng(1)
    init = TargetState(position=np.array([0.0, 0.0, 0.0]), velocity=np.zeros(3))
    ctrl = RoadScenarioController(
        occ,
        init,
        rng,
        RoadScenarioConfig(name="high_maneuver", route_len_m=90.0),
    )
    drone = DroneState(position=np.array([-12.0, 0.0, -20.0]), velocity=np.zeros(3))
    state = ctrl.initial_state()
    for i in range(30):
        state, action = ctrl.step(drone, i * 0.1, 0.1)
        assert not occ.is_bev_occupied_ned(state.position)
        assert abs(float(state.position[2])) < 1e-9
        assert action.behavior_state == "high_maneuver"


def test_tracker_predicts_before_search():
    occ = FakeRoadOcc()
    tracker = TrackingDroneController(Args(), occupancy=occ)
    drone = DroneState(position=np.array([0.0, -12.0, -20.0]),
                       velocity=np.zeros(3), heading=0.0)
    target = TargetState(position=np.array([20.0, 0.0, 0.0]),
                         velocity=np.array([4.0, 0.0, 0.0]))
    tracker.reset(drone, target)
    modes = []
    for i in range(6):
        target = target.copy_with(timestamp=i * 0.2)
        drone, log = tracker.step(drone, target, 0.2)
        modes.append(log["tracker_mode"])
    assert "predict" in modes
    assert "search" not in modes[:4]

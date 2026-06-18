# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Tests for occlusion-seeking route planning (PCD-aware hide leg)."""

from __future__ import annotations

import numpy as np
import pytest

from flyseek.adversary import DroneState, TargetState
from flyseek.bench.target_policy import (
    RouteFollowingTargetPolicy,
    create_target_policy,
)
from flyseek.utils.occlusion_route import (
    analyze_route_occlusion,
    build_occlusion_seeking_route,
    refine_route_hide_leg,
)
from flyseek.utils.road_graph import build_route


class AlleyOcc:
    """Main E-W road + N-side alley pocket hidden from a south-side drone."""

    cfg = type("Cfg", (), {
        "map_elevation": 0.0,
        "min_drone_clearance": 8.0,
    })()

    def is_bev_occupied_ned(self, pos_ned: np.ndarray) -> bool:
        return not self.is_drivable_ned(pos_ned)

    def is_drivable_ned(self, pos_ned: np.ndarray) -> bool:
        x, y = float(pos_ned[0]), float(pos_ned[1])
        # E-W main road
        if abs(y) <= 10.0 and -200.0 <= x <= 200.0:
            return True
        # N-side alley + connector from main road at x≈50
        if 40.0 <= x <= 75.0 and 8.0 <= y <= 42.0:
            return True
        return False

    def _segment_drivable(self, a, b, *, keep_z):
        steps = max(2, int(np.linalg.norm(b[:2] - a[:2]) / 2.0))
        for i in range(steps + 1):
            t = i / steps
            p = a + t * (b - a)
            if not self.is_drivable_ned(p):
                return False
        return True

    def snap_car_to_ground_ned(self, pos_ned: np.ndarray) -> np.ndarray:
        p = np.asarray(pos_ned, dtype=np.float64).reshape(3).copy()
        p[2] = -0.35
        return p

    def resolve_bev_move_ned(self, _prev, proposed, *, keep_z=None):
        return self.snap_car_to_ground_ned(proposed)

    def los_blocked_ned(self, drone_ned, target_ned, **kwargs) -> bool:
        # Drone south of main road; alley (y>12) is behind occluders.
        return float(target_ned[1]) > 12.0 and float(drone_ned[1]) < 5.0

    def los_blocked_by_building_ned(self, drone_ned, target_ned, **kwargs) -> bool:
        # Only the wide north "building" strip counts — not a pole at x=20.
        ty = float(target_ned[1])
        if float(drone_ned[1]) >= 5.0:
            return False
        return ty > 18.0

    def building_occludes_between_ned(self, drone_ned, target_ned, **kwargs) -> bool:
        return self.los_blocked_by_building_ned(drone_ned, target_ned, **kwargs)

    def first_building_occluder_on_ray_ned(self, drone_ned, target_ned, **kwargs):
        if self.los_blocked_by_building_ned(drone_ned, target_ned, **kwargs):
            t = 0.55
            return t, np.array([float(target_ned[0]), 15.0, 10.0])
        return None

    def has_adjacent_building_wall_ned(
        self, pos_ned, *, keep_z, min_footprint_cells=9, probe_dist_m=7.5, **kwargs,
    ) -> bool:
        return float(pos_ned[1]) > 16.0 and 40.0 <= float(pos_ned[0]) <= 75.0

    def find_hide_goal_ned(
        self, target_ned, drone_ned, *, keep_z, search_radius_m=28.0,
        building_only=False, require_adjacent_building=True, **kwargs,
    ):
        base = np.asarray(target_ned, dtype=np.float64).reshape(3).copy()
        base[2] = keep_z
        for y in (18.0, 24.0, 30.0):
            cand = np.array([55.0, y, keep_z], dtype=np.float64)
            hidden = (
                self.los_blocked_by_building_ned(drone_ned, cand)
                if building_only else self.los_blocked_ned(drone_ned, cand)
            )
            if self.is_drivable_ned(cand) and hidden:
                return cand
        return None

    def _alley_hide_bonus_ned(self, pos_ned, *, keep_z, max_width=16.0, step=2.0):
        p = np.asarray(pos_ned, dtype=np.float64).reshape(3).copy()
        side_h = 0.0
        for sign in (-1.0, 1.0):
            cur = p.copy()
            walked = 0.0
            while walked + step <= max_width:
                cur[0] += sign * step
                if self.is_bev_occupied_ned(cur):
                    break
                walked += step
            side_h += walked
        return max(0.0, 12.0 - side_h) * 1.5


def test_build_occlusion_seeking_route_has_occluded_hide_leg():
    occ = AlleyOcc()
    anchor = np.array([0.0, 0.0, -0.35])
    drone = np.array([-12.0, -5.0, -14.0])
    route, meta = build_occlusion_seeking_route(
        occ, anchor, np.random.default_rng(3),
        keep_z=-0.35, drone_ned=drone, route_len_m=120.0,
        anchor_heading_rad=0.0, max_attempts=4,
    )
    assert route.waypoints.shape[0] >= 4
    assert meta.get("hide_goal") is not None or meta.get("building_occluded_frac", 0) > 0.0


def test_refine_route_hide_leg_appends_toward_goal():
    occ = AlleyOcc()
    anchor = np.array([0.0, 0.0, -0.35])
    drone = np.array([-12.0, -5.0, -14.0])
    base = build_route(
        occ, anchor, np.random.default_rng(0),
        keep_z=-0.35, route_len_m=90.0, maneuver="open_then_hide",
        start_at_anchor=True, anchor_heading_rad=0.0,
    )
    refined = refine_route_hide_leg(base, occ, drone, keep_z=-0.35)
    meta = analyze_route_occlusion(refined, occ, drone, keep_z=-0.35)
    assert meta["hide_goal"] is not None


def test_route_following_policy_uses_road_controller():
    occ = AlleyOcc()
    pol = create_target_policy(
        "occlusion_seeking",
        config={"difficulty": "medium", "dt": 0.2},
        scene_context={"occupancy": occ, "drone_eye_agl_m": 14.0},
        seed=1,
    )
    assert isinstance(pol, RouteFollowingTargetPolicy)
    target = TargetState(position=np.array([0.0, 0.0, -0.35]), velocity=np.zeros(3))
    uav = DroneState(position=np.array([-12.0, -5.0, -14.0]), velocity=np.zeros(3))
    pol.reset(target, uav_state=uav)
    st = target
    modes = []
    for i in range(20):
        st = pol.get_next_target_state(i * 0.2, st, uav)
        modes.append(pol.last_action.behavior_state)
    assert not occ.is_bev_occupied_ned(st.position)
    assert any(m in ("open_road", "approach_alley", "normal_drive") for m in modes)


def test_create_target_policy_reactive_for_other_behaviors():
    pol = create_target_policy("sharp_turn", config={"difficulty": "easy"}, seed=0)
    from flyseek.bench.target_policy import TargetPolicy
    assert isinstance(pol, TargetPolicy)

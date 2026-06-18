# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Tests for the visibility-aware expert viewpoint planner."""

from __future__ import annotations

import json
import math

import numpy as np

from flyseek.bench.expert_trajectory import (
    ExpertTrajectoryConfig,
    ExpertViewpointPlanner,
    build_expert_trajectory_for_episode,
)


def _straight_target(n=20, dt=0.2, speed=3.0):
    return [
        {"t": i * dt, "pos": [i * speed * dt, 0.0, -0.3],
         "vel": [speed, 0.0, 0.0]}
        for i in range(n)
    ]


def test_plan_basic_structure():
    planner = ExpertViewpointPlanner(scene_context={})  # no occupancy
    out = planner.plan(_straight_target())
    assert set(out) >= {
        "uav_trajectory", "target_trajectory", "expert_viewpoints",
        "selected_scores", "config",
    }
    assert len(out["expert_viewpoints"]) == len(out["selected_scores"])
    vp = out["expert_viewpoints"][0]
    assert set(vp) >= {"t", "position", "heading"}
    assert len(vp["position"]) == 3


def test_viewpoints_keep_follow_distance():
    cfg = ExpertTrajectoryConfig(follow_distance_m=12.0)
    planner = ExpertViewpointPlanner(config=cfg, scene_context={})
    out = planner.plan(_straight_target())
    tgt = out["target_trajectory"]
    for vp in out["expert_viewpoints"]:
        i = vp["frame_idx"]
        p = np.array(vp["position"])
        t = np.array(tgt[i]["pos"])
        d = math.hypot(p[0] - t[0], p[1] - t[1])
        # within the sampled radius band around the desired follow distance
        assert 8.0 <= d <= 16.0


def test_heading_points_at_target():
    planner = ExpertViewpointPlanner(scene_context={})
    out = planner.plan(_straight_target())
    tgt = out["target_trajectory"]
    for vp in out["expert_viewpoints"][:5]:
        i = vp["frame_idx"]
        p = np.array(vp["position"])
        t = np.array(tgt[i]["pos"])
        bearing = math.atan2(t[1] - p[1], t[0] - p[0])
        assert abs((bearing - vp["heading"] + math.pi) % (2 * math.pi) - math.pi) < 1e-6


def test_determinism():
    p1 = ExpertViewpointPlanner(scene_context={}, seed=1).plan(_straight_target())
    p2 = ExpertViewpointPlanner(scene_context={}, seed=1).plan(_straight_target())
    assert p1["expert_viewpoints"] == p2["expert_viewpoints"]


def test_score_components_present():
    planner = ExpertViewpointPlanner(scene_context={})
    out = planner.plan(_straight_target())
    comp = out["selected_scores"][0]["components"]
    assert set(comp) == {
        "expected_visibility", "occlusion_risk", "distance_cost",
        "collision_risk", "smoothness_cost",
    }


class _LosOccupancy:
    """Stub: LoS blocked iff target y>0 region (so a side viewpoint stays clear)."""

    def __init__(self):
        self.cfg = type("C", (), {"min_drone_clearance": 8.0})()

    def los_blocked_ned(self, observer, target, *, drone_eye_agl_m, target_agl_m):
        # Blocked when the observer is on the -X side of the target (behind a
        # "wall" at the target). Encourages the planner to pick +X / lateral.
        return bool(observer[0] < target[0] - 1.0)

    def is_3d_occupied_map(self, p):
        return False

    def local_roof_map_z(self, p):
        return 0.0


def test_occlusion_aware_avoids_blocked_side():
    cfg = ExpertTrajectoryConfig(beta_occlusion=2.0, mu_smoothness=0.0)
    planner = ExpertViewpointPlanner(config=cfg, scene_context={"occupancy": _LosOccupancy()})
    out = planner.plan(_straight_target(n=6))
    # With LoS blocked from the -X side, expert viewpoints should avoid sitting
    # well behind the target on -X.
    behind = [vp for vp in out["expert_viewpoints"]
              if vp["position"][0] < out["target_trajectory"][vp["frame_idx"]]["pos"][0] - 2.0]
    assert len(behind) <= 1
    assert out["occupancy_available"] is True


def test_preemptive_horizon_used():
    cfg = ExpertTrajectoryConfig(horizon_steps=5)
    planner = ExpertViewpointPlanner(config=cfg, scene_context={})
    out = planner.plan(_straight_target())
    assert out["preemptive_horizon_steps"] == 5


def test_build_for_episode_writes_file(tmp_path):
    ep = tmp_path / "ep"
    ep.mkdir()
    with (ep / "flyseek_meta.jsonl").open("w") as f:
        for i in range(12):
            rec = {
                "timestamp": i * 0.2,
                "drone_state": {"pos": [i * 0.6 - 12, 0, -18], "heading": 0.0},
                "target_state": {"pos": [i * 0.6, 0, -0.3], "vel": [3.0, 0, 0]},
            }
            f.write(json.dumps(rec) + "\n")
    out = build_expert_trajectory_for_episode(ep, scene_context={})
    assert (ep / "trajectories.json").exists()
    loaded = json.loads((ep / "trajectories.json").read_text())
    assert loaded["uav_trajectory"] is not None
    assert len(loaded["expert_viewpoints"]) > 0

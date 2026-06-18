# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Tests for episode-level FlySeek-Bench metrics."""

from __future__ import annotations

import json

from flyseek.bench.metrics import (
    MetricsConfig,
    compute_metrics,
    evaluate_episode_dir,
)


def _frame(i, visible, collision=False, x=0.0):
    return {
        "frame_id": i,
        "timestamp": i * 0.1,
        "target_visible": visible,
        "collision": collision,
        "uav_pose": [x, 0.0, -18.0, 0.0],
    }


def test_empty_episode():
    m = compute_metrics([], difficulty="medium")
    assert m["total_frames"] == 0
    assert m["tracking_success"] is False


def test_visibility_ratio_and_success():
    frames = [_frame(i, i % 10 != 0, x=i * 1.0) for i in range(100)]  # 90% visible
    m = compute_metrics(frames, difficulty="medium")
    assert abs(m["target_visibility_ratio"] - 0.9) < 1e-6
    assert m["tracking_success"] is True  # 0.9 >= 0.70, no collision


def test_success_fails_below_threshold():
    frames = [_frame(i, i % 2 == 0, x=i) for i in range(100)]  # 50% visible
    m = compute_metrics(frames, difficulty="medium")
    assert m["target_visibility_ratio"] == 0.5
    assert m["tracking_success"] is False


def test_collision_fails_success():
    frames = [_frame(i, True, collision=(i == 5), x=i) for i in range(50)]
    m = compute_metrics(frames, difficulty="easy")
    assert m["target_visibility_ratio"] == 1.0
    assert m["collision_flag"] is True
    assert m["tracking_success"] is False


def test_hard_threshold_lower():
    frames = [_frame(i, i % 100 < 65, x=i) for i in range(100)]  # 65% visible
    easy = compute_metrics(frames, difficulty="easy")     # thr 0.70 -> fail
    hard = compute_metrics(frames, difficulty="hard")     # thr 0.60 -> pass
    assert easy["tracking_success"] is False
    assert hard["tracking_success"] is True


def test_configurable_thresholds():
    frames = [_frame(i, i % 100 < 65, x=i) for i in range(100)]
    cfg = MetricsConfig(success_thresholds={"medium": 0.5})
    m = compute_metrics(frames, difficulty="medium", config=cfg)
    assert m["success_threshold"] == 0.5
    assert m["tracking_success"] is True


def test_los_continuity_longest_segment():
    # visible pattern: 5 visible, 2 invisible, 3 visible -> longest run 5 / 10
    pattern = [True]*5 + [False]*2 + [True]*3
    frames = [_frame(i, v, x=i) for i, v in enumerate(pattern)]
    m = compute_metrics(frames, difficulty="medium")
    assert m["line_of_sight_continuity"] == 0.5
    assert m["target_lost_frames"] == 2


def test_re_acquisition_time():
    # visible, then lost 3 frames, recover; lost 1 frame, recover.
    pattern = [True, True] + [False]*3 + [True, True] + [False] + [True, True]
    frames = [_frame(i, v, x=i) for i, v in enumerate(pattern)]
    m = compute_metrics(frames, difficulty="medium")
    assert m["re_acquisition_events"] == 2
    assert abs(m["re_acquisition_time_frames"] - 2.0) < 1e-6  # mean(3,1)=2
    assert "re_acquisition_time_s" in m


def test_trailing_loss_not_counted_as_reacquisition():
    pattern = [True, True, False, False]  # never recovers
    frames = [_frame(i, v, x=i) for i, v in enumerate(pattern)]
    m = compute_metrics(frames, difficulty="medium")
    assert m["re_acquisition_events"] == 0


def test_path_length_and_efficiency():
    # UAV moves straight along +x by 1 m per frame -> path length == net disp.
    frames = [_frame(i, True, x=float(i)) for i in range(11)]
    m = compute_metrics(frames, difficulty="medium")
    assert abs(m["path_length_m"] - 10.0) < 1e-6
    assert abs(m["path_efficiency"] - 1.0) < 1e-6


def test_evaluate_episode_dir(tmp_path):
    ep = tmp_path / "ep"
    ep.mkdir()
    (ep / "metadata.json").write_text(json.dumps(
        {"episode_id": "ep_test", "difficulty_level": "hard"}))
    with (ep / "frames.jsonl").open("w") as f:
        for i in range(50):
            f.write(json.dumps(_frame(i, i % 100 < 62, x=i)) + "\n")
    m = evaluate_episode_dir(ep, write=True)
    assert (ep / "metrics.json").exists()
    assert m["episode_id"] == "ep_test"
    assert m["difficulty_level"] == "hard"
    assert m["success_threshold"] == 0.60

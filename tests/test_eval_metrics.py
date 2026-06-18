# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Tests for tracking evaluation metrics."""

from __future__ import annotations

import numpy as np

from flyseek.eval.metrics import (
    aggregate_metrics,
    compute_episode_metrics,
    compute_metrics_from_series,
)


def test_all_visible():
    vis = np.ones(20, dtype=bool)
    m = compute_metrics_from_series(vis, dt=0.1)
    assert m.track_auc == 1.0
    assert m.lost_rate == 0.0
    assert m.redetection_events == 0
    assert m.redetection_time_s == 0.0


def test_track_auc_and_lost_rate():
    vis = np.array([1, 1, 0, 0, 1, 1, 1, 0, 1, 1], dtype=bool)
    m = compute_metrics_from_series(vis, dt=0.5)
    assert abs(m.track_auc - 0.7) < 1e-9
    assert abs(m.lost_rate - 0.3) < 1e-9


def test_redetection_time():
    # One lost run of length 2 (indices 2,3) recovered at index 4, and one lost
    # run of length 1 (index 7) recovered at 8. dt=1 => latencies 3 and 2 -> mean 2.5.
    vis = np.array([1, 1, 0, 0, 1, 1, 1, 0, 1, 1], dtype=bool)
    m = compute_metrics_from_series(vis, dt=1.0)
    assert m.redetection_events == 2
    assert abs(m.redetection_time_s - 2.5) < 1e-9


def test_trailing_lost_run_not_counted():
    vis = np.array([1, 1, 0, 0, 0], dtype=bool)  # never recovers
    m = compute_metrics_from_series(vis, dt=1.0)
    assert m.redetection_events == 0


def test_collision_rate():
    vis = np.ones(10, dtype=bool)
    coll = np.zeros(10, dtype=bool)
    coll[3] = True
    m = compute_metrics_from_series(vis, dt=0.1, collisions=coll)
    assert abs(m.collision_rate - 0.1) < 1e-9


def test_compute_from_meta_records():
    records = []
    for i in range(10):
        records.append({
            "timestamp": i * 0.1,
            "target_visible": (i % 2 == 0),
        })
    m = compute_episode_metrics(records)
    assert 0.0 < m.track_auc < 1.0
    assert m.frames == 10


def test_aggregate():
    a = compute_metrics_from_series(np.ones(10, dtype=bool), dt=0.1)
    b = compute_metrics_from_series(
        np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=bool), dt=0.1
    )
    agg = aggregate_metrics([a, b])
    assert agg["episodes"] == 2
    assert "track_auc" in agg
    assert agg["track_auc"]["max"] == 1.0

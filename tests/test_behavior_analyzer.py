# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Tests for numeric target-behavior classification."""

from __future__ import annotations

import math

import numpy as np

from flyseek.instruction.behavior_analyzer import classify, classify_from_meta


def test_straight_line():
    pos = np.zeros((50, 3))
    pos[:, 0] = np.arange(50) * 0.2  # constant +X heading
    res = classify(pos, dt=0.05)
    assert res.behavior_class == "straight"
    assert res.direction_changes == 0


def test_zigzag():
    # Alternating left/right heading every few steps -> many direction changes.
    pos = [(0.0, 0.0, 0.0)]
    x = y = 0.0
    for i in range(60):
        ang = math.radians(40.0 if (i // 4) % 2 == 0 else -40.0)
        x += math.cos(ang)
        y += math.sin(ang)
        pos.append((x, y, 0.0))
    res = classify(np.asarray(pos), dt=0.1)
    assert res.behavior_class in ("zigzag", "dodging")
    assert res.direction_changes >= 4


def test_cover_using_from_occlusion():
    pos = np.zeros((40, 3))
    pos[:, 0] = np.arange(40) * 0.1
    occ = np.zeros(40, dtype=bool)
    occ[10:25] = True  # > 12% occluded
    res = classify(pos, dt=0.05, occluded_flags=occ)
    assert res.behavior_class == "cover_using"
    assert res.occluded_fraction > 0.12


def test_classify_from_meta_records():
    records = []
    for i in range(30):
        records.append({
            "timestamp": i * 0.05,
            "target_state": {"pos": [i * 0.2, 0.0, -0.3], "is_occluded": False},
        })
    res = classify_from_meta(records)
    assert res.behavior_class == "straight"
    assert res.descriptor

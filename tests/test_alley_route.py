# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Tests for hutong / alley route planning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def occ_and_seg():
    import sys
    sys.path.insert(0, str(REPO / "flyseek_extend"))
    from flyseek.adapters.pcd_occupancy import PcdOccupancyMap
    from flyseek.utils.seg_buildings import SegBuildingMap

    occ = PcdOccupancyMap.load_or_build(REPO, env_name="env_airsim_16")
    seg = SegBuildingMap.from_jsonl(
        REPO / "scene_data" / "seg_map" / "env_airsim_16.jsonl",
        footprint_radius_m=10.0,
    )
    return occ, seg


def test_find_best_alley_scene(occ_and_seg):
    occ, seg = occ_and_seg
    alley, anchor = __import__(
        "flyseek.utils.alley_route", fromlist=["find_best_alley_scene"]
    ).find_best_alley_scene(occ, seg, keep_z=-0.6)
    assert alley is not None
    assert anchor is not None
    assert 3.0 <= alley.corridor_width_m <= 14.0
    assert alley.depth_m >= 10.0


def test_build_alley_hutong_route(occ_and_seg):
    from flyseek.utils.alley_route import build_alley_hutong_route, find_best_alley_scene

    occ, seg = occ_and_seg
    alley, anchor = find_best_alley_scene(occ, seg, keep_z=-0.6)
    route, meta = build_alley_hutong_route(
        occ, seg, anchor, np.random.default_rng(42),
        keep_z=-0.6, preset_alley=alley,
    )
    assert route is not None
    assert meta["planner"] == "alley_hutong"
    assert route.waypoints.shape[0] >= 4
    assert meta["corridor_width_m"] <= 14.0

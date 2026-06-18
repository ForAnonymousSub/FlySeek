# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Tests for seg_map annotated building loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from flyseek.utils.seg_buildings import (
    SegBuildingMap,
    load_seg_buildings,
    parse_landmark_filename,
)
from flyseek.utils.coords import map_to_airsim_ned

REPO = Path(__file__).resolve().parents[2]
JSONL = REPO / "scene_data" / "seg_map" / "env_airsim_16.jsonl"


def test_parse_landmark_filename():
    xyz = parse_landmark_filename("X=281.78Y=-816.76Z=31.15.png")
    assert xyz is not None
    assert np.allclose(xyz, [281.78, -816.76, 31.15])
    ned = map_to_airsim_ned(xyz)
    assert np.isclose(ned[1], 816.76)


def test_load_env_airsim_16_buildings():
    if not JSONL.is_file():
        return
    b = load_seg_buildings(JSONL)
    assert len(b) >= 100
    assert all(x.type == "building" for x in b)


def test_seg_los_blocked():
    seg = SegBuildingMap(
        buildings=[],
        footprint_radius_m=10.0,
        min_occluder_height_m=8.0,
    )
    from flyseek.utils.seg_buildings import SegBuilding
    bd = SegBuilding(
        index=0, type="building",
        map_xyz=np.array([0.0, 0.0, 25.0]),
        ned_xyz=np.array([0.0, 0.0, -25.0]),
        height_map_z=25.0,
    )
    seg = SegBuildingMap(
        buildings=[bd], footprint_radius_m=10.0, min_occluder_height_m=8.0,
    )
    observer = np.array([-40.0, 0.0, -14.0])
    target = np.array([10.0, 0.0, -0.6])
    assert seg.los_blocked_by_annotated_building_ned(observer, target)

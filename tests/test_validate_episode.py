# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Tests for the episode validator + paper-consistency report."""

from __future__ import annotations

import json

from flyseek_bench.run_generate_episodes import main as gen_main
from flyseek.bench.validate_episode import (
    build_consistency_report,
    find_episode_dirs,
    validate_episode,
)


def _gen(tmp_path, *, behavior="direct_escape", difficulty="medium", num=1, seed=5):
    rc = gen_main([
        "--scene_id", "unit_v", "--behavior", behavior, "--difficulty", difficulty,
        "--num_episodes", str(num), "--seed", str(seed),
        "--output_dir", str(tmp_path), "--dry_run",
    ])
    assert rc == 0
    return [p for p in sorted(tmp_path.iterdir()) if p.is_dir()]


def test_valid_episode_passes(tmp_path):
    ep = _gen(tmp_path, num=1)[0]
    rep = validate_episode(ep)
    assert rep["passed"], rep["issues"]
    assert rep["difficulty"] == "medium"
    assert rep["behavior"] == "direct_escape"
    assert 0.0 <= rep["visibility_ratio"] <= 1.0


def test_missing_required_file_fails(tmp_path):
    ep = _gen(tmp_path, num=1)[0]
    (ep / "metrics.json").unlink()
    rep = validate_episode(ep)
    assert not rep["passed"]
    assert any("metrics.json" in i for i in rep["issues"])


def test_bad_visibility_ratio_fails(tmp_path):
    ep = _gen(tmp_path, num=1)[0]
    m = json.loads((ep / "metrics.json").read_text())
    m["target_visibility_ratio"] = 1.5
    (ep / "metrics.json").write_text(json.dumps(m))
    rep = validate_episode(ep)
    assert not rep["passed"]
    assert any("out of [0,1]" in i for i in rep["issues"])


def test_empty_instruction_fails(tmp_path):
    ep = _gen(tmp_path, num=1)[0]
    instr = json.loads((ep / "instruction.json").read_text())
    instr["instruction"] = ""
    (ep / "instruction.json").write_text(json.dumps(instr))
    rep = validate_episode(ep)
    assert not rep["passed"]
    assert any("empty instruction" in i for i in rep["issues"])


def test_missing_image_fails(tmp_path):
    ep = _gen(tmp_path, num=1)[0]
    imgs = sorted((ep / "images").glob("*.png"))
    imgs[0].unlink()
    rep = validate_episode(ep)
    assert not rep["passed"]
    assert any("image paths do not exist" in i for i in rep["issues"])


def test_no_image_check_skips(tmp_path):
    ep = _gen(tmp_path, num=1)[0]
    sorted((ep / "images").glob("*.png"))[0].unlink()
    rep = validate_episode(ep, check_images=False)
    assert rep["passed"], rep["issues"]


def test_trajectories_required_keys(tmp_path):
    ep = _gen(tmp_path, num=1)[0]
    (ep / "trajectories.json").write_text(json.dumps({"target_trajectory": []}))
    rep = validate_episode(ep)
    assert not rep["passed"]
    assert any("target_trajectory" in i or "uav_trajectory" in i for i in rep["issues"])


def test_batch_discovery_and_consistency_report(tmp_path):
    _gen(tmp_path, behavior="direct_escape", difficulty="easy", num=2, seed=1)
    _gen(tmp_path, behavior="occlusion_seeking", difficulty="hard", num=1, seed=10)
    eps = find_episode_dirs(tmp_path)
    assert len(eps) == 3
    reports = [validate_episode(e) for e in eps]
    consistency = build_consistency_report(reports)
    assert consistency["num_episodes"] == 3
    assert consistency["by_difficulty"].get("easy") == 2
    assert consistency["by_difficulty"].get("hard") == 1
    assert consistency["by_behavior"].get("direct_escape") == 2
    assert consistency["by_behavior"].get("occlusion_seeking") == 1
    assert 0.0 <= consistency["mean_visibility_ratio"] <= 1.0
    assert consistency["success_rate"] is not None
    assert consistency["collision_count"] == 0

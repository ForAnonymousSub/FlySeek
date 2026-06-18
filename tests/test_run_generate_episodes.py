# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Tests for the batch episode generation entry point (dry-run, no simulator)."""

from __future__ import annotations

import json

from flyseek_bench.run_generate_episodes import main

REQUIRED = ["metadata.json", "frames.jsonl", "trajectories.json",
            "instruction.json", "visibility.json", "metrics.json", "config.yaml"]


def _run(tmp_path, **kw):
    # Use a scene id with no PCD so occupancy load returns None (fast, offline).
    argv = [
        "--scene_id", kw.get("scene", "unit_test_scene"),
        "--difficulty", kw.get("difficulty", "medium"),
        "--behavior", kw.get("behavior", "direct_escape"),
        "--seed", str(kw.get("seed", 5)),
        "--num_episodes", str(kw.get("num", 1)),
        "--output_dir", str(tmp_path),
        "--dry_run",
    ]
    return main(argv)


def test_single_episode_structure(tmp_path):
    rc = _run(tmp_path, num=1)
    assert rc == 0
    eps = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(eps) == 1
    ep = eps[0]
    for name in REQUIRED:
        assert (ep / name).exists(), f"missing {name}"
    assert (ep / "images").is_dir()
    n_images = len(list((ep / "images").glob("*.png")))
    n_frames = sum(1 for ln in (ep / "frames.jsonl").read_text().splitlines() if ln.strip())
    assert n_images == n_frames > 0


def test_batch_manifest_and_seed_logged(tmp_path):
    rc = _run(tmp_path, num=3, seed=10, behavior="sharp_turn")
    assert rc == 0
    manifest = json.loads((tmp_path / "batch_manifest.json").read_text())
    assert manifest["num_episodes"] == 3
    assert manifest["passed"] == 3
    seeds = sorted(e["seed"] for e in manifest["episodes"])
    assert seeds == [10, 11, 12]
    # each episode logs its seed in config.yaml + metadata.json
    for ep in manifest["episodes"]:
        d = tmp_path / ep["episode_dir"].split("/")[-1]
        assert str(ep["seed"]) in (d / "config.yaml").read_text()
        assert json.loads((d / "metadata.json").read_text())["random_seed"] == ep["seed"]


def test_metrics_valid_and_instruction_in_metadata(tmp_path):
    _run(tmp_path, num=1, behavior="detour_feint", difficulty="hard", seed=1)
    ep = next(p for p in tmp_path.iterdir() if p.is_dir())
    metrics = json.loads((ep / "metrics.json").read_text())
    assert isinstance(metrics["tracking_success"], bool)
    assert 0.0 <= metrics["target_visibility_ratio"] <= 1.0
    assert metrics["total_frames"] > 0
    meta = json.loads((ep / "metadata.json").read_text())
    instr = json.loads((ep / "instruction.json").read_text())
    assert meta["instruction"] == instr["instruction"]
    assert meta["instruction"]  # non-empty


def test_all_required_files_and_sanity_pass(tmp_path):
    rc = _run(tmp_path, num=1, behavior="occlusion_seeking", difficulty="easy")
    assert rc == 0
    manifest = json.loads((tmp_path / "batch_manifest.json").read_text())
    assert all(e["passed"] for e in manifest["episodes"])
    assert all(not e["issues"] for e in manifest["episodes"])

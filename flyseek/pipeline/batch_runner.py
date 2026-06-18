# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Batch episode generation for FlySeek-Bench.

Runs N episodes through :func:`flyseek.pipeline.single_episode.run_episode`,
distributing a difficulty mix and writing each episode into the canonical
``output/trajectories/<env>/<difficulty>/<traj_id>/`` layout, plus a manifest.

This requires a running AirVLN/AirSim instance (the per-episode loop teleports
and captures). It is intentionally single-process; multi-worker concurrency
(SKILL §7.1) can wrap this later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from flyseek.pipeline.single_episode import (
    DEFAULT_TRAJ_ROOT,
    canonical_episode_dir,
    run_episode,
)

DEFAULT_DIFFICULTY_MIX = {"easy": 1, "medium": 1, "hard": 1}


@dataclass
class BatchConfig:
    episodes: int = 3
    env: str = "env_airsim_16"
    difficulty_mix: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_DIFFICULTY_MIX)
    )
    seed: int = 42
    traj_root: Path = DEFAULT_TRAJ_ROOT
    base_overrides: dict[str, Any] = field(default_factory=dict)


def _difficulty_schedule(mix: dict[str, int], n: int) -> list[str]:
    """Round-robin a weighted difficulty mix into a length-n schedule."""
    expanded: list[str] = []
    for diff, weight in mix.items():
        expanded.extend([diff] * max(0, int(weight)))
    if not expanded:
        expanded = ["medium"]
    return [expanded[i % len(expanded)] for i in range(n)]


def _episode_overrides(env: str, difficulty: str, seed: int,
                       base: dict[str, Any]) -> dict[str, Any]:
    """Map a difficulty label to demo args (tracking-difficulty preset + scenario)."""
    overrides = dict(base)
    overrides.setdefault("env", env)
    overrides.setdefault("seed", seed)
    overrides.setdefault("auto_from_scout", True)
    if difficulty == "easy":
        overrides.update(scenario="hide_seek", tracking_difficulty="easy")
    elif difficulty == "medium":
        overrides.update(scenario="hide_seek", tracking_difficulty="medium")
    else:  # hard == hide_seek de-facto hard tier
        overrides.update(scenario="hide_seek", tracking_difficulty="hard")
    return overrides


def run_batch(cfg: BatchConfig, *, batch_id: str | None = None) -> dict:
    """Generate ``cfg.episodes`` episodes and return a manifest dict."""
    batch_id = batch_id or datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    schedule = _difficulty_schedule(cfg.difficulty_mix, cfg.episodes)

    results: list[dict] = []
    for i, difficulty in enumerate(schedule):
        traj_id = f"{batch_id}_{i:05d}"
        overrides = _episode_overrides(
            cfg.env, difficulty, cfg.seed + i, cfg.base_overrides
        )
        ep_dir = canonical_episode_dir(
            env=cfg.env, difficulty=difficulty, traj_id=traj_id,
            traj_root=cfg.traj_root,
        )
        try:
            report = run_episode(overrides, traj_id=traj_id, traj_root=cfg.traj_root)
            results.append({
                "traj_id": traj_id,
                "difficulty": difficulty,
                "episode_dir": str(ep_dir),
                "success": bool(report.success),
                "frames": int(report.frames_captured),
                "errors": list(report.errors),
            })
        except Exception as e:  # keep the batch going on a single failure
            results.append({
                "traj_id": traj_id,
                "difficulty": difficulty,
                "episode_dir": str(ep_dir),
                "success": False,
                "errors": [f"{type(e).__name__}: {e}"],
            })

    manifest = {
        "batch_id": batch_id,
        "env": cfg.env,
        "episodes": cfg.episodes,
        "difficulty_mix": cfg.difficulty_mix,
        "traj_root": str(cfg.traj_root),
        "results": results,
    }
    manifest_path = cfg.traj_root / cfg.env / f"batch_{batch_id}_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest


__all__ = ["BatchConfig", "run_batch", "DEFAULT_DIFFICULTY_MIX"]

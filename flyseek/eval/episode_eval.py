# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Load episodes from disk and score them with the tracking metrics."""

from __future__ import annotations

import json
from pathlib import Path

from flyseek.eval.metrics import (
    EpisodeMetrics,
    aggregate_metrics,
    compute_episode_metrics,
)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def find_episode_dirs(root: Path) -> list[Path]:
    """All directories under ``root`` that contain a ``flyseek_meta.jsonl``."""
    root = Path(root)
    if (root / "flyseek_meta.jsonl").exists():
        return [root]
    return sorted(
        p.parent for p in root.rglob("flyseek_meta.jsonl")
    )


def evaluate_episode(episode_dir: Path) -> EpisodeMetrics:
    meta = _read_jsonl(Path(episode_dir) / "flyseek_meta.jsonl")
    return compute_episode_metrics(meta)


def evaluate_batch(root: Path) -> dict:
    """Evaluate every episode under ``root``; return a SKILL §7.3-style report."""
    root = Path(root)
    episodes = find_episode_dirs(root)
    per_episode: list[EpisodeMetrics] = []
    details: list[dict] = []
    for ep in episodes:
        m = evaluate_episode(ep)
        per_episode.append(m)
        details.append({"episode_dir": str(ep), **m.as_dict()})

    return {
        "root": str(root),
        "summary": aggregate_metrics(per_episode),
        "episodes": details,
    }


__all__ = [
    "find_episode_dirs",
    "evaluate_episode",
    "evaluate_batch",
]

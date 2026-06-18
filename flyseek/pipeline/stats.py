# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Dataset-level quality report (SKILL §7.3).

Combines tracking metrics (``flyseek.eval``) with instruction statistics across
a trajectories root and writes ``output/stats/<batch>_report.json``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from flyseek.eval.episode_eval import evaluate_batch, find_episode_dirs


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def instruction_stats(episode_dirs: list[Path]) -> dict:
    """Aggregate per-episode ``instruction.json`` files."""
    total = 0
    word_counts: list[int] = []
    blacklist_rejected = 0
    by_tier: Counter[str] = Counter()
    episodes_with_instr = 0

    for ep in episode_dirs:
        rec = _read_json(ep / "instruction.json")
        if not rec:
            continue
        episodes_with_instr += 1
        instrs = rec.get("instructions", [])
        total += len(instrs := instrs)
        for s in instrs:
            word_counts.append(len(str(s).split()))
        by_tier[rec.get("tier", "?")] += len(instrs)
        for r in rec.get("rejected", []):
            if str(r.get("reason", "")).startswith("blacklist"):
                blacklist_rejected += 1

    avg_len = round(sum(word_counts) / len(word_counts), 2) if word_counts else 0.0
    avg_per = round(total / episodes_with_instr, 2) if episodes_with_instr else 0.0
    return {
        "episodes_with_instructions": episodes_with_instr,
        "total_instructions": total,
        "avg_per_episode": avg_per,
        "avg_length_words": avg_len,
        "blacklist_rejected": blacklist_rejected,
        "by_tier": dict(by_tier),
    }


def difficulty_distribution(episode_dirs: list[Path]) -> dict:
    counts: Counter[str] = Counter()
    for ep in episode_dirs:
        summary = _read_json(ep / "summary.json")
        if summary and summary.get("difficulty"):
            counts[str(summary["difficulty"])] += 1
        else:
            # Infer from canonical layout: .../<env>/<difficulty>/<traj_id>/
            counts[ep.parent.name] += 1
    return dict(counts)


def build_report(root: Path) -> dict:
    root = Path(root)
    episode_dirs = find_episode_dirs(root)
    eval_report = evaluate_batch(root)
    return {
        "root": str(root),
        "total_trajectories": len(episode_dirs),
        "by_difficulty": difficulty_distribution(episode_dirs),
        "quality": eval_report.get("summary", {}),
        "instructions": instruction_stats(episode_dirs),
    }


__all__ = ["build_report", "instruction_stats", "difficulty_distribution"]

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""End-to-end orchestration for FlySeek-Bench data generation.

Public API:
  - ``run_episode`` / ``canonical_episode_dir`` (single episode)
  - ``BatchConfig`` / ``run_batch``             (batch generation)
  - ``build_report``                            (dataset stats)

Note: ``single_episode`` and ``batch_runner`` import the demo loop, which pulls
in ``airsim``; they are imported lazily so the offline submodules (stats/eval)
remain usable without a simulator installed.
"""

__all__ = [
    "run_episode",
    "canonical_episode_dir",
    "BatchConfig",
    "run_batch",
    "build_report",
]


def __getattr__(name: str):  # lazy re-exports (PEP 562)
    if name in ("run_episode", "canonical_episode_dir"):
        from flyseek.pipeline import single_episode
        return getattr(single_episode, name)
    if name in ("BatchConfig", "run_batch"):
        from flyseek.pipeline import batch_runner
        return getattr(batch_runner, name)
    if name == "build_report":
        from flyseek.pipeline import stats
        return getattr(stats, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""FlySeek tracking-evaluation metrics (paper §3 evaluation).

Computes Track-AUC, Lost-Rate, Redetection-Time, Collision-Rate and FOV-keep
from generated episodes' ``flyseek_meta.jsonl``.
"""

from flyseek.eval.metrics import (
    EpisodeMetrics,
    aggregate_metrics,
    compute_episode_metrics,
)

__all__ = [
    "EpisodeMetrics",
    "compute_episode_metrics",
    "aggregate_metrics",
]

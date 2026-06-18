# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Tracking metrics over a generated episode (SKILL §8.2).

All metrics derive from the per-frame visibility / collision signals already
recorded in ``flyseek_meta.jsonl`` (and the demo summary), so evaluation is
fully offline and unit-testable on synthetic series.

  - Track-AUC          : mean(target_in_fov) over the episode  -> [0, 1]
  - Lost-Rate          : fraction of frames with the target fully lost
                         (not visible). lost_frames / total_frames
  - Redetection-Time   : mean seconds from the start of a lost run to the next
                         visible frame (re-lock latency)
  - Collision-Rate     : collisions / frames (from collision flags if present)
  - FOV-keep-rate      : alias of Track-AUC for parity with OpenFly reporting
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class EpisodeMetrics:
    frames: int
    duration_s: float
    track_auc: float
    lost_rate: float
    redetection_time_s: float
    redetection_events: int
    collision_rate: float
    fov_keep_rate: float

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


def compute_metrics_from_series(
    visible: np.ndarray,
    *,
    dt: float = 0.05,
    collisions: np.ndarray | None = None,
) -> EpisodeMetrics:
    """Compute metrics from a boolean ``visible`` series (length N).

    ``collisions`` is an optional boolean series of per-frame collision events.
    """
    visible = np.asarray(visible, dtype=bool).reshape(-1)
    n = int(visible.size)
    if n == 0:
        return EpisodeMetrics(0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)

    track_auc = float(np.mean(visible))
    lost_rate = float(np.mean(~visible))

    # Re-detection: for each maximal run of lost frames that is followed by a
    # visible frame, the latency is (run_length + 1) * dt (lost duration until
    # the first visible frame). Trailing lost runs that never recover are not
    # counted (no re-detection happened).
    redetect_times: list[float] = []
    run_len = 0
    for i in range(n):
        if not visible[i]:
            run_len += 1
        else:
            if run_len > 0:
                redetect_times.append((run_len + 1) * dt)
            run_len = 0
    redetection_events = len(redetect_times)
    redetection_time_s = float(np.mean(redetect_times)) if redetect_times else 0.0

    if collisions is not None and len(collisions) > 0:
        collision_rate = float(np.mean(np.asarray(collisions, dtype=bool)))
    else:
        collision_rate = 0.0

    return EpisodeMetrics(
        frames=n,
        duration_s=float(n * dt),
        track_auc=track_auc,
        lost_rate=lost_rate,
        redetection_time_s=redetection_time_s,
        redetection_events=redetection_events,
        collision_rate=collision_rate,
        fov_keep_rate=track_auc,
    )


def compute_episode_metrics(meta_records: list[dict]) -> EpisodeMetrics:
    """Compute metrics from parsed ``flyseek_meta.jsonl`` records."""
    if not meta_records:
        return EpisodeMetrics(0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)

    visible = np.asarray(
        [bool(r.get("target_visible", False)) for r in meta_records], dtype=bool
    )
    ts = [float(r.get("timestamp", 0.0)) for r in meta_records]
    dt = (ts[-1] - ts[0]) / max(len(ts) - 1, 1) if len(ts) > 1 else 0.05
    dt = dt if dt > 1e-6 else 0.05

    collisions = None
    coll_flags = [
        bool(r.get("agent_decision", {}).get("collision", False)) for r in meta_records
    ]
    if any(coll_flags):
        collisions = np.asarray(coll_flags, dtype=bool)

    return compute_metrics_from_series(visible, dt=dt, collisions=collisions)


def aggregate_metrics(per_episode: list[EpisodeMetrics]) -> dict:
    """Aggregate per-episode metrics into mean/min/max summary (SKILL §7.3)."""
    if not per_episode:
        return {"episodes": 0}

    def stat(values: list[float]) -> dict:
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": round(float(arr.mean()), 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "p10": round(float(np.percentile(arr, 10)), 4),
        }

    return {
        "episodes": len(per_episode),
        "track_auc": stat([m.track_auc for m in per_episode]),
        "lost_rate": stat([m.lost_rate for m in per_episode]),
        "redetection_time_s": stat([m.redetection_time_s for m in per_episode]),
        "collision_rate": stat([m.collision_rate for m in per_episode]),
        "fov_keep_rate": stat([m.fov_keep_rate for m in per_episode]),
        "frames": stat([float(m.frames) for m in per_episode]),
    }


__all__ = [
    "EpisodeMetrics",
    "compute_metrics_from_series",
    "compute_episode_metrics",
    "aggregate_metrics",
]

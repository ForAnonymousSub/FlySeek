# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Episode-level FlySeek-Bench evaluation metrics (paper §3).

Computes, from an episode's ``metadata.json`` + ``frames.jsonl`` (the
``FrameMetadata`` schema):

  1. tracking_success         : visibility_ratio >= threshold(difficulty) AND no collision
  2. target_visibility_ratio  : visible_frames / total_frames
  3. line_of_sight_continuity : longest continuous visible segment / total_frames
                                 (avg_visible_segment_frames also reported)
  4. target_lost_duration     : invisible frames (+ invisible time ratio / seconds)
  5. re_acquisition_time      : avg frames to recover visibility after a
                                 visible -> invisible transition (+ seconds)
  6. collision_flag / collision_rate : any collision / collision_frames / total
  7. path_length              : summed UAV inter-frame displacement (m)
  8. path_efficiency          : net_displacement / path_length  (<=1; None if path_length==0)

Success thresholds (configurable via :class:`MetricsConfig`):
    easy = 0.70, medium = 0.70, hard = 0.60  (unknown difficulty -> default 0.70)

CLI:  python -m flyseek_bench.metrics --episode_dir path/to/episode
 (equivalently: python -m flyseek.bench.metrics --episode_dir ...)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class MetricsConfig:
    success_thresholds: dict[str, float] = field(
        default_factory=lambda: {"easy": 0.70, "medium": 0.70, "hard": 0.60}
    )
    default_threshold: float = 0.70

    def threshold_for(self, difficulty: str) -> float:
        return float(self.success_thresholds.get(str(difficulty), self.default_threshold))


# --------------------------------------------------------------------------- #
# IO helpers                                                                  #
# --------------------------------------------------------------------------- #
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


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _pose_xyz(pose: Any) -> np.ndarray | None:
    if pose is None:
        return None
    if isinstance(pose, dict):
        if "pos" in pose:
            return np.asarray(pose["pos"], dtype=np.float64).reshape(3)
        if {"x", "y", "z"} <= set(pose):
            return np.array([pose["x"], pose["y"], pose["z"]], dtype=np.float64)
        return None
    arr = np.asarray(pose, dtype=np.float64).reshape(-1)
    return arr[:3] if arr.size >= 3 else None


# --------------------------------------------------------------------------- #
# Core computation                                                            #
# --------------------------------------------------------------------------- #
def _visible_runs(visible: list[bool]) -> list[int]:
    """Lengths of maximal runs of consecutive True (visible) frames."""
    runs: list[int] = []
    cur = 0
    for v in visible:
        if v:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def _reacquisition_gaps(visible: list[bool]) -> list[int]:
    """Lengths of invisible runs that follow a visible frame AND recover.

    A gap is counted only when the target was visible before going invisible and
    becomes visible again (a genuine re-acquisition). Trailing never-recovered
    losses are excluded.
    """
    gaps: list[int] = []
    n = len(visible)
    i = 0
    seen_visible = False
    while i < n:
        if visible[i]:
            seen_visible = True
            i += 1
            continue
        # start of an invisible run
        j = i
        while j < n and not visible[j]:
            j += 1
        if seen_visible and j < n:  # recovered
            gaps.append(j - i)
        i = j
    return gaps


def compute_metrics(
    frames: list[dict],
    *,
    difficulty: str = "medium",
    config: MetricsConfig | None = None,
    dt_hint: float | None = None,
) -> dict[str, Any]:
    """Compute the 8 episode-level metrics from a list of frame records."""
    config = config or MetricsConfig()
    total = len(frames)
    threshold = config.threshold_for(difficulty)

    if total == 0:
        return {
            "total_frames": 0,
            "tracking_success": False,
            "target_visibility_ratio": 0.0,
            "line_of_sight_continuity": 0.0,
            "avg_visible_segment_frames": 0.0,
            "target_lost_frames": 0,
            "target_lost_ratio": 0.0,
            "re_acquisition_time_frames": 0.0,
            "re_acquisition_events": 0,
            "collision_flag": False,
            "collision_rate": 0.0,
            "path_length_m": 0.0,
            "path_efficiency": None,
            "success_threshold": threshold,
            "difficulty_level": difficulty,
        }

    visible = [bool(f.get("target_visible", False)) for f in frames]
    collisions = [bool(f.get("collision", False)) for f in frames]
    visible_frames = int(sum(visible))

    # timestamps -> mean dt (for seconds conversions).
    ts = [f.get("timestamp") for f in frames if f.get("timestamp") is not None]
    if dt_hint is not None and dt_hint > 0:
        dt = float(dt_hint)
    elif len(ts) >= 2:
        dt = (float(ts[-1]) - float(ts[0])) / max(len(ts) - 1, 1)
        dt = dt if dt > 1e-6 else 0.0
    else:
        dt = 0.0

    # (2) visibility ratio
    visibility_ratio = visible_frames / total

    # (3) line-of-sight continuity
    runs = _visible_runs(visible)
    longest = max(runs) if runs else 0
    los_continuity = longest / total
    avg_segment = float(np.mean(runs)) if runs else 0.0

    # (4) target lost duration
    lost_frames = total - visible_frames
    lost_ratio = lost_frames / total

    # (5) re-acquisition time
    gaps = _reacquisition_gaps(visible)
    reacq_frames = float(np.mean(gaps)) if gaps else 0.0

    # (6) collision
    collision_flag = any(collisions)
    collision_rate = sum(collisions) / total

    # (7) path length (UAV)
    uav_pts = [p for p in (_pose_xyz(f.get("uav_pose")) for f in frames) if p is not None]
    path_length = 0.0
    for a, b in zip(uav_pts[:-1], uav_pts[1:]):
        path_length += float(np.linalg.norm(b - a))

    # (8) path efficiency = net displacement / path length
    path_efficiency: float | None = None
    if len(uav_pts) >= 2 and path_length > 1e-6:
        net = float(np.linalg.norm(uav_pts[-1] - uav_pts[0]))
        path_efficiency = round(min(1.0, net / path_length), 4)

    # (1) tracking success
    tracking_success = bool(visibility_ratio >= threshold and not collision_flag)

    result = {
        "total_frames": total,
        "tracking_success": tracking_success,
        "target_visibility_ratio": round(visibility_ratio, 4),
        "line_of_sight_continuity": round(los_continuity, 4),
        "avg_visible_segment_frames": round(avg_segment, 3),
        "target_lost_frames": lost_frames,
        "target_lost_ratio": round(lost_ratio, 4),
        "re_acquisition_time_frames": round(reacq_frames, 3),
        "re_acquisition_events": len(gaps),
        "collision_flag": collision_flag,
        "collision_rate": round(collision_rate, 4),
        "path_length_m": round(path_length, 3),
        "path_efficiency": path_efficiency,
        "success_threshold": threshold,
        "difficulty_level": difficulty,
    }
    if dt > 0:
        result["target_lost_duration_s"] = round(lost_frames * dt, 3)
        result["re_acquisition_time_s"] = round(reacq_frames * dt, 3)
    return result


def evaluate_episode_dir(
    episode_dir: str | Path,
    *,
    config: MetricsConfig | None = None,
    difficulty: str | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Read ``metadata.json`` + ``frames.jsonl`` from a dir and compute metrics."""
    episode_dir = Path(episode_dir)
    meta = _read_json(episode_dir / "metadata.json")
    frames = _read_jsonl(episode_dir / "frames.jsonl")
    if difficulty is None:
        difficulty = str(meta.get("difficulty_level", "medium"))

    metrics = compute_metrics(frames, difficulty=difficulty, config=config)
    metrics["episode_id"] = meta.get("episode_id", episode_dir.name)
    metrics["episode_dir"] = str(episode_dir)
    if write:
        (episode_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return metrics


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute FlySeek-Bench episode metrics from metadata.json + frames.jsonl."
    )
    parser.add_argument("--episode_dir", "--episode-dir", dest="episode_dir",
                        type=Path, required=True, help="Episode directory.")
    parser.add_argument("--difficulty", default=None,
                        help="Override difficulty (else read from metadata.json).")
    parser.add_argument("--easy-threshold", type=float, default=0.70)
    parser.add_argument("--medium-threshold", type=float, default=0.70)
    parser.add_argument("--hard-threshold", type=float, default=0.60)
    parser.add_argument("--default-threshold", type=float, default=0.70)
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path (default: <episode_dir>/metrics.json).")
    args = parser.parse_args(argv)

    config = MetricsConfig(
        success_thresholds={
            "easy": args.easy_threshold,
            "medium": args.medium_threshold,
            "hard": args.hard_threshold,
        },
        default_threshold=args.default_threshold,
    )
    metrics = evaluate_episode_dir(
        args.episode_dir, config=config, difficulty=args.difficulty, write=(args.out is None),
    )
    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                                  encoding="utf-8")

    print(f"[ok] episode {metrics.get('episode_id')} "
          f"({metrics['total_frames']} frames, difficulty={metrics['difficulty_level']})")
    print(f"  tracking_success        : {metrics['tracking_success']} "
          f"(thr={metrics['success_threshold']})")
    print(f"  target_visibility_ratio : {metrics['target_visibility_ratio']}")
    print(f"  line_of_sight_continuity: {metrics['line_of_sight_continuity']}")
    print(f"  target_lost_ratio       : {metrics['target_lost_ratio']}")
    print(f"  re_acquisition (frames) : {metrics['re_acquisition_time_frames']}")
    print(f"  collision_rate          : {metrics['collision_rate']}")
    print(f"  path_length_m           : {metrics['path_length_m']}")
    print(f"  path_efficiency         : {metrics['path_efficiency']}")
    out_path = args.out or (args.episode_dir / "metrics.json")
    print(f"  metrics -> {out_path}")
    return 0


__all__ = [
    "MetricsConfig",
    "compute_metrics",
    "evaluate_episode_dir",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())

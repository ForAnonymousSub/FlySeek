# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Episode validator + paper-consistency report for FlySeek-Bench.

Validates a generated episode against the standardized layout and schema, and
aggregates a short paper-consistency report over a batch.

Per-episode checks:
  1. required files exist
  2. metadata.json has the required episode fields
  3. frames.jsonl has the required per-frame fields
  4. metrics.target_visibility_ratio is in [0, 1]
  5. every recorded image path exists
  6. trajectories.json contains UAV and target trajectories
  7. instruction.json contains a valid (non-empty, blacklist-clean) instruction
  8. metrics.json contains the required metrics

CLI:
    python -m flyseek_bench.validate_episode --episode_dir path/to/episode
    python -m flyseek_bench.validate_episode --batch_dir   path/to/batch
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from flyseek.bench import schema
from flyseek.instruction import blacklist

REQUIRED_FILES = (
    "metadata.json", "frames.jsonl", "trajectories.json",
    "instruction.json", "metrics.json",
)
RECOMMENDED_FILES = ("visibility.json", "config.yaml")

REQUIRED_METRIC_KEYS = (
    "tracking_success", "target_visibility_ratio", "line_of_sight_continuity",
    "target_lost_ratio", "re_acquisition_time_frames", "collision_rate",
    "path_length_m",
)


# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _resolve_image(episode_dir: Path, image_path: str) -> Path:
    p = Path(image_path)
    return p if p.is_absolute() else (episode_dir / p)


def validate_episode(
    episode_dir: str | Path,
    *,
    check_images: bool = True,
    max_image_checks: int | None = None,
) -> dict[str, Any]:
    """Validate one episode directory; return a report dict (never raises)."""
    episode_dir = Path(episode_dir)
    issues: list[str] = []
    warnings: list[str] = []

    report: dict[str, Any] = {
        "episode_dir": str(episode_dir),
        "passed": False,
        "issues": issues,
        "warnings": warnings,
        "difficulty": None,
        "behavior": None,
        "visibility_ratio": None,
        "lost_ratio": None,
        "lost_duration_s": None,
        "reacq_frames": None,
        "reacq_s": None,
        "collision": None,
        "success": None,
    }

    if not episode_dir.is_dir():
        issues.append("episode_dir does not exist")
        return report

    # (1) required files
    for name in REQUIRED_FILES:
        if not (episode_dir / name).exists():
            issues.append(f"missing required file: {name}")
    for name in RECOMMENDED_FILES:
        if not (episode_dir / name).exists():
            warnings.append(f"missing recommended file: {name}")

    # (2) metadata fields
    meta: dict[str, Any] = {}
    if (episode_dir / "metadata.json").exists():
        try:
            meta = _read_json(episode_dir / "metadata.json")
            schema.validate_episode(meta)
            report["difficulty"] = meta.get("difficulty_level")
            report["behavior"] = meta.get("target_behavior_type")
        except schema.SchemaValidationError as e:
            issues.append(f"metadata invalid: {e}")
        except Exception as e:
            issues.append(f"metadata unreadable: {e}")

    # (3) frames fields + (5) image paths
    frames: list[dict] = []
    if (episode_dir / "frames.jsonl").exists():
        try:
            frames = _read_jsonl(episode_dir / "frames.jsonl")
        except Exception as e:
            issues.append(f"frames.jsonl unreadable: {e}")
        if not frames:
            issues.append("frames.jsonl is empty")
        frame_field_errors = 0
        for i, fr in enumerate(frames):
            try:
                schema.validate_frame(fr)
            except schema.SchemaValidationError as e:
                frame_field_errors += 1
                if frame_field_errors <= 3:
                    issues.append(f"frame {i} invalid: {e}")
        if frame_field_errors > 3:
            issues.append(f"... {frame_field_errors} frames failed field validation")

        if check_images and frames:
            missing = 0
            checked = 0
            for fr in frames:
                if max_image_checks is not None and checked >= max_image_checks:
                    break
                checked += 1
                ip = fr.get("image_path")
                if not ip or not _resolve_image(episode_dir, ip).exists():
                    missing += 1
            if missing:
                issues.append(f"{missing}/{checked} recorded image paths do not exist")

    # (6) trajectories
    if (episode_dir / "trajectories.json").exists():
        try:
            traj = _read_json(episode_dir / "trajectories.json")
            if not traj.get("target_trajectory"):
                issues.append("trajectories.json: missing/empty target_trajectory")
            if not traj.get("uav_trajectory"):
                issues.append("trajectories.json: missing/empty uav_trajectory")
            if not traj.get("expert_viewpoints"):
                warnings.append("trajectories.json: no expert_viewpoints")
        except Exception as e:
            issues.append(f"trajectories.json unreadable: {e}")

    # (7) instruction
    if (episode_dir / "instruction.json").exists():
        try:
            instr = _read_json(episode_dir / "instruction.json")
            text = str(instr.get("instruction", "")).strip()
            if not text:
                issues.append("instruction.json: empty instruction")
            elif not blacklist.is_clean(text):
                warnings.append(
                    f"instruction violates a blacklist rule: "
                    f"{blacklist.violations(text)[:1]}"
                )
        except Exception as e:
            issues.append(f"instruction.json unreadable: {e}")

    # (4) + (8) metrics
    if (episode_dir / "metrics.json").exists():
        try:
            m = _read_json(episode_dir / "metrics.json")
            for key in REQUIRED_METRIC_KEYS:
                if key not in m:
                    issues.append(f"metrics.json missing key: {key}")
            vr = m.get("target_visibility_ratio")
            if not (isinstance(vr, (int, float)) and 0.0 <= float(vr) <= 1.0):
                issues.append(f"target_visibility_ratio out of [0,1]: {vr}")
            # collect report fields
            report["visibility_ratio"] = vr
            report["lost_ratio"] = m.get("target_lost_ratio")
            report["lost_duration_s"] = m.get("target_lost_duration_s")
            report["reacq_frames"] = m.get("re_acquisition_time_frames")
            report["reacq_s"] = m.get("re_acquisition_time_s")
            report["collision"] = bool(m.get("collision_flag",
                                              (m.get("collision_rate", 0) or 0) > 0))
            report["success"] = m.get("tracking_success")
            if report["difficulty"] is None:
                report["difficulty"] = m.get("difficulty_level")
        except Exception as e:
            issues.append(f"metrics.json unreadable: {e}")

    report["passed"] = len(issues) == 0
    return report


# --------------------------------------------------------------------------- #
# Batch discovery + paper-consistency report                                  #
# --------------------------------------------------------------------------- #
def find_episode_dirs(batch_dir: str | Path) -> list[Path]:
    batch_dir = Path(batch_dir)
    if (batch_dir / "metadata.json").exists():
        return [batch_dir]
    return sorted({
        p.parent for p in batch_dir.rglob("metadata.json")
        if (p.parent / "frames.jsonl").exists()
    })


def build_consistency_report(episode_reports: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(episode_reports)
    by_difficulty = Counter(r.get("difficulty") or "unknown" for r in episode_reports)
    by_behavior = Counter(r.get("behavior") or "unknown" for r in episode_reports)

    def _mean(key: str) -> float | None:
        vals = [float(r[key]) for r in episode_reports
                if isinstance(r.get(key), (int, float))]
        return round(float(np.mean(vals)), 4) if vals else None

    successes = [bool(r["success"]) for r in episode_reports
                 if r.get("success") is not None]
    collision_count = sum(1 for r in episode_reports if r.get("collision"))

    return {
        "num_episodes": n,
        "num_valid": sum(1 for r in episode_reports if r.get("passed")),
        "by_difficulty": dict(by_difficulty),
        "by_behavior": dict(by_behavior),
        "mean_visibility_ratio": _mean("visibility_ratio"),
        "mean_lost_ratio": _mean("lost_ratio"),
        "mean_lost_duration_s": _mean("lost_duration_s"),
        "mean_reacquisition_frames": _mean("reacq_frames"),
        "mean_reacquisition_s": _mean("reacq_s"),
        "collision_count": collision_count,
        "success_rate": (round(float(np.mean(successes)), 4) if successes else None),
    }


def _print_consistency(report: dict[str, Any]) -> None:
    print("\n" + "=" * 56)
    print("FlySeek-Bench Paper-Consistency Report")
    print("=" * 56)
    print(f"episodes             : {report['num_episodes']} "
          f"({report['num_valid']} passed validation)")
    print(f"by difficulty        : {report['by_difficulty']}")
    print(f"by behavior          : {report['by_behavior']}")
    print(f"mean visibility ratio: {report['mean_visibility_ratio']}")
    print(f"mean lost ratio      : {report['mean_lost_ratio']}")
    if report["mean_lost_duration_s"] is not None:
        print(f"mean lost duration(s): {report['mean_lost_duration_s']}")
    print(f"mean re-acq (frames) : {report['mean_reacquisition_frames']}")
    if report["mean_reacquisition_s"] is not None:
        print(f"mean re-acq (s)      : {report['mean_reacquisition_s']}")
    print(f"collision count      : {report['collision_count']}")
    print(f"success rate         : {report['success_rate']}")
    print("=" * 56)


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate FlySeek-Bench episodes and print a consistency report."
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--episode_dir", "--episode-dir", dest="episode_dir", type=Path)
    g.add_argument("--batch_dir", "--batch-dir", dest="batch_dir", type=Path)
    parser.add_argument("--no-image-check", action="store_true",
                        help="Skip the image-path existence check.")
    parser.add_argument("--report-out", type=Path, default=None,
                        help="Write the consistency report JSON here.")
    args = parser.parse_args(argv)

    check_images = not args.no_image_check

    if args.episode_dir is not None:
        rep = validate_episode(args.episode_dir, check_images=check_images)
        status = "PASS" if rep["passed"] else "FAIL"
        print(f"[{status}] {args.episode_dir}")
        for w in rep["warnings"]:
            print(f"  [warn] {w}")
        for issue in rep["issues"]:
            print(f"  [issue] {issue}")
        episode_reports = [rep]
    else:
        episode_dirs = find_episode_dirs(args.batch_dir)
        if not episode_dirs:
            print(f"[ERR] no episodes (metadata.json + frames.jsonl) under {args.batch_dir}")
            return 1
        episode_reports = []
        for ep in episode_dirs:
            rep = validate_episode(ep, check_images=check_images)
            episode_reports.append(rep)
            status = "PASS" if rep["passed"] else "FAIL"
            extra = "" if rep["passed"] else f"  issues={len(rep['issues'])}"
            print(f"[{status}] {ep.name}{extra}")
            if not rep["passed"]:
                for issue in rep["issues"][:5]:
                    print(f"    - {issue}")

    consistency = build_consistency_report(episode_reports)
    _print_consistency(consistency)

    if args.report_out is not None:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(
            json.dumps({"consistency": consistency,
                        "episodes": episode_reports}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"report -> {args.report_out}")

    all_passed = all(r["passed"] for r in episode_reports)
    return 0 if all_passed else 1


__all__ = [
    "validate_episode",
    "find_episode_dirs",
    "build_consistency_report",
    "REQUIRED_FILES",
    "REQUIRED_METRIC_KEYS",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Record alley-chase demos and build the FlySeek-vs-baseline comparison.

The target car drives into a narrow hutong between buildings. By default this
runs the *same* seeded scene twice through the proven teleport pipeline —

  1. **FlySeek** run  : adaptive predictive FSM tracker (active viewpoint
     adjustment, occlusion-risk-aware PEEK/PREDICT/REACQUIRE) → keeps line of
     sight, succeeds.
  2. **Reactive baseline** run: chase-current-pose follower (no occlusion
     handling) → flies blindly into the occluder, loses the target.

Both runs share identical scene/camera parameters (only the drone controller
differs), then ``viz_chase_compare`` renders the occlusion-risk map and the
two-panel paper comparison figure.

Requires AirSim running and annotated seg_map buildings for the environment.

Example (env_airsim_16)::

    python flyseek_extend/scripts/demo_alley_chase.py \\
      --env env_airsim_16 \\
      --auto-from-scout \\
      --seed 66 \\
      --duration 55

Single run only (no baseline / comparison)::

    python flyseek_extend/scripts/demo_alley_chase.py ... --no-compare
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "flyseek_extend" / "scripts" / "demo_adversary_chase.py"
VIZ = REPO / "flyseek_extend" / "scripts" / "viz_chase_compare.py"
DEFAULT_SEG = REPO / "scene_data" / "seg_map" / "env_airsim_16.jsonl"
DEFAULT_OUT_ROOT = REPO / "flyseek_extend" / "output" / "demo_alley_compare"


def _base_cmd(args: argparse.Namespace) -> list[str]:
    """Shared demo_adversary_chase command (scene + car identical per run)."""
    cmd = [
        sys.executable, str(DEMO),
        "--env", args.env,
        "--duration", str(args.duration),
        "--tick-hz", str(args.tick_hz),
        "--target-behavior", "alley_hutong",
        "--target-policy-difficulty", "hard",
        "--seg-building-jsonl", str(args.seg_building_jsonl),
        "--open-approach-m", str(args.open_approach_m),
        "--max-corridor-width-m", str(args.max_corridor_width_m),
        "--route-len-m", str(args.route_len_m),
        "--route-search-radius-m", "220",
        "--seed", str(args.seed),
        "--airsim-ip", args.airsim_ip,
        "--airsim-port", str(args.airsim_port),
    ]
    if args.auto_from_scout:
        cmd.append("--auto-from-scout")
    if args.target:
        cmd.extend(["--target", args.target])
    if args.target_regex:
        cmd.extend(["--target-regex", args.target_regex])
    if args.label:
        cmd.extend(["--label", args.label])
    if args.no_alley_near_entry:
        cmd.append("--no-alley-near-entry")
    if args.los_include_trees:
        cmd.append("--los-include-trees")
    else:
        cmd.append("--no-los-include-trees")
    return cmd


def _run_episode(
    args: argparse.Namespace,
    *,
    tracker_mode: str,
    episode_tag: str,
) -> int:
    """Run one demo_adversary_chase episode with the given drone tracker."""
    cmd = _base_cmd(args)
    cmd += [
        "--force-tracker",
        "--tracker-mode", tracker_mode,
        "--no-topdown",
        "--output", str(args.out_root),
        "--episode-tag", episode_tag,
    ]
    print(f"\n[alley] === episode '{episode_tag}' (tracker={tracker_mode}) ===")
    print("[alley] launching:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO))


def main() -> int:
    p = argparse.ArgumentParser(
        description="Alley chase: FlySeek vs reactive baseline + comparison figures.",
    )
    p.add_argument("--env", default="env_airsim_16")
    p.add_argument("--duration", type=float, default=55.0,
                   help="Seconds to record (≥50 recommended for alley dive).")
    p.add_argument("--tick-hz", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--auto-from-scout", action="store_true",
                   help="Pick the target from the scout JSON "
                        "(flyseek_extend/output/assets/scene_targets_latest.json). "
                        "If that file is missing, run scout once or reuse a "
                        "known car via --target / --target-regex instead.")
    p.add_argument("--target", default=None,
                   help="Reuse a specific target car by exact scene-object name "
                        "(no scout file needed), e.g. SM_ClassicCar02_Drivable6_4.")
    p.add_argument("--target-regex", default=None,
                   help="Reuse a target car by matching live scene objects with "
                        "a regex (no scout file needed), e.g. "
                        "'ClassicCar|Car0|Suv'.")
    p.add_argument("--label", default=None,
                   help="Natural-language label for the reused target "
                        "(e.g. 'a classic car').")
    p.add_argument("--seg-building-jsonl", type=Path, default=DEFAULT_SEG)
    p.add_argument("--open-approach-m", type=float, default=35.0,
                   help="Shorter open road before hutong entry (default 35m).")
    p.add_argument("--max-corridor-width-m", type=float, default=12.0)
    p.add_argument("--route-len-m", type=float, default=120.0)
    p.add_argument("--airsim-ip", default="127.0.0.1")
    p.add_argument("--airsim-port", type=int, default=41451)
    p.add_argument("--no-alley-near-entry", action="store_true",
                   help="Keep scout spawn instead of teleporting near hutong entry.")
    p.add_argument("--los-include-trees", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Count trees / foliage / poles (tall non-building PCD "
                        "columns) as line-of-sight occluders in both runs so the "
                        "reactive baseline visibly loses the target when it is "
                        "occluded by a tree (tagged 'los_blocked_occluder'). "
                        "Default on; pass --no-los-include-trees for "
                        "buildings-only occlusion.")

    # Comparison orchestration.
    p.add_argument("--compare", action=argparse.BooleanOptionalAction, default=True,
                   help="Run FlySeek + reactive baseline and build comparison "
                        "figures (default). --no-compare runs a single episode.")
    p.add_argument("--flyseek-tracker", default="adaptive",
                   choices=["adaptive", "inline", "legacy", "reactive"],
                   help="Tracker for the FlySeek run (default adaptive).")
    p.add_argument("--baseline-tracker", default="reactive",
                   choices=["adaptive", "inline", "legacy", "reactive"],
                   help="Tracker for the baseline run (default reactive).")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                   help="Root directory for the two episodes + comparison.")
    p.add_argument("--grid-step-m", type=float, default=2.0,
                   help="Occlusion-risk field grid resolution (m).")
    p.add_argument("--frustum-every", type=int, default=18)
    args = p.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    if not args.compare:
        return _run_episode(
            args, tracker_mode=args.flyseek_tracker,
            episode_tag=f"flyseek_seed{args.seed}",
        )

    fly_tag = f"flyseek_seed{args.seed}"
    base_tag = f"baseline_seed{args.seed}"

    rc = _run_episode(args, tracker_mode=args.flyseek_tracker, episode_tag=fly_tag)
    if rc != 0:
        print(f"[alley] FlySeek episode failed (rc={rc}); aborting comparison.")
        return rc
    rc = _run_episode(args, tracker_mode=args.baseline_tracker, episode_tag=base_tag)
    if rc != 0:
        print(f"[alley] baseline episode failed (rc={rc}); aborting comparison.")
        return rc

    viz_cmd = [
        sys.executable, str(VIZ),
        "--flyseek-dir", str(args.out_root / fly_tag),
        "--baseline-dir", str(args.out_root / base_tag),
        "--env", args.env,
        "--seg-building-jsonl", str(args.seg_building_jsonl),
        "--out-dir", str(args.out_root / f"compare_seed{args.seed}"),
        "--grid-step-m", str(args.grid_step_m),
        "--frustum-every", str(args.frustum_every),
    ]
    print("\n[alley] === building comparison figures ===")
    print("[alley] launching:", " ".join(viz_cmd))
    rc = subprocess.call(viz_cmd, cwd=str(REPO))
    if rc == 0:
        print(f"\n[alley] DONE. Figures in {args.out_root / f'compare_seed{args.seed}'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

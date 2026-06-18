#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Record a "car hides behind a building" chase + FlySeek-vs-baseline comparison.

The target car drives down an open road and then ducks **behind an annotated
building** (target behavior ``occlusion_seeking``). By default this runs the
same seeded scene twice through the proven teleport pipeline —

  1. **FlySeek** run  : adaptive predictive FSM tracker (PREDICT / PEEK /
     REACQUIRE) → anticipates the hide, repositions, keeps / regains the target.
  2. **Reactive baseline** run: memoryless ``reactive_lost`` follower → chases
     while the car is visible, then once the car is hidden behind the building
     it stalls near the last-seen spot and wanders aimlessly, never reacquiring.

Both runs share identical scene/camera parameters (only the drone controller
differs), then ``viz_chase_compare`` renders the occlusion-risk map and the
two-panel paper comparison figure.

Requires AirSim running and annotated seg_map buildings for the environment.

Example (env_airsim_16)::

    # terminal 1 — start the sim headless
    bash flyseek_extend/shell/start_airsim.sh env_airsim_16
    # terminal 2 — run the comparison
    python flyseek_extend/scripts/demo_building_chase.py \\
      --env env_airsim_16 \\
      --target-regex 'ClassicCar|Car0|Suv' \\
      --seed 66 --duration 75
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
DEFAULT_OUT_ROOT = REPO / "flyseek_extend" / "output" / "demo_building_compare"


def _base_cmd(args: argparse.Namespace) -> list[str]:
    """Shared demo_adversary_chase command (scene + car identical per run).

    Two target trajectories are supported:

      * ``alley_hutong`` (default) — the car dives into a narrow gap *between
        annotated buildings* and parks deep inside, so the trailing drone's
        line of sight is genuinely broken by the flanking building walls. This
        is the proven occluding route (the reactive baseline reliably loses the
        target here).
      * ``occlusion_seeking`` — open road then duck behind a single annotated
        building. NOTE: this planner can pick a hide goal that does not actually
        occlude the chase drone (the car then parks in the open and stays
        visible), so the baseline may *not* fail. Prefer ``alley_hutong`` for a
        guaranteed tracking-failure result.
    """
    cmd = [
        sys.executable, str(DEMO),
        "--env", args.env,
        "--duration", str(args.duration),
        "--tick-hz", str(args.tick_hz),
        "--target-behavior", args.target_behavior,
        "--target-policy-difficulty", args.difficulty,
        "--seg-building-jsonl", str(args.seg_building_jsonl),
        "--route-len-m", str(args.route_len_m),
        "--route-search-radius-m", str(args.route_search_radius_m),
        "--seed", str(args.seed),
        "--airsim-ip", args.airsim_ip,
        "--airsim-port", str(args.airsim_port),
    ]
    if args.target_behavior == "alley_hutong":
        cmd += [
            "--open-approach-m", str(args.open_approach_m),
            "--max-corridor-width-m", str(args.max_corridor_width_m),
        ]
    else:  # occlusion_seeking
        cmd += [
            "--open-road-frac", str(args.open_road_frac),
            "--route-max-attempts", str(args.route_max_attempts),
            "--min-building-height-m", str(args.min_building_height_m),
            "--min-building-footprint-cells", str(args.min_building_footprint_cells),
            "--hide-search-radius-m", str(args.hide_search_radius_m),
            "--min-building-occluded-frac", str(args.min_building_occluded_frac),
            "--building-probe-dist-m", str(args.building_probe_dist_m),
        ]
        if args.require_adjacent_building:
            cmd.append("--require-adjacent-building")
    if args.auto_from_scout:
        cmd.append("--auto-from-scout")
    if args.target:
        cmd.extend(["--target", args.target])
    if args.target_regex:
        cmd.extend(["--target-regex", args.target_regex])
    if args.label:
        cmd.extend(["--label", args.label])
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
    cmd = _base_cmd(args)
    cmd += [
        "--force-tracker",
        "--tracker-mode", tracker_mode,
        "--no-topdown",
        "--output", str(args.out_root),
        "--episode-tag", episode_tag,
    ]
    if tracker_mode == "reactive_lost":
        cmd += [
            "--lost-wander-radius-m", str(args.lost_wander_radius_m),
            "--lost-wander-scan-dps", str(args.lost_wander_scan_dps),
        ]
    print(f"\n[building] === episode '{episode_tag}' (tracker={tracker_mode}) ===")
    print("[building] launching:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO))


def main() -> int:
    p = argparse.ArgumentParser(
        description="Behind-building chase: FlySeek vs reactive_lost baseline.",
    )
    p.add_argument("--env", default="env_airsim_16")
    p.add_argument("--target-behavior",
                   choices=["alley_hutong", "occlusion_seeking"],
                   default="alley_hutong",
                   help="alley_hutong (default): car dives into a gap between "
                        "buildings and parks hidden — guaranteed occlusion / "
                        "baseline failure. occlusion_seeking: duck behind a "
                        "single building (planner can leave the car visible).")
    p.add_argument("--duration", type=float, default=75.0,
                   help="Seconds to record (≥55 for alley_hutong, ≥70 for "
                        "occlusion_seeking so the car reaches + dwells hidden).")
    p.add_argument("--tick-hz", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=66)
    # alley_hutong route knobs.
    p.add_argument("--open-approach-m", type=float, default=35.0,
                   help="alley_hutong: open road before the hutong entry.")
    p.add_argument("--max-corridor-width-m", type=float, default=12.0,
                   help="alley_hutong: max corridor width of the gap.")
    p.add_argument("--difficulty", default="hard",
                   choices=["easy", "medium", "hard"])
    p.add_argument("--auto-from-scout", action="store_true",
                   help="Pick the target from the scout JSON. If missing, reuse "
                        "a known car via --target / --target-regex instead.")
    p.add_argument("--target", default=None,
                   help="Reuse a specific target car by exact scene-object name.")
    p.add_argument("--target-regex", default=None,
                   help="Reuse a target car by matching live scene objects with "
                        "a regex, e.g. 'ClassicCar|Car0|Suv' (no scout needed).")
    p.add_argument("--label", default=None,
                   help="Natural-language label for the reused target.")
    p.add_argument("--seg-building-jsonl", type=Path, default=DEFAULT_SEG)

    # occlusion_seeking (behind-building) route knobs.
    p.add_argument("--route-len-m", type=float, default=120.0)
    p.add_argument("--open-road-frac", type=float, default=0.4,
                   help="Fraction of the route on the open road before hiding.")
    p.add_argument("--route-search-radius-m", type=float, default=240.0)
    p.add_argument("--route-max-attempts", type=int, default=20)
    p.add_argument("--min-building-height-m", type=float, default=20.0)
    p.add_argument("--min-building-footprint-cells", type=int, default=12)
    p.add_argument("--hide-search-radius-m", type=float, default=55.0)
    p.add_argument("--min-building-occluded-frac", type=float, default=0.65)
    p.add_argument("--building-probe-dist-m", type=float, default=9.0)
    p.add_argument("--require-adjacent-building",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Require the hide goal to sit against a building wall.")

    p.add_argument("--airsim-ip", default="127.0.0.1")
    p.add_argument("--airsim-port", type=int, default=41451)

    # Comparison orchestration.
    p.add_argument("--compare", action=argparse.BooleanOptionalAction, default=True,
                   help="Run FlySeek + baseline and build comparison figures.")
    p.add_argument("--flyseek-tracker", default="adaptive",
                   choices=["adaptive", "inline", "legacy", "reactive",
                            "reactive_lost"],
                   help="Tracker for the FlySeek run (default adaptive).")
    p.add_argument("--baseline-tracker", default="reactive_lost",
                   choices=["adaptive", "inline", "legacy", "reactive",
                            "reactive_lost"],
                   help="Tracker for the baseline run (default reactive_lost: "
                        "stalls + wanders near the last-seen spot once hidden).")
    p.add_argument("--los-include-trees", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Count trees / foliage as LoS occluders too (default on).")
    p.add_argument("--lost-wander-radius-m", type=float, default=6.0,
                   help="reactive_lost baseline: loiter radius (m) around the "
                        "last-seen spot once the target is lost.")
    p.add_argument("--lost-wander-scan-dps", type=float, default=35.0,
                   help="reactive_lost baseline: yaw sweep rate (deg/s) while "
                        "wandering.")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--grid-step-m", type=float, default=2.0)
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
        print(f"[building] FlySeek episode failed (rc={rc}); aborting.")
        return rc
    rc = _run_episode(args, tracker_mode=args.baseline_tracker, episode_tag=base_tag)
    if rc != 0:
        print(f"[building] baseline episode failed (rc={rc}); aborting.")
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
    print("\n[building] === building comparison figures ===")
    print("[building] launching:", " ".join(viz_cmd))
    rc = subprocess.call(viz_cmd, cwd=str(REPO))
    if rc == 0:
        print(f"\n[building] DONE. Figures in "
              f"{args.out_root / f'compare_seed{args.seed}'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

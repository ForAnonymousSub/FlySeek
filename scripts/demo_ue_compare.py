#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""FlySeek-vs-reactive-baseline comparison on a UE5 City Sample scene.

This is the ``env_ue_smallcity`` (UnrealCV backend) counterpart of
``demo_building_chase.py`` / ``demo_alley_chase.py``. It reuses the *same*
simulator-agnostic closed-loop trackers as the env_airsim_16 pipeline —

  * **FlySeek** run  : ``--tracker-mode adaptive`` (predictive FSM), and
  * **Reactive baseline** run: ``--tracker-mode reactive_lost`` (stalls + wanders
    near the last-seen spot once the target is occluded),

drives them through ``demo_unrealcv_chase`` (UnrealCV render), and renders the
shared occlusion-risk map + two-panel comparison figure with the same
``viz_chase_compare`` tool.

Only the render backend differs from the AirSim pipeline; the target route, the
trackers, the visibility/metrics and the figures are identical code.

Prereq — launch the UE City Sample (UnrealCV server on :9000) in another shell::

    bash envs/ue/env_ue_smallcity/CitySample.sh

Then::

    python flyseek_extend/scripts/demo_ue_compare.py \\
      --env env_ue_smallcity \\
      --target-behavior occlusion_seeking \\
      --seed 66 --duration 45

NOTE: UE City Sample scenes need one live tuning pass (which car is
controllable, camera framing, whether the chosen behaviour genuinely occludes).
Run once and adjust ``--target``, ``--follow-*`` and ``--target-behavior``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
UE_DEMO = REPO / "flyseek_extend" / "scripts" / "demo_unrealcv_chase.py"
VIZ = REPO / "flyseek_extend" / "scripts" / "viz_chase_compare.py"
DEFAULT_OUT_ROOT = REPO / "flyseek_extend" / "output" / "demo_ue_compare"


def _episode_dir(run_out: Path, args, seed: int, tracker_mode: str) -> Path:
    """The episode subdir demo_unrealcv_chase writes for one tracker mode."""
    eid = (f"{args.env}_{args.target_behavior}_{args.difficulty}"
           f"_seed{seed}_000_{tracker_mode}")
    return run_out / eid


def _run_pair(args: argparse.Namespace, run_out: Path) -> int:
    """Render BOTH trackers in ONE UnrealCV connection (single client slot).

    UnrealCV allows only one client at a time, so we must not spawn a second
    process; ``--tracker-modes`` renders each tracker over the same seeded
    route within a single connection.
    """
    run_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(UE_DEMO),
        "--env", args.env,
        "--ue-ip", args.ue_ip,
        "--ue-port", str(args.ue_port),
        "--target-behavior", args.target_behavior,
        "--target-policy-difficulty", args.difficulty,
        "--tracker-modes", f"{args.flyseek_tracker},{args.baseline_tracker}",
        "--duration", str(args.duration),
        "--tick-hz", str(args.tick_hz),
        "--follow-distance", str(args.follow_distance),
        "--follow-altitude", str(args.follow_altitude),
        "--camera-pitch-deg", str(args.camera_pitch_deg),
        "--camera-hfov-deg", str(args.camera_hfov_deg),
        "--route-len-m", str(args.route_len_m),
        "--road-search-m", str(args.road_search_m),
        "--seg-building-jsonl", str(args.seg_building_jsonl),
        "--num-episodes", "1",
        "--seed", str(args.seed),
        "--out", str(run_out),
        "--lost-wander-radius-m", str(args.lost_wander_radius_m),
        "--lost-wander-scan-dps", str(args.lost_wander_scan_dps),
    ]
    if args.target_behavior == "alley_hutong":
        cmd += [
            "--open-approach-m", str(args.open_approach_m),
            "--max-corridor-width-m", str(args.max_corridor_width_m),
        ]
    if args.target:
        cmd += ["--target", args.target]
    if args.building_min_h is not None:
        cmd += ["--building-min-h", str(args.building_min_h)]
    print(f"\n[ue-compare] === rendering both trackers "
          f"({args.flyseek_tracker} + {args.baseline_tracker}) → {run_out} ===")
    print("[ue-compare] launching:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO))


def main() -> int:
    p = argparse.ArgumentParser(
        description="env_ue_smallcity FlySeek vs reactive_lost baseline + figures.",
    )
    p.add_argument("--env", default="env_ue_smallcity")
    p.add_argument("--ue-ip", default="127.0.0.1")
    p.add_argument("--ue-port", type=int, default=9000)
    p.add_argument("--target", default=None,
                   help="Exact controllable car actor name (else auto-probe).")
    p.add_argument("--target-behavior", default="occlusion_seeking",
                   help="Target route behavior. occlusion_seeking (default): duck "
                        "behind an annotated building (use a seed that occludes, "
                        "e.g. 80/83/90). alley_hutong: dive into a gap between "
                        "buildings — ONLY works in scenes that have narrow "
                        "drivable alleys (env_airsim_16); UE City Sample scenes "
                        "(smallcity/bigcity) have none, so it will fail there.")
    p.add_argument("--difficulty", default="hard",
                   choices=["easy", "medium", "hard"])
    p.add_argument("--open-approach-m", type=float, default=35.0,
                   help="alley_hutong: open road before the hutong entry (m).")
    p.add_argument("--max-corridor-width-m", type=float, default=12.0,
                   help="alley_hutong: max corridor width of the gap (m).")
    p.add_argument("--duration", type=float, default=55.0)
    p.add_argument("--tick-hz", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=80,
                   help="Default 80 (offline-verified to produce a building-"
                        "occluded hide leg in smallcity; 83/90 also occlude).")
    p.add_argument("--follow-distance", type=float, default=12.0)
    p.add_argument("--follow-altitude", type=float, default=14.0)
    p.add_argument("--camera-pitch-deg", type=float, default=55.0)
    p.add_argument("--camera-hfov-deg", type=float, default=70.0)
    p.add_argument("--route-len-m", type=float, default=240.0)
    p.add_argument("--road-search-m", type=float, default=180.0)
    p.add_argument("--building-min-h", type=float, default=None,
                   help="Building height threshold (m); UE yaml=6 over-blocks, "
                        "try 30 if the car gets boxed in.")
    p.add_argument("--flyseek-tracker", default="adaptive",
                   choices=["adaptive", "inline", "reactive", "reactive_lost"])
    p.add_argument("--baseline-tracker", default="reactive_lost",
                   choices=["adaptive", "inline", "reactive", "reactive_lost"])
    p.add_argument("--lost-wander-radius-m", type=float, default=6.0)
    p.add_argument("--lost-wander-scan-dps", type=float, default=35.0)
    p.add_argument("--compare", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--grid-step-m", type=float, default=2.0)
    p.add_argument("--frustum-every", type=int, default=18)
    p.add_argument("--seg-building-jsonl", type=Path,
                   default=REPO / "scene_data" / "seg_map" / "env_ue_smallcity.jsonl")
    args = p.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    run_out = args.out_root / f"runs_seed{args.seed}"

    rc = _run_pair(args, run_out)
    if rc != 0:
        print(f"[ue-compare] UE render failed (rc={rc}); aborting.")
        return rc
    if not args.compare:
        return 0

    fly_dir = _episode_dir(run_out, args, args.seed, args.flyseek_tracker)
    base_dir = _episode_dir(run_out, args, args.seed, args.baseline_tracker)
    if not (fly_dir / "frames.jsonl").exists():
        print(f"[ue-compare] expected episode dir missing: {fly_dir}")
        return 1
    if not (base_dir / "frames.jsonl").exists():
        print(f"[ue-compare] expected episode dir missing: {base_dir}")
        return 1

    viz_cmd = [
        sys.executable, str(VIZ),
        "--flyseek-dir", str(fly_dir),
        "--baseline-dir", str(base_dir),
        "--env", args.env,
        "--seg-building-jsonl", str(args.seg_building_jsonl),
        "--out-dir", str(args.out_root / f"compare_seed{args.seed}"),
        "--grid-step-m", str(args.grid_step_m),
        "--frustum-every", str(args.frustum_every),
    ]
    print("\n[ue-compare] === building comparison figures ===")
    print("[ue-compare] launching:", " ".join(viz_cmd))
    rc = subprocess.call(viz_cmd, cwd=str(REPO))
    if rc == 0:
        print(f"\n[ue-compare] DONE. Figures in "
              f"{args.out_root / f'compare_seed{args.seed}'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

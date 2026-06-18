#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Batch hide-and-seek demos: multiple episodes + OpenFly-style logs.

Each episode writes under::

    flyseek_extend/output/demo_hide_and_seek_batch/<batch_id>/episode_XXX/
        image_*.png
        pose.jsonl
        flyseek_meta.jsonl
        trajectory.jsonl
        hide_seek.mp4
        topdown.mp4   (if --topdown)

Usage (AirVLN running)::

    python flyseek_extend/scripts/run_hide_seek_batch.py \\
        --auto-from-scout --episodes 5
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEMO = REPO_ROOT / "flyseek_extend" / "scripts" / "demo_adversary_chase.py"
DEFAULT_BATCH_ROOT = REPO_ROOT / "flyseek_extend" / "output" / "demo_hide_and_seek_batch"


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch hide-and-seek episode runner.")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to record.")
    parser.add_argument("--batch-id", default=None,
                        help="Output folder name (default: timestamp).")
    parser.add_argument("--start-index", type=int, default=0,
                        help="First --target-index for scout car rotation.")
    parser.add_argument("--auto-from-scout", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42,
                        help="Base RNG seed; each episode uses seed+i.")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--tick-hz", type=float, default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--topdown", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--topdown-fps", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--no-camera-smoothing", action="store_true",
                        help="Forward --no-camera-smoothing to demo (raw tracker pose).")
    parser.add_argument("--camera-pos-smooth-tau", type=float, default=None,
                        help="Forward --camera-pos-smooth-tau (s).")
    parser.add_argument("--camera-yaw-smooth-tau", type=float, default=None,
                        help="Forward --camera-yaw-smooth-tau (s).")
    parser.add_argument("--camera-max-yaw-rate-dps", type=float, default=None,
                        help="Forward --camera-max-yaw-rate-dps (deg/s).")
    parser.add_argument("--no-use-inline-tracker", action="store_true",
                        help="Forward --no-use-inline-tracker (legacy tracker).")
    parser.add_argument("--tracker-motion-dir-tau", type=float, default=None,
                        help="Forward --tracker-motion-dir-tau (s).")
    parser.add_argument("--tracker-lead-s", type=float, default=None,
                        help="Forward --tracker-lead-s (s).")
    parser.add_argument("--tracker-yaw-gain", type=float, default=None,
                        help="Forward --tracker-yaw-gain (1/s).")
    parser.add_argument("--search-orbit-speed-dps", type=float, default=None,
                        help="Forward --search-orbit-speed-dps (deg/s).")
    args, extra = parser.parse_known_args()

    batch_id = args.batch_id or datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    batch_dir = DEFAULT_BATCH_ROOT / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    exit_code = 0

    for ep in range(args.episodes):
        ep_tag = f"episode_{ep:03d}"
        ep_dir = batch_dir / ep_tag
        argv = [
            str(DEMO),
            "--scenario", "hide_seek",
            "--output", str(batch_dir),
            "--episode-tag", ep_tag,
            "--target-index", str(args.start_index + ep),
            "--seed", str(args.seed + ep),
        ]
        if args.auto_from_scout:
            argv.append("--auto-from-scout")
        if args.duration is not None:
            argv.extend(["--duration", str(args.duration)])
        if args.tick_hz is not None:
            argv.extend(["--tick-hz", str(args.tick_hz)])
        if args.frames is not None:
            argv.extend(["--frames", str(args.frames)])
        if args.topdown is not None:
            argv.extend(["--topdown" if args.topdown else "--no-topdown"])
        if args.topdown_fps is not None:
            argv.extend(["--topdown-fps", str(args.topdown_fps)])
        if args.timeout is not None:
            argv.extend(["--timeout", str(args.timeout)])
        if args.no_camera_smoothing:
            argv.append("--no-camera-smoothing")
        if args.camera_pos_smooth_tau is not None:
            argv.extend(["--camera-pos-smooth-tau", str(args.camera_pos_smooth_tau)])
        if args.camera_yaw_smooth_tau is not None:
            argv.extend(["--camera-yaw-smooth-tau", str(args.camera_yaw_smooth_tau)])
        if args.camera_max_yaw_rate_dps is not None:
            argv.extend(["--camera-max-yaw-rate-dps", str(args.camera_max_yaw_rate_dps)])
        if args.no_use_inline_tracker:
            argv.append("--no-use-inline-tracker")
        if args.tracker_motion_dir_tau is not None:
            argv.extend(["--tracker-motion-dir-tau", str(args.tracker_motion_dir_tau)])
        if args.tracker_lead_s is not None:
            argv.extend(["--tracker-lead-s", str(args.tracker_lead_s)])
        if args.tracker_yaw_gain is not None:
            argv.extend(["--tracker-yaw-gain", str(args.tracker_yaw_gain)])
        if args.search_orbit_speed_dps is not None:
            argv.extend(["--search-orbit-speed-dps", str(args.search_orbit_speed_dps)])
        argv.extend(extra)

        print("\n" + "=" * 72)
        print(f"BATCH {batch_id} — {ep_tag} ({ep + 1}/{args.episodes})")
        print("=" * 72)

        old_argv = sys.argv
        sys.argv = argv
        try:
            runpy.run_path(str(DEMO), run_name="__main__")
            summary_path = ep_dir / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                results.append(summary)
                if not summary.get("success"):
                    exit_code = 1
            else:
                results.append({"episode": ep_tag, "success": False,
                                "errors": ["summary.json missing"]})
                exit_code = 1
        except SystemExit as e:
            code = int(e.code) if e.code is not None else 1
            results.append({"episode": ep_tag, "success": code == 0,
                            "exit_code": code})
            if code != 0:
                exit_code = code
        finally:
            sys.argv = old_argv

    manifest = {
        "batch_id": batch_id,
        "batch_dir": str(batch_dir),
        "episodes": args.episodes,
        "results": results,
    }
    (batch_dir / "batch_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\n" + "=" * 72)
    print(f"BATCH DONE — {batch_dir}")
    print(f"  episodes : {args.episodes}")
    print(f"  manifest : {batch_dir / 'batch_manifest.json'}")
    print("=" * 72)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Record a paired chase demo: drone tracking success vs failure.

Uses annotated ``seg_map/*.jsonl`` buildings for the *failure* episode (car hides
behind a labeled building → drone loses track). The *success* episode keeps the
car on an open-road escape route so the drone maintains visibility.

Both episodes share the same target init (--auto-from-scout / --seed) when
``--shared-seed`` is set.

Example (AirSim env_airsim_16 running):

    python flyseek_extend/scripts/demo_chase_pair.py \\
      --env env_airsim_16 \\
      --auto-from-scout \\
      --seg-building-jsonl scene_data/seg_map/env_airsim_16.jsonl \\
      --duration 75 \\
      --shared-seed 66
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "flyseek_extend" / "scripts" / "demo_adversary_chase.py"


def _run_episode(name: str, extra: list[str], args: argparse.Namespace) -> int:
    cmd = [
        sys.executable, str(DEMO),
        "--env", args.env,
        "--duration", str(args.duration),
        "--tick-hz", str(args.tick_hz),
        "--episode-tag", f"{args.episode_prefix}_{name}",
        "--airsim-ip", args.airsim_ip,
        "--airsim-port", str(args.airsim_port),
    ]
    if args.auto_from_scout:
        cmd.append("--auto-from-scout")
    if args.target:
        cmd.extend(["--target", args.target])
    if args.shared_seed is not None:
        cmd.extend(["--seed", str(args.shared_seed)])
    if args.init_profile:
        cmd.extend(["--init-profile", args.init_profile])
    cmd.extend(extra)
    print(f"\n{'='*60}\n[pair] episode: {name}\n[pair] cmd: {' '.join(cmd)}\n{'='*60}")
    return subprocess.call(cmd, cwd=str(REPO))


def main() -> int:
    p = argparse.ArgumentParser(description="Record track-success + track-fail chase pair.")
    p.add_argument("--env", default="env_airsim_16")
    p.add_argument("--duration", type=float, default=75.0)
    p.add_argument("--tick-hz", type=float, default=5.0)
    p.add_argument("--shared-seed", type=int, default=66)
    p.add_argument("--episode-prefix", default="chase_pair")
    p.add_argument("--auto-from-scout", action="store_true")
    p.add_argument("--target", default=None)
    p.add_argument("--init-profile", default="standard")
    p.add_argument("--seg-building-jsonl", type=Path,
                    default=REPO / "scene_data" / "seg_map" / "env_airsim_16.jsonl")
    p.add_argument("--airsim-ip", default="127.0.0.1")
    p.add_argument("--airsim-port", type=int, default=41451)
    p.add_argument("--route-len-m", type=float, default=150.0)
    p.add_argument("--open-road-frac", type=float, default=0.4)
    args = p.parse_args()

    seg = str(args.seg_building_jsonl)
    common_hide = [
        "--target-policy-difficulty", "hard",
        "--route-len-m", str(args.route_len_m),
        "--open-road-frac", str(args.open_road_frac),
        "--duration", str(args.duration),
    ]

    # 跟踪成功：开阔逃逸，无人机持续可见
    rc_ok = _run_episode("track_success", [
        "--target-behavior", "direct_escape",
        *common_hide,
    ], args)

    # 跟踪失败：标注建筑后躲藏，无人机丢失目标
    rc_fail = _run_episode("track_fail", [
        "--target-behavior", "occlusion_seeking",
        "--seg-building-jsonl", seg,
        "--seg-building-radius-m", "10",
        *common_hide,
    ], args)

    print(f"\n[pair] done  track_success rc={rc_ok}  track_fail rc={rc_fail}")
    print(f"[pair] outputs under flyseek_extend/output/ with prefix {args.episode_prefix}_")
    return 0 if rc_ok == 0 and rc_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

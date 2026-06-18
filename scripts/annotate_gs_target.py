#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Mark the target car on already-rendered GS chase frames (CPU, no GPU).

For each frame i it loads frames/frame_{i}/<png>, rebuilds the UAV camera from
trajectories.json (uav_trajectory[i]) and draws the target box/crosshair/label
at target_trajectory[i]. Frames MUST have been rendered from the same
trajectories.json (same UAV poses), otherwise the marker won't align.

Outputs <frames>/../frames_marked/frame_XXXX.png + a contact sheet.

Usage:
    python flyseek_extend/scripts/annotate_gs_target.py \
        --traj flyseek_extend/output/gs_debug/chase_geom/trajectories.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from flyseek.render.gs_camera import Intrinsics, bridge_extrinsics
from flyseek.render.target_overlay import draw_target_marker

REPO_ROOT = Path(__file__).resolve().parents[2]
FULL_INTR = "0 PINHOLE 2048 1536 1335.1645731732658 1335.4075753200657 1024.0 768.0"


def newest_png(d):
    p = sorted(glob.glob(os.path.join(d, "*.png")), key=os.path.getmtime)
    return p[-1] if p else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--frames", default=None, help="frames dir (default: <traj_dir>/frames)")
    ap.add_argument("--render-scale", type=float, default=1.0)
    args = ap.parse_args()

    traj = json.load(open(args.traj))
    car = traj["target_trajectory"]
    uav = traj["uav_trajectory"]
    frames_dir = args.frames or os.path.join(os.path.dirname(args.traj), "frames")
    out_dir = os.path.join(os.path.dirname(args.traj), "frames_marked")
    os.makedirs(out_dir, exist_ok=True)

    intr0 = Intrinsics.from_colmap_str(FULL_INTR).scaled(args.render_scale)
    car_xy = np.array([[c["pos"][0], c["pos"][1], c["pos"][2]] for c in car])

    marked, n_inview = [], 0
    for i, (c, u) in enumerate(zip(car, uav)):
        d = os.path.join(frames_dir, f"frame_{i:04d}")
        png = newest_png(d)
        if not png:
            print(f"[annotate] frame {i}: no png in {d}, skip")
            continue
        im = Image.open(png).convert("RGB")
        intr = intr0
        if im.size != (intr0.w, intr0.h):
            intr = Intrinsics(im.size[0], im.size[1],
                              intr0.fx * im.size[0] / intr0.w, intr0.fy * im.size[1] / intr0.h,
                              intr0.cx * im.size[0] / intr0.w, intr0.cy * im.size[1] / intr0.h)
        R, t = bridge_extrinsics(u["pos"], u["yaw_input"], u["pitch_deg"])
        trail = car_xy[max(0, i - 4): i + 3]
        info = draw_target_marker(im, np.array(c["pos"]), c["heading"], R, t, intr,
                                  label="TARGET", trail_world=trail)
        n_inview += int(info["in_view"])
        op = os.path.join(out_dir, f"frame_{i:04d}.png")
        im.save(op)
        marked.append(op)
        print(f"[annotate] frame {i:04d} center_uv={info['center_uv']} in_view={info['in_view']}")

    # contact sheet
    if marked:
        cols = min(5, len(marked))
        rows = (len(marked) + cols - 1) // cols
        cell = 360
        sheet = Image.new("RGB", (cols * cell, rows * cell), (15, 15, 15))
        for k, p in enumerate(marked):
            im = Image.open(p).convert("RGB"); im.thumbnail((cell, cell))
            sheet.paste(im, ((k % cols) * cell, (k // cols) * cell))
        sp = os.path.join(os.path.dirname(args.traj), "frames_marked_sheet.png")
        sheet.save(sp)
        print(f"[annotate] {len(marked)} frames, {n_inview} in-view. sheet -> {sp}")


if __name__ == "__main__":
    main()

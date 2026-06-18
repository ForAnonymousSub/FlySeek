#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Replay a recorded expert trajectory and re-render it to verify reproducibility.

Reads ``pose.jsonl`` (OpenFly action records: map-frame ``pos`` + ``yaw``) from an
episode directory, teleports the drone along the recorded poses via the AirSim
whitelist (``simSetVehiclePose``), captures RGB each frame, and renders an mp4.

Usage (AirVLN must be running)::

    python flyseek_extend/scripts/replay_trajectory.py \\
        flyseek_extend/output/demo_hide_and_seek/<ts>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT / "flyseek_extend") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "flyseek_extend"))

from flyseek.utils.coords import map_to_airsim_ned  # noqa: E402


def _load_poses(pose_path: Path):
    poses = []
    for line in pose_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        act = rec.get("action")
        if not act:
            continue
        pos_map = act["pos"]
        ned = map_to_airsim_ned(np.asarray(pos_map, dtype=float))
        poses.append((float(ned[0]), float(ned[1]), float(ned[2]), float(act["yaw"])))
    return poses


def _render_mp4(frames_dir: Path, out_path: Path, fps: float) -> bool:
    import shutil
    import subprocess
    if shutil.which("ffmpeg") is None:
        return False
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-pattern_type", "glob", "-i", str(frames_dir / "replay_*.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return out_path.exists()
    except subprocess.CalledProcessError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay an expert trajectory.")
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--airsim-ip", default="127.0.0.1")
    parser.add_argument("--airsim-port", type=int, default=41451)
    parser.add_argument("--camera-name", default="front_custom")
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    pose_path = args.episode_dir / "pose.jsonl"
    if not pose_path.exists():
        print(f"[ERR] no pose.jsonl in {args.episode_dir}")
        return 1
    poses = _load_poses(pose_path)
    if not poses:
        print("[ERR] no action poses found in pose.jsonl")
        return 1

    import airsim  # type: ignore
    import numpy as np  # noqa: F401

    client = airsim.MultirotorClient(ip=args.airsim_ip, port=args.airsim_port)
    client.confirmConnection()

    frames_dir = args.episode_dir / "replay_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for i, (x, y, z, yaw) in enumerate(poses):
        pose = airsim.Pose(airsim.Vector3r(x, y, z),
                           airsim.to_quaternion(0.0, 0.0, yaw))
        client.simSetVehiclePose(pose, ignore_collision=True)
        resp = client.simGetImages([
            airsim.ImageRequest(args.camera_name, airsim.ImageType.Scene, False, False)
        ])
        if resp and resp[0].image_data_uint8:
            import numpy as _np
            img = _np.frombuffer(resp[0].image_data_uint8, dtype=_np.uint8)
            img = img.reshape(resp[0].height, resp[0].width, 3)
            import cv2  # type: ignore
            cv2.imwrite(str(frames_dir / f"replay_{i:04d}.png"), img)

    out_mp4 = args.episode_dir / "replay.mp4"
    if _render_mp4(frames_dir, out_mp4, args.fps):
        print(f"[ok] replay mp4 -> {out_mp4}")
    else:
        print(f"[ok] replay frames -> {frames_dir} (ffmpeg unavailable for mp4)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

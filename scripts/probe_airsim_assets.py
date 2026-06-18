# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Probe an AirSim instance for available assets and scene objects.

This script is the **D1 asset survey** entry point. It connects to a running
AirSim instance (started by OpenFly's bridge or this script itself) and
catalogues every asset the binary knows about, so we can decide which
pedestrians/vehicles to use as adversarial targets.

Whitelist policy (see SKILL.md §6.1): this script touches AirSim only through
read-only listing APIs. It does NOT call `enableApiControl`, `armDisarm`, or
any movement command.

Usage:
    # Option A: AirSim already running (preferred)
    python scripts/probe_airsim_assets.py \\
        --env env_airsim_16 \\
        --output output/assets/env_airsim_16_assets.json

    # Option B: auto-launch the binary, probe, then kill
    python scripts/probe_airsim_assets.py \\
        --env env_airsim_16 \\
        --auto-launch \\
        --launch-timeout 20

    # Filter only FlySeek-injected assets (validate pak overlay)
    python scripts/probe_airsim_assets.py \\
        --env env_airsim_16 --filter FlySeek
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # OpenFly-Platform/


@dataclass
class ProbeReport:
    timestamp: str
    env: str
    airsim_ip: str
    airsim_rpc_port: int
    assets_total: int = 0
    scene_objects_total: int = 0
    assets: list[str] = field(default_factory=list)
    scene_objects: list[str] = field(default_factory=list)
    # Categorized views (heuristics on name)
    characters: list[str] = field(default_factory=list)
    vehicles: list[str] = field(default_factory=list)
    drone_pawns: list[str] = field(default_factory=list)
    cover_objects: list[str] = field(default_factory=list)  # TrafficLight, RoadBarrier, ...
    flyseek_injected: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _categorize_asset(name: str) -> str | None:
    n = name.lower()
    if "flyseek" in n:
        return "flyseek_injected"
    if any(k in n for k in ("human", "pedestrian", "person", "citizen", "character", "mannequin")):
        return "characters"
    if any(k in n for k in ("car", "vehicle", "suv", "truck", "van", "bus", "bicycle", "moto")):
        return "vehicles"
    if any(k in n for k in ("flyingpawn", "quad", "computervision", "drone", "uav")):
        return "drone_pawns"
    if any(k in n for k in ("trafficlight", "roadbarrier", "tramline", "fence", "pole", "sign")):
        return "cover_objects"
    return None


def _launch_airsim(env_name: str, timeout_s: float) -> subprocess.Popen[bytes] | None:
    """Best-effort launch of the AirVLN binary for `env_name`.

    Mirrors OpenFly's pattern in `scripts/sim/airsim_bridge.py:30-39`:
    `bash envs/airsim/<env>/LinuxNoEditor/start.sh`.
    """
    start_sh = REPO_ROOT / "envs" / "airsim" / env_name / "LinuxNoEditor" / "start.sh"
    if not start_sh.exists():
        print(f"[warn] {start_sh} not found; assuming AirSim is already running.")
        return None

    print(f"[info] Launching {start_sh}")
    proc = subprocess.Popen(
        ["bash", str(start_sh)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(REPO_ROOT),
    )
    print(f"[info] Waiting {timeout_s}s for AirSim to come up...")
    time.sleep(timeout_s)
    return proc


def _probe(airsim_ip: str, airsim_port: int, report: ProbeReport, asset_filter: str | None) -> None:
    try:
        import airsim  # type: ignore
    except ImportError as e:
        report.errors.append(f"airsim package not importable: {e}")
        return

    try:
        client = airsim.MultirotorClient(ip=airsim_ip, port=airsim_port)
        client.confirmConnection()
    except Exception as e:
        report.errors.append(f"failed to connect to AirSim at {airsim_ip}:{airsim_port}: {e}")
        return

    # ⚠️ We do NOT call enableApiControl / armDisarm — read-only probe.

    # 1. simListAssets — every spawn-able asset baked into the cooked pak
    try:
        assets: list[str] = client.simListAssets() or []
    except Exception as e:
        report.errors.append(f"simListAssets failed: {e}")
        assets = []

    # 2. simListSceneObjects — objects already placed in the level
    try:
        scene_objects: list[str] = client.simListSceneObjects() or []
    except Exception as e:
        report.errors.append(f"simListSceneObjects failed: {e}")
        scene_objects = []

    if asset_filter:
        f = asset_filter.lower()
        assets = [a for a in assets if f in a.lower()]
        scene_objects = [o for o in scene_objects if f in o.lower()]

    report.assets = sorted(assets)
    report.scene_objects = sorted(scene_objects)
    report.assets_total = len(report.assets)
    report.scene_objects_total = len(report.scene_objects)

    for name in report.assets + report.scene_objects:
        bucket = _categorize_asset(name)
        if bucket == "characters":
            report.characters.append(name)
        elif bucket == "vehicles":
            report.vehicles.append(name)
        elif bucket == "drone_pawns":
            report.drone_pawns.append(name)
        elif bucket == "cover_objects":
            report.cover_objects.append(name)
        elif bucket == "flyseek_injected":
            report.flyseek_injected.append(name)

    # Dedup categorized lists
    for attr in ("characters", "vehicles", "drone_pawns", "cover_objects", "flyseek_injected"):
        setattr(report, attr, sorted(set(getattr(report, attr))))


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe AirSim assets and scene objects.")
    parser.add_argument("--env", default="env_airsim_16", help="OpenFly environment name")
    parser.add_argument("--airsim-ip", default=os.environ.get("AIRSIM_IP", "127.0.0.1"))
    parser.add_argument(
        "--airsim-port",
        type=int,
        default=int(os.environ.get("AIRSIM_RPC_PORT", 41451)),
    )
    parser.add_argument(
        "--auto-launch",
        action="store_true",
        help="Launch the AirVLN binary before probing (using start.sh).",
    )
    parser.add_argument(
        "--launch-timeout",
        type=float,
        default=20.0,
        help="Seconds to wait after launching before probing.",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Case-insensitive substring filter (e.g. 'FlySeek' to validate pak overlay).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to output/assets/<env>_assets.json",
    )
    args = parser.parse_args()

    output_path = args.output or REPO_ROOT / "flyseek_extend" / "output" / "assets" / f"{args.env}_assets.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = ProbeReport(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        env=args.env,
        airsim_ip=args.airsim_ip,
        airsim_rpc_port=args.airsim_port,
    )

    launched: subprocess.Popen[bytes] | None = None
    if args.auto_launch:
        launched = _launch_airsim(args.env, args.launch_timeout)

    try:
        _probe(args.airsim_ip, args.airsim_port, report, args.filter)
    finally:
        if launched is not None:
            print("[info] Terminating launched AirSim process...")
            launched.terminate()
            try:
                launched.wait(timeout=10)
            except subprocess.TimeoutExpired:
                launched.kill()

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)

    # Human-readable summary
    print("\n" + "=" * 60)
    print(f"AirSim Asset Probe Report — env={args.env}")
    print("=" * 60)
    print(f"Total assets:        {report.assets_total}")
    print(f"Total scene objects: {report.scene_objects_total}")
    print(f"Characters:          {len(report.characters):4d}  {report.characters[:5]}")
    print(f"Vehicles:            {len(report.vehicles):4d}  {report.vehicles[:5]}")
    print(f"Drone pawns:         {len(report.drone_pawns):4d}  {report.drone_pawns}")
    print(f"Cover objects:       {len(report.cover_objects):4d}  {report.cover_objects[:5]}")
    print(f"FlySeek injected:    {len(report.flyseek_injected):4d}  {report.flyseek_injected[:5]}")
    if report.errors:
        print(f"\nErrors ({len(report.errors)}):")
        for e in report.errors:
            print(f"  - {e}")
    print(f"\nFull JSON written to: {output_path}")
    return 0 if not report.errors else 2


if __name__ == "__main__":
    sys.exit(main())

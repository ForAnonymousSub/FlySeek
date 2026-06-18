#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Scan scene targets and list which pass PCD road init (env preset profiles).

Requires AirSim / AirVLN running (reads spawn poses via simGetObjectPose).

Usage:

    python flyseek_extend/scripts/list_initializable_targets.py \\
        --env env_airsim_16 --init-profile standard

    # Motorized cars only (recommended for chase / hide_seek):
    python flyseek_extend/scripts/list_initializable_targets.py \\
        --motorized-cars-only

    # Save JSON for batch generation:
    python flyseek_extend/scripts/list_initializable_targets.py \\
        -o flyseek_extend/output/assets/initializable_targets.json

    # Reuse scout output (no live pose RPC per object if JSON has poses):
    python flyseek_extend/scripts/list_initializable_targets.py \\
        --from-scout-json flyseek_extend/output/assets/scene_targets_latest.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (REPO_ROOT / "flyseek_extend", _SCRIPTS_DIR):
  if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

from flyseek.adapters.pcd_occupancy import PcdOccupancyMap  # noqa: E402
from flyseek.utils.target_init import (  # noqa: E402
  _heading_from_quaternion,
  resolve_target_init_pose,
  score_init_pose_ned,
)
from flyseek.utils.target_init_presets import (  # noqa: E402
  default_profile_name,
  list_profiles,
  load_target_init_profile,
)

# Reuse scout category patterns
from scout_scene_targets import (  # noqa: E402
  CATEGORY_PATTERNS,
  MOTORIZED_CAR_CATEGORIES,
  is_motorized_car_name,
)


def _classify(name: str) -> tuple[str, str]:
  import re
  for cat, regex, label in CATEGORY_PATTERNS:
    if re.search(regex, name):
      return cat, label
  return "other", "a scene object"


def _load_from_scout(path: Path) -> list[dict]:
  doc = json.loads(path.read_text(encoding="utf-8"))
  if isinstance(doc, list):
    return doc
  return doc.get("targets") or doc.get("candidates") or []


def _anchors_from_airsim(client, name_regex: str) -> list[dict]:
  names = sorted(client.simListSceneObjects(name_regex=name_regex) or [])
  rows: list[dict] = []
  for name in names:
    try:
      pose = client.simGetObjectPose(name)
    except Exception:
      continue
    if math.isnan(pose.position.x_val):
      continue
    cat, label = _classify(name)
    rows.append({
        "name": name,
        "category": cat,
        "suggested_label": label,
        "default_pose": {
            "x": pose.position.x_val,
            "y": pose.position.y_val,
            "z": pose.position.z_val,
            "qw": pose.orientation.w_val,
            "qx": pose.orientation.x_val,
            "qy": pose.orientation.y_val,
            "qz": pose.orientation.z_val,
        },
    })
  return rows


def main() -> int:
  parser = argparse.ArgumentParser(
      description="List scene targets that pass PCD road initialization.",
  )
  parser.add_argument("--env", default="env_airsim_16")
  parser.add_argument("--init-profile", default=None)
  parser.add_argument("--list-profiles", action="store_true")
  parser.add_argument("--motorized-cars-only", action="store_true",
                      help="Only SM_* cars / taxis (excludes carts, buses).")
  parser.add_argument("--categories", default=None,
                      help="Comma-separated scout categories (e.g. car_drivable,taxi).")
  parser.add_argument("--name-regex", default=".*",
                      help="AirSim simListSceneObjects regex (default: all).")
  parser.add_argument("--from-scout-json", type=Path, default=None,
                      help="Use poses from scout_scene_targets.py output.")
  parser.add_argument("--max-targets", type=int, default=0,
                      help="Stop after N candidates (0 = no limit).")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--rebuild-occupancy-cache", action="store_true")
  parser.add_argument("-o", "--output", type=Path, default=None)
  parser.add_argument("--airsim-ip", default="127.0.0.1")
  parser.add_argument("--airsim-port", type=int, default=41451)
  args = parser.parse_args()

  if args.list_profiles:
    for p in list_profiles(args.env):
      prof = load_target_init_profile(args.env, p)
      d = " (default)" if p == default_profile_name(args.env) else ""
      print(f"{p}{d}: {prof.description}")
    return 0

  profile_name = args.init_profile or default_profile_name(args.env)
  profile = load_target_init_profile(args.env, profile_name)
  print(f"[ok] env={args.env} profile={profile.name} strategy={profile.strategy}")

  occupancy = PcdOccupancyMap.load_or_build(
      REPO_ROOT, env_name=args.env, rebuild=args.rebuild_occupancy_cache,
  )
  rng = np.random.default_rng(args.seed)

  if args.from_scout_json:
    if not args.from_scout_json.is_file():
      print(f"[ERR] scout JSON not found: {args.from_scout_json}")
      return 1
    candidates = _load_from_scout(args.from_scout_json)
    print(f"[ok] loaded {len(candidates)} entries from {args.from_scout_json}")
  else:
    try:
      import airsim  # type: ignore
    except ImportError:
      print("[ERR] airsim not installed")
      return 1
    client = airsim.MultirotorClient(ip=args.airsim_ip, port=args.airsim_port)
    client.confirmConnection()
    candidates = _anchors_from_airsim(client, args.name_regex)
    print(f"[ok] listed {len(candidates)} scene objects (regex={args.name_regex!r})")

  allowed_cats: set[str] | None = None
  if args.categories:
    allowed_cats = {c.strip() for c in args.categories.split(",") if c.strip()}

  ok_rows: list[dict] = []
  fail_rows: list[dict] = []
  scanned = 0

  for entry in candidates:
    name = entry["name"] if isinstance(entry, dict) else str(entry)
    cat = entry.get("category") if isinstance(entry, dict) else "other"
    if args.motorized_cars_only and not is_motorized_car_name(name):
      continue
    if allowed_cats is not None and cat not in allowed_cats:
      continue

    pose_d = entry.get("default_pose") if isinstance(entry, dict) else None
    if not pose_d:
      continue

    scanned += 1
    if args.max_targets and scanned > args.max_targets:
      break

    anchor = np.array([pose_d["x"], pose_d["y"], pose_d["z"]])
    hint_h = _heading_from_quaternion(
        pose_d["qw"], pose_d["qx"], pose_d["qy"], pose_d["qz"],
    )
    spawn_score, spawn_reason = score_init_pose_ned(
        occupancy, anchor, hint_h, cfg=profile.config,
    )
    # Per-target seed so results are reproducible but not identical
    trng = np.random.default_rng(args.seed + hash(name) % 10_000)
    result = resolve_target_init_pose(
        occupancy, anchor, trng, profile, hint_heading=hint_h,
    )
    shift = float(np.linalg.norm(result.position_ned[:2] - anchor[:2]))
    row = {
        "name": name,
        "category": cat,
        "suggested_label": entry.get("suggested_label", ""),
        "spawn_ned": anchor.tolist(),
        "spawn_score": spawn_score,
        "spawn_reason": spawn_reason,
        "init_ok": result.ok,
        "init_score": result.score,
        "init_reason": result.reason,
        "init_method": result.init_method,
        "init_profile": profile.name,
        "position_ned": result.position_ned.tolist(),
        "heading_deg": math.degrees(result.heading_rad),
        "shift_from_spawn_m": round(shift, 2),
        "samples_tried": result.samples_tried,
    }
    if result.ok:
      ok_rows.append(row)
    else:
      fail_rows.append(row)

  ok_rows.sort(key=lambda r: (-r["init_score"], r["name"]))
  fail_rows.sort(key=lambda r: r["name"])

  print(f"\n[summary] scanned={scanned}  init_ok={len(ok_rows)}  failed={len(fail_rows)}")
  print(f"\n{'=' * 72}")
  print("INITIALIZABLE TARGETS (ok=True) — use --target <name> for generation")
  print(f"{'=' * 72}")
  if not ok_rows:
    print("  (none — try --init-profile major_road or widen preset in YAML)")
  for i, r in enumerate(ok_rows, 1):
    p = r["position_ned"]
    print(
        f"  {i:3d}. {r['name']}\n"
        f"       score={r['init_score']:.1f}  method={r['init_method']}  "
        f"shift={r['shift_from_spawn_m']:.0f}m  "
        f"pos=({p[0]:.1f}, {p[1]:.1f}, {p[2]:.2f})  "
        f"hdg={r['heading_deg']:.0f}°"
    )

  if fail_rows and len(fail_rows) <= 15:
    print(f"\n[failed] {len(fail_rows)} targets (spawn or search could not pass gates)")

  doc = {
      "generated_at": datetime.now().isoformat(timespec="seconds"),
      "env": args.env,
      "init_profile": profile.name,
      "scanned": scanned,
      "initializable": ok_rows,
      "failed": fail_rows,
  }

  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"\n[ok] wrote {args.output}")

  return 0 if ok_rows else 2


if __name__ == "__main__":
  sys.exit(main())

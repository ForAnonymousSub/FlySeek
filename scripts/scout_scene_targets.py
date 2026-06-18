# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Scout: discover usable tracking targets from the existing NYC scene.

We don't inject new assets (mod pak route proven unstable). Instead we
**reuse the 35 528 actors already placed in env_airsim_16**, picking
vehicles / carts / signs etc. that we can later teleport along adversarial
trajectories via ``simSetObjectPose`` (no asset registry / spawn involvement
=> zero SEGV risk).

Output:
    flyseek_extend/output/assets/scene_targets_<timestamp>.json
    flyseek_extend/output/assets/scene_targets_latest.json   (symlink convenience)

The JSON has one entry per candidate:
    {
        "name":            "Cart_v1_low2_634",
        "category":        "cart",
        "default_pose": {  // NED frame, AirSim raw
            "x": 12.34, "y": 56.78, "z": -1.20,
            "qw": 0.707, "qx": 0, "qy": 0, "qz": 0.707
        },
        "approx_size_m": null,            // unknown without bbox API
        "suggested_label": "a wooden cart"
    }

Whitelist policy (SKILL.md §6.1):
    - simListSceneObjects(name_regex)  ✓ read-only
    - simGetObjectPose                 ✓ read-only
    NOT used: simSpawnObject, simSetObjectPose (those come later in smoke_test)

Usage:
    # Assumes AirVLN is up (start.sh).
    python flyseek_extend/scripts/scout_scene_targets.py

    # Narrow down to a single category, save fewer to JSON:
    python flyseek_extend/scripts/scout_scene_targets.py \\
        --categories car,truck --max-per-category 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "flyseek_extend" / "output" / "assets"


# Patterns we search via simListSceneObjects. Order matters: a name may match
# multiple, we keep the *first* match → so put the most specific first.
CATEGORY_PATTERNS: list[tuple[str, str, str]] = [
    # (category, regex,                            human-friendly label template)
    # Order matters — first wins on dedup. Motorized cars must come BEFORE
    # generic "car" / "vehicle" / "cart" patterns so they get classified first.
    ("car_drivable", r"^SM_.*Car.*Drivable.*",     "a small motorized car"),
    ("car_nyc",      r"^SM_NYC.*Car0?[0-9]+.*",    "a small motorized car"),
    ("car_classic",  r"^SM_(Classic|classic).*Car0?[0-9]+.*", "a small motorized car"),
    ("taxi",   r".*[Tt]axi.*",                     "a yellow taxi cab"),
    ("bus",    r".*[Bb]us(?![a-zA-Z]).*",          "a city bus"),
    ("truck",  r".*[Tt]ruck.*",                    "a delivery truck"),
    ("suv",    r".*[Ss]UV.*",                      "an SUV"),
    ("sedan",  r".*[Ss]edan.*",                    "a sedan car"),
    ("car",    r".*Car0?[0-9]+.*",                 "a parked car"),
    ("van",    r".*[Vv]an(?![a-zA-Z]).*",          "a van"),
    ("bike",   r".*([Bb]ike|[Bb]icycle).*",        "a bicycle"),
    ("cart",   r".*[Cc]art.*",                     "a small cart"),
    ("stall",  r".*[Bb]uffet.*",                   "a street food stall"),
    ("sign",   r".*[Ss]ign.*",                     "a street sign"),
    # Catch-all generic vehicle word
    ("vehicle", r".*[Vv]ehicle.*",                 "a vehicle"),
]


# Sub-set of categories considered "motorized small car" — used by the
# hide_seek demo when --motorized-cars-only is passed (default ON).
MOTORIZED_CAR_CATEGORIES = {"car_drivable", "car_nyc", "car_classic", "sedan"}


def is_motorized_car_name(name: str) -> bool:
    """True if the actor name looks like a motorized small car (not Cart, not Bus)."""
    if "Cart" in name or "cart" in name:
        return False
    if "Bus" in name or "bus" in name:
        return False
    if "Truck" in name or "truck" in name:
        return False
    if "Buffet" in name or "Stand" in name or "Sign" in name:
        return False
    import re
    for pat in (
        r"^SM_.*Car.*Drivable.*",
        r"^SM_NYC.*Car0?[0-9]+",
        r"^SM_(Classic|classic).*Car0?[0-9]+",
        r"^SM_.*Sedan",
    ):
        if re.match(pat, name):
            return True
    return False


def _pose_to_dict(p) -> dict[str, float]:
    return {
        "x": float(p.position.x_val),
        "y": float(p.position.y_val),
        "z": float(p.position.z_val),
        "qw": float(p.orientation.w_val),
        "qx": float(p.orientation.x_val),
        "qy": float(p.orientation.y_val),
        "qz": float(p.orientation.z_val),
    }


def _is_valid_pose(p) -> bool:
    """AirSim returns NaNs for non-existent / inaccessible actors."""
    for v in (p.position.x_val, p.position.y_val, p.position.z_val,
              p.orientation.w_val, p.orientation.x_val,
              p.orientation.y_val, p.orientation.z_val):
        if math.isnan(v) or math.isinf(v):
            return False
    # Also reject objects at the exact origin (often a sentinel/error value)
    if (abs(p.position.x_val) < 1e-6
            and abs(p.position.y_val) < 1e-6
            and abs(p.position.z_val) < 1e-6
            and abs(p.orientation.w_val - 1.0) < 1e-6):
        return False
    return True


def _select_active_categories(args) -> list[tuple[str, str, str]]:
    if not args.categories:
        return CATEGORY_PATTERNS
    wanted = {c.strip().lower() for c in args.categories.split(",") if c.strip()}
    return [t for t in CATEGORY_PATTERNS if t[0] in wanted]


def run_scout(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import airsim  # type: ignore
    except ImportError as e:
        return {"error": f"airsim not importable: {e}"}

    try:
        client = airsim.MultirotorClient(ip=args.airsim_ip, port=args.airsim_port)
        client.confirmConnection()
    except Exception as e:
        return {"error": f"failed to connect: {e}"}

    active_cats = _select_active_categories(args)
    print(f"[info] scanning {len(active_cats)} categories: "
          f"{[c[0] for c in active_cats]}")

    # ---- Step 1: regex list per category -----------------------------------
    raw_by_cat: dict[str, list[str]] = {}
    seen: set[str] = set()
    for cat, pattern, _label in active_cats:
        try:
            t0 = time.time()
            matches = client.simListSceneObjects(name_regex=pattern) or []
        except Exception as e:
            print(f"[warn] simListSceneObjects('{pattern}') raised: {e}")
            matches = []
        # Drop duplicates across categories (first-wins)
        dedup = [m for m in matches if m not in seen]
        seen.update(dedup)
        raw_by_cat[cat] = dedup
        print(f"  [{cat:8s}] pattern={pattern:30s} → {len(matches):4d} matches "
              f"({len(dedup)} new, {(time.time()-t0)*1000:.0f} ms)")

    # ---- Step 2: cap by --max-per-category & query poses -------------------
    candidates: list[dict[str, Any]] = []
    pose_query_count = 0
    pose_query_skip = 0
    for cat, pattern, label_tpl in active_cats:
        names = raw_by_cat.get(cat, [])
        if args.max_per_category > 0:
            names = names[: args.max_per_category]
        for name in names:
            try:
                pose = client.simGetObjectPose(name)
            except Exception as e:
                pose_query_skip += 1
                if args.verbose:
                    print(f"  [skip] {name}: getPose raised {e}")
                continue
            pose_query_count += 1
            if not _is_valid_pose(pose):
                pose_query_skip += 1
                continue
            candidates.append({
                "name": name,
                "category": cat,
                "default_pose": _pose_to_dict(pose),
                "approx_size_m": None,
                "suggested_label": label_tpl,
            })

    print(f"\n[info] pose queries: {pose_query_count} ok, "
          f"{pose_query_skip} skipped (invalid/non-existent)")

    # ---- Step 3: summary ---------------------------------------------------
    cat_counts: dict[str, int] = defaultdict(int)
    for c in candidates:
        cat_counts[c["category"]] += 1

    return {
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
        "airsim_endpoint": f"{args.airsim_ip}:{args.airsim_port}",
        "total_candidates": len(candidates),
        "by_category": dict(cat_counts),
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Scout tracking targets in env_airsim_16.")
    parser.add_argument("--airsim-ip", default=os.environ.get("AIRSIM_IP", "127.0.0.1"))
    parser.add_argument("--airsim-port", type=int,
                        default=int(os.environ.get("AIRSIM_RPC_PORT", 41451)))
    parser.add_argument("--categories", default="",
                        help="Comma list of categories to keep (default: all known).")
    parser.add_argument("--max-per-category", type=int, default=50,
                        help="Cap candidates per category (0 = no cap).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Hard timeout (s) — AirVLN may crash mid-scan.")
    args = parser.parse_args()

    import signal

    def _on_timeout(_sig, _frm):
        print(f"\n[FATAL] scout timed out after {args.timeout}s. "
              "AirVLN 可能已 crash。", flush=True)
        sys.exit(124)

    signal.signal(signal.SIGALRM, _on_timeout)
    signal.alarm(int(args.timeout))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    result = run_scout(args)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output_dir / f"scene_targets_{ts}.json"
    latest = args.output_dir / "scene_targets_latest.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    # latest convenience link
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(out_path.name)
    except OSError:
        # symlink not allowed on this fs — fall back to a copy
        with latest.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 64)
    print(f"SCOUT RESULT — saved to: {out_path}")
    print("=" * 64)
    if "error" in result:
        print(f"ERROR: {result['error']}")
        return 1
    print(f"Total candidates : {result['total_candidates']}")
    for cat, n in sorted(result["by_category"].items(),
                         key=lambda kv: -kv[1]):
        print(f"  {cat:10s} : {n}")
    if result["candidates"]:
        print("\nFirst 5 candidates (peek):")
        for c in result["candidates"][:5]:
            p = c["default_pose"]
            print(f"  - {c['name']:40s} "
                  f"[{c['category']:8s}] "
                  f"pos=({p['x']:7.1f}, {p['y']:7.1f}, {p['z']:6.1f})  "
                  f"label='{c['suggested_label']}'")
    print(f"\nNext step:")
    print(f"  python flyseek_extend/scripts/smoke_test_teleport_and_capture.py")
    print("=" * 64)

    return 0 if result.get("total_candidates", 0) > 0 else 2


if __name__ == "__main__":
    sys.exit(main())

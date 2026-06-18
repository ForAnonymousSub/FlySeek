#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Render the FlySeek-vs-reactive-baseline comparison figures for an alley chase.

Consumes two episode directories produced by ``demo_adversary_chase`` for the
*same* seeded scene (one adaptive/FlySeek run, one reactive baseline run) and
writes publication-quality figures:

  * ``occlusion_risk_map.png`` — the standalone occlusion / track-loss risk
    field with intersection / alley / building-edge structure highlighted,
  * ``comparison.png``         — two-panel BEV (FlySeek | baseline) over the
    risk field with drone & target trajectories, camera frustums, candidate
    hiding zones, selected observation points, plus visibility & occlusion-risk
    timelines.

This reuses the existing offline PCD occupancy map + seg-building annotations;
it does **not** run AirSim or re-simulate anything.

Example::

    python flyseek_extend/scripts/viz_chase_compare.py \\
        --flyseek-dir  flyseek_extend/output/demo_alley_compare/flyseek_seed66 \\
        --baseline-dir flyseek_extend/output/demo_alley_compare/baseline_seed66 \\
        --env env_airsim_16 \\
        --seg-building-jsonl scene_data/seg_map/env_airsim_16.jsonl \\
        --out-dir flyseek_extend/output/demo_alley_compare/compare_seed66
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "flyseek_extend") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "flyseek_extend"))

from flyseek.adapters.pcd_occupancy import PcdOccupancyMap  # noqa: E402
from flyseek.utils.chase_compare_viz import (  # noqa: E402
    compute_occlusion_risk_field,
    compute_path_occlusion_risk,
    load_episode,
    region_bounds_from_episodes,
    render_comparison_figure,
    render_occlusion_risk_map,
)


def _alley_markers(occupancy, seg_jsonl: Path | None, keep_z: float):
    """Best-effort hutong (narrow alley) annotation for the risk map."""
    if not seg_jsonl:
        return None
    try:
        from flyseek.utils.alley_route import find_best_alley_scene
        from flyseek.utils.seg_buildings import SegBuildingMap
        seg = SegBuildingMap.from_jsonl(seg_jsonl, footprint_radius_m=10.0)
        alley, _anchor = find_best_alley_scene(
            occupancy, seg, keep_z=keep_z,
            max_corridor_width_m=12.0, min_depth_m=12.0,
        )
        if alley is None:
            return None
        markers = []
        entry = np.asarray(alley.entry_ned, dtype=np.float64).reshape(3)
        deep = np.asarray(alley.deep_ned, dtype=np.float64).reshape(3)
        markers.append((float(entry[0]), float(entry[1]), "alley entry"))
        markers.append((float(deep[0]), float(deep[1]), "alley deep"))
        return markers
    except Exception as e:  # pragma: no cover
        print(f"[warn] alley annotation skipped: {e}")
        return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flyseek-dir", type=Path, required=True)
    p.add_argument("--baseline-dir", type=Path, required=True)
    p.add_argument("--env", default="env_airsim_16")
    p.add_argument("--seg-building-jsonl", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--grid-step-m", type=float, default=2.0)
    p.add_argument("--frustum-every", type=int, default=18,
                   help="Draw a camera frustum wedge every N frames.")
    p.add_argument("--no-collision", action="store_true",
                   help="Skip the PCD occupancy map (risk field becomes empty).")
    args = p.parse_args()

    fly = load_episode(args.flyseek_dir, name="flyseek", label="FlySeek")
    base = load_episode(args.baseline_dir, name="baseline", label="Reactive baseline")
    if fly is None:
        print(f"[FATAL] no frames.jsonl in {args.flyseek_dir}")
        return 1
    if base is None:
        print(f"[FATAL] no frames.jsonl in {args.baseline_dir}")
        return 1
    print(f"[ok] FlySeek frames={fly.drone_xy.shape[0]}  "
          f"baseline frames={base.drone_xy.shape[0]}")

    occupancy: PcdOccupancyMap | None = None
    if not args.no_collision:
        try:
            occupancy = PcdOccupancyMap.load_or_build(REPO_ROOT, env_name=args.env)
            print("[ok] PCD occupancy ready")
        except Exception as e:
            print(f"[warn] PCD occupancy unavailable ({e}); risk field disabled")

    keep_z = float(np.median(fly.target_z)) if fly.target_z.size else -0.6
    bounds = region_bounds_from_episodes([fly, base], margin_m=35.0)
    print(f"[ok] region bounds (NED x/y) = "
          f"({bounds[0]:.0f},{bounds[1]:.0f},{bounds[2]:.0f},{bounds[3]:.0f})  "
          f"keep_z={keep_z:.2f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if occupancy is not None:
        eye = max(fly.metrics.get("follow_altitude", 12.0) if fly.metrics else 12.0,
                  float(np.median(-fly.drone_z)) if fly.drone_z.size else 12.0)
        print("[..] computing occlusion / track-loss risk field "
              f"(step={args.grid_step_m}m) — this scans the street grid")
        rf = compute_occlusion_risk_field(
            occupancy, bounds=bounds, keep_z=keep_z,
            step_m=float(args.grid_step_m),
        )
        print(f"[ok] risk field {rf.risk.shape}  "
              f"intersections={rf.intersection_pts.shape[0]}")

        print("[..] computing per-frame occlusion risk along both paths")
        fly.occlusion_risk_path = compute_path_occlusion_risk(
            fly, occupancy, drone_eye_agl_m=eye)
        base.occlusion_risk_path = compute_path_occlusion_risk(
            base, occupancy, drone_eye_agl_m=eye)

        markers = _alley_markers(occupancy, args.seg_building_jsonl, keep_z)
        risk_png = render_occlusion_risk_map(
            rf, out_dir / "occlusion_risk_map.png",
            target_xy=fly.target_xy, alley_markers=markers,
        )
        print(f"[ok] occlusion risk map → {risk_png}")

        cmp_png = render_comparison_figure(
            fly, base, rf, out_dir / "comparison.png",
            frustum_every=int(args.frustum_every),
        )
        print(f"[ok] comparison figure → {cmp_png}")
    else:
        print("[warn] no occupancy — skipping risk-field figures")

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    for ep in (fly, base):
        m = ep.metrics or {}
        print(f"  {ep.label:20s} success={m.get('tracking_success')}  "
              f"vis_ratio={m.get('target_visibility_ratio')}  "
              f"los_continuity={m.get('line_of_sight_continuity')}  "
              f"lost_ratio={m.get('target_lost_ratio')}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

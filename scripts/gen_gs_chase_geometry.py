#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Phase A2/A3: build + visualise the GS UAV-tracks-car geometry (no render).

Loads the chase config, builds the car route and the UAV tracking trajectory
(camera yaw/pitch solved to keep the car centered), reports how often the car
stays in frame, writes trajectories.json, and draws a BEV overlay (car route,
UAV route, building footprint, good-region box) for a quick sanity check.

Usage:
    python flyseek_extend/scripts/gen_gs_chase_geometry.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from flyseek.adapters.pcd_occupancy import PcdOccupancyMap
from flyseek.render.gs_camera import Intrinsics
from flyseek.render.gs_camera import bridge_extrinsics  # noqa: F401 (used indirectly)
from flyseek.render.gs_chase import (build_car_track, build_car_track_policy,
                                     build_uav_track)
from flyseek.render.pcd_depth import SceneGeometry

REPO_ROOT = Path(__file__).resolve().parents[2]
# fallback when cameras.bin is missing
_DEFAULT_INTR = "0 PINHOLE 2048 1536 1335.1645731732658 1335.4075753200657 1024.0 768.0"


def _load_intrinsics(env: str, render_scale: float) -> Intrinsics:
    """Read per-env COLMAP cameras.bin (same path gs_bridge uses)."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts" / "sim"))
    from gs_bridge import load_colmap_intrinsics  # noqa: WPS433

    cams = REPO_ROOT / "envs" / "gs" / env / "camera_calibration" / "aligned" / "sparse" / "0" / "cameras.bin"
    s = load_colmap_intrinsics(str(cams), scale=render_scale)
    if s is None:
        print(f"[warn] no cameras.bin for {env}, using default intrinsics")
        return Intrinsics.from_colmap_str(_DEFAULT_INTR).scaled(render_scale)
    return Intrinsics.from_colmap_str(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(
        REPO_ROOT / "flyseek_extend" / "configs" / "gs_chase_env_gs_urban_dense.yaml"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    env = cfg["env"]
    intr = _load_intrinsics(env, float(cfg["camera"]["render_scale"]))

    geom = SceneGeometry(REPO_ROOT / "scene_data" / "pcd_map" / f"{env}.pcd")
    cars_json = REPO_ROOT / "flyseek_extend" / "output" / "gs_debug" / f"{env}_cars.json"
    if cars_json.exists():
        max_bz = float(cfg.get("car_anchor_max_base_z", -12.0))
        geom.set_car_anchors(json.loads(cars_json.read_text()), max_base_z=max_bz)
        print(f"loaded {0 if geom.car_anchors is None else len(geom.car_anchors)} car height anchors (base_z<={max_bz})")
    region = cfg.get("good_region")
    motion = str(cfg.get("target_motion", "waypoints"))
    if motion == "policy":
        tb = cfg.get("target", {})
        print(f"target motion: policy ({tb.get('behavior')}/{tb.get('difficulty')}, "
              f"seed={tb.get('seed')}, {tb.get('episode_seconds')}s @ {tb.get('dt')}s)")
        car = build_car_track_policy(cfg, geom=geom)
    else:
        print("target motion: fixed waypoints")
        car = build_car_track(cfg, geom=geom)
    uav = build_uav_track(car, cfg, intr, geom=geom, region=region)

    in_view = sum(u.car_in_view for u in uav)
    print("=" * 60)
    print(f"env {env}: car frames={len(car)} uav frames={len(uav)}")
    print(f"car z (ground-hugged): [{min(c.pos[2] for c in car):.1f}, {max(c.pos[2] for c in car):.1f}]")
    print(f"car visible (in-frame + LOS clear): {in_view}/{len(uav)} ({100*in_view/len(uav):.0f}%)")
    pit = [u.pitch_deg for u in uav]
    print(f"uav pitch range: [{min(pit):.0f}, {max(pit):.0f}] deg")
    print("=" * 60)

    out_dir = Path(args.out) if args.out else REPO_ROOT / "flyseek_extend" / "output" / "gs_debug" / "chase_geom"
    out_dir.mkdir(parents=True, exist_ok=True)

    traj = {
        "env": env,
        "frame": "gs_world",
        "ground_z": cfg["ground_z"],
        "target_motion": motion,
        "target_meta": cfg.get("target", {}) if motion == "policy" else None,
        "car_model": cfg.get("car_model", {}),
        "dt": float(cfg.get("target", {}).get("dt", 0.2)) if motion == "policy" else None,
        "target_trajectory": [
            {"t": c.t, "pos": c.pos.tolist(), "heading": c.heading} for c in car
        ],
        "uav_trajectory": [
            {"t": u.t, "pos": u.pos.tolist(), "yaw_input": u.yaw_input,
             "pitch_deg": u.pitch_deg, "car_uv": u.car_uv, "car_in_view": u.car_in_view}
            for u in uav
        ],
    }
    (out_dir / "trajectories.json").write_text(json.dumps(traj, indent=2))
    print(f"trajectories -> {out_dir / 'trajectories.json'}")

    # ---- BEV overlay ----
    occ = PcdOccupancyMap.load_or_build(REPO_ROOT, env_name=env)
    x0, y0, vw = occ._x0, occ._y0, occ._vw
    nx = int(np.floor((occ._x1 - x0) / vw)) + 1
    ny = int(np.floor((occ._y1 - y0) / vw)) + 1
    img = np.full((ny, nx, 3), 25, dtype=np.uint8)
    for (ix, iy) in occ._bev2d:
        if 0 <= ix < nx and 0 <= iy < ny:
            img[iy, ix] = (95, 95, 105)
    im = Image.fromarray(np.flipud(img)).convert("RGB")
    scale = max(1, int(round(1000 / max(nx, ny))))
    im = im.resize((nx * scale, ny * scale), Image.NEAREST)
    dr = ImageDraw.Draw(im)

    def to_px(x, y):
        ix = (x - x0) / vw
        iy = (y - y0) / vw
        return ix * scale, (ny - 1 - iy) * scale

    gr = cfg["good_region"]
    bx = [to_px(gr["xmin"], gr["ymin"]), to_px(gr["xmax"], gr["ymin"]),
          to_px(gr["xmax"], gr["ymax"]), to_px(gr["xmin"], gr["ymax"])]
    dr.polygon(bx, outline=(240, 60, 60))

    cpx = [to_px(c.pos[0], c.pos[1]) for c in car]
    dr.line(cpx, fill=(40, 200, 90), width=3)
    upx = [to_px(u.pos[0], u.pos[1]) for u in uav]
    dr.line(upx, fill=(80, 160, 255), width=2)
    # connect uav->car every few frames + mark in/out of view
    for c, u in list(zip(car, uav))[::3]:
        a = to_px(u.pos[0], u.pos[1]); b = to_px(c.pos[0], c.pos[1])
        dr.line([a, b], fill=(255, 220, 60) if u.car_in_view else (255, 80, 80))
    dr.ellipse([cpx[0][0]-4, cpx[0][1]-4, cpx[0][0]+4, cpx[0][1]+4], fill=(0, 255, 0))   # car start
    dr.ellipse([cpx[-1][0]-4, cpx[-1][1]-4, cpx[-1][0]+4, cpx[-1][1]+4], fill=(255, 0, 0))  # car end

    out_png = out_dir / "chase_bev.png"
    im.save(out_png)
    print(f"BEV -> {out_png}  (green=car route, blue=uav route, yellow/red=LOS in/out view)")


if __name__ == "__main__":
    main()

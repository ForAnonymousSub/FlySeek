#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""FlySeek-Bench batch episode generation entry point.

Generates one or more standardized episodes. Each episode runs the full pipeline:
scene init -> UAV/target init -> instruction -> target behavior policy -> render
(existing OpenFly teleport pipeline) -> per-frame visibility -> standardized frame
metadata -> trajectories -> expert (visibility-aware) annotation -> metrics.

Output layout per episode::

    <episode_root>/
        images/              # one PNG per frame (frame_XXXX.png)
        metadata.json
        frames.jsonl
        trajectories.json
        instruction.json
        visibility.json
        metrics.json
        config.yaml

``--dry_run`` performs the entire pipeline OFFLINE (no AirSim): the UAV reference
is the expert viewpoint sequence, visibility is computed geometrically, and small
placeholder PNGs stand in for rendered images so the structure and sanity checks
are fully exercised without a simulator.

CLI::

    python -m flyseek_bench.run_generate_episodes \\
        --scene_id env_airsim_16 --difficulty medium --behavior sharp_turn \\
        --seed 42 --num_episodes 3 --output_dir flyseek_extend/output/bench

Equivalently runnable as a file:
    python flyseek_extend/flyseek_bench/run_generate_episodes.py ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "flyseek_extend") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "flyseek_extend"))

from flyseek.bench.schema import CameraConfig, EpisodeMetadata, FrameMetadata  # noqa: E402
from flyseek.bench.export import (  # noqa: E402
    append_frame_jsonl,
    save_metadata_json,
)
from flyseek.bench.instruction_generator import (  # noqa: E402
    InstructionGenerator,
    attributes_from_label,
    write_instruction_json,
)
from flyseek.bench.target_policy import generate_target_waypoints  # noqa: E402
from flyseek.bench.expert_trajectory import (  # noqa: E402
    ExpertTrajectoryConfig,
    ExpertViewpointPlanner,
    save_trajectories,
)
from flyseek.bench.visibility import VisibilityEvaluator  # noqa: E402
from flyseek.bench.metrics import evaluate_episode_dir  # noqa: E402

BEHAVIORS = ("direct_escape", "sharp_turn", "detour_feint", "occlusion_seeking")
DIFFICULTIES = ("easy", "medium", "hard")
DEFAULT_OUTPUT = REPO_ROOT / "flyseek_extend" / "output" / "bench"


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #
def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "scene_id": args.scene_id,
        "difficulty": args.difficulty,
        "behavior": args.behavior,
        "seed": args.seed,
        "num_episodes": args.num_episodes,
        "output_dir": str(args.output_dir),
        "dry_run": bool(args.dry_run),
        "render": {
            "tick_hz": 20.0,
            "duration_s": None,
            "num_frames": 60,
            "step_dt": 0.2,
            "follow_distance_m": 12.0,
            "follow_altitude_m": 18.0,
            "camera_hfov_deg": 70.0,
            "image_width": 256,
            "image_height": 144,
        },
    }
    if args.config and Path(args.config).exists():
        try:
            import yaml  # type: ignore
            loaded = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
            for k, v in loaded.items():
                if k == "render" and isinstance(v, dict):
                    cfg["render"].update(v)
                else:
                    cfg[k] = v
        except Exception as e:
            print(f"[warn] failed to read --config {args.config}: {e}")
    return cfg


def _write_config_yaml(episode_root: Path, cfg: dict[str, Any]) -> Path:
    path = episode_root / "config.yaml"
    try:
        import yaml  # type: ignore
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                        encoding="utf-8")
    except Exception:
        path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Visibility summary                                                          #
# --------------------------------------------------------------------------- #
def _write_visibility_json(episode_root: Path) -> Path:
    frames_path = episode_root / "frames.jsonl"
    per_frame: list[dict] = []
    reason_counts: dict[str, int] = {}
    visible = 0
    total = 0
    if frames_path.exists():
        for line in frames_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            f = json.loads(line)
            total += 1
            vis = bool(f.get("target_visible", False))
            visible += int(vis)
            reason = (f.get("extra") or {}).get("vis_reason", "")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            per_frame.append({
                "frame_id": f.get("frame_id"),
                "target_visible": vis,
                "in_camera_frustum": f.get("in_camera_frustum"),
                "line_of_sight_clear": f.get("line_of_sight_clear"),
                "visibility_score": f.get("visibility_score"),
                "vis_reason": reason,
            })
    summary = {
        "total_frames": total,
        "visible_frames": visible,
        "target_visibility_ratio": round(visible / total, 4) if total else 0.0,
        "vis_reason_counts": reason_counts,
        "per_frame": per_frame,
    }
    path = episode_root / "visibility.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Placeholder image writer (dry-run)                                          #
# --------------------------------------------------------------------------- #
def _write_placeholder_png(path: Path, w: int = 16, h: int = 16) -> bool:
    try:
        import cv2  # type: ignore
        cv2.imwrite(str(path), np.zeros((h, w, 3), dtype=np.uint8))
        return True
    except Exception:
        try:
            from PIL import Image  # type: ignore
            Image.new("RGB", (w, h)).save(path)
            return True
        except Exception:
            # Last resort: write a tiny valid (1x1) PNG by bytes.
            path.write_bytes(bytes.fromhex(
                "89504e470d0a1a0a0000000d49484452000000010000000108020000"
                "00907753de0000000c4944415408d76360000002000154a24f600000"
                "0000049454e44ae426082"
            ))
            return True


# --------------------------------------------------------------------------- #
# Dry-run generation (no simulator)                                           #
# --------------------------------------------------------------------------- #
def _load_occupancy(scene_id: str):
    try:
        from flyseek.adapters.pcd_occupancy import PcdOccupancyMap
        return PcdOccupancyMap.load_or_build(REPO_ROOT, env_name=scene_id, rebuild=False)
    except Exception:
        return None


def _dry_run_generate(episode_root: Path, cfg: dict[str, Any], seed: int) -> None:
    r = cfg["render"]
    behavior = cfg["behavior"]
    difficulty = cfg["difficulty"]
    scene_id = cfg["scene_id"]
    n = int(r["num_frames"])
    dt = float(r["step_dt"])
    fd = float(r["follow_distance_m"])
    fa = float(r["follow_altitude_m"])
    hfov = float(r["camera_hfov_deg"])
    iw, ih = int(r["image_width"]), int(r["image_height"])

    images_dir = episode_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    occupancy = _load_occupancy(scene_id)

    # (1-2) scene + initial poses (synthetic, documented for dry-run).
    target0 = [0.0, 0.0, -0.3]
    uav0 = [-fd, 0.0, -fa, 0.0]

    # (4) target behavior policy -> waypoints -> target trajectory.
    waypoints = generate_target_waypoints(
        initial_target_pose=target0, initial_uav_pose=uav0,
        behavior_type=behavior, difficulty=difficulty, seed=seed,
        scene_context=({"occupancy": occupancy} if occupancy is not None else None),
        n_waypoints=n, step_dt=dt,
    )
    wp = np.asarray(waypoints, dtype=np.float64)
    if wp.shape[0] < n:  # pad by repeating the last point
        pad = np.repeat(wp[-1:], n - wp.shape[0], axis=0)
        wp = np.concatenate([wp, pad], axis=0)
    wp = wp[:n]
    target_traj = [
        {"t": i * dt, "pos": [float(x) for x in wp[i]],
         "vel": [float(v) for v in ((wp[min(i + 1, n - 1)] - wp[max(i - 1, 0)])
                                    / max((min(i + 1, n - 1) - max(i - 1, 0)) * dt, 1e-6))]}
        for i in range(n)
    ]

    # (9) expert (visibility-aware) viewpoints; also serve as the dry-run UAV.
    expert_cfg = ExpertTrajectoryConfig(
        follow_distance_m=fd, follow_altitude_m=fa, hfov_deg=hfov, plan_stride=1,
    )
    planner = ExpertViewpointPlanner(
        config=expert_cfg,
        scene_context=({"occupancy": occupancy} if occupancy is not None else {}),
        seed=seed,
    )
    expert_out = planner.plan(target_traj)  # uav filled below from viewpoints
    viewpoints = expert_out["expert_viewpoints"]

    uav_traj = [
        {"t": vp["t"], "pos": vp["position"], "heading": vp["heading"]}
        for vp in viewpoints
    ]
    expert_out["uav_trajectory"] = uav_traj
    save_trajectories(expert_out, episode_root / "trajectories.json")

    # (3) instruction.
    label = "a small car"
    ctx: dict[str, Any] = {}
    if occupancy is not None:
        ctx["motion"] = "the street"
        if behavior == "occlusion_seeking":
            ctx["occlusion"] = "an occluded street"
    instr = InstructionGenerator(seed=seed).generate(
        target_class=label, target_attributes=attributes_from_label(label),
        initial_context=ctx, behavior_type=behavior,
        difficulty_level=difficulty, seed=seed,
    )
    write_instruction_json(instr, episode_root / "instruction.json")

    # (5-8) per-frame: render placeholder, evaluate visibility, write frame meta.
    cam_cfg = {"name": "front_custom", "hfov_deg": hfov, "pitch_deg": 55.0,
               "body_forward_m": 0.45, "body_down_m": 0.25,
               "width": iw, "height": ih}
    vis_eval = VisibilityEvaluator(max_range_m=100.0, drone_eye_agl_m=fa)
    frames_path = episode_root / "frames.jsonl"
    if frames_path.exists():
        frames_path.unlink()

    for i in range(n):
        tpos = list(wp[i])
        vp = viewpoints[i]
        upose = list(vp["position"]) + [float(vp["heading"])]
        img_rel = f"images/frame_{i:04d}.png"
        _write_placeholder_png(images_dir / f"frame_{i:04d}.png", iw, ih)

        vstd = vis_eval.evaluate_frame(
            uav_pose=upose, target_pose=tpos, camera_config=cam_cfg,
            scene_context=({"occupancy": occupancy} if occupancy is not None
                           else {}),
        )
        tvel = target_traj[i]["vel"]
        uvel = ([0.0, 0.0, 0.0] if i == 0 else
                [(upose[k] - uav_traj[i - 1]["pos"][k]) / dt for k in range(3)])
        collision = False
        fm = FrameMetadata(
            frame_id=i, image_path=img_rel, step_index=i, timestamp=round(i * dt, 4),
            uav_pose=[float(x) for x in upose],
            target_pose=[float(x) for x in tpos] + [0.0],
            uav_velocity=[float(x) for x in uvel],
            target_velocity=[float(x) for x in tvel],
            target_visible=bool(vstd["target_visible"]),
            in_camera_frustum=vstd["in_camera_frustum"],
            line_of_sight_clear=vstd["line_of_sight_clear"],
            visibility_score=vstd["visibility_score"],
            distance_to_target=float(vstd["distance_to_target"]),
            relative_bearing=float(vstd["relative_bearing"]),
            occlusion_risk=vstd["occlusion_risk"],
            selected_viewpoint=[float(x) for x in vp["position"]],
            collision=collision,
            target_behavior_type=behavior,
            difficulty_level=difficulty,
            extra={"vis_reason": vstd["visibility_source"].split("|")[0],
                   "visibility_source": vstd["visibility_source"]},
        )
        append_frame_jsonl(fm, frames_path)

    # metadata.json (with the instruction).
    meta = EpisodeMetadata(
        episode_id=episode_root.name,
        scene_id=scene_id,
        difficulty_level=difficulty,
        target_behavior_type=behavior,
        target_class=label,
        instruction=instr["instruction"],
        random_seed=int(seed),
        max_steps=n,
        camera_config=CameraConfig(name="front_custom", hfov_deg=hfov,
                                   pitch_deg=55.0, width=iw, height=ih),
        uav_initial_pose=uav0,
        target_initial_pose=target0 + [0.0],
        environment_summary={"env": scene_id,
                             "occupancy_available": occupancy is not None,
                             "dry_run": True},
        occluder_summary={"source": "pcd_occupancy" if occupancy else "none"},
    )
    save_metadata_json(meta, episode_root / "metadata.json")


# --------------------------------------------------------------------------- #
# Real (simulator) generation                                                 #
# --------------------------------------------------------------------------- #
def _real_generate(episode_root: Path, cfg: dict[str, Any], seed: int) -> None:
    from flyseek.pipeline.single_episode import run_episode

    r = cfg["render"]
    overrides: dict[str, Any] = {
        "env": cfg["scene_id"],
        "target_behavior": cfg["behavior"],
        "target_policy_difficulty": cfg["difficulty"],
        "seed": seed,
        "auto_from_scout": True,
        "output": episode_root.parent,
        "episode_tag": episode_root.name,
        "follow_distance": float(r["follow_distance_m"]),
        "follow_altitude": float(r["follow_altitude_m"]),
        "camera_hfov_deg": float(r["camera_hfov_deg"]),
    }
    if r.get("duration_s"):
        overrides["duration"] = float(r["duration_s"])
    if r.get("tick_hz"):
        overrides["tick_hz"] = float(r["tick_hz"])
    # run_episode -> demo loop writes metadata/frames/trajectories/instruction/metrics.
    run_episode(overrides, write_summary=True)
    # Assemble images/ from the demo's per-frame RGB captures.
    _assemble_images_from_frames(episode_root)


def _assemble_images_from_frames(episode_root: Path) -> None:
    import shutil
    images_dir = episode_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = episode_root / "frames"
    srcs = sorted(frames_dir.glob("frame_*_rgb.png")) if frames_dir.exists() else []
    if not srcs:  # fall back to OpenFly-style image_<idx>.png at root
        srcs = sorted(episode_root.glob("image_*.png"))
    for i, src in enumerate(srcs):
        try:
            shutil.copy2(src, images_dir / f"frame_{i:04d}.png")
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Sanity checks                                                               #
# --------------------------------------------------------------------------- #
def _sanity_check(episode_root: Path, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    issues: list[str] = []

    required = ["metadata.json", "frames.jsonl", "trajectories.json",
                "instruction.json", "visibility.json", "metrics.json", "config.yaml"]
    for name in required:
        if not (episode_root / name).exists():
            issues.append(f"missing {name}")

    # image count == frame metadata count
    n_images = len(list((episode_root / "images").glob("*.png"))) \
        if (episode_root / "images").exists() else 0
    fpath = episode_root / "frames.jsonl"
    n_frames = sum(1 for ln in fpath.read_text(encoding="utf-8").splitlines()
                   if ln.strip()) if fpath.exists() else 0
    if n_images != n_frames:
        issues.append(f"image_count({n_images}) != frame_count({n_frames})")

    # metrics valid
    mpath = episode_root / "metrics.json"
    if mpath.exists():
        try:
            m = json.loads(mpath.read_text(encoding="utf-8"))
            if not isinstance(m.get("tracking_success"), bool):
                issues.append("metrics.tracking_success not bool")
            vr = m.get("target_visibility_ratio")
            if not (isinstance(vr, (int, float)) and 0.0 <= vr <= 1.0):
                issues.append("metrics.target_visibility_ratio out of [0,1]")
            if int(m.get("total_frames", 0)) <= 0:
                issues.append("metrics.total_frames <= 0")
        except Exception as e:
            issues.append(f"metrics.json unreadable: {e}")

    # seed logged
    seed_logged = False
    cpath = episode_root / "config.yaml"
    if cpath.exists() and str(seed) in cpath.read_text(encoding="utf-8"):
        seed_logged = True
    meta_path = episode_root / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("random_seed") == seed:
                seed_logged = True
        except Exception:
            pass
    if not seed_logged:
        issues.append("random seed not logged")

    return {
        "episode_dir": str(episode_root),
        "passed": len(issues) == 0,
        "n_images": n_images,
        "n_frames": n_frames,
        "seed": seed,
        "issues": issues,
    }


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def generate_episode(cfg: dict[str, Any], episode_index: int, seed: int) -> dict[str, Any]:
    episode_id = (f"{cfg['scene_id']}_{cfg['behavior']}_{cfg['difficulty']}"
                  f"_seed{seed}_{episode_index:03d}")
    episode_root = Path(cfg["output_dir"]) / episode_id
    episode_root.mkdir(parents=True, exist_ok=True)

    if cfg["dry_run"]:
        _dry_run_generate(episode_root, cfg, seed)
    else:
        _real_generate(episode_root, cfg, seed)

    # metrics.json (compute/refresh from the written frames + metadata).
    try:
        evaluate_episode_dir(episode_root, difficulty=cfg["difficulty"], write=True)
    except Exception as e:
        print(f"[warn] metrics computation failed: {e}")

    # visibility.json + config.yaml (+ seed logging).
    _write_visibility_json(episode_root)
    ep_cfg = {**cfg, "episode_id": episode_id, "episode_index": episode_index,
              "seed": seed}
    _write_config_yaml(episode_root, ep_cfg)

    report = _sanity_check(episode_root, cfg, seed)
    status = "PASS" if report["passed"] else "FAIL"
    print(f"[{status}] {episode_id}  frames={report['n_frames']} "
          f"images={report['n_images']}"
          + ("" if report["passed"] else f"  issues={report['issues']}"))
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FlySeek-Bench batch episode generator.")
    p.add_argument("--scene_id", "--scene-id", dest="scene_id", default="env_airsim_16")
    p.add_argument("--difficulty", choices=list(DIFFICULTIES), default="medium")
    p.add_argument("--behavior", choices=list(BEHAVIORS), default="direct_escape")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_episodes", "--num-episodes", dest="num_episodes",
                   type=int, default=1)
    p.add_argument("--output_dir", "--output-dir", dest="output_dir",
                   type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--config", default=None, help="Optional YAML config (merged).")
    p.add_argument("--dry_run", "--dry-run", dest="dry_run", action="store_true",
                   help="Offline generation (no AirSim); placeholder images.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = resolve_config(args)
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    print(f"[gen] scene={cfg['scene_id']} behavior={cfg['behavior']} "
          f"difficulty={cfg['difficulty']} base_seed={cfg['seed']} "
          f"episodes={cfg['num_episodes']} dry_run={cfg['dry_run']}")

    reports: list[dict] = []
    for i in range(int(cfg["num_episodes"])):
        seed = int(cfg["seed"]) + i
        try:
            reports.append(generate_episode(cfg, i, seed))
        except Exception as e:
            print(f"[FAIL] episode {i} (seed={seed}): {type(e).__name__}: {e}")
            reports.append({"episode_index": i, "seed": seed, "passed": False,
                            "issues": [f"{type(e).__name__}: {e}"]})

    manifest = {
        "scene_id": cfg["scene_id"], "behavior": cfg["behavior"],
        "difficulty": cfg["difficulty"], "base_seed": cfg["seed"],
        "num_episodes": cfg["num_episodes"], "dry_run": cfg["dry_run"],
        "output_dir": str(cfg["output_dir"]),
        "episodes": reports,
        "passed": sum(1 for r in reports if r.get("passed")),
    }
    (Path(cfg["output_dir"]) / "batch_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    n_pass = manifest["passed"]
    print(f"[done] {n_pass}/{cfg['num_episodes']} episodes passed sanity checks "
          f"-> {cfg['output_dir']}")
    return 0 if n_pass == int(cfg["num_episodes"]) else 1


if __name__ == "__main__":
    sys.exit(main())

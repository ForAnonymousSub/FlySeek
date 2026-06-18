# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""End-to-end demo: adversarial car evades a chasing drone.

This is the first **closed-loop** Phase 1 demo. It plugs the new
``flyseek.adversary`` module into the proven teleport pipeline from Phase 0.5
and records 15-60 s of frames showing the car running away from the drone in
an S-curve pattern.

Pipeline per simulation tick (default 5 Hz):

    1. Read drone pose (we move it ourselves below — no apiControl).
    2. Adversary decides next target velocity / heading.
    3. Integrate target state (kinematics-limited).
    4. Compute drone's next pose with a *dumb* chase controller (NOT PID — that
       comes in the next module). The drone tries to stay ``follow_distance``
       behind the target, ``follow_altitude`` above ground, yawed toward target.
    5. Teleport both via simSetVehiclePose / simSetObjectPose.
    6. Capture RGB (always) + optional Depth/Seg (every ``--capture-modalities-stride``).
    7. Append to trajectory.jsonl.

After the run, the target is restored to its original world pose.

Whitelist policy (SKILL.md §6.1):
    - simSetVehiclePose, simGetVehiclePose, simSetObjectPose, simGetObjectPose,
      simSetCameraPose, simGetImages, simListSceneObjects ✓
    NOT used: enableApiControl, armDisarm, moveByVelocity*

Quickstart:
    # 1. scout a target (or reuse last)
    python flyseek_extend/scripts/scout_scene_targets.py

    # 2. run the 30s demo
    python flyseek_extend/scripts/demo_adversary_chase.py \\
        --auto-from-scout --duration 30 --difficulty medium

    # 3. eyeball the frames
    ls flyseek_extend/output/demo_adversary_chase/<timestamp>/frames/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT / "flyseek_extend") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "flyseek_extend"))
if str(REPO_ROOT / "flyseek_extend" / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "flyseek_extend" / "scripts"))

from flyseek.adapters.pcd_occupancy import PcdOccupancyMap  # noqa: E402
from flyseek.utils.coords import airsim_altitude_m, airsim_ned_to_map  # noqa: E402
from flyseek.adversary import (  # noqa: E402
    AgentAction,
    DroneState,
    PlayBox,
    TargetState,
    create_adversarial_agent,
    horizontal_distance,
    integrate_target,
    wrap_to_pi,
)
from flyseek.adapters.output_writer import EpisodeWriter  # noqa: E402
from flyseek.expert.adaptive_tracker import AdaptiveTracker  # noqa: E402
from flyseek.expert.drone_altitude import OpenFlyDroneAltitude  # noqa: E402
from flyseek.expert.tracking_drone import TrackingDroneController  # noqa: E402
from flyseek.utils.street_motion import StreetMotionHelper, stabilize_car_state  # noqa: E402
from flyseek.utils.target_init import (  # noqa: E402
    _heading_from_quaternion,
    apply_init_pose_to_sim,
    resolve_target_init_pose,
    score_init_pose_ned,
)
from flyseek.utils.target_init_presets import (  # noqa: E402
    default_profile_name,
    load_target_init_profile,
)
from flyseek.utils.visibility import fov_centering_offset_xy  # noqa: E402
from flyseek.utils.visibility import target_visible, visibility_status  # noqa: E402
from flyseek.utils.seg_bbox import bbox_from_segmentation, project_ned_to_pixel  # noqa: E402
from flyseek.bench.schema import CameraConfig, EpisodeMetadata, FrameMetadata  # noqa: E402
from flyseek.bench.export import append_frame_jsonl, save_metadata_json  # noqa: E402
from flyseek.bench.visibility import VisibilityEvaluator  # noqa: E402
from flyseek.bench.target_policy import BEHAVIOR_TYPES, create_target_policy  # noqa: E402
from flyseek.utils.occlusion_route import occlusion_route_kwargs_from_args  # noqa: E402
from flyseek.bench.instruction_generator import (  # noqa: E402
    InstructionGenerator,
    attributes_from_label,
    write_instruction_json,
)
from flyseek.bench.expert_trajectory import (  # noqa: E402
    ExpertTrajectoryConfig,
    ExpertViewpointPlanner,
    save_trajectories,
)
from flyseek.bench.metrics import evaluate_episode_dir as _bench_eval_episode  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "flyseek_extend" / "output" / "demo_adversary_chase"
DEFAULT_SCOUT_FILE = (
    REPO_ROOT / "flyseek_extend" / "output" / "assets" / "scene_targets_latest.json"
)


# --------------------------------------------------------------------------- #
# Records                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class FrameRecord:
    frame_idx: int
    timestamp_s: float
    drone_pose_world: dict[str, float]
    target_pose_world: dict[str, float]
    target_velocity: list[float]
    drone_distance_m: float
    drone_bearing_to_target_rad: float
    adversary_log: dict[str, Any] = field(default_factory=dict)
    tracker_mode: str = ""
    target_visible: bool = True
    images_saved: dict[str, str] = field(default_factory=dict)


@dataclass
class DemoReport:
    timestamp: str
    airsim_endpoint: str
    target_name: str | None = None
    target_label: str | None = None
    difficulty: str = "medium"
    duration_s: float = 0.0
    sim_dt: float = 0.0
    frames_captured: int = 0
    output_dir: str = ""
    initial_drone_pose: dict[str, float] = field(default_factory=dict)
    initial_target_pose: dict[str, float] = field(default_factory=dict)
    final_drone_target_distance_m: float = 0.0
    target_restored: bool = False
    success: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# AirSim helpers                                                              #
# --------------------------------------------------------------------------- #
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


def _make_airsim_pose(x: float, y: float, z: float, yaw_rad: float):
    import airsim  # type: ignore
    return airsim.Pose(
        airsim.Vector3r(x, y, z),
        airsim.to_quaternion(0.0, 0.0, yaw_rad),
    )


def _make_tracking_camera_pose(args):
    """Nose-mounted downward camera — keeps propellers out of frame."""
    import airsim  # type: ignore
    return airsim.Pose(
        airsim.Vector3r(
            float(args.camera_body_forward_m),
            0.0,
            float(args.camera_body_down_m),
        ),
        airsim.to_quaternion(math.radians(-float(args.camera_pitch_deg)), 0.0, 0.0),
    )


def _apply_tracking_camera(client, args) -> str:
    """Set a fixed body-relative downward camera; returns camera name used."""
    cam_pose = _make_tracking_camera_pose(args)
    chosen = args.camera_name
    for cam in (args.camera_name, "front_custom", "0"):
        try:
            client.simSetCameraPose(cam, cam_pose)
            chosen = cam
            break
        except Exception:
            continue
    return chosen


def _make_topdown_camera_pose():
    """Camera pointing straight down (pitch -90°), mounted at vehicle origin."""
    import airsim  # type: ignore
    return airsim.Pose(
        airsim.Vector3r(0.0, 0.0, 0.0),
        airsim.to_quaternion(math.radians(-90.0), 0.0, 0.0),
    )


class _TopdownAnchor:
    """Low-pass filter the (x, y, yaw) the overhead camera teleports to.

    The drivable-cell snapping done every tick by ``stabilize_car_state``
    introduces ~1 voxel of XY jitter in ``target.position``, and
    ``target.heading`` swings whenever the car re-aligns to a new route
    segment. Captured directly, these tiny per-tick wobbles compound across
    consecutive top-down frames and look like global camera shake. We EMA the
    anchor pose and (by default) lock the yaw to 0° (world north up) so the
    overhead video stays steady regardless of car wiggle.
    """

    def __init__(self, xy_tau_s: float = 0.6, yaw_tau_s: float = 1.0,
                 fix_yaw_north: bool = True) -> None:
        self.xy_tau = max(1e-3, float(xy_tau_s))
        self.yaw_tau = max(1e-3, float(yaw_tau_s))
        self.fix_yaw_north = bool(fix_yaw_north)
        self._x: float = 0.0
        self._y: float = 0.0
        self._yaw: float = 0.0
        self._has_state: bool = False

    def update(self, target_state, dt: float) -> tuple[float, float, float]:
        if not self._has_state:
            self._x = float(target_state.position[0])
            self._y = float(target_state.position[1])
            self._yaw = float(target_state.heading)
            self._has_state = True
            return self._x, self._y, (0.0 if self.fix_yaw_north else self._yaw)
        ax = float(np.clip(dt / self.xy_tau, 0.0, 1.0))
        self._x += ax * (float(target_state.position[0]) - self._x)
        self._y += ax * (float(target_state.position[1]) - self._y)
        if self.fix_yaw_north:
            return self._x, self._y, 0.0
        ay = float(np.clip(dt / self.yaw_tau, 0.0, 1.0))
        self._yaw = wrap_to_pi(
            self._yaw + ay * wrap_to_pi(float(target_state.heading) - self._yaw)
        )
        return self._x, self._y, self._yaw


def _capture_topdown_frame(
    client,
    args,
    target_state,
    drone_state,
    chase_cam_pose,
    frame_path: Path,
    *,
    anchor_xy_yaw: tuple[float, float, float] | None = None,
) -> bool:
    """Briefly teleport drone above the car, snap a top-down RGB, restore drone.

    ``anchor_xy_yaw`` overrides the raw (target.x, target.y, target.heading)
    so the caller can feed an EMA-smoothed anchor (kills tick-level jitter).

    Side effects (must be undone before next tick):
        - drone pose moves to (anchor.x, anchor.y, -altitude)
        - camera pose tilts to straight down
    Both are restored at the end.
    """
    import airsim  # type: ignore

    topdown_altitude_ned = -abs(float(args.topdown_altitude))
    if anchor_xy_yaw is not None:
        ax, ay, ayaw = anchor_xy_yaw
    else:
        ax = float(target_state.position[0])
        ay = float(target_state.position[1])
        ayaw = float(target_state.heading)
    overhead_pose = airsim.Pose(
        airsim.Vector3r(ax, ay, topdown_altitude_ned),
        airsim.to_quaternion(0.0, 0.0, ayaw),
    )
    chase_pose = airsim.Pose(
        airsim.Vector3r(
            float(drone_state.position[0]),
            float(drone_state.position[1]),
            float(drone_state.position[2]),
        ),
        airsim.to_quaternion(0.0, 0.0, float(drone_state.heading)),
    )
    try:
        client.simSetVehiclePose(overhead_pose, ignore_collision=True)
        client.simSetCameraPose(args.camera_name, _make_topdown_camera_pose())
        resp = client.simGetImages([
            airsim.ImageRequest(args.camera_name, airsim.ImageType.Scene, False, False),
        ])
        ok = bool(resp) and _save_image(resp[0], frame_path, "topdown")
    except Exception as e:
        print(f"[warn] topdown capture failed: {e}")
        ok = False
    finally:
        try:
            client.simSetCameraPose(args.camera_name, chase_cam_pose)
            client.simSetVehiclePose(chase_pose, ignore_collision=True)
        except Exception as e:
            print(f"[warn] topdown restore failed: {e}")
    return ok


def _render_mp4(
    frames_dir: Path,
    out_path: Path,
    framerate: float,
    glob_suffix: str = "_rgb.png",
    *,
    output_fps: float | None = None,
) -> bool:
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        return False
    pattern = str(frames_dir / f"frame_*{glob_suffix}")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(framerate),
        "-pattern_type", "glob",
        "-i", pattern,
    ]
    if output_fps is not None and output_fps > 0:
        cmd.extend(["-r", str(output_fps)])
    cmd.extend([
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ])
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return out_path.exists()
    except subprocess.CalledProcessError as e:
        print(f"[warn] ffmpeg failed: {e.stderr[:500] if e.stderr else e}")
        return False


FLYSEEK_TARGET_SEG_ID = 200


def _seg_array_from_response(response):
    """Return an (H, W, 3) uint8 array from a Scene/Segmentation response."""
    import numpy as np
    try:
        if getattr(response, "pixels_as_float", False):
            return None
        buf = response.image_data_uint8
        if not buf or len(buf) == 0:
            return None
        img = np.frombuffer(buf, dtype=np.uint8)
        return img.reshape(response.height, response.width, 3)
    except Exception:
        return None


def _save_image(response, out_path: Path, kind: str) -> bool:
    import numpy as np
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if response.pixels_as_float:
            depth = np.array(response.image_data_float, dtype=np.float32)
            if depth.size == 0:
                return False
            depth = depth.reshape(response.height, response.width)
            np.save(out_path.with_suffix(".npy"), depth)
            depth_vis = np.clip(depth, 0.0, 100.0) / 100.0
            depth_u8 = (depth_vis * 255).astype(np.uint8)
            import cv2  # type: ignore
            cv2.imwrite(str(out_path), depth_u8)
        else:
            if not response.image_data_uint8 or len(response.image_data_uint8) == 0:
                return False
            img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img = img.reshape(response.height, response.width, 3)
            import cv2  # type: ignore
            cv2.imwrite(str(out_path), img)
        return True
    except Exception as e:
        print(f"[warn] save {kind} failed → {out_path}: {e}")
        return False


# --------------------------------------------------------------------------- #
# Pose smoother — post-control low-pass + yaw-rate cap                        #
# --------------------------------------------------------------------------- #
class _PoseSmoother:
    """Damps drone teleport pose to remove camera jitter.

    Applied AFTER the tracker / chase controller computes the next drone pose,
    and BEFORE ``simSetVehiclePose``. The tracker itself is not modified —
    next-tick feedback (it receives the smoothed pose) just adds extra damping.

    Three knobs:
      - pos_tau_s : position low-pass time constant (s)
      - yaw_tau_s : yaw      low-pass time constant (s)
      - max_yaw_rate_dps : hard cap on camera angular velocity (deg/s)

    Discrete first-order EMA with α = clip(dt/τ, 0, 1). τ ≈ 0.3–0.6 s gives
    a noticeably smooth pan without feeling laggy on a 20 Hz tick loop.
    """

    def __init__(
        self,
        pos_tau_s: float,
        yaw_tau_s: float,
        max_yaw_rate_dps: float,
        z_tau_s: float | None = None,
    ) -> None:
        self._pos_tau = max(1e-3, float(pos_tau_s))
        self._z_tau = max(1e-3, float(z_tau_s if z_tau_s is not None else pos_tau_s * 2.5))
        self._yaw_tau = max(1e-3, float(yaw_tau_s))
        self._max_yaw_rate = math.radians(max(0.0, float(max_yaw_rate_dps)))
        self._pos: np.ndarray | None = None
        self._yaw: float | None = None

    def reset(self, position: np.ndarray, yaw: float) -> None:
        self._pos = np.asarray(position, dtype=np.float64).copy()
        self._yaw = float(yaw)

    def filter(
        self,
        position: np.ndarray,
        yaw: float,
        dt: float,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        target_pos = np.asarray(position, dtype=np.float64)
        target_yaw = float(yaw)
        if self._pos is None or self._yaw is None:
            self.reset(target_pos, target_yaw)
            return self._pos.copy(), self._yaw, np.zeros(3, dtype=np.float64)  # type: ignore[union-attr]

        prev_pos = self._pos.copy()
        a_xy = float(np.clip(dt / self._pos_tau, 0.0, 1.0))
        a_z = float(np.clip(dt / self._z_tau, 0.0, 1.0))
        self._pos = np.array([
            prev_pos[0] + a_xy * (target_pos[0] - prev_pos[0]),
            prev_pos[1] + a_xy * (target_pos[1] - prev_pos[1]),
            prev_pos[2] + a_z * (target_pos[2] - prev_pos[2]),
        ], dtype=np.float64)

        a_yaw = float(np.clip(dt / self._yaw_tau, 0.0, 1.0))
        delta = wrap_to_pi(target_yaw - self._yaw)
        step = a_yaw * delta
        if self._max_yaw_rate > 0.0:
            cap = self._max_yaw_rate * max(dt, 1e-6)
            step = max(-cap, min(cap, step))
        self._yaw = wrap_to_pi(self._yaw + step)

        velocity = (self._pos - prev_pos) / max(dt, 1e-6)
        return self._pos.copy(), self._yaw, velocity


# --------------------------------------------------------------------------- #
# Target picker (mirrors smoke_test_teleport_and_capture)                     #
# --------------------------------------------------------------------------- #
def _pick_target(
    client,
    args,
    occupancy: PcdOccupancyMap | None = None,
) -> tuple[str, str] | None:
    if args.target:
        return (args.target, args.label or "a tracked vehicle")

    candidates: list[dict[str, Any]] = []
    if args.auto_from_scout:
        scout_file = args.scout_file or DEFAULT_SCOUT_FILE
        if not scout_file.exists():
            print(f"[ERR] --auto-from-scout but missing: {scout_file}")
            print("      Run scout first: "
                  "python flyseek_extend/scripts/scout_scene_targets.py")
            return None
        data = json.loads(scout_file.read_text(encoding="utf-8"))
        candidates = data.get("candidates", [])
        if not candidates:
            print("[ERR] scout JSON has no candidates")
            return None

    if args.target_regex:
        matches = client.simListSceneObjects(name_regex=args.target_regex) or []
        if not matches:
            print(f"[ERR] regex '{args.target_regex}' → 0 matches")
            return None
        candidates = [{"name": m, "suggested_label": args.label or "a small motorized car"}
                        for m in matches]

    if args.motorized_cars_only:
        from scout_scene_targets import is_motorized_car_name  # type: ignore
        kept: list[dict[str, Any]] = []
        for c in candidates:
            n = c["name"] if isinstance(c, dict) else str(c)
            if is_motorized_car_name(n):
                if isinstance(c, dict):
                    c.setdefault("suggested_label", "a small motorized car")
                    if "Cart" in c.get("suggested_label", ""):
                        c["suggested_label"] = "a small motorized car"
                kept.append(c)
        if not kept:
            print("[ERR] --motorized-cars-only filtered out every candidate. "
                  "Re-run scout, or pass --no-motorized-cars-only, or --target.")
            return None
        if len(kept) < len(candidates):
            print(f"[ok] motorized-car filter: {len(candidates)} → {len(kept)}")
        candidates = kept

    if not candidates:
        print("[ERR] one of --target / --auto-from-scout / --target-regex required")
        return None

    street_safe: list[tuple[str, str]] = []
    for c in candidates:
        name = c["name"] if isinstance(c, dict) else str(c)
        label = (c.get("suggested_label", "a tracked vehicle")
                 if isinstance(c, dict) else (args.label or "a tracked vehicle"))
        if occupancy is None:
            street_safe.append((name, label))
            continue
        try:
            pose = client.simGetObjectPose(name)
        except Exception:
            continue
        if math.isnan(pose.position.x_val):
            continue
        pos = np.array([pose.position.x_val, pose.position.y_val,
                        pose.position.z_val])
        hint_h = _heading_from_quaternion(
            pose.orientation.w_val,
            pose.orientation.x_val,
            pose.orientation.y_val,
            pose.orientation.z_val,
        )
        score, reason = score_init_pose_ned(occupancy, pos, hint_h)
        if score > -1e8 and reason == "ok":
            street_safe.append((name, label))

    idx = int(getattr(args, "target_index", 0) or 0)
    if street_safe:
        pick = street_safe[idx % len(street_safe)]
        print(f"[ok] picked street-safe target [{idx % len(street_safe)}/{len(street_safe)}]: "
              f"{pick[0]}")
        return pick

    name = candidates[0]["name"] if isinstance(candidates[0], dict) else candidates[0]
    label = (candidates[0].get("suggested_label", "a tracked vehicle")
             if isinstance(candidates[0], dict) else "a tracked vehicle")
    print(f"[warn] no init-valid candidate at spawn; using first: {name}")
    return (name, label)


def _initialize_target_on_road(
    client,
    target_name: str,
    target_pose_air,
    occupancy: PcdOccupancyMap,
    args,
    report: DemoReport,
    *,
    anchor_override: np.ndarray | None = None,
) -> tuple[np.ndarray, float, bool]:
    """Teleport target to a scored drivable road pose; return (pos, heading, ok)."""
    if anchor_override is not None:
        anchor = np.asarray(anchor_override, dtype=np.float64).reshape(3).copy()
    else:
        anchor = np.array([
            target_pose_air.position.x_val,
            target_pose_air.position.y_val,
            target_pose_air.position.z_val,
        ])
    hint_h = _heading_from_quaternion(
        target_pose_air.orientation.w_val,
        target_pose_air.orientation.x_val,
        target_pose_air.orientation.y_val,
        target_pose_air.orientation.z_val,
    )
    env_name = str(getattr(args, "env", None) or "env_airsim_16")
    profile_name = getattr(args, "init_profile", None) or default_profile_name(env_name)
    try:
        profile = load_target_init_profile(env_name, profile_name)
    except (KeyError, FileNotFoundError):
        profile = load_target_init_profile(
            "env_airsim_16",
            "standard",
        )
        report.warnings.append(
            f"init profile {profile_name!r} unavailable; using standard"
        )

    cfg = profile.config
    if getattr(args, "init_search_radius_m", None) is not None:
        from dataclasses import replace
        cfg = replace(cfg, search_radius_m=float(args.init_search_radius_m))
    if getattr(args, "init_min_corridor_width_m", None) is not None:
        from dataclasses import replace
        cfg = replace(cfg, min_corridor_width_m=float(args.init_min_corridor_width_m))
    if cfg is not profile.config:
        from flyseek.utils.target_init_presets import TargetInitProfile
        profile = TargetInitProfile(
            name=profile.name,
            env=profile.env,
            description=profile.description,
            config=cfg,
            strategy=profile.strategy,
            use_road_seed_fallback=profile.use_road_seed_fallback,
            road_seed_search_radius_m=profile.road_seed_search_radius_m,
            road_seed_sample_step_m=profile.road_seed_sample_step_m,
        )

    cur_score, cur_reason = score_init_pose_ned(
        occupancy, anchor, hint_h, cfg=profile.config,
    )
    print(f"[info] target spawn score={cur_score:.1f} ({cur_reason})")

    # ---- robust init: multi-seed retry + profile fallback chain ----------
    # The road-seed search uses a stochastic shuffle, so one bad seed can
    # silently fail and leave the car in a (typically not_drivable) spawn
    # pose. Retry up to ``max_seed_retries`` times per profile and, if the
    # requested profile cannot find a road, automatically widen to ``major_road``.
    max_seed_retries = 8
    profile_chain: list[Any] = [profile]
    if profile.name != "major_road":
        try:
            profile_chain.append(load_target_init_profile(env_name, "major_road"))
        except (KeyError, FileNotFoundError):
            pass

    best_attempt = None
    base_seed = int(args.seed) if args.seed is not None else None
    for prof in profile_chain:
        for attempt in range(max_seed_retries):
            if base_seed is None:
                # OS entropy + attempt index → varied but reproducible-per-run.
                rng = np.random.default_rng(
                    [int(time.time_ns() & 0xFFFFFFFF), attempt * 1009 + 17]
                )
            else:
                rng = np.random.default_rng(base_seed + attempt * 23 + 29)
            r = resolve_target_init_pose(
                occupancy, anchor, rng, prof, hint_heading=hint_h,
            )
            if r.ok:
                result = r
                if attempt > 0 or prof.name != profile.name:
                    print(f"[info] init succeeded on retry "
                          f"(profile={prof.name}, attempts={attempt+1})")
                break
            if best_attempt is None or r.score > best_attempt.score:
                best_attempt = r
        else:
            # All seeds failed under this profile; try next profile in chain.
            continue
        break
    else:
        # All profiles exhausted with no success.
        fail_reason = best_attempt.reason if best_attempt else "no_candidate"
        fail_score = best_attempt.score if best_attempt else -1e9
        report.errors.append(
            f"target init FAILED after {max_seed_retries} retries × "
            f"{len(profile_chain)} profiles (best score={fail_score:.1f}, "
            f"reason={fail_reason}). Refusing to run demo with non-drivable "
            f"spawn pose — pick a different --target or relax the profile."
        )
        return anchor, hint_h, False

    if apply_init_pose_to_sim(
        client, target_name, result, make_pose_fn=_make_airsim_pose
    ):
        shift = float(np.linalg.norm(result.position_ned[:2] - anchor[:2]))
        print(
            f"[ok] target init [{result.profile}/{result.init_method}] "
            f"score={result.score:.1f} "
            f"@ ({result.position_ned[0]:.1f}, {result.position_ned[1]:.1f}), "
            f"heading {math.degrees(result.heading_rad):.1f}° "
            f"(shift {shift:.0f}m, {result.samples_tried} samples)"
        )
        return result.position_ned.copy(), float(result.heading_rad), True

    report.errors.append("target init pose found but simSetObjectPose failed")
    return result.position_ned.copy(), float(result.heading_rad), False


# --------------------------------------------------------------------------- #
# Dumb chase controller — used only to MOVE the drone for the demo            #
# --------------------------------------------------------------------------- #
def _chase_drone_pose(
    target: TargetState,
    args,
    dt: float,
    prev_drone: DroneState,
    occupancy: PcdOccupancyMap | None = None,
) -> tuple[DroneState, float]:
    """Compute the next drone pose with optional PCD collision resolution."""
    # Behind the target relative to its motion direction; if target is still,
    # fall back to a fixed offset along +X.
    v_xy = float(np.linalg.norm(target.velocity[:2]))
    if v_xy > 0.2:
        motion_dir = target.velocity[:2] / v_xy
    else:
        # Use target heading as a fallback orientation
        motion_dir = np.array([math.cos(target.heading), math.sin(target.heading)])

    back = -motion_dir  # behind target
    desired_pos_xy = target.position[:2] + back * args.follow_distance

    desired_z = -abs(args.follow_altitude)
    if occupancy is not None:
        probe_map = airsim_ned_to_map(
            np.array([desired_pos_xy[0], desired_pos_xy[1], desired_z])
        )
        min_map_z = occupancy.min_safe_map_z(probe_map)
        min_ned_z = -min_map_z
        if desired_z > min_ned_z:
            desired_z = min_ned_z

    alpha = float(np.clip(args.drone_smoothing * dt, 0.0, 1.0))
    new_x = prev_drone.position[0] + alpha * (desired_pos_xy[0] - prev_drone.position[0])
    new_y = prev_drone.position[1] + alpha * (desired_pos_xy[1] - prev_drone.position[1])
    new_z = prev_drone.position[2] + alpha * (desired_z - prev_drone.position[2])

    # Yaw must point from the drone's ACTUAL position toward the target so the
    # camera centers the car. Computing yaw from desired_pos (as the old code
    # did) collapses algebraically to "yaw = target motion direction", which
    # left the camera misaligned whenever the drone wasn't on its ideal path.
    yaw_des = math.atan2(
        target.position[1] - new_y,
        target.position[0] - new_x,
    )
    new_yaw = wrap_to_pi(
        prev_drone.heading + alpha * wrap_to_pi(yaw_des - prev_drone.heading)
    )

    proposed = np.array([new_x, new_y, new_z], dtype=np.float64)
    if occupancy is not None and not args.no_collision:
        proposed = occupancy.resolve_drone_ned(prev_drone.position, proposed)

    new_state = DroneState(
        position=proposed,
        velocity=np.array([
            (proposed[0] - prev_drone.position[0]) / dt,
            (proposed[1] - prev_drone.position[1]) / dt,
            (proposed[2] - prev_drone.position[2]) / dt,
        ]),
        heading=new_yaw,
        timestamp=prev_drone.timestamp + dt,
    )
    return new_state, new_yaw


# --------------------------------------------------------------------------- #
# Inline robust tracker — replaces legacy TrackingDroneController             #
# --------------------------------------------------------------------------- #
class _InlineTracker:
    """Drone tracking controller with TRACK / SEARCH state machine.

    Fixes three concrete bugs in the legacy chase / tracker code:
      1. Desired position used the target's *instantaneous* velocity to compute
         the "behind" offset. Any small turn flipped that vector, swinging the
         drone around the target → the "drone is orbiting in place" symptom.
         Here we low-pass the motion direction (τ ≈ 1.5 s) and *only update it
         when target is actually moving* (speed > 0.5 m/s). Otherwise we hold
         the last good direction, so a stopped/turning car does NOT cause the
         drone to circle.
      2. Yaw was derived from a geometric identity that collapsed to "yaw =
         target motion direction", so the camera did not actually face the
         target unless the drone was perfectly on its ideal trajectory. Here
         yaw is computed from the **drone's actual position** toward the target
         (or last-seen position in SEARCH), so the car stays centered in the
         frame.
      3. There was no SEARCH behavior on occlusion. Here we use
         ``target_visible`` (same helper the recorder uses) every tick. After
         ``lost_after_s`` seconds without LOS, we orbit the last-seen position
         at ``search_orbit_radius`` until we reacquire.

    Extra robustness:
      - Lead-pursuit: aim for ``target.position + velocity * lead_s`` so the
        drone does not lag behind a moving car.
      - Altitude has its own low-pass τ + climb/drop rate limits.
      - Internal yaw rate cap (120°/s) — outer `_PoseSmoother` caps it again
        for the rendered camera.

    Public surface mirrors the legacy controller:
        tracker.reset(drone_state, target_state)
        new_drone_state, log_dict = tracker.step(drone, target, dt)
    """

    def __init__(self, args, occupancy: PcdOccupancyMap | None) -> None:
        self.args = args
        self.occupancy = occupancy
        self._fd = float(args.follow_distance)
        self._fa = float(args.follow_altitude)
        self._hfov = float(args.camera_hfov_deg)
        self._lost_after = float(args.lost_after_s)
        self._orbit_r = float(args.search_orbit_radius)
        self._alt_tau = max(1e-3, float(args.altitude_smooth_tau))
        self._max_climb = float(args.max_climb_mps)
        self._max_drop = float(args.max_drop_mps)
        self._k_xy = max(0.1, float(args.drone_smoothing))
        self._k_yaw = max(0.1, float(getattr(args, "tracker_yaw_gain",
                                             args.drone_smoothing)))
        self._max_yaw_rate = math.radians(120.0)
        self._mdir_tau = max(0.2, float(getattr(args, "tracker_motion_dir_tau", 1.5)))
        self._mspeed_min = 0.5
        self._orbit_omega = math.radians(
            float(getattr(args, "search_orbit_speed_dps", 30.0))
        )
        self._lead_s = max(0.0, float(getattr(args, "tracker_lead_s", 0.7)))
        self._fov_gain = float(getattr(args, "tracker_fov_center_gain", 10.0))
        self._altitude = OpenFlyDroneAltitude(
            float(args.follow_altitude),
            occupancy,
            roof_smooth_tau_s=float(getattr(args, "roof_smooth_tau", 6.0)),
            alt_smooth_tau_s=float(args.altitude_smooth_tau),
            max_climb_mps=float(args.max_climb_mps),
            max_drop_mps=float(args.max_drop_mps),
            roof_probe_range_m=float(getattr(args, "roof_probe_range_m", 2.0)),
        )

        self._mode: str = "track"
        self._mdir = np.array([1.0, 0.0], dtype=np.float64)
        self._last_seen_pos: np.ndarray | None = None
        self._lost_since: float = -1.0
        self._t: float = 0.0
        self._orbit_phase: float = 0.0

    def reset(self, drone: DroneState, target: TargetState) -> None:
        self._mode = "track"
        d = np.array([math.cos(float(target.heading)),
                      math.sin(float(target.heading))], dtype=np.float64)
        n = float(np.linalg.norm(d))
        self._mdir = (d / n) if n > 1e-9 else np.array([1.0, 0.0])
        self._last_seen_pos = target.position.copy()
        self._lost_since = -1.0
        self._t = 0.0
        dx = float(drone.position[0] - target.position[0])
        dy = float(drone.position[1] - target.position[1])
        self._orbit_phase = math.atan2(dy, dx)
        self._altitude.reset(drone, target)

    def step(
        self,
        drone: DroneState,
        target: TargetState,
        dt: float,
    ) -> tuple[DroneState, dict[str, Any]]:
        self._t += dt

        visible = bool(target_visible(
            self.occupancy, drone, target, hfov_deg=self._hfov
        ))
        if visible:
            self._last_seen_pos = target.position.copy()
            self._lost_since = -1.0
        elif self._lost_since < 0:
            self._lost_since = self._t

        v_xy = np.asarray(target.velocity[:2], dtype=np.float64)
        speed = float(np.linalg.norm(v_xy))
        if speed > self._mspeed_min:
            raw = v_xy / speed
            a = float(np.clip(dt / self._mdir_tau, 0.0, 1.0))
            blended = (1.0 - a) * self._mdir + a * raw
            n = float(np.linalg.norm(blended))
            if n > 1e-6:
                self._mdir = blended / n

        lost_dur = (self._t - self._lost_since) if self._lost_since >= 0 else 0.0
        if self._mode == "track" and not visible and lost_dur > self._lost_after:
            self._mode = "search"
            anchor = (self._last_seen_pos if self._last_seen_pos is not None
                      else target.position)
            self._orbit_phase = math.atan2(
                float(drone.position[1] - anchor[1]),
                float(drone.position[0] - anchor[0]),
            )
        elif self._mode == "search" and visible:
            self._mode = "track"

        if self._mode == "track":
            lead_xy = (target.position[:2].astype(np.float64)
                       + v_xy * self._lead_s)
            back = -self._mdir
            desired_xy = lead_xy + back * self._fd
            if visible:
                desired_xy = desired_xy + fov_centering_offset_xy(
                    drone.position,
                    drone.heading,
                    target.position,
                    gain_m_per_rad=self._fov_gain,
                )
            face_xy = target.position[:2].astype(np.float64)
        else:
            anchor = (self._last_seen_pos if self._last_seen_pos is not None
                      else target.position).astype(np.float64)
            self._orbit_phase = wrap_to_pi(
                self._orbit_phase + self._orbit_omega * dt
            )
            desired_xy = np.array([
                anchor[0] + self._orbit_r * math.cos(self._orbit_phase),
                anchor[1] + self._orbit_r * math.sin(self._orbit_phase),
            ], dtype=np.float64)
            face_xy = anchor[:2]

        alt_target = target.copy_with(
            position=np.array([
                float(face_xy[0]), float(face_xy[1]), target.position[2],
            ], dtype=np.float64),
        )
        new_z, _alt_log = self._altitude.step(drone, alt_target, dt)

        alpha_xy = float(np.clip(self._k_xy * dt, 0.0, 1.0))
        new_x = drone.position[0] + alpha_xy * (desired_xy[0] - drone.position[0])
        new_y = drone.position[1] + alpha_xy * (desired_xy[1] - drone.position[1])

        yaw_des = math.atan2(float(face_xy[1] - new_y),
                             float(face_xy[0] - new_x))
        alpha_yaw = float(np.clip(self._k_yaw * dt, 0.0, 1.0))
        d_yaw = alpha_yaw * wrap_to_pi(yaw_des - drone.heading)
        cap = self._max_yaw_rate * dt
        d_yaw = max(-cap, min(cap, d_yaw))
        new_yaw = wrap_to_pi(drone.heading + d_yaw)

        proposed = np.array([new_x, new_y, new_z], dtype=np.float64)
        if (self.occupancy is not None
                and not getattr(self.args, "no_collision", False)):
            proposed = self.occupancy.resolve_drone_ned(drone.position, proposed)

        new_state = DroneState(
            position=proposed,
            velocity=np.array([
                (proposed[0] - drone.position[0]) / max(dt, 1e-6),
                (proposed[1] - drone.position[1]) / max(dt, 1e-6),
                (proposed[2] - drone.position[2]) / max(dt, 1e-6),
            ]),
            heading=new_yaw,
            timestamp=drone.timestamp + dt,
        )

        log: dict[str, Any] = {
            "tracker_mode": self._mode,
            "visible": visible,
            "lost_s": round(lost_dur, 2),
            "follow_dist_m": round(
                float(np.linalg.norm(target.position[:2] - proposed[:2])), 2
            ),
            "target_speed_mps": round(speed, 2),
        }
        return new_state, log


class _ReactiveTracker:
    """Reactive baseline follower (comparison against the adaptive FlySeek FSM).

    This is the OpenFly-style *reactive* policy: the drone is driven straight
    toward a fixed offset behind the target's **current** position (via
    ``_chase_drone_pose``) and its yaw points at the target's instantaneous
    location. There is **no** occlusion handling — no motion-direction
    low-pass, no lead/predict, no PEEK side-step, no REACQUIRE/SEARCH. When the
    car ducks behind a building or makes a sudden turn at an intersection /
    into a narrow alley, the reactive follower flies blindly into the occluder,
    the line of sight breaks, and the target is lost. This is the intended
    failure mode used as the baseline in the FlySeek paper comparison.

    Public surface mirrors the other trackers: ``reset`` + ``step``.
    """

    def __init__(self, args, occupancy: PcdOccupancyMap | None) -> None:
        self.args = args
        self.occupancy = occupancy
        self._hfov = float(args.camera_hfov_deg)

    def reset(self, drone: DroneState, target: TargetState) -> None:
        # Stateless reactive controller — nothing to reset.
        return None

    def step(
        self,
        drone: DroneState,
        target: TargetState,
        dt: float,
    ) -> tuple[DroneState, dict[str, Any]]:
        new_state, _yaw = _chase_drone_pose(
            target, self.args, dt, drone, occupancy=self.occupancy
        )
        visible = bool(target_visible(
            self.occupancy, new_state, target, hfov_deg=self._hfov
        ))
        log: dict[str, Any] = {
            "tracker_mode": "reactive",
            "visible": visible,
            "follow_dist_m": round(
                float(np.linalg.norm(target.position[:2] - new_state.position[:2])), 2
            ),
            "target_speed_mps": round(float(np.linalg.norm(target.velocity[:2])), 2),
        }
        return new_state, log


class _ReactiveLostTracker:
    """Memoryless reactive baseline that genuinely *loses* the target.

    Unlike ``_ReactiveTracker`` (which always knows the true target pose), this
    baseline only acts on the target while it is actually *visible*:

      * **TRACK** — target visible: chase the current target pose, remember it.
      * **GIVE_UP** — just lost line of sight: keep coasting toward the
        last-seen position for ``lost_after_s`` (it does not yet know the target
        is gone).
      * **WANDER** — lost longer than that: with no memory / prediction it
        cannot follow the now-hidden car, so it loiters near the last-seen spot,
        drifting to short random waypoints and sweeping its yaw back and forth —
        the camera "stalls near where the target was lost and mills around".
        It only resumes if the target happens to re-enter view (which it does
        not, once the car is parked behind a building).

    This is the requested failure mode for the "car drives behind a building"
    scenario, contrasted with the adaptive FlySeek FSM that predicts / peeks and
    reacquires.
    """

    def __init__(self, args, occupancy: PcdOccupancyMap | None) -> None:
        self.args = args
        self.occupancy = occupancy
        self._hfov = float(args.camera_hfov_deg)
        self._k_xy = max(0.1, float(args.drone_smoothing))
        self._k_yaw = max(0.1, float(getattr(args, "tracker_yaw_gain",
                                              args.drone_smoothing)))
        self._lost_after = float(getattr(args, "lost_after_s", 0.6))
        self._max_yaw_rate = math.radians(120.0)
        # Aimless-wander knobs.
        self._wander_radius_m = float(getattr(args, "lost_wander_radius_m", 6.0))
        self._wander_scan_dps = float(getattr(args, "lost_wander_scan_dps", 35.0))
        self._rng = np.random.default_rng(
            (int(args.seed) + 101) if args.seed is not None else None
        )
        self._altitude = OpenFlyDroneAltitude(
            float(args.follow_altitude),
            occupancy,
            roof_smooth_tau_s=float(getattr(args, "roof_smooth_tau", 6.0)),
            alt_smooth_tau_s=float(args.altitude_smooth_tau),
            max_climb_mps=float(args.max_climb_mps),
            max_drop_mps=float(args.max_drop_mps),
            roof_probe_range_m=float(getattr(args, "roof_probe_range_m", 2.0)),
        )
        self._last_seen: np.ndarray = np.zeros(3)
        self._lost_since: float = -1.0
        self._t: float = 0.0
        self._wander_target: np.ndarray | None = None
        self._scan_phase: float = 0.0

    def reset(self, drone: DroneState, target: TargetState) -> None:
        self._last_seen = target.position.copy()
        self._lost_since = -1.0
        self._t = 0.0
        self._wander_target = None
        self._scan_phase = 0.0
        self._altitude.reset(drone, target)

    def step(
        self,
        drone: DroneState,
        target: TargetState,
        dt: float,
    ) -> tuple[DroneState, dict[str, Any]]:
        self._t += dt
        visible = bool(target_visible(
            self.occupancy, drone, target, hfov_deg=self._hfov
        ))
        if visible:
            self._last_seen = target.position.copy()
            self._lost_since = -1.0
            self._wander_target = None
        elif self._lost_since < 0:
            self._lost_since = self._t
        lost_dur = (self._t - self._lost_since) if self._lost_since >= 0 else 0.0

        if visible:
            mode = "track"
            # Reactive chase of the (visible) target pose.
            new_state, _yaw = _chase_drone_pose(
                target, self.args, dt, drone, occupancy=self.occupancy
            )
            log: dict[str, Any] = {
                "tracker_mode": mode,
                "visible": True,
                "follow_dist_m": round(float(np.linalg.norm(
                    target.position[:2] - new_state.position[:2])), 2),
                "target_speed_mps": round(
                    float(np.linalg.norm(target.velocity[:2])), 2),
            }
            return new_state, log

        anchor = self._last_seen.astype(np.float64)
        if lost_dur <= self._lost_after:
            mode = "give_up"
            desired_xy = anchor[:2]
        else:
            mode = "lost_wander"
            reached = (self._wander_target is not None and float(np.linalg.norm(
                drone.position[:2] - self._wander_target)) < 1.5)
            if self._wander_target is None or reached:
                ang = float(self._rng.uniform(-math.pi, math.pi))
                rad = float(self._rng.uniform(0.3, 1.0)) * self._wander_radius_m
                self._wander_target = anchor[:2] + rad * np.array(
                    [math.cos(ang), math.sin(ang)], dtype=np.float64)
            desired_xy = self._wander_target

        alpha_xy = float(np.clip(self._k_xy * dt, 0.0, 1.0))
        new_x = drone.position[0] + alpha_xy * (desired_xy[0] - drone.position[0])
        new_y = drone.position[1] + alpha_xy * (desired_xy[1] - drone.position[1])

        alt_target = target.copy_with(position=np.array(
            [float(anchor[0]), float(anchor[1]), float(anchor[2])],
            dtype=np.float64))
        new_z, _alt_log = self._altitude.step(drone, alt_target, dt)

        # Aimless yaw: sweep around the bearing to the last-seen spot.
        self._scan_phase = wrap_to_pi(
            self._scan_phase + math.radians(self._wander_scan_dps) * dt)
        base_yaw = math.atan2(float(anchor[1] - new_y), float(anchor[0] - new_x))
        yaw_des = wrap_to_pi(base_yaw + 0.7 * math.sin(self._scan_phase))
        alpha_yaw = float(np.clip(self._k_yaw * dt, 0.0, 1.0))
        d_yaw = alpha_yaw * wrap_to_pi(yaw_des - drone.heading)
        cap = self._max_yaw_rate * dt
        d_yaw = max(-cap, min(cap, d_yaw))
        new_yaw = wrap_to_pi(drone.heading + d_yaw)

        proposed = np.array([new_x, new_y, new_z], dtype=np.float64)
        if (self.occupancy is not None
                and not getattr(self.args, "no_collision", False)):
            proposed = self.occupancy.resolve_drone_ned(drone.position, proposed)
        new_state = DroneState(
            position=proposed,
            velocity=np.array([
                (proposed[0] - drone.position[0]) / max(dt, 1e-6),
                (proposed[1] - drone.position[1]) / max(dt, 1e-6),
                (proposed[2] - drone.position[2]) / max(dt, 1e-6),
            ]),
            heading=new_yaw,
            timestamp=drone.timestamp + dt,
        )
        log = {
            "tracker_mode": mode,
            "visible": False,
            "lost_s": round(lost_dur, 2),
            "follow_dist_m": round(float(np.linalg.norm(
                target.position[:2] - proposed[:2])), 2),
            "target_speed_mps": round(float(np.linalg.norm(target.velocity[:2])), 2),
        }
        return new_state, log


# --------------------------------------------------------------------------- #
# Main loop                                                                   #
# --------------------------------------------------------------------------- #
def run_demo(args: argparse.Namespace) -> DemoReport:
    report = DemoReport(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        airsim_endpoint=f"{args.airsim_ip}:{args.airsim_port}",
        difficulty=args.difficulty,
        duration_s=args.duration,
        sim_dt=1.0 / args.tick_hz,
    )

    try:
        import airsim  # type: ignore
    except ImportError as e:
        report.errors.append(f"airsim not importable: {e}")
        return report

    try:
        client = airsim.MultirotorClient(ip=args.airsim_ip, port=args.airsim_port)
        client.confirmConnection()
    except Exception as e:
        report.errors.append(f"failed to connect: {e}")
        return report

    # ---- 0. load PCD occupancy (before target pick for street-safe spawn) -
    occupancy: PcdOccupancyMap | None = None
    if not args.no_collision:
        try:
            occupancy = PcdOccupancyMap.load_or_build(
                REPO_ROOT,
                env_name=args.env,
                rebuild=args.rebuild_occupancy_cache,
            )
            if getattr(args, "seg_building_jsonl", None):
                from flyseek.utils.seg_buildings import SegBuildingMap
                args._seg_building_map = SegBuildingMap.from_jsonl(
                    args.seg_building_jsonl,
                    footprint_radius_m=float(args.seg_building_radius_m),
                    min_occluder_height_m=float(args.seg_building_min_height_m),
                )
                print(f"[ok] seg buildings: {len(args._seg_building_map)} from "
                      f"{args.seg_building_jsonl}")
            else:
                args._seg_building_map = None
            print("[ok] PCD occupancy ready")
        except Exception as e:
            report.warnings.append(f"PCD occupancy unavailable ({e}); no collision checks.")

    # ---- 1. pick target ---------------------------------------------------
    picked = _pick_target(client, args, occupancy=occupancy)
    if picked is None:
        report.errors.append("no target picked — see stderr")
        return report
    target_name, target_label = picked
    report.target_name = target_name
    report.target_label = target_label
    print(f"[ok] target: '{target_name}' — \"{target_label}\"")

    # ---- 2. snapshot original poses ---------------------------------------
    try:
        drone_pose_air = client.simGetVehiclePose()
        target_pose_air = client.simGetObjectPose(target_name)
    except Exception as e:
        report.errors.append(f"snapshot poses failed: {e}")
        return report

    if math.isnan(target_pose_air.position.x_val):
        report.errors.append(f"target '{target_name}' returned NaN pose")
        return report

    # Assign a unique segmentation id so we can extract a 2D bbox for the
    # target from the segmentation frames (SKILL §5.4 / §6.3). Whitelisted API.
    target_seg_id: int | None = FLYSEEK_TARGET_SEG_ID
    try:
        ok_seg = client.simSetSegmentationObjectID(
            target_name, FLYSEEK_TARGET_SEG_ID, False
        )
        if not ok_seg:
            target_seg_id = None
            report.warnings.append(
                f"simSetSegmentationObjectID returned False for {target_name}; "
                "bbox extraction disabled"
            )
    except Exception as e:
        target_seg_id = None
        report.warnings.append(f"simSetSegmentationObjectID failed: {e}")

    report.initial_drone_pose = _pose_to_dict(drone_pose_air)
    report.initial_target_pose = _pose_to_dict(target_pose_air)
    print(f"[ok] drone  @ ({drone_pose_air.position.x_val:.1f}, "
          f"{drone_pose_air.position.y_val:.1f}, "
          f"{drone_pose_air.position.z_val:.1f})")
    print(f"[ok] target @ ({target_pose_air.position.x_val:.1f}, "
          f"{target_pose_air.position.y_val:.1f}, "
          f"{target_pose_air.position.z_val:.1f})")

    # ---- 3. seed kinematic states ----------------------------------------
    target_state = TargetState(
        position=np.array([target_pose_air.position.x_val,
                           target_pose_air.position.y_val,
                           target_pose_air.position.z_val]),
        velocity=np.zeros(3),
        heading=0.0,
        timestamp=0.0,
    )

    drone_state = DroneState(
        position=np.array([
            target_pose_air.position.x_val - args.follow_distance,
            target_pose_air.position.y_val,
            -abs(args.follow_altitude),
        ]),
        velocity=np.zeros(3),
        heading=0.0,
        timestamp=0.0,
    )

    # ---- 4. build adversary ----------------------------------------------
    play_box = None
    if args.play_box_half_extent > 0:
        cx, cy = float(target_state.position[0]), float(target_state.position[1])
        e = float(args.play_box_half_extent)
        play_box = PlayBox(cx - e, cx + e, cy - e, cy + e)

    cfg: dict[str, Any] | None = None
    if args.config:
        try:
            import yaml  # type: ignore
            cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        except ImportError:
            report.warnings.append("PyYAML not installed; using built-in defaults.")
        except Exception as e:
            report.warnings.append(f"config load failed: {e}; using defaults.")

    diff = args.difficulty
    if args.scenario == "hide_seek":
        diff = "hide_seek"
    hide_overrides = dict(getattr(args, "hide_seek_overrides", {}) or {})
    if hide_overrides:
        if cfg is None:
            cfg = {}
        # Merge tracking-difficulty overrides on top of yaml/default config.
        # Keys like ``open_road_duration_s`` belong to HideSeekCarAgent.DEFAULTS;
        # SCurveEvasionAgent ignores unknown keys gracefully.
        cfg = {**cfg, **hide_overrides}
        print(f"[ok] adversary overrides applied: "
              f"{ {k: v for k, v in hide_overrides.items()} }")
    agent = create_adversarial_agent(
        difficulty=diff,
        config=cfg,
        play_box=play_box,
        seed=args.seed,
        occupancy=occupancy,
    )
    agent.reset(target_state)

    tracker: (TrackingDroneController | _InlineTracker | AdaptiveTracker
              | _ReactiveTracker | _ReactiveLostTracker | None) = None
    # A drone tracker is built for the hide_seek scenario, or whenever
    # ``--force-tracker`` is set (e.g. the alley-chase comparison runs the
    # adaptive FlySeek FSM and the reactive baseline in the default ``chase``
    # scenario so every other scene/camera parameter is held identical).
    if args.scenario == "hide_seek" or getattr(args, "force_tracker", False):
        tracker_mode = getattr(args, "tracker_mode", "adaptive")
        # Backwards compat: --no-use-inline-tracker forces legacy controller.
        if not getattr(args, "use_inline_tracker", True):
            tracker_mode = "legacy"
        if tracker_mode == "adaptive":
            tracker = AdaptiveTracker.from_args(args, occupancy=occupancy)
            print("[ok] tracker: adaptive FSM "
                  "(TRACK/PREDICT/REACQUIRE/PEEK/SEARCH/HOLD, "
                  f"predict={args.tracker_predict_after_s}s, "
                  f"peek={args.tracker_peek_after_s}s, "
                  f"search={args.tracker_search_after_s}s, "
                  f"hold_dwell={args.tracker_hold_dwell_s}s)")
        elif tracker_mode == "inline":
            tracker = _InlineTracker(args, occupancy=occupancy)
            print("[ok] tracker: inline (TRACK/SEARCH, lead-pursuit, "
                  f"motion-dir τ={args.tracker_motion_dir_tau}s, "
                  f"lead={args.tracker_lead_s}s)")
        elif tracker_mode == "reactive":
            tracker = _ReactiveTracker(args, occupancy=occupancy)
            print("[ok] tracker: reactive baseline (chase current target pose, "
                  "no predict / peek / search — loses target on occlusion)")
        elif tracker_mode == "reactive_lost":
            tracker = _ReactiveLostTracker(args, occupancy=occupancy)
            print("[ok] tracker: reactive_lost baseline (chase while visible; on "
                  "occlusion it stalls near the last-seen spot and wanders "
                  "aimlessly — never reacquires)")
        else:
            tracker = TrackingDroneController(args, occupancy=occupancy)
            print("[ok] tracker: legacy TrackingDroneController")
        tracker.reset(drone_state, target_state)

    street_helper: StreetMotionHelper | None = None
    if occupancy is not None and getattr(args, "street_follow", True):
        street_rng = np.random.default_rng(
            (int(args.seed) + 17) if args.seed is not None else None
        )
        street_helper = StreetMotionHelper(
            occupancy=occupancy,
            rng=street_rng,
            street_blend=float(getattr(args, "street_blend", 0.4)),
        )
        street_helper.reset(target_state)
        print(f"[ok] street-follow ON (blend={street_helper.street_blend:.2f})")

    integrator_cfg = (cfg or {}).get("integrator", {}) if cfg else {}
    max_speed = float(integrator_cfg.get("max_speed_mps", 6.0))
    max_turn_rate = math.radians(
        float(integrator_cfg.get("max_turn_rate_deg_s", 90.0))
    )

    # FlySeek-Bench adversarial target policy (overrides the legacy adversary
    # agent / street-follow when --target-behavior is given). Reuses the same
    # integrate_target + stabilize_car_state pipeline internally.
    target_policy = None
    occ_kw: dict = {}
    if getattr(args, "target_behavior", None):
        occ_kw = (
            occlusion_route_kwargs_from_args(args)
            if args.target_behavior == "occlusion_seeking" and occupancy is not None
            else {}
        )
        if args.target_behavior == "occlusion_seeking" and getattr(args, "seg_building_jsonl", None):
            occ_kw["seg_building_jsonl"] = str(args.seg_building_jsonl)
            occ_kw["seg_building_radius_m"] = float(args.seg_building_radius_m)
            occ_kw["seg_building_min_height_m"] = float(args.seg_building_min_height_m)
        if args.target_behavior == "alley_hutong":
            if not getattr(args, "seg_building_jsonl", None):
                report.errors.append(
                    "--target-behavior alley_hutong requires --seg-building-jsonl"
                )
                return report
            occ_kw["seg_building_jsonl"] = str(args.seg_building_jsonl)
            occ_kw["seg_building_radius_m"] = float(args.seg_building_radius_m)
            occ_kw["seg_building_min_height_m"] = float(args.seg_building_min_height_m)
            if getattr(args, "open_approach_m", None):
                occ_kw["open_approach_m"] = float(args.open_approach_m)
            if getattr(args, "max_corridor_width_m", None):
                occ_kw["max_corridor_width_m"] = float(args.max_corridor_width_m)
            occ_kw["search_radius_m"] = float(
                getattr(args, "route_search_radius_m", 220.0)
            )
        target_policy = create_target_policy(
            args.target_behavior,
            config={
                "difficulty": args.target_policy_difficulty,
                "dt": 1.0 / float(args.tick_hz),
                "max_speed_mps": max_speed,
                **occ_kw,
            },
            scene_context={
                "occupancy": occupancy,
                "keep_z": None,
                "drone_eye_agl_m": float(args.follow_altitude),
                "follow_distance_m": float(args.follow_distance),
            },
            seed=args.seed,
        )
        mode = ("route/alley_hutong" if args.target_behavior == "alley_hutong"
                and occupancy is not None
                else "route/annotated_buildings" if args.target_behavior == "occlusion_seeking"
                and getattr(args, "seg_building_jsonl", None)
                else "route/open_then_hide" if args.target_behavior == "occlusion_seeking"
                and occupancy is not None else "reactive")
        print(f"[ok] target policy: {args.target_behavior} "
              f"(difficulty={args.target_policy_difficulty}, mode={mode})")

    if occupancy is not None and not getattr(args, "skip_target_init", False):
        init_anchor_override = None
        if (getattr(args, "target_behavior", None) == "alley_hutong"
                and getattr(args, "_seg_building_map", None) is not None
                and getattr(args, "alley_near_entry", True)):
            from flyseek.utils.alley_route import find_best_alley_scene
            _alley, init_anchor_override = find_best_alley_scene(
                occupancy,
                args._seg_building_map,
                keep_z=float(target_state.position[2]),
            )
            if _alley is not None and init_anchor_override is not None:
                print(f"[ok] alley hutong: buildings {_alley.building_a.index}/"
                      f"{_alley.building_b.index}  corridor="
                      f"{_alley.corridor_width_m:.1f}m  depth={_alley.depth_m:.0f}m  "
                      f"init near entry ({init_anchor_override[0]:.1f}, "
                      f"{init_anchor_override[1]:.1f})")
        init_pos, init_h, init_ok = _initialize_target_on_road(
            client, target_name, target_pose_air, occupancy, args, report,
            anchor_override=init_anchor_override,
        )
        if not init_ok and not bool(getattr(args, "allow_bad_init", False)):
            # Refuse to record an episode where the car cannot move (this is
            # the root cause of "car stuck in/under a building" videos).
            report.success = False
            print("\n[FATAL] target init failed and --allow-bad-init not set; "
                  "aborting demo. See errors above. Try a different --target, "
                  "another --init-profile, or pass --allow-bad-init to record "
                  "a degenerate scene anyway.")
            return report
        target_state = target_state.copy_with(
            position=init_pos,
            heading=init_h,
        )
        # Re-seat the drone to a CANONICAL chase pose behind the (re-initialized)
        # target — directly behind its heading at ``follow_distance`` and facing
        # it. This is independent of the tracker, so every run (adaptive /
        # reactive / reactive_lost / …) starts from the *same* initial camera
        # pose with the target already in view; the policies only diverge once
        # the chase begins. Without this the drone_state was left at the stale
        # pre-init spot, and the one-shot dt=1.0 pre-roll snapped each tracker to
        # a *different* opening pose (e.g. reactive_lost jumped onto the target).
        back = float(target_state.heading) + math.pi
        drone_state = drone_state.copy_with(
            position=np.array([
                float(target_state.position[0] + math.cos(back) * args.follow_distance),
                float(target_state.position[1] + math.sin(back) * args.follow_distance),
                -abs(float(args.follow_altitude)),
            ], dtype=np.float64),
            velocity=np.zeros(3),
            heading=float(target_state.heading),
        )
        agent.reset(target_state)
        if tracker is not None:
            tracker.reset(drone_state, target_state)
        if street_helper is not None:
            street_helper.reset(target_state)
        if target_policy is not None:
            target_policy.reset(target_state, uav_state=drone_state)
            if hasattr(target_policy, "route_meta") and target_policy.route_meta:
                meta = target_policy.route_meta
                if meta.get("planner") == "alley_hutong":
                    print(f"[ok] alley route: corridor="
                          f"{meta.get('corridor_width_m')}m  depth="
                          f"{meta.get('alley_depth_m')}m  "
                          f"buildings={meta.get('alley_building_a')}/"
                          f"{meta.get('alley_building_b')}  "
                          f"alley_starts≈{meta.get('est_alley_start_s')}s "
                          f"(route≈{meta.get('est_route_total_s')}s, "
                          f"recommend duration≥{meta.get('recommended_duration_s')}s)")
                else:
                    print(f"[ok] hide route: frustum_hidden="
                          f"{meta.get('frustum_hidden_frac', 0):.0%} "
                          f"planner={meta.get('planner', 'pcd')} "
                          f"building_idx={meta.get('building_index', '-')} "
                          f"hide_goal_frustum={meta.get('hide_goal_frustum_hidden')} "
                          f"hide_starts≈{meta.get('est_hide_start_s')}s "
                          f"(route≈{meta.get('est_route_total_s')}s, "
                          f"recommend duration≥{meta.get('recommended_duration_s')}s) "
                          f"chase_drones={meta.get('chase_drone_samples', '-')}")
    elif target_policy is not None:
        target_policy.reset(target_state, uav_state=drone_state)

    if occupancy is not None:
        roof = occupancy.local_roof_map_z(airsim_ned_to_map(target_state.position))
        print(f"[ok] local roof ≈ {roof:.1f} m (map), "
              f"min drone clearance {occupancy.cfg.min_drone_clearance:.1f} m")

    if occupancy is not None:
        if tracker is not None:
            drone_state, _ = tracker.step(drone_state, target_state, 1.0)
        else:
            drone_state, _ = _chase_drone_pose(
                target_state, args, 1.0, drone_state, occupancy=occupancy
            )
        alt = airsim_altitude_m(drone_state.position)
        print(f"[ok] initial drone altitude after PCD clearance: {alt:.1f} m AGL")

    # ---- 5. set camera (nose mount, steep downward pitch) ----------------
    chase_cam_pose = _make_tracking_camera_pose(args)
    try:
        chosen_cam = _apply_tracking_camera(client, args)
        args.camera_name = chosen_cam
        print(f"[ok] camera '{chosen_cam}' → forward {args.camera_body_forward_m}m, "
              f"down {args.camera_body_down_m}m, pitch {args.camera_pitch_deg}°")
    except Exception as e:
        report.warnings.append(f"simSetCameraPose error: {e}")

    capture_topdown = bool(getattr(args, "topdown", False))
    topdown_stride = max(1, int(round(float(args.tick_hz) / float(args.topdown_fps))))
    effective_topdown_fps = float(args.tick_hz) / topdown_stride
    topdown_anchor = _TopdownAnchor(
        xy_tau_s=float(getattr(args, "topdown_xy_smooth_tau", 0.6)),
        yaw_tau_s=float(getattr(args, "topdown_yaw_smooth_tau", 1.0)),
        fix_yaw_north=bool(getattr(args, "topdown_lock_north", True)),
    )
    if capture_topdown:
        print(f"[ok] topdown capture ON → altitude {args.topdown_altitude} m AGL, "
              f"{effective_topdown_fps:.1f} FPS (stride={topdown_stride})")

    # ---- 5b. post-control pose smoother (camera anti-jitter) -------------
    smoother: _PoseSmoother | None = None
    if not args.no_camera_smoothing:
        smoother = _PoseSmoother(
            pos_tau_s=float(args.camera_pos_smooth_tau),
            yaw_tau_s=float(args.camera_yaw_smooth_tau),
            max_yaw_rate_dps=float(args.camera_max_yaw_rate_dps),
            z_tau_s=float(getattr(args, "camera_z_smooth_tau", 1.2)),
        )
        smoother.reset(drone_state.position, drone_state.heading)
        print(f"[ok] camera smoother ON → pos_τ={args.camera_pos_smooth_tau}s, "
              f"yaw_τ={args.camera_yaw_smooth_tau}s, "
              f"yaw_rate≤{args.camera_max_yaw_rate_dps}°/s")
    else:
        print("[warn] camera smoother OFF (raw tracker pose teleported each tick)")

    # ---- 6. simulation loop ----------------------------------------------
    episode_tag = getattr(args, "episode_tag", None) or report.timestamp.replace(":", "-")
    out_dir = args.output / episode_tag
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    report.output_dir = str(out_dir)
    episode_writer = EpisodeWriter(out_dir, difficulty=args.difficulty)

    # Horizontal FOV of the rendered camera (for bbox projection). Falls back
    # to AirSim's default 90° if the camera info call is unavailable.
    render_hfov_deg = 90.0
    try:
        cam_info = client.simGetCameraInfo(args.camera_name)
        fov = float(getattr(cam_info, "fov", 0.0) or 0.0)
        if fov > 1.0:
            render_hfov_deg = fov
    except Exception:
        pass

    dt = 1.0 / float(args.tick_hz)
    total_ticks = int(args.duration * args.tick_hz)
    frames: list[FrameRecord] = []

    # ---- FlySeek-Bench standardized visibility + frame export ------------
    # Converts the demo's existing view judgment into paper-consistent fields
    # (in_camera_frustum / line_of_sight_clear / target_visible /
    # visibility_score / occlusion_risk) and writes them to frames.jsonl.
    vis_evaluator = VisibilityEvaluator(
        max_range_m=float(getattr(args, "vis_max_range_m", 100.0)),
        drone_eye_agl_m=float(args.follow_altitude),
    )
    bench_camera_cfg = CameraConfig(
        name=str(args.camera_name),
        hfov_deg=float(render_hfov_deg),
        pitch_deg=float(args.camera_pitch_deg),
        body_forward_m=float(args.camera_body_forward_m),
        body_down_m=float(args.camera_body_down_m),
    )
    frames_jsonl_path = out_dir / "frames.jsonl"
    bench_export_warned = False

    # ---- language-conditioned instruction (attribute-grounded) -----------
    instr_behavior = getattr(args, "target_behavior", None) or str(args.scenario)
    instr_difficulty = (
        str(args.target_policy_difficulty)
        if getattr(args, "target_behavior", None) else str(args.difficulty)
    )
    instr_context: dict[str, Any] = {}
    if occupancy is not None:
        # The car is road-constrained, so "moving along the street" is a valid,
        # non-hallucinated motion context. Occlusion context only when the
        # target actively seeks cover.
        instr_context["motion"] = "the street"
        if instr_behavior == "occlusion_seeking":
            instr_context["occlusion"] = "an occluded street"
    instruction_text = ""
    try:
        instr_record = InstructionGenerator(seed=args.seed).generate(
            target_class=str(target_label),
            target_attributes=attributes_from_label(target_label),
            initial_context=instr_context,
            behavior_type=str(instr_behavior),
            difficulty_level=instr_difficulty,
        )
        instruction_text = str(instr_record["instruction"])
        write_instruction_json(instr_record, out_dir / "instruction.json")
        report.target_label = report.target_label or target_label
        print(f"[ok] instruction [{instr_record['template_family']}]: "
              f"\"{instruction_text}\"")
    except Exception as e:
        report.warnings.append(f"instruction generation failed: {e}")

    try:
        episode_meta = EpisodeMetadata(
            episode_id=str(episode_tag),
            scene_id=str(args.env),
            difficulty_level=str(args.difficulty),
            target_behavior_type=str(instr_behavior),
            target_class=str(target_label),
            instruction=instruction_text,
            random_seed=(int(args.seed) if args.seed is not None else None),
            max_steps=int(total_ticks),
            camera_config=bench_camera_cfg,
            uav_initial_pose=[float(v) for v in drone_state.position]
            + [float(drone_state.heading)],
            target_initial_pose=[float(v) for v in target_state.position]
            + [float(target_state.heading)],
            environment_summary={"env": str(args.env)},
            occluder_summary={"source": "pcd_occupancy" if occupancy else "none"},
        )
        save_metadata_json(episode_meta, out_dir / "metadata.json")
    except Exception as e:
        report.warnings.append(f"bench metadata.json export failed: {e}")

    print(f"\n[loop] {total_ticks} ticks at {args.tick_hz} Hz "
          f"({args.duration} s wall), output → {out_dir}")

    try:
        for tick in range(total_ticks):
            t_sim = tick * dt
            target_state = target_state.copy_with(timestamp=t_sim)
            drone_state = drone_state.copy_with(timestamp=t_sim)

            prev_target_pos = target_state.position.copy()
            if target_policy is not None:
                # FlySeek-Bench adversarial policy path. It performs integration
                # + ground/collision stabilization internally, so we bypass the
                # legacy adversary/street/stabilize chain here.
                target_state = target_policy.get_next_target_state(
                    t_sim, target_state, drone_state, history=frames,
                )
                action = target_policy.last_action
            else:
                action: AgentAction = agent.step(drone_state, target_state, dt=dt)
                if (street_helper is not None
                        and action.behavior_state not in ("open_road", "goto_hide")):
                    street_helper.update(t_sim, target_state)
                    action = street_helper.bias_action(action)
                target_state = integrate_target(
                    target_state, action, dt=dt,
                    keep_z=None,
                    max_speed=max_speed,
                    max_turn_rate_rad_s=max_turn_rate,
                )
                if occupancy is not None:
                    target_state = stabilize_car_state(
                        prev_target_pos,
                        target_state,
                        occupancy,
                        keep_z=None,
                        max_turn_rate_rad_s=max_turn_rate * 0.5,
                        dt=dt,
                    )

            tracker_log: dict[str, Any] = {}
            if tracker is not None:
                drone_state, tracker_log = tracker.step(
                    drone_state, target_state, dt=dt
                )
            else:
                drone_state, _ = _chase_drone_pose(
                    target_state, args, dt, drone_state, occupancy=occupancy
                )

            # 6b'. post-control low-pass: kill camera jitter without touching
            # the tracker. We re-pack the smoothed pose into drone_state so the
            # NEXT tick's tracker sees the same pose AirSim was teleported to.
            if smoother is not None:
                smooth_pos, smooth_yaw, smooth_vel = smoother.filter(
                    drone_state.position, drone_state.heading, dt
                )
                drone_state = drone_state.copy_with(
                    position=smooth_pos,
                    heading=smooth_yaw,
                    velocity=smooth_vel,
                )

            # 6c. teleport both
            t_world_pose = _make_airsim_pose(
                target_state.position[0], target_state.position[1],
                target_state.position[2], target_state.heading,
            )
            d_world_pose = _make_airsim_pose(
                drone_state.position[0], drone_state.position[1],
                drone_state.position[2], drone_state.heading,
            )
            try:
                try:
                    client.simSetObjectPose(target_name, t_world_pose, teleport=True)
                except TypeError:
                    client.simSetObjectPose(target_name, t_world_pose)
                client.simSetVehiclePose(d_world_pose, ignore_collision=True)
            except Exception as e:
                report.errors.append(f"tick {tick} teleport failed: {e}")
                break

            # 6d. capture
            saved: dict[str, str] = {}
            requests = [airsim.ImageRequest(
                args.camera_name, airsim.ImageType.Scene, False, False)]
            include_extra = (tick % max(1, args.modalities_stride) == 0)
            if include_extra:
                requests.append(airsim.ImageRequest(
                    args.camera_name, airsim.ImageType.DepthPlanar, True, False))
                requests.append(airsim.ImageRequest(
                    args.camera_name, airsim.ImageType.Segmentation, False, False))
            try:
                responses = client.simGetImages(requests)
            except Exception as e:
                report.errors.append(f"tick {tick} simGetImages failed: {e}")
                break

            kinds = ["rgb", "depth", "segmentation"][:len(responses)]
            bbox_2d: list[float] | None = None
            for resp, kind in zip(responses, kinds):
                fname = f"frame_{tick:04d}_{kind}.png"
                if _save_image(resp, frames_dir / fname, kind):
                    saved[kind] = str(frames_dir / fname)
                if kind == "segmentation" and target_seg_id is not None:
                    seg_arr = _seg_array_from_response(resp)
                    if seg_arr is not None:
                        uv = project_ned_to_pixel(
                            target_state.position,
                            drone_state.position,
                            drone_state.heading,
                            width=int(seg_arr.shape[1]),
                            height=int(seg_arr.shape[0]),
                            hfov_deg=render_hfov_deg,
                            cam_forward_m=float(args.camera_body_forward_m),
                            cam_down_m=float(args.camera_body_down_m),
                            cam_pitch_deg=float(args.camera_pitch_deg),
                        )
                        if uv is not None:
                            bbox_2d = bbox_from_segmentation(seg_arr, uv)

            # Smooth the topdown anchor every tick (not only on capture ticks)
            # so the EMA state stays up-to-date with car motion between snaps.
            top_anchor_xy_yaw = topdown_anchor.update(target_state, dt)
            if capture_topdown and tick % topdown_stride == 0:
                top_path = frames_dir / f"frame_{tick:04d}_topdown.png"
                if _capture_topdown_frame(
                    client, args, target_state, drone_state,
                    chase_cam_pose, top_path,
                    anchor_xy_yaw=top_anchor_xy_yaw,
                ):
                    saved["topdown"] = str(top_path)

            # 6e. record
            r = horizontal_distance(target_state.position, drone_state.position)
            from flyseek.adversary import bearing_xy as _bearing
            bearing = _bearing(target_state.position, drone_state.position)
            vis, vis_reason = visibility_status(
                occupancy, drone_state, target_state,
                hfov_deg=float(args.camera_hfov_deg),
                max_range_m=float(getattr(args, "vis_max_range_m", 100.0)),
                drone_eye_agl_m=max(
                    float(args.follow_altitude),
                    float(-drone_state.position[2]),
                ),
                seg_building_map=getattr(args, "_seg_building_map", None),
                include_pcd_occluders=bool(getattr(args, "los_include_trees", False)),
            )
            rec = FrameRecord(
                frame_idx=tick,
                timestamp_s=t_sim,
                drone_pose_world={
                    "x": float(drone_state.position[0]),
                    "y": float(drone_state.position[1]),
                    "z": float(drone_state.position[2]),
                    "yaw": float(drone_state.heading),
                },
                target_pose_world={
                    "x": float(target_state.position[0]),
                    "y": float(target_state.position[1]),
                    "z": float(target_state.position[2]),
                    "yaw": float(target_state.heading),
                },
                target_velocity=[float(v) for v in target_state.velocity],
                drone_distance_m=r,
                drone_bearing_to_target_rad=float(bearing),
                adversary_log={
                    **action.decision_log,
                    **(tracker_log if tracker is not None else {}),
                },
                tracker_mode=tracker_log.get("tracker_mode", "track")
                if tracker is not None else "",
                target_visible=vis,
                images_saved=saved,
            )
            frames.append(rec)
            rgb_path = Path(saved["rgb"]) if "rgb" in saved else None
            episode_writer.write_frame(
                tick,
                drone_state,
                target_state,
                trajectory_record=asdict(rec),
                target_visible=vis,
                tracker_mode=rec.tracker_mode,
                adversary_log=rec.adversary_log,
                rgb_src=rgb_path,
                vis_reason=vis_reason,
                target_name=target_name,
                asset_name=target_name,
                seg_id=target_seg_id,
                bbox_2d=bbox_2d,
            )

            # 6f. standardized FlySeek-Bench FrameMetadata export.
            try:
                img_w = int(responses[0].width) if responses else None
                img_h = int(responses[0].height) if responses else None
                cam_cfg = {
                    "name": str(args.camera_name),
                    "hfov_deg": float(render_hfov_deg),
                    "pitch_deg": float(args.camera_pitch_deg),
                    "body_forward_m": float(args.camera_body_forward_m),
                    "body_down_m": float(args.camera_body_down_m),
                    "width": img_w,
                    "height": img_h,
                }
                eye_agl = max(float(args.follow_altitude),
                              float(-drone_state.position[2]))
                vstd = vis_evaluator.evaluate_frame(
                    uav_pose=[float(v) for v in drone_state.position]
                    + [float(drone_state.heading)],
                    target_pose=[float(v) for v in target_state.position],
                    camera_config=cam_cfg,
                    scene_context={
                        "occupancy": occupancy,
                        "max_range_m": float(getattr(args, "vis_max_range_m", 100.0)),
                        "drone_eye_agl_m": eye_agl,
                    },
                    existing_visibility_metadata={
                        "target_visible": vis,
                        "vis_reason": vis_reason,
                    },
                )
                collision_flag = False
                if occupancy is not None:
                    try:
                        collision_flag = bool(occupancy.is_3d_occupied_map(
                            airsim_ned_to_map(drone_state.position)
                        ))
                    except Exception:
                        collision_flag = False
                frame_meta = FrameMetadata(
                    frame_id=tick,
                    image_path=str(saved.get("rgb")
                                   or (frames_dir / f"frame_{tick:04d}_rgb.png")),
                    step_index=tick,
                    timestamp=float(t_sim),
                    uav_pose=[float(v) for v in drone_state.position]
                    + [float(drone_state.heading)],
                    target_pose=[float(v) for v in target_state.position]
                    + [float(target_state.heading)],
                    uav_velocity=[float(v) for v in drone_state.velocity],
                    target_velocity=[float(v) for v in target_state.velocity],
                    target_visible=bool(vstd["target_visible"]),
                    in_camera_frustum=vstd["in_camera_frustum"],
                    line_of_sight_clear=vstd["line_of_sight_clear"],
                    visibility_score=vstd["visibility_score"],
                    distance_to_target=float(vstd["distance_to_target"]),
                    relative_bearing=float(vstd["relative_bearing"]),
                    occlusion_risk=vstd["occlusion_risk"],
                    selected_viewpoint=[float(v) for v in drone_state.position],
                    collision=collision_flag,
                    target_behavior_type=str(action.behavior_state),
                    difficulty_level=str(args.difficulty),
                    extra={
                        "tracker_mode": rec.tracker_mode,
                        "vis_reason": vis_reason,
                        "visibility_source": vstd["visibility_source"],
                        "bbox_2d": bbox_2d,
                        "seg_id": target_seg_id,
                    },
                )
                append_frame_jsonl(frame_meta, frames_jsonl_path)
            except Exception as e:
                if not bench_export_warned:
                    bench_export_warned = True
                    report.warnings.append(
                        f"bench frames.jsonl export failed (frame {tick}): {e}"
                    )

            log_stride = max(1, int(round(float(args.tick_hz))))
            if tick % log_stride == 0:
                extra = ""
                if tracker is not None:
                    extra = (f"  vis={'Y' if vis else 'N'}  "
                             f"drone={tracker_log.get('tracker_mode', '?')}")
                print(f"  tick {tick:4d}/{total_ticks}  "
                      f"t={t_sim:5.2f}s  "
                      f"target_xy=({target_state.position[0]:7.1f},"
                      f"{target_state.position[1]:7.1f})  "
                      f"r={r:5.1f}m  "
                      f"car={action.behavior_state}{extra}")
    finally:
        episode_writer.close(
            target_name=target_name,
            target_label=target_label,
            target_pos_ned=np.array([
                float(target_pose_air.position.x_val),
                float(target_pose_air.position.y_val),
                float(target_pose_air.position.z_val),
            ]),
        )

    report.frames_captured = len(frames)
    if frames:
        report.final_drone_target_distance_m = frames[-1].drone_distance_m

    # ---- visibility-aware expert viewpoint annotation -> trajectories.json --
    # Offline oracle: scores candidate UAV viewpoints over a short horizon of the
    # target's upcoming positions (preemptive), so the reference is visibility-
    # aware rather than shortest-path follow.
    if frames:
        try:
            target_traj = [
                {"t": fr.timestamp_s,
                 "pos": [fr.target_pose_world["x"], fr.target_pose_world["y"],
                         fr.target_pose_world["z"]],
                 "vel": fr.target_velocity}
                for fr in frames
            ]
            uav_traj = [
                {"t": fr.timestamp_s,
                 "pos": [fr.drone_pose_world["x"], fr.drone_pose_world["y"],
                         fr.drone_pose_world["z"]],
                 "heading": fr.drone_pose_world.get("yaw", 0.0)}
                for fr in frames
            ]
            stride = max(1, len(frames) // 200)
            expert_cfg = ExpertTrajectoryConfig(
                follow_distance_m=float(args.follow_distance),
                follow_altitude_m=float(args.follow_altitude),
                hfov_deg=float(args.camera_hfov_deg),
                max_range_m=float(getattr(args, "vis_max_range_m", 100.0)),
                plan_stride=stride,
            )
            planner = ExpertViewpointPlanner(
                config=expert_cfg,
                scene_context={"occupancy": occupancy},
                seed=args.seed,
            )
            expert_out = planner.plan(target_traj, uav_trajectory=uav_traj)
            save_trajectories(expert_out, out_dir / "trajectories.json")
            print(f"[ok] expert viewpoints → trajectories.json "
                  f"({len(expert_out['expert_viewpoints'])} planned, stride={stride})")
        except Exception as e:
            report.warnings.append(f"expert trajectory annotation failed: {e}")

        # ---- episode-level evaluation metrics -> metrics.json -------------
        try:
            m = _bench_eval_episode(out_dir, write=True)
            print(f"[ok] metrics → metrics.json (success={m['tracking_success']}, "
                  f"vis_ratio={m['target_visibility_ratio']}, "
                  f"los_continuity={m['line_of_sight_continuity']})")
        except Exception as e:
            report.warnings.append(f"metrics computation failed: {e}")

    # ---- 7. restore target -------------------------------------------------
    try:
        try:
            client.simSetObjectPose(target_name, target_pose_air, teleport=True)
        except TypeError:
            client.simSetObjectPose(target_name, target_pose_air)
        report.target_restored = True
        print(f"\n[ok] target restored to original pose")
    except Exception as e:
        report.warnings.append(f"target restore failed: {e}")

    report.success = (report.frames_captured > 0 and not report.errors)

    if report.success and args.make_mp4:
        mp4_name = "hide_seek.mp4" if args.scenario == "hide_seek" else "chase.mp4"
        mp4_path = out_dir / mp4_name
        if _render_mp4(frames_dir, mp4_path, float(args.tick_hz),
                       glob_suffix="_rgb.png"):
            print(f"[ok] drone-view mp4 → {mp4_path}")
        else:
            report.warnings.append("ffmpeg not available or mp4 render failed")

        if capture_topdown:
            top_mp4 = out_dir / "topdown.mp4"
            if _render_mp4(
                frames_dir,
                top_mp4,
                effective_topdown_fps,
                glob_suffix="_topdown.png",
                output_fps=float(args.tick_hz),
            ):
                print(f"[ok] top-down mp4 → {top_mp4}")
            else:
                report.warnings.append("topdown mp4 render failed")

    return report


# --------------------------------------------------------------------------- #
# Tracking-difficulty presets                                                 #
# --------------------------------------------------------------------------- #
TRACKING_DIFFICULTY_PRESETS: dict[str, dict[str, Any]] = {
    "easy": {
        # Drone: high + close, very forgiving FSM (effectively always TRACK).
        "follow_altitude": 24.0,
        "follow_distance": 10.0,
        "camera_hfov_deg": 60.0,
        "tracker_predict_after_s": 1.0,
        "tracker_reacquire_after_s": 2.5,
        "tracker_peek_after_s": 2.0,
        "tracker_search_after_s": 10.0,
        # Smaller tau → drone reacts within ~0.4s of a heading change.
        "tracker_motion_dir_tau": 0.5,
        "tracker_lead_s": 0.6,
        "tracker_yaw_gain": 4.0,
        # Snappier position controller (was 3.0). Larger = position responds
        # faster to desired anchor changes, less trailing on corners.
        "drone_smoothing": 5.0,
        "vis_max_range_m": 120.0,
        # Car: open road forever, no hiding.
        "hide_overrides": {
            "use_open_road_phase": True,
            "open_road_duration_s": 9999.0,
            "open_road_route_len_m": 240.0,
            "open_road_min_route_frac": 1.5,   # never satisfied → never hide
            "open_road_speed_mps": 4.0,
            "hide_trigger_range_m": 0.0,       # never trigger hide
        },
    },
    "medium": {
        # Drone: moderate altitude + standard FSM (PEEK enabled).
        "follow_altitude": 18.0,
        "follow_distance": 12.0,
        "camera_hfov_deg": 55.0,
        "tracker_predict_after_s": 0.4,
        "tracker_reacquire_after_s": 1.2,
        "tracker_peek_after_s": 0.8,
        "tracker_search_after_s": 4.0,
        "tracker_motion_dir_tau": 0.8,
        "tracker_lead_s": 0.7,
        "tracker_yaw_gain": 3.5,
        "drone_smoothing": 4.5,
        "vis_max_range_m": 100.0,
        # Car: drive open road then duck behind a building once.
        "hide_overrides": {
            "use_open_road_phase": True,
            "open_road_duration_s": 12.0,
            "open_road_route_len_m": 140.0,
            "open_road_speed_mps": 4.5,
            "hide_trigger_range_m": 40.0,
            "hide_duration_s": 12.0,
            "peek_after_hide_s": 6.0,
        },
    },
    "hard": {
        # Drone: low altitude, twitchy give-up (transitions to SEARCH fast).
        "follow_altitude": 14.0,
        "follow_distance": 14.0,
        "camera_hfov_deg": 45.0,
        "tracker_predict_after_s": 0.2,
        "tracker_reacquire_after_s": 0.6,
        "tracker_peek_after_s": 99.0,          # PEEK disabled — straight to SEARCH
        "tracker_search_after_s": 1.5,
        # Snappy but stable — tau=0.5 strikes the balance for fast cars.
        "tracker_motion_dir_tau": 0.5,
        "tracker_lead_s": 0.5,
        "tracker_yaw_gain": 4.0,
        "drone_smoothing": 5.0,
        "vis_max_range_m": 80.0,
        # Car: fast evasive — short open road then sharp dive into cover.
        "hide_overrides": {
            "use_open_road_phase": True,
            "open_road_duration_s": 6.0,
            "open_road_route_len_m": 90.0,
            "open_road_min_route_frac": 0.25,
            "open_road_speed_mps": 6.0,
            "hide_trigger_range_m": 80.0,      # trigger hide regardless of distance
            "hide_speed": 5.5,
            "hide_duration_s": 25.0,
            "hide_search_radius_m": 40.0,
        },
    },
}


def _apply_tracking_difficulty_preset(args: Any) -> dict[str, Any]:
    """Apply ``--tracking-difficulty`` preset to ``args``; return car overrides."""
    preset_name = getattr(args, "tracking_difficulty", None)
    if not preset_name:
        return {}
    preset = TRACKING_DIFFICULTY_PRESETS[preset_name]
    overrides: dict[str, Any] = {}
    for key, value in preset.items():
        if key == "hide_overrides":
            overrides = dict(value)
            continue
        if hasattr(args, key):
            setattr(args, key, value)
    print(f"[ok] tracking-difficulty preset = '{preset_name}': "
          f"follow_alt={args.follow_altitude:.0f}m follow_dist={args.follow_distance:.0f}m "
          f"search_after={args.tracker_search_after_s:.1f}s "
          f"peek_after={args.tracker_peek_after_s:.1f}s")
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FlySeek Phase 1 demo — adversarial car evades chasing drone.",
    )
    parser.add_argument("--airsim-ip", default=os.environ.get("AIRSIM_IP", "127.0.0.1"))
    parser.add_argument("--airsim-port", type=int,
                        default=int(os.environ.get("AIRSIM_RPC_PORT", 41451)))

    tg = parser.add_argument_group("target picking")
    tg.add_argument("--target", default=None,
                    help="Exact scene actor name (skips scout/regex).")
    tg.add_argument("--auto-from-scout", action="store_true",
                    help=f"Use first candidate from {DEFAULT_SCOUT_FILE}.")
    tg.add_argument("--scout-file", type=Path, default=None)
    tg.add_argument("--target-regex", default=None)
    tg.add_argument("--label", default=None,
                    help="Override natural-language label.")
    tg.add_argument("--motorized-cars-only", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Only allow real motorized cars (excludes Cart/Bus/etc.). "
                         "Default ON. Use --no-motorized-cars-only to disable.")
    tg.add_argument("--target-index", type=int, default=0,
                    help="Pick Nth street-safe motorized car from scout list (batch).")
    tg.add_argument("--episode-tag", default=None,
                    help="Output sub-folder name (default: timestamp).")

    ag = parser.add_argument_group("adversary")
    ag.add_argument("--scenario", choices=["chase", "hide_seek"], default="chase",
                    help="chase=S-evade only; hide_seek=car hides, drone searches.")
    ag.add_argument("--difficulty", choices=["easy", "medium", "hide_seek"],
                    default="medium",
                    help="Adversary difficulty (hide_seek auto-set by --scenario).")
    ag.add_argument("--config", type=Path,
                    default=REPO_ROOT / "flyseek_extend" / "configs"
                            / "adversarial_agent.yaml",
                    help="YAML config; falls back to module defaults if missing.")
    ag.add_argument("--seed", type=int, default=None,
                    help="RNG seed (only matters for easy / cruise jitter).")
    ag.add_argument("--play-box-half-extent", type=float, default=0.0,
                    help="Optional XY play box (m). 0 = disabled (use PCD instead).")
    ag.add_argument("--target-behavior",
                    choices=list(BEHAVIOR_TYPES), default=None,
                    help="Use a FlySeek-Bench adversarial target policy instead "
                         "of the built-in adversary agent: direct_escape | "
                         "sharp_turn | detour_feint | occlusion_seeking | "
                         "alley_hutong. "
                         "When unset, the legacy scenario/difficulty agent runs.")
    ag.add_argument("--target-policy-difficulty",
                   choices=["easy", "medium", "hard"], default="medium",
                   help="Difficulty preset for --target-behavior "
                        "(speed / turn frequency / occlusion preference).")

    oh = parser.add_argument_group(
        "occlusion_seeking building hide (PCD route; only with --target-behavior occlusion_seeking)")
    oh.add_argument("--min-building-height-m", type=float, default=18.0,
                    help="Min PCD column height (m) to count as a large building occluder. "
                         "Raise to ignore lamp posts / thin poles (try 20–25).")
    oh.add_argument("--min-building-footprint-cells", type=int, default=9,
                    help="Min building BEV cells in 5×5 window on the blocker. "
                         "Raise to require wider footprints (try 12–16).")
    oh.add_argument("--hide-search-radius-m", type=float, default=48.0,
                    help="Radius (m) to search for a building hide goal behind a wall.")
    oh.add_argument("--route-search-radius-m", type=float, default=220.0,
                    help="Radius (m) for open_then_hide major-road route seed search.")
    oh.add_argument("--route-max-attempts", type=int, default=16,
                    help="Rebuild attempts until building hide leg validates.")
    oh.add_argument("--min-building-occluded-frac", type=float, default=0.55,
                    help="Min fraction of hide-leg waypoints building-occluded (0–1).")
    oh.add_argument("--building-probe-dist-m", type=float, default=7.5,
                    help="Distance (m) to probe for an adjacent building wall at hide goal.")
    oh.add_argument("--require-adjacent-building",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Hide goal must sit beside a wide building footprint (default on).")
    oh.add_argument("--open-road-frac", type=float, default=0.45,
                    help="Fraction of route on open main road before entering "
                         "alley/hide leg (0.45 → hide starts ~45%% into route; "
                         "default 0.45, was 0.68). Lower = earlier hutong entry.")
    oh.add_argument("--route-len-m", type=float, default=0.0,
                    help="Override PCD route length (m). 0 = use difficulty preset "
                         "(hard≈180). Try 140–160 for shorter total run.")
    oh.add_argument("--los-include-trees", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Count tall non-building PCD columns (trees / foliage / "
                         "poles / structures) as line-of-sight occluders in the "
                         "recorded visibility judgment, not only annotated "
                         "buildings. A tree/foliage block is tagged with reason "
                         "'los_blocked_occluder' (vs 'los_blocked' for buildings) "
                         "so the comparison figure can label it. Default off "
                         "(buildings/seg_map only).")
    oh.add_argument("--seg-building-jsonl", type=Path, default=None,
                    help="Use ONLY annotated buildings from seg_map JSONL for hide "
                         "planning (e.g. scene_data/seg_map/env_airsim_16.jsonl).")
    oh.add_argument("--seg-building-radius-m", type=float, default=10.0,
                    help="Annotated building footprint radius (m) for LoS blocking.")
    oh.add_argument("--seg-building-min-height-m", type=float, default=8.0,
                    help="Min annotated building height (m) to count as occluder.")

    ah = parser.add_argument_group(
        "alley_hutong (car drives into narrow gap between buildings)")
    ah.add_argument("--alley-near-entry", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Teleport target near the hutong entry road before chase "
                         "(default on; strongly recommended for visible alley dive).")
    ah.add_argument("--open-approach-m", type=float, default=45.0,
                    help="Open-road approach length (m) before entering the alley.")
    ah.add_argument("--max-corridor-width-m", type=float, default=12.0,
                    help="Max drivable corridor width (m) to count as a hutong "
                         "(lower = narrower alleys only).")

    cc = parser.add_argument_group("chase camera (drone)")
    cc.add_argument("--follow-distance", type=float, default=12.0,
                    help="m, ideal distance drone keeps behind target.")
    cc.add_argument("--follow-altitude", type=float, default=12.0,
                    help="m, nominal drone altitude (PCD may lift higher).")
    cc.add_argument("--drone-smoothing", type=float, default=3.0,
                    help="Drone pose 1st-order smoothing (1/s). Bigger = snappier.")
    cc.add_argument("--camera-name", default="front_custom")
    cc.add_argument("--camera-pitch-deg", type=float, default=55.0,
                    help="Downward pitch (deg). 55° keeps propellers out of frame.")
    cc.add_argument("--camera-body-forward-m", type=float, default=0.45,
                    help="Camera mount forward offset on drone body (+X, m).")
    cc.add_argument("--camera-body-down-m", type=float, default=0.25,
                    help="Camera mount downward offset on drone body (+Z NED, m).")
    cc.add_argument("--camera-hfov-deg", type=float, default=50.0,
                    help="Horizontal FOV for visibility / lost-target detection.")
    cc.add_argument("--lost-after-s", type=float, default=0.6,
                    help="Seconds target occluded before drone enters SEARCH.")
    cc.add_argument("--search-orbit-radius", type=float, default=14.0,
                    help="Orbit radius (m) around last known position in SEARCH.")
    cc.add_argument("--altitude-smooth-tau", type=float, default=3.0,
                    help="Drone altitude low-pass time constant (s).")
    cc.add_argument("--roof-smooth-tau", type=float, default=6.0,
                    help="EMA time constant for roof ceiling (OpenFly-style, s).")
    cc.add_argument("--roof-probe-range-m", type=float, default=2.0,
                    help="Local window (m) for getMaxZinP-style roof probe.")
    cc.add_argument("--camera-z-smooth-tau", type=float, default=1.2,
                    help="Extra vertical low-pass on rendered camera pose (s).")
    cc.add_argument("--max-climb-mps", type=float, default=1.5,
                    help="Max upward NED speed (reduces vertical jitter).")
    cc.add_argument("--max-drop-mps", type=float, default=2.0,
                    help="Max downward NED speed.")
    cc.add_argument("--camera-pos-smooth-tau", type=float, default=0.35,
                    help="Post-control drone position low-pass τ (s). "
                         "Larger = smoother camera, more lag. Default 0.35.")
    cc.add_argument("--camera-yaw-smooth-tau", type=float, default=0.50,
                    help="Post-control drone yaw low-pass τ (s). "
                         "Larger = smoother pan, more lag. Default 0.50.")
    cc.add_argument("--camera-max-yaw-rate-dps", type=float, default=60.0,
                    help="Hard cap on camera yaw rate (deg/s) to suppress "
                         "snap rotations. Default 60.")
    cc.add_argument("--no-camera-smoothing", action="store_true",
                    help="Disable post-control pose smoothing (debug).")
    cc.add_argument("--tracker-mode",
                    choices=["adaptive", "inline", "legacy", "reactive",
                             "reactive_lost"],
                    default="adaptive",
                    help="adaptive=6-state FSM (TRACK/PREDICT/REACQUIRE/PEEK/"
                         "SEARCH/HOLD, decides when to chase vs hold). "
                         "inline=TRACK+SEARCH (orbits last_seen anchor). "
                         "reactive=baseline follower (chase current target pose, "
                         "no occlusion handling). reactive_lost=memoryless "
                         "baseline that stalls near the last-seen spot and "
                         "wanders aimlessly once occluded (never reacquires; "
                         "use for the 'car hides behind a building' scenario). "
                         "legacy=TrackingDroneController. Default adaptive.")
    cc.add_argument("--lost-wander-radius-m", type=float, default=6.0,
                    help="reactive_lost: radius (m) of the aimless loiter around "
                         "the last-seen position after the target is lost.")
    cc.add_argument("--lost-wander-scan-dps", type=float, default=35.0,
                    help="reactive_lost: yaw sweep rate (deg/s) while wandering.")
    cc.add_argument("--force-tracker", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Build a drone tracker even in the 'chase' scenario "
                         "(default only hide_seek builds one). Lets the "
                         "alley-chase comparison run --tracker-mode adaptive "
                         "(FlySeek) vs reactive (baseline) with identical "
                         "scene/camera parameters.")
    cc.add_argument("--use-inline-tracker", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Deprecated: use --tracker-mode instead. "
                         "--no-use-inline-tracker forces legacy controller.")
    cc.add_argument("--tracker-predict-after-s", type=float, default=0.4,
                    help="Adaptive tracker: enter PREDICT after this many "
                         "seconds without visibility. Default 0.4.")
    cc.add_argument("--tracker-reacquire-after-s", type=float, default=1.2,
                    help="Adaptive tracker: enter REACQUIRE (yaw-scan toward "
                         "predicted pose) after this many seconds. Default 1.2.")
    cc.add_argument("--tracker-peek-after-s", type=float, default=0.8,
                    help="Adaptive tracker: enter PEEK (side-step around "
                         "occluder) when LOS-blocked this long. Default 0.8.")
    cc.add_argument("--tracker-search-after-s", type=float, default=3.0,
                    help="Adaptive tracker: enter SEARCH (expanding spiral on "
                         "predicted pose) after this many seconds. Default 3.0.")
    cc.add_argument("--tracker-hold-speed", type=float, default=0.3,
                    help="Adaptive tracker: target speed (m/s) below which "
                         "HOLD is allowed. Default 0.3.")
    cc.add_argument("--tracker-hold-dwell-s", type=float, default=4.0,
                    help="Adaptive tracker: hold-eligible dwell duration (s) "
                         "before drone stops chasing. Default 4.0.")
    cc.add_argument("--tracker-hold-resume-speed", type=float, default=1.0,
                    help="Adaptive tracker: target speed (m/s) that wakes "
                         "HOLD back to TRACK. Default 1.0.")
    cc.add_argument("--tracker-motion-dir-tau", type=float, default=1.5,
                    help="Low-pass τ (s) for target motion direction in the "
                         "inline tracker. Larger = drone holds its bearing "
                         "better when target turns. Default 1.5.")
    cc.add_argument("--tracker-lead-s", type=float, default=0.7,
                    help="Lead-pursuit lookahead (s). Drone aims for the "
                         "predicted target position lead_s seconds ahead. "
                         "Default 0.7. Set 0 to disable.")
    cc.add_argument("--tracker-yaw-gain", type=float, default=2.0,
                    help="Yaw control gain (1/s) for the inline tracker. "
                         "Default 2.0.")
    cc.add_argument("--search-orbit-speed-dps", type=float, default=30.0,
                    help="Orbit angular velocity (deg/s) in SEARCH mode. "
                         "Default 30.")
    cc.add_argument("--tracker-fov-center-gain", type=float, default=10.0,
                    help="Lateral position correction (m/rad) to keep the "
                         "target centered in the camera FOV. Default 10.")
    cc.add_argument("--vis-max-range-m", type=float, default=100.0,
                    help="Max distance (m) at which target is considered "
                         "visible. Default 100m.")
    cc.add_argument("--tracking-difficulty",
                    choices=["easy", "medium", "hard"], default=None,
                    help=(
                        "One-shot preset that wires tracker + car-agent "
                        "parameters to a chosen tracking scenario:\n"
                        "  easy   — drone flies high (~24m) and close (~10m); "
                        "car drives straight on the open road, NEVER hides; "
                        "tracker stays in TRACK throughout.\n"
                        "  medium — drone flies at moderate alt (~18m); car "
                        "drives the open road then ducks behind a building; "
                        "tracker uses PEEK/REACQUIRE to recover.\n"
                        "  hard   — drone flies low (~14m); car drives fast "
                        "and makes a sudden evasive maneuver; tracker should "
                        "lose target and end in SEARCH.\n"
                        "Overrides the corresponding flags when set."
                    ))

    st = parser.add_argument_group("street motion (car)")
    st.add_argument("--street-follow", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Bias car motion along PCD street rays with random turns.")
    st.add_argument("--street-blend", type=float, default=0.4,
                    help="Blend weight toward street heading [0,1].")

    ini = parser.add_argument_group("target init (PCD road placement)")
    ini.add_argument("--skip-target-init", action="store_true",
                     help="Keep target at scene spawn pose (debug).")
    ini.add_argument("--init-profile", default=None,
                     help="Target init preset: strict | standard | major_road "
                          "(default from env YAML).")
    ini.add_argument("--init-search-radius-m", type=float, default=None,
                     help="Override spiral search radius (m); default from preset.")
    ini.add_argument("--init-min-corridor-width-m", type=float, default=None,
                     help="Override min corridor width (m); default from preset.")
    ini.add_argument("--allow-bad-init", action="store_true",
                     help="Record a demo even if target init failed (the car "
                          "would stay at its scene spawn, often inside a "
                          "building). Off by default — demo aborts instead.")

    td = parser.add_argument_group("top-down (car-overhead) view")
    td.add_argument("--topdown", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="Synchronously capture a top-down view above the car. "
                         "Default ON for hide_seek, OFF for chase.")
    td.add_argument("--topdown-altitude", type=float, default=30.0,
                    help="Altitude (m AGL) of overhead camera above the car.")
    td.add_argument("--topdown-fps", type=float, default=5.0,
                    help="Top-down capture FPS. It is re-timed to --tick-hz in "
                         "topdown.mp4. Default 5 keeps 30Hz demos practical.")
    td.add_argument("--topdown-xy-smooth-tau", type=float, default=0.6,
                    help="EMA τ (s) for top-down camera XY anchor. Bigger = "
                         "smoother but more lag. Default 0.6.")
    td.add_argument("--topdown-yaw-smooth-tau", type=float, default=1.0,
                    help="EMA τ (s) for top-down camera yaw when "
                         "--no-topdown-lock-north is set. Default 1.0.")
    td.add_argument("--topdown-lock-north", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Lock top-down yaw to world north (most stable). "
                         "Disable to follow the car heading via EMA.")

    sim = parser.add_argument_group("simulation")
    sim.add_argument("--frames", type=int, default=None,
                     help="Exact frame count (overrides --duration).")
    sim.add_argument("--duration", type=float, default=None,
                     help="Simulation seconds. Default = 40 s (chase) / 35 s (hide_seek).")
    sim.add_argument("--tick-hz", type=float, default=None,
                     help="Simulation tick rate (= rendered FPS). "
                          "Default: 5 Hz (chase), 30 Hz (hide_seek).")
    sim.add_argument("--modalities-stride", type=int, default=10,
                     help="Capture depth+seg every N ticks (RGB every frame).")

    col = parser.add_argument_group("collision (OpenFly PCD)")
    col.add_argument("--env", default="env_airsim_16",
                     help="Scene env: PCD map, occupancy cache, and target-init "
                          "presets (configs/target_init_<env>.yaml).")
    col.add_argument("--no-collision", action="store_true",
                     help="Disable PCD collision checks (debug only).")
    col.add_argument("--rebuild-occupancy-cache", action="store_true",
                     help="Force rebuild voxel cache from PCD.")

    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--make-mp4", action="store_true", default=True,
                        help="Auto-render chase.mp4 via ffmpeg (default on).")
    parser.add_argument("--no-mp4", action="store_true",
                        help="Skip ffmpeg mp4 render.")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Hard timeout (s). Default auto-scales with frame count.")
    return parser


def finalize_args(args: argparse.Namespace) -> None:
    """Apply post-parse defaults: tracking-difficulty preset, per-scenario
    tick/duration/topdown defaults, and the auto-scaled timeout. Shared by the
    CLI ``main()`` and the library ``flyseek.pipeline.single_episode``."""
    if args.no_mp4:
        args.make_mp4 = False

    args.hide_seek_overrides = _apply_tracking_difficulty_preset(args)

    # Per-scenario defaults (tick_hz, duration, output dir, topdown ON/OFF).
    if args.scenario == "hide_seek":
        args.difficulty = "hide_seek"
        if args.tick_hz is None:
            # 20 Hz: AirSim teleport + simGetImages RPC roundtrip is typically
            # 30–80 ms, so 30 Hz (33 ms budget) routinely overruns and produces
            # irregular frame intervals → visible camera jitter. 20 Hz gives
            # 50 ms headroom, keeps RPC pacing even, and still looks smooth.
            args.tick_hz = 20.0
        if args.duration is None:
            args.duration = 40.0
        if args.follow_distance == 12.0:
            args.follow_distance = 14.0
        if args.tracker_yaw_gain == 2.0:
            args.tracker_yaw_gain = 3.0
        if args.tracker_fov_center_gain == 10.0:
            args.tracker_fov_center_gain = 12.0
        if args.street_blend == 0.4:
            args.street_blend = 0.15
        if args.altitude_smooth_tau == 3.0:
            args.altitude_smooth_tau = 5.0
        if args.roof_smooth_tau == 6.0:
            args.roof_smooth_tau = 8.0
        if args.camera_z_smooth_tau == 1.2:
            args.camera_z_smooth_tau = 1.8
        if args.output == DEFAULT_OUTPUT_DIR:
            args.output = REPO_ROOT / "flyseek_extend" / "output" / "demo_hide_and_seek"
        if args.topdown is None:
            args.topdown = True
    else:
        if args.tick_hz is None:
            args.tick_hz = 5.0
        if args.duration is None:
            args.duration = 40.0
        if args.topdown is None:
            args.topdown = False

    if args.frames is not None and args.frames > 0:
        args.duration = args.frames / args.tick_hz

    if args.topdown_fps <= 0:
        args.topdown_fps = args.tick_hz
    args.topdown_fps = min(float(args.topdown_fps), float(args.tick_hz))

    if args.timeout is None:
        total_ticks = int(args.duration * args.tick_hz)
        topdown_ticks = int(args.duration * args.topdown_fps) if args.topdown else 0
        # AirVLN RPC/image throughput varies a lot by GPU load. Use a generous
        # upper bound so demos finish, while still catching real hangs.
        args.timeout = max(900.0, 0.65 * total_ticks + 1.2 * topdown_ticks + 180.0)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    finalize_args(args)

    def _on_timeout(_s, _f):
        print(f"\n[FATAL] demo timed out after {args.timeout}s.", flush=True)
        sys.exit(124)
    signal.signal(signal.SIGALRM, _on_timeout)
    signal.alarm(int(args.timeout))

    args.output.mkdir(parents=True, exist_ok=True)

    report = run_demo(args)

    out_dir = Path(report.output_dir) if report.output_dir else args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 64)
    print(f"DEMO RESULT — {'PASS' if report.success else 'FAIL'}")
    print("=" * 64)
    print(f"Target               : {report.target_name} — \"{report.target_label}\"")
    print(f"Difficulty           : {report.difficulty}")
    print(f"Duration / dt        : {report.duration_s}s @ {1/report.sim_dt:.1f} Hz")
    print(f"Frames captured      : {report.frames_captured}")
    print(f"Final drone-target Δ : {report.final_drone_target_distance_m:.1f} m")
    print(f"Output directory     : {report.output_dir}")
    if report.warnings:
        print(f"\nWarnings ({len(report.warnings)}):")
        for w in report.warnings:
            print(f"  - {w}")
    if report.errors:
        print(f"\nErrors ({len(report.errors)}):")
        for e in report.errors:
            print(f"  - {e}")
    if report.success:
        mp4 = Path(report.output_dir) / (
            "hide_seek.mp4" if args.scenario == "hide_seek" else "chase.mp4"
        )
        if mp4.exists():
            print(f"\n📽  Showcase video: {mp4}")
            print(f"     xdg-open {mp4}")
        else:
            print(f"\n📽  生成 mp4（需要 ffmpeg）：")
            print(f"     cd {report.output_dir}/frames")
            print(f'     ffmpeg -framerate {args.tick_hz:.0f} '
                  f'-pattern_type glob -i "frame_*_rgb.png" '
                  f"-c:v libx264 -pix_fmt yuv420p ../chase.mp4")
    print("=" * 64)
    return 0 if report.success else 1


if __name__ == "__main__":
    sys.exit(main())

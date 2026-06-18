# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""End-to-end smoke test: teleport an existing NYC scene actor + capture images.

This is the **Phase 0.5 proof-of-life** — the spawn-based smoke test
(``smoke_test_spawn_and_capture.py``) is shelved because it requires
asset registry injection that we proved unstable. Instead we reuse one of
the 35 528 actors already placed in env_airsim_16 (a parked car, a cart,
etc.) and teleport it to the drone's field of view.

Pipeline:
    1. Pick a target either via --target=<name>, --auto-from-scout, or
       --category=<cat>+probe.
    2. Read drone pose (read-only).
    3. Snapshot the target's original world pose (for later restore).
    4. Compute a "tracking pose" in the drone's local frame (default: 10 m
       ahead, same altitude, yaw=180° so target faces drone).
    5. simSetObjectPose(target, tracking_pose, teleport=True).
    6. Capture RGB + Depth + Segmentation from front_custom camera.
    7. Restore target to its original world pose (clean exit).
    8. Write JSON summary + PNGs.

Whitelist policy (SKILL.md §6.1):
    - simListSceneObjects   ✓
    - simGetVehiclePose     ✓
    - simGetObjectPose      ✓
    - simSetObjectPose      ✓   (this is the **whole point** of teleport mode)
    - simSetCameraPose      ✓
    - simGetImages          ✓
    NOT used: simSpawnObject, enableApiControl, armDisarm, moveByVelocity*

Usage examples:
    # Full happy path: scout first, then auto-pick.
    python flyseek_extend/scripts/scout_scene_targets.py
    python flyseek_extend/scripts/smoke_test_teleport_and_capture.py --auto-from-scout

    # Specific actor (no scout file needed):
    python flyseek_extend/scripts/smoke_test_teleport_and_capture.py \\
        --target Cart_v1_low2_634

    # Pick first vehicle live on-the-fly (no scout file, single regex):
    python flyseek_extend/scripts/smoke_test_teleport_and_capture.py \\
        --target-regex '.*Car0?[0-9]+.*'
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


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "flyseek_extend" / "output" / "smoke_test_teleport"
DEFAULT_SCOUT_FILE = (
    REPO_ROOT / "flyseek_extend" / "output" / "assets" / "scene_targets_latest.json"
)


@dataclass
class TeleportReport:
    timestamp: str
    airsim_ip: str
    airsim_rpc_port: int
    target_name: str | None = None
    target_category: str | None = None
    target_label: str | None = None
    drone_pose_world: dict[str, float] = field(default_factory=dict)
    target_orig_pose_world: dict[str, float] = field(default_factory=dict)
    target_teleport_pose_world: dict[str, float] = field(default_factory=dict)
    teleport_offset_m: float = 0.0
    camera_name: str = ""
    camera_pitch_deg: float = 0.0
    images_saved: list[str] = field(default_factory=list)
    target_restored: bool = False
    success: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


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


def _dict_to_pose(d: dict[str, float]):
    import airsim  # type: ignore
    return airsim.Pose(
        airsim.Vector3r(d["x"], d["y"], d["z"]),
        airsim.Quaternionr(d.get("qx", 0.0), d.get("qy", 0.0),
                           d.get("qz", 0.0), d.get("qw", 1.0)),
    )


def _is_valid_pose(p) -> bool:
    for v in (p.position.x_val, p.position.y_val, p.position.z_val):
        if math.isnan(v) or math.isinf(v):
            return False
    return True


def _pick_target(client, args, report: TeleportReport) -> tuple[str, str, str] | None:
    """Returns (target_name, category, suggested_label) or None on failure."""
    # priority 1: explicit --target
    if args.target:
        return (args.target, "unknown", args.label or "a tracked object")

    # priority 2: --auto-from-scout (read previously saved scout JSON)
    if args.auto_from_scout:
        scout_file = args.scout_file or DEFAULT_SCOUT_FILE
        if not scout_file.exists():
            report.errors.append(
                f"--auto-from-scout 但找不到 scout 输出文件：{scout_file}。"
                f"请先跑 `python flyseek_extend/scripts/scout_scene_targets.py`。"
            )
            return None
        try:
            data = json.loads(scout_file.read_text(encoding="utf-8"))
        except Exception as e:
            report.errors.append(f"scout JSON 解析失败：{e}")
            return None
        candidates = data.get("candidates", [])
        if not candidates:
            report.errors.append("scout JSON 里 candidates 为空")
            return None
        # Pick first one (sorted by category priority in scout output)
        c = candidates[0]
        return (c["name"], c.get("category", "unknown"),
                c.get("suggested_label", "a tracked object"))

    # priority 3: --target-regex (live regex query, no scout needed)
    if args.target_regex:
        try:
            matches = client.simListSceneObjects(name_regex=args.target_regex) or []
        except Exception as e:
            report.errors.append(f"simListSceneObjects('{args.target_regex}') 失败: {e}")
            return None
        if not matches:
            report.errors.append(
                f"正则 '{args.target_regex}' 在场景里 0 匹配。"
                f"试试 `--target-regex .*[Cc]art.*` 或 `--target-regex .*Car0.*`"
            )
            return None
        # First match, no pose verification (we'll fail fast in main flow)
        return (matches[0], "unknown",
                args.label or f"a {args.target_regex.replace('.*', '').strip()}")

    report.errors.append(
        "must provide one of: --target / --auto-from-scout / --target-regex"
    )
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
            import numpy as _np
            _np.save(out_path.with_suffix(".npy"), depth)
            depth_vis = np.clip(depth, 0, 100.0) / 100.0
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


def run_teleport_smoke(args: argparse.Namespace) -> TeleportReport:
    report = TeleportReport(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        airsim_ip=args.airsim_ip,
        airsim_rpc_port=args.airsim_port,
        teleport_offset_m=args.teleport_offset,
        camera_name=args.camera_name,
        camera_pitch_deg=args.camera_pitch_deg,
    )

    try:
        import airsim  # type: ignore
    except ImportError as e:
        report.errors.append(f"airsim not importable: {e}; pip install airsim==1.8.1")
        return report

    try:
        client = airsim.MultirotorClient(ip=args.airsim_ip, port=args.airsim_port)
        client.confirmConnection()
    except Exception as e:
        report.errors.append(f"failed to connect: {e}")
        return report

    # ---- 1. pick target ----------------------------------------------------
    picked = _pick_target(client, args, report)
    if picked is None:
        return report
    target_name, category, label = picked
    report.target_name = target_name
    report.target_category = category
    report.target_label = label
    print(f"[ok] target picked: '{target_name}' ({category}) — \"{label}\"")

    # ---- 2. drone pose -----------------------------------------------------
    try:
        drone_pose = client.simGetVehiclePose()
    except Exception as e:
        report.errors.append(f"simGetVehiclePose failed: {e}")
        return report
    if not _is_valid_pose(drone_pose):
        report.errors.append("drone pose is NaN/invalid")
        return report
    report.drone_pose_world = _pose_to_dict(drone_pose)
    print(f"[ok] drone @ ({drone_pose.position.x_val:.1f}, "
          f"{drone_pose.position.y_val:.1f}, {drone_pose.position.z_val:.1f})")

    # ---- 3. snapshot target original pose ----------------------------------
    try:
        orig_pose = client.simGetObjectPose(target_name)
    except Exception as e:
        report.errors.append(f"simGetObjectPose('{target_name}') failed: {e}")
        return report
    if not _is_valid_pose(orig_pose):
        report.errors.append(
            f"target '{target_name}' pose is NaN — 该 actor 不存在或不可访问。"
            f"换一个 --target 或重跑 scout。"
        )
        return report
    report.target_orig_pose_world = _pose_to_dict(orig_pose)
    print(f"[ok] target original @ ({orig_pose.position.x_val:.1f}, "
          f"{orig_pose.position.y_val:.1f}, {orig_pose.position.z_val:.1f})")

    # ---- 4. compute teleport pose -----------------------------------------
    # NED: +X north, +Y east, +Z down (down=positive). Drone yaw=0 → faces +X.
    # We put the target args.teleport_offset m ahead (+X), same altitude.
    # 选项 --target-z-mode:
    #   same   = target_z = drone_z              （和无人机同高，最朴素的"在视野里"）
    #   below  = target_z = drone_z + 5          （5m below drone, 适合无人机俯视）
    #   keep   = target_z = orig_pose.z          （保持目标原本的高度，最真实）
    if args.target_z_mode == "same":
        tz = drone_pose.position.z_val
    elif args.target_z_mode == "below":
        tz = drone_pose.position.z_val + 5.0
    else:  # keep
        tz = orig_pose.position.z_val

    teleport_pose = airsim.Pose(
        airsim.Vector3r(
            drone_pose.position.x_val + args.teleport_offset,
            drone_pose.position.y_val + args.teleport_lateral,
            tz,
        ),
        airsim.to_quaternion(0, 0, math.radians(args.target_yaw_deg)),
    )
    report.target_teleport_pose_world = _pose_to_dict(teleport_pose)
    print(f"[ok] teleporting to ({teleport_pose.position.x_val:.1f}, "
          f"{teleport_pose.position.y_val:.1f}, "
          f"{teleport_pose.position.z_val:.1f})  yaw={args.target_yaw_deg}°")

    # ---- 5. teleport! ------------------------------------------------------
    try:
        # airsim>=1.8 supports the 'teleport=True' kwarg; older silently no-op.
        try:
            client.simSetObjectPose(target_name, teleport_pose, teleport=True)
        except TypeError:
            client.simSetObjectPose(target_name, teleport_pose)
    except Exception as e:
        report.errors.append(f"simSetObjectPose failed: {e}")
        return report

    time.sleep(args.settle_s)  # let the renderer redraw 1-2 frames

    # ---- 6. set camera + capture ------------------------------------------
    chosen_cam = args.camera_name
    try:
        cam_pose = airsim.Pose(
            airsim.Vector3r(0, 0, 0),
            airsim.to_quaternion(math.radians(-args.camera_pitch_deg), 0, 0),
        )
        # 兜底相机名
        for cam in (args.camera_name, "front_custom", "0"):
            try:
                client.simSetCameraPose(cam, cam_pose)
                chosen_cam = cam
                break
            except Exception:
                continue
        print(f"[ok] camera '{chosen_cam}' pitched {args.camera_pitch_deg}° down")
    except Exception as e:
        report.warnings.append(f"simSetCameraPose error: {e}")

    requests = [
        airsim.ImageRequest(chosen_cam, airsim.ImageType.Scene, False, False),
        airsim.ImageRequest(chosen_cam, airsim.ImageType.DepthPlanar, True, False),
        airsim.ImageRequest(chosen_cam, airsim.ImageType.Segmentation, False, False),
    ]

    try:
        responses = client.simGetImages(requests)
    except Exception as e:
        report.errors.append(f"simGetImages failed: {e}")
        responses = []

    out_dir = args.output / report.timestamp.replace(":", "-")
    saved: list[str] = []
    for resp, kind in zip(responses, ("rgb", "depth", "segmentation")):
        out_path = out_dir / f"{kind}.png"
        if _save_image(resp, out_path, kind):
            saved.append(str(out_path))
            print(f"[ok] saved {kind}: {out_path}")
        else:
            report.warnings.append(f"{kind} 图像为空或保存失败")
    report.images_saved = saved

    # ---- 7. restore target -------------------------------------------------
    if not args.no_restore:
        try:
            try:
                client.simSetObjectPose(target_name, orig_pose, teleport=True)
            except TypeError:
                client.simSetObjectPose(target_name, orig_pose)
            report.target_restored = True
            print(f"[ok] target restored to original pose")
        except Exception as e:
            report.warnings.append(f"target restore failed: {e}")
    else:
        print("[info] --no-restore set, leaving target at teleport pose")

    report.success = (len(saved) > 0) and (not report.errors)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FlySeek Phase 0.5 — teleport-based smoke test (no pak needed).",
    )
    parser.add_argument("--airsim-ip", default=os.environ.get("AIRSIM_IP", "127.0.0.1"))
    parser.add_argument("--airsim-port", type=int,
                        default=int(os.environ.get("AIRSIM_RPC_PORT", 41451)))

    target_grp = parser.add_argument_group("target picking")
    target_grp.add_argument("--target", default=None,
                            help="Exact actor name to teleport (skips scout/regex).")
    target_grp.add_argument("--auto-from-scout", action="store_true",
                            help=f"Read first candidate from {DEFAULT_SCOUT_FILE}.")
    target_grp.add_argument("--scout-file", type=Path, default=None,
                            help="Override the scout JSON path.")
    target_grp.add_argument("--target-regex", default=None,
                            help="Live regex query (e.g. '.*Car0?[0-9]+.*'), picks first match.")
    target_grp.add_argument("--label", default=None,
                            help="Override the natural-language label (default: from scout JSON).")

    pose_grp = parser.add_argument_group("teleport geometry")
    pose_grp.add_argument("--teleport-offset", type=float, default=10.0,
                          help="Forward (NED +X) distance (m). Default 10.")
    pose_grp.add_argument("--teleport-lateral", type=float, default=0.0,
                          help="Lateral (NED +Y) offset (m). Default 0.")
    pose_grp.add_argument("--target-yaw-deg", type=float, default=180.0,
                          help="World-frame yaw of target. 180 = facing drone. Default 180.")
    pose_grp.add_argument("--target-z-mode", choices=["same", "below", "keep"],
                          default="same",
                          help="Target altitude: 'same'=drone z (default), "
                               "'below'=drone+5m down, 'keep'=target's original z.")
    pose_grp.add_argument("--settle-s", type=float, default=0.3,
                          help="Sleep after teleport before capture.")

    cam_grp = parser.add_argument_group("camera")
    cam_grp.add_argument("--camera-name", default="front_custom")
    cam_grp.add_argument("--camera-pitch-deg", type=float, default=15.0,
                         help="Downward pitch in degrees (positive = look down).")

    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-restore", action="store_true",
                        help="Don't move target back to original pose after capture.")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Hard timeout (s) — AirVLN may crash mid-RPC.")
    args = parser.parse_args()

    def _on_timeout(_sig, _frm):
        print(f"\n[FATAL] smoke test timed out after {args.timeout}s. "
              "AirVLN 可能已 crash。", flush=True)
        sys.exit(124)

    signal.signal(signal.SIGALRM, _on_timeout)
    signal.alarm(int(args.timeout))

    args.output.mkdir(parents=True, exist_ok=True)

    report = run_teleport_smoke(args)

    out_dir = args.output / report.timestamp.replace(":", "-")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 64)
    print(f"TELEPORT SMOKE TEST — {'PASS' if report.success else 'FAIL'}")
    print("=" * 64)
    print(f"Target               : {report.target_name}  "
          f"({report.target_category}) — \"{report.target_label}\"")
    print(f"Images saved         : {len(report.images_saved)}")
    for p in report.images_saved:
        print(f"                       - {p}")
    if report.warnings:
        print(f"\nWarnings ({len(report.warnings)}):")
        for w in report.warnings:
            print(f"  - {w}")
    if report.errors:
        print(f"\nErrors ({len(report.errors)}):")
        for e in report.errors:
            print(f"  - {e}")
    print(f"\nSummary JSON         : {summary_path}")
    if report.success:
        rgb = next((p for p in report.images_saved if p.endswith("rgb.png")), None)
        if rgb:
            print(f"\n👁  打开看一下能不能看到车：")
            print(f"     xdg-open {rgb}")
    print("=" * 64)

    return 0 if report.success else 1


if __name__ == "__main__":
    sys.exit(main())

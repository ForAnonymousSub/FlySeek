# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""OpenFly-compatible trajectory writers (pose.jsonl + flyseek_meta.jsonl)."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from flyseek.adversary.base import DroneState, TargetState, wrap_to_pi


def _ned_to_openfly_pos(pos_ned: np.ndarray) -> list[float]:
    """OpenFly pose.jsonl stores planner-style [x, y, z] (map frame z up)."""
    p = np.asarray(pos_ned, dtype=np.float64).reshape(3)
    return [float(p[0]), float(p[1]), float(-p[2])]


def build_openfly_action_record(
    frame_idx: int,
    drone: DroneState,
    *,
    prev_drone: DroneState | None = None,
) -> dict[str, Any]:
    """One line of ``pose.jsonl`` — mirrors traj_gen ``saveActionsToJson``."""
    pos = _ned_to_openfly_pos(drone.position)
    yaw = float(drone.heading)
    action_type = "move"
    value = 0
    if prev_drone is not None:
        dyaw = wrap_to_pi(yaw - prev_drone.heading)
        if abs(dyaw) > math.radians(8.0):
            action_type = "turn left" if dyaw > 0 else "turn right"
            value = int(round(math.degrees(abs(dyaw))))
    return {
        "action": {
            "imageid": int(frame_idx),
            "type": action_type,
            "value": int(value),
            "pos": pos,
            "yaw": yaw,
        }
    }


def build_flyseek_meta_record(
    frame_idx: int,
    drone: DroneState,
    target: TargetState,
    *,
    prev_drone: DroneState | None = None,
    target_visible: bool = True,
    tracker_mode: str = "",
    adversary_log: dict[str, Any] | None = None,
    images: dict[str, str] | None = None,
    vis_reason: str = "",
    is_occluded: bool | None = None,
    difficulty: str = "",
    target_name: str | None = None,
    asset_name: str | None = None,
    seg_id: int | None = None,
    bbox_2d: list[float] | None = None,
) -> dict[str, Any]:
    """FlySeek incremental metadata (8-D action semantics, SKILL D3 + §6.3).

    ``bbox_2d`` is ``[u_min, v_min, u_max, v_max]`` in pixels or ``None`` when
    the target could not be localised in the segmentation frame (occluded,
    off-screen, or a non-segmentation tick). ``vis_reason`` mirrors
    ``visibility_status`` (``ok``/``out_of_range``/``out_of_fov``/``los_blocked``).
    """
    if prev_drone is not None:
        delta = drone.position - prev_drone.position
        delta_yaw = wrap_to_pi(drone.heading - prev_drone.heading)
        dt = max(drone.timestamp - prev_drone.timestamp, 1e-6)
    else:
        delta = np.zeros(3)
        delta_yaw = 0.0
        dt = 1.0

    action_8d = [
        float(delta[0]),
        float(delta[1]),
        float(delta[2]),
        float(delta_yaw),
        float(target.velocity[0]),
        float(target.velocity[1]),
        float(target.velocity[2]),
        1.0 if target_visible else 0.0,
    ]

    # Derive occlusion from the visibility reason when not provided explicitly.
    if is_occluded is None:
        is_occluded = (vis_reason == "los_blocked")

    target_state: dict[str, Any] = {
        "pos": [float(v) for v in target.position],
        "vel": [float(v) for v in target.velocity],
        "heading": float(target.heading),
        "is_in_fov": bool(target_visible),
        "is_occluded": bool(is_occluded),
        "bbox_2d": [float(c) for c in bbox_2d] if bbox_2d is not None else None,
    }
    if target_name is not None:
        target_state["name"] = target_name
    if asset_name is not None:
        target_state["asset_name"] = asset_name
    if seg_id is not None:
        target_state["seg_id"] = int(seg_id)

    return {
        "frame_idx": int(frame_idx),
        "timestamp": float(drone.timestamp),
        "difficulty": difficulty,
        "drone_state": {
            "pos": [float(v) for v in drone.position],
            "vel": [float(v) for v in drone.velocity],
            "heading": float(drone.heading),
        },
        "target_state": target_state,
        "action_8d": action_8d,
        "action_8d_labels": [
            "delta_x", "delta_y", "delta_z", "delta_yaw",
            "target_vx", "target_vy", "target_vz", "target_in_fov",
        ],
        "tracker_mode": tracker_mode,
        "target_visible": bool(target_visible),
        "vis_reason": vis_reason,
        "agent_decision": adversary_log or {},
        "images": images or {},
    }


def write_aim_landmark(
    path: Path,
    *,
    target_name: str,
    target_label: str,
    target_pos_ned: np.ndarray,
) -> None:
    """Final line of pose.jsonl (OpenFly ``aim_landmark``)."""
    pos = _ned_to_openfly_pos(target_pos_ned)
    record = {
        "aim_landmark": {
            "type": "vehicle",
            "color": "unknown",
            "size": "small",
            "shape": "car",
            "feature": target_label,
            "name": target_name,
            "position": pos,
        }
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class EpisodeWriter:
    """Append-only writers for one demo episode directory."""

    def __init__(self, episode_dir: Path, *, difficulty: str = "") -> None:
        self.episode_dir = Path(episode_dir)
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.episode_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.pose_path = self.episode_dir / "pose.jsonl"
        self.meta_path = self.episode_dir / "flyseek_meta.jsonl"
        self.trajectory_path = self.episode_dir / "trajectory.jsonl"
        self._pose_f = self.pose_path.open("w", encoding="utf-8")
        self._meta_f = self.meta_path.open("w", encoding="utf-8")
        self._traj_f = self.trajectory_path.open("w", encoding="utf-8")
        self._prev_drone: DroneState | None = None
        self.difficulty = difficulty

    def close(
        self,
        *,
        target_name: str,
        target_label: str,
        target_pos_ned: np.ndarray,
    ) -> None:
        write_aim_landmark(
            self.pose_path,
            target_name=target_name,
            target_label=target_label,
            target_pos_ned=target_pos_ned,
        )
        self._pose_f.close()
        self._meta_f.close()
        self._traj_f.close()

    def write_frame(
        self,
        frame_idx: int,
        drone: DroneState,
        target: TargetState,
        *,
        trajectory_record: dict[str, Any],
        target_visible: bool = True,
        tracker_mode: str = "",
        adversary_log: dict[str, Any] | None = None,
        rgb_src: Path | None = None,
        vis_reason: str = "",
        is_occluded: bool | None = None,
        target_name: str | None = None,
        asset_name: str | None = None,
        seg_id: int | None = None,
        bbox_2d: list[float] | None = None,
    ) -> Path | None:
        pose_rec = build_openfly_action_record(
            frame_idx, drone, prev_drone=self._prev_drone,
        )
        meta_rec = build_flyseek_meta_record(
            frame_idx,
            drone,
            target,
            prev_drone=self._prev_drone,
            target_visible=target_visible,
            tracker_mode=tracker_mode,
            adversary_log=adversary_log,
            vis_reason=vis_reason,
            is_occluded=is_occluded,
            difficulty=self.difficulty,
            target_name=target_name,
            asset_name=asset_name,
            seg_id=seg_id,
            bbox_2d=bbox_2d,
        )
        self._pose_f.write(json.dumps(pose_rec, ensure_ascii=False) + "\n")
        self._meta_f.write(json.dumps(meta_rec, ensure_ascii=False) + "\n")
        self._traj_f.write(json.dumps(trajectory_record, ensure_ascii=False) + "\n")
        self._prev_drone = DroneState(
            position=drone.position.copy(),
            velocity=drone.velocity.copy(),
            heading=drone.heading,
            timestamp=drone.timestamp,
        )

        openfly_img: Path | None = None
        if rgb_src is not None and rgb_src.exists():
            dst = self.episode_dir / f"image_{frame_idx}.png"
            try:
                shutil.copy2(rgb_src, dst)
                openfly_img = dst
            except OSError:
                openfly_img = None
        return openfly_img


__all__ = [
    "EpisodeWriter",
    "build_openfly_action_record",
    "build_flyseek_meta_record",
    "write_aim_landmark",
]

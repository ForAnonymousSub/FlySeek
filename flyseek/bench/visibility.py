# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""FlySeek-Bench visibility evaluator.

Converts the demo's existing *view judgment* into the paper-consistent visibility
fields, computing real geometry when the inputs allow it and falling back to the
recorded judgment otherwise — never inventing geometry silently.

How the current demo stores view judgment (inspected):
  - ``flyseek/utils/visibility.py::visibility_status(...)`` returns
    ``(visible, reason)`` with ``reason`` in
    ``{"ok", "out_of_range", "out_of_fov", "los_blocked"}``. The reason already
    separates FOV failure from LoS occlusion.
  - The demo loop writes ``target_visible`` (bool) and ``vis_reason`` (the reason
    string) into ``flyseek_meta.jsonl`` per frame, plus ``bbox_2d`` / ``seg_id``.
  - Line-of-sight is computed by
    ``PcdOccupancyMap.los_blocked_ned(observer_ned, target_ned, ...)``.

This evaluator therefore prefers, in order:
  1. **Real geometry** — frustum from camera intrinsics/extrinsics (projection),
     LoS from the PCD occupancy map (ray-march), when ``camera_config`` /
     ``scene_context["occupancy"]`` are supplied.
  2. **Recorded reason** — derive frustum/LoS from ``vis_reason`` when geometry
     is unavailable but the reason string is present.
  3. **Binary fallback** — ``target_visible = existing_view_judgment``,
     ``visibility_score = 1.0/0.0``, frustum/LoS = ``None`` when they cannot be
     separated. A one-time warning is emitted in this mode.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from flyseek.utils.seg_bbox import project_ned_to_pixel


def _parse_pose(pose: Any) -> tuple[np.ndarray, float | None]:
    """Return ``(position_xyz, yaw_or_None)`` from a list or dict pose."""
    if pose is None:
        raise ValueError("pose is None")
    if isinstance(pose, dict):
        if "pos" in pose:
            p = np.asarray(pose["pos"], dtype=np.float64).reshape(3)
            yaw = pose.get("yaw", pose.get("heading"))
        else:
            p = np.array([float(pose["x"]), float(pose["y"]), float(pose["z"])])
            yaw = pose.get("yaw", pose.get("heading"))
        return p, (float(yaw) if yaw is not None else None)
    arr = np.asarray(pose, dtype=np.float64).reshape(-1)
    pos = arr[:3]
    yaw = float(arr[3]) if arr.size >= 4 else None
    return pos, yaw


def _cfg_get(camera_config: Any, key: str, default: Any) -> Any:
    if camera_config is None:
        return default
    if isinstance(camera_config, dict):
        return camera_config.get(key, default)
    return getattr(camera_config, key, default)


def _wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class VisibilityEvaluator:
    """Standardize per-frame visibility metadata.

    ``max_range_m`` / ``drone_eye_agl_m`` mirror the demo's visibility settings so
    the geometric path matches the recorded judgment closely.
    """

    def __init__(
        self,
        *,
        max_range_m: float = 100.0,
        drone_eye_agl_m: float = 12.0,
        graded_score: bool = True,
    ) -> None:
        self.max_range_m = float(max_range_m)
        self.drone_eye_agl_m = float(drone_eye_agl_m)
        self.graded_score = bool(graded_score)
        self._warned: set[str] = set()

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            warnings.warn(f"[VisibilityEvaluator] {msg}", stacklevel=2)

    # ------------------------------------------------------------------ #
    def evaluate_frame(
        self,
        uav_pose: Any,
        target_pose: Any,
        camera_config: Any = None,
        scene_context: dict[str, Any] | None = None,
        existing_visibility_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return standardized visibility fields for one frame.

        Output keys: ``in_camera_frustum``, ``line_of_sight_clear``,
        ``target_visible``, ``visibility_score``, ``occlusion_risk``, plus a
        ``visibility_source`` tag describing which path produced the result.
        """
        scene_context = scene_context or {}
        existing = existing_visibility_metadata or {}

        uav_pos, uav_yaw = _parse_pose(uav_pose)
        tgt_pos, _ = _parse_pose(target_pose)

        existing_visible = existing.get("target_visible")
        existing_reason = existing.get("vis_reason")

        distance = float(np.hypot(*(tgt_pos[:2] - uav_pos[:2])))
        in_range = distance <= self.max_range_m

        # ---- (2) camera-frustum check (intrinsics + extrinsics) --------- #
        in_frustum, frustum_src, bearing_offset = self._frustum_check(
            uav_pos, uav_yaw, tgt_pos, camera_config, in_range
        )

        # ---- (3) line-of-sight check (ray cast / PCD) ------------------- #
        los_clear, los_src = self._los_check(
            uav_pos, tgt_pos, scene_context, existing_reason
        )

        # ---- (4) target_visible ---------------------------------------- #
        if existing_visible is not None:
            # The demo's recorded judgment is authoritative.
            target_visible = bool(existing_visible)
            vis_src = "recorded_judgment"
        elif in_frustum is not None and los_clear is not None:
            target_visible = bool(in_frustum and los_clear and in_range)
            vis_src = "geometry"
        elif in_frustum is not None:
            target_visible = bool(in_frustum and in_range)
            vis_src = "frustum_only"
            self._warn_once(
                "no_los",
                "no LoS source (occupancy/reason) — target_visible derived from "
                "frustum only; line_of_sight_clear left as None.",
            )
        else:
            self._warn_once(
                "fallback",
                "FALLBACK MODE: no geometry and no recorded judgment — cannot "
                "evaluate visibility; returning None/0.0.",
            )
            target_visible = False
            vis_src = "fallback_none"

        # ---- visibility_score ------------------------------------------- #
        visibility_score = self._score(
            target_visible, bearing_offset, distance, camera_config
        )

        # ---- (7) occlusion risk ----------------------------------------- #
        occlusion_risk = self.estimate_occlusion_risk(
            uav_pos, scene_context, existing
        )

        return {
            "in_camera_frustum": in_frustum,
            "line_of_sight_clear": los_clear,
            "target_visible": bool(target_visible),
            "visibility_score": visibility_score,
            "occlusion_risk": occlusion_risk,
            "distance_to_target": distance,
            "relative_bearing": (bearing_offset if bearing_offset is not None else 0.0),
            "visibility_source": f"{vis_src}|frustum:{frustum_src}|los:{los_src}",
        }

    # ------------------------------------------------------------------ #
    def _frustum_check(
        self, uav_pos, uav_yaw, tgt_pos, camera_config, in_range,
    ) -> tuple[bool | None, str, float | None]:
        """Return ``(in_frustum, source, bearing_offset_rad)``.

        Uses a real pinhole projection when image width/height are known
        (intrinsics + the body-relative extrinsics the demo configures). Falls
        back to a horizontal-FOV bearing test when only the FOV is known, and to
        ``None`` when even yaw is unavailable.
        """
        hfov = float(_cfg_get(camera_config, "hfov_deg", 90.0))
        width = _cfg_get(camera_config, "width", None)
        height = _cfg_get(camera_config, "height", None)

        bearing_offset = None
        if uav_yaw is not None:
            bearing = math.atan2(
                float(tgt_pos[1] - uav_pos[1]), float(tgt_pos[0] - uav_pos[0])
            )
            bearing_offset = abs(_wrap_to_pi(bearing - uav_yaw))

        if width and height and uav_yaw is not None:
            uv = project_ned_to_pixel(
                tgt_pos, uav_pos, uav_yaw,
                width=int(width), height=int(height), hfov_deg=hfov,
                cam_forward_m=float(_cfg_get(camera_config, "body_forward_m", 0.45)),
                cam_down_m=float(_cfg_get(camera_config, "body_down_m", 0.25)),
                cam_pitch_deg=float(_cfg_get(camera_config, "pitch_deg", 55.0)),
            )
            if uv is None:
                return False, "projection", bearing_offset
            u, v = uv
            inside = (0 <= u < int(width)) and (0 <= v < int(height))
            return bool(inside and in_range), "projection", bearing_offset

        if uav_yaw is not None:
            # Horizontal-only FOV test; vertical extent NOT checked.
            in_h = bearing_offset <= math.radians(hfov / 2.0)
            return bool(in_h and in_range), "hfov_bearing", bearing_offset

        return None, "unavailable", bearing_offset

    def _los_check(
        self, uav_pos, tgt_pos, scene_context, existing_reason,
    ) -> tuple[bool | None, str]:
        """Return ``(line_of_sight_clear, source)``."""
        occ = scene_context.get("occupancy")
        if occ is not None and hasattr(occ, "los_blocked_ned"):
            try:
                blocked = occ.los_blocked_ned(
                    np.asarray(uav_pos, dtype=np.float64),
                    np.asarray(tgt_pos, dtype=np.float64),
                    drone_eye_agl_m=float(
                        scene_context.get("drone_eye_agl_m", self.drone_eye_agl_m)
                    ),
                    target_agl_m=max(0.5, -float(tgt_pos[2])),
                )
                return (not bool(blocked)), "pcd_raycast"
            except Exception:
                pass  # fall through to reason-based derivation

        if existing_reason:
            if existing_reason == "los_blocked":
                return False, "recorded_reason"
            if existing_reason == "ok":
                return True, "recorded_reason"
            # out_of_fov / out_of_range -> LoS was not evaluated.
            return None, "recorded_reason_unevaluated"

        return None, "unavailable"

    def _score(
        self, target_visible, bearing_offset, distance, camera_config,
    ) -> float:
        """Visibility confidence in [0, 1].

        Binary fallback (1.0/0.0) unless graded scoring is enabled AND we have a
        bearing offset to centre against — then a heuristic confidence based on
        how centred the target is within the horizontal FOV (documented; this is
        a confidence proxy, not a physical measurement).
        """
        if not target_visible:
            return 0.0
        if not self.graded_score or bearing_offset is None:
            return 1.0
        hfov_half = math.radians(float(_cfg_get(camera_config, "hfov_deg", 90.0)) / 2.0)
        if hfov_half <= 0:
            return 1.0
        centering = max(0.0, 1.0 - (bearing_offset / hfov_half))
        # Floor so a visible-but-edge target keeps a non-zero score.
        return round(0.1 + 0.9 * centering, 4)

    # ------------------------------------------------------------------ #
    def estimate_occlusion_risk(
        self,
        uav_pos: np.ndarray,
        scene_context: dict[str, Any] | None = None,
        existing_visibility_metadata: dict[str, Any] | None = None,
    ) -> float | None:
        """Estimate near-future occlusion risk in [0, 1], or ``None``.

        If ``scene_context`` provides ``future_target_positions`` (a list of NED
        points the target is predicted to reach) AND a PCD ``occupancy`` map, the
        risk is the fraction of those future positions whose line of sight from
        the current UAV position would be blocked. If a pre-computed
        ``occlusion_risk`` is present in the existing metadata it is passed
        through. Otherwise returns ``None``.

        LIMITATION: without future target positions or candidate occluders this
        cannot be estimated; we deliberately return ``None`` rather than guess.
        """
        scene_context = scene_context or {}
        existing = existing_visibility_metadata or {}

        if existing.get("occlusion_risk") is not None:
            return float(existing["occlusion_risk"])

        future = scene_context.get("future_target_positions")
        occ = scene_context.get("occupancy")
        if not future or occ is None or not hasattr(occ, "los_blocked_ned"):
            return None

        try:
            blocked = 0
            total = 0
            for fp in future:
                fp = np.asarray(fp, dtype=np.float64).reshape(3)
                total += 1
                if occ.los_blocked_ned(
                    np.asarray(uav_pos, dtype=np.float64), fp,
                    drone_eye_agl_m=float(
                        scene_context.get("drone_eye_agl_m", self.drone_eye_agl_m)
                    ),
                    target_agl_m=max(0.5, -float(fp[2])),
                ):
                    blocked += 1
            if total == 0:
                return None
            return round(blocked / total, 4)
        except Exception:
            return None


def evaluate_frame(
    uav_pose: Any,
    target_pose: Any,
    camera_config: Any = None,
    scene_context: dict[str, Any] | None = None,
    existing_visibility_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Module-level convenience wrapper using a default evaluator."""
    return VisibilityEvaluator().evaluate_frame(
        uav_pose, target_pose, camera_config,
        scene_context, existing_visibility_metadata,
    )


__all__ = ["VisibilityEvaluator", "evaluate_frame"]

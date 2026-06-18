# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""2D bounding-box extraction from AirSim segmentation frames.

The capture loop assigns the tracked target a unique segmentation id and renders
a segmentation image every ``modalities_stride`` ticks. To turn that into a
``bbox_2d`` we:

1. Project the target world position (AirSim NED) into the rendered camera using
   the *same* body-relative camera mount the demo configures
   (forward/down offset + downward pitch), so we do not depend on the exact
   semantics of ``simGetCameraInfo().pose`` across AirSim builds.
2. Sample the segmentation colour at that pixel.
3. Take the axis-aligned bounding box of all pixels sharing that colour.

Everything is best-effort: any projection that lands outside the frame, hits the
background colour, or raises is reported as ``None`` (target not localisable).
All maths is pure numpy so the projection half can be unit-tested without a sim.
"""

from __future__ import annotations

import math

import numpy as np


def project_ned_to_pixel(
    target_ned: np.ndarray,
    drone_pos_ned: np.ndarray,
    drone_yaw_rad: float,
    *,
    width: int,
    height: int,
    hfov_deg: float = 90.0,
    cam_forward_m: float = 0.45,
    cam_down_m: float = 0.25,
    cam_pitch_deg: float = 55.0,
) -> tuple[int, int] | None:
    """Project a world NED point into pixel ``(u, v)``; ``None`` if behind cam.

    Uses a pinhole model with AirSim camera conventions (camera looks along body
    +X, +Y right, +Z down). Returns integer pixel coordinates which may be
    outside ``[0, width) x [0, height)`` so the caller can decide to clamp or
    reject; only points strictly behind the image plane (x <= 0) return None.
    """
    target_ned = np.asarray(target_ned, dtype=np.float64).reshape(3)
    drone_pos_ned = np.asarray(drone_pos_ned, dtype=np.float64).reshape(3)

    cy, sy = math.cos(drone_yaw_rad), math.sin(drone_yaw_rad)
    # Camera mount offset expressed in the world frame (yaw only).
    mount_world = np.array([
        cy * cam_forward_m,
        sy * cam_forward_m,
        cam_down_m,
    ], dtype=np.float64)
    cam_pos = drone_pos_ned + mount_world

    rel = target_ned - cam_pos

    # Undo yaw, then undo the downward pitch, to land in the camera frame.
    x_y = cy * rel[0] + sy * rel[1]
    y_c = -sy * rel[0] + cy * rel[1]
    z_world_down = rel[2]

    pitch = math.radians(cam_pitch_deg)  # downward tilt about camera +Y
    cp, sp = math.cos(pitch), math.sin(pitch)
    # Rotate (forward, down) by +pitch so the optical axis points down-forward.
    x_cam = cp * x_y + sp * z_world_down
    z_cam = -sp * x_y + cp * z_world_down
    y_cam = y_c

    if x_cam <= 1e-3:
        return None

    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    fy = fx  # assume square pixels
    u = width / 2.0 + fx * (y_cam / x_cam)
    v = height / 2.0 + fy * (z_cam / x_cam)
    return int(round(u)), int(round(v))


def _is_background(color: np.ndarray) -> bool:
    """AirSim's default scene background / sky segments tend to be near-black."""
    return bool(np.all(color <= 2))


def bbox_from_segmentation(
    seg_img: np.ndarray,
    pixel_uv: tuple[int, int],
    *,
    min_pixels: int = 12,
) -> list[float] | None:
    """Bbox ``[u_min, v_min, u_max, v_max]`` of the colour at ``pixel_uv``.

    ``seg_img`` is an ``(H, W, 3)`` uint8 array (BGR or RGB — colour identity is
    all that matters). Returns ``None`` when the sampled pixel is background or
    the matching region is smaller than ``min_pixels``.
    """
    if seg_img is None or seg_img.ndim != 3 or seg_img.shape[2] < 3:
        return None
    h, w = seg_img.shape[:2]
    u, v = pixel_uv
    if not (0 <= u < w and 0 <= v < h):
        return None
    color = seg_img[v, u, :3].astype(np.int32)
    if _is_background(color):
        return None
    mask = np.all(seg_img[:, :, :3].astype(np.int32) == color, axis=-1)
    if int(mask.sum()) < min_pixels:
        return None
    ys, xs = np.nonzero(mask)
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


__all__ = ["project_ned_to_pixel", "bbox_from_segmentation"]

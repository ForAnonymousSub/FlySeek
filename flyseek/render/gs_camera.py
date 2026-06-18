# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Exact camera model for the OpenFly 3DGS bridge (SIBR hierarchy viewer).

Replicates the world->camera convention that ``scripts/sim/gs_bridge.py``
sends over HTTP, so we can:
  * project world points (the target car, a road, a coordinate grid) into the
    exact pixel frame the viewer renders  -> Phase C car compositing + labels;
  * inverse-project pixels onto the ground plane -> read road coords from a
    rendered top-down tile without a new render.

gs_bridge builds, for drone world position C=(x,y,z) and angles:
    yaw_used   = -yaw_input            (gs_bridge negates yaw)
    pitch_used = pitch_deg             (default -40 in the original bridge)
    R = M @ (Rpitch(pitch_used) @ Ryaw(yaw_used) @ Rroll(roll))
    M = [[0,-1,0],[0,0,-1],[1,0,0]]
    tvec = -R @ C
COLMAP convention: X_cam = R @ X_world + tvec = R @ (X_world - C).
PINHOLE projection: u = fx*Xc.x/Xc.z + cx,  v = fy*Xc.y/Xc.z + cy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_M = np.array([[0.0, -1.0, 0.0],
               [0.0, 0.0, -1.0],
               [1.0, 0.0, 0.0]])


def _rx(a):  # roll
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):  # pitch
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):  # yaw
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@dataclass(frozen=True)
class Intrinsics:
    w: int
    h: int
    fx: float
    fy: float
    cx: float
    cy: float

    @staticmethod
    def from_colmap_str(s: str) -> "Intrinsics":
        # "0 PINHOLE W H fx fy cx cy"
        p = s.split()
        return Intrinsics(int(p[2]), int(p[3]), float(p[4]), float(p[5]),
                          float(p[6]), float(p[7]))

    def scaled(self, k: float) -> "Intrinsics":
        return Intrinsics(int(round(self.w * k)), int(round(self.h * k)),
                          self.fx * k, self.fy * k, self.cx * k, self.cy * k)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1.0]])


def bridge_extrinsics(cam_xyz, yaw_input_deg: float, pitch_deg: float,
                      roll_deg: float = 0.0):
    """Return (R, tvec) exactly as gs_bridge sends to the viewer.

    R, tvec map world->camera: X_cam = R @ X_world + tvec.
    """
    C = np.asarray(cam_xyz, dtype=np.float64).reshape(3)
    yaw_used = np.radians(-yaw_input_deg)
    pitch_used = np.radians(pitch_deg)
    roll = np.radians(roll_deg)
    R_comb = _ry(pitch_used) @ _rz(yaw_used) @ _rx(roll)
    R = _M @ R_comb
    tvec = -R @ C
    return R, tvec


def project(world_pts: np.ndarray, R: np.ndarray, tvec: np.ndarray,
            intr: Intrinsics):
    """Project Nx3 world points to pixels.

    Returns (uv [N,2], depth [N], valid [N] bool) where valid = in front of
    camera AND inside the image rectangle.
    """
    P = np.asarray(world_pts, dtype=np.float64).reshape(-1, 3)
    Xc = (R @ P.T).T + tvec  # [N,3]
    z = Xc[:, 2]
    safe = z > 1e-6
    u = np.where(safe, intr.fx * Xc[:, 0] / np.where(safe, z, 1.0) + intr.cx, -1.0)
    v = np.where(safe, intr.fy * Xc[:, 1] / np.where(safe, z, 1.0) + intr.cy, -1.0)
    uv = np.stack([u, v], axis=1)
    valid = safe & (u >= 0) & (u < intr.w) & (v >= 0) & (v < intr.h)
    return uv, z, valid


def pixel_to_ground(uv, R: np.ndarray, tvec: np.ndarray, intr: Intrinsics,
                    ground_z: float):
    """Inverse-project pixel(s) onto the world plane Z = ground_z.

    Returns world XYZ [N,3] (NaN where the ray is parallel / behind).
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    C = (-R.T @ tvec)  # camera center in world
    # ray dir in camera frame, then to world
    x = (uv[:, 0] - intr.cx) / intr.fx
    y = (uv[:, 1] - intr.cy) / intr.fy
    d_cam = np.stack([x, y, np.ones_like(x)], axis=1)      # [N,3]
    d_world = (R.T @ d_cam.T).T                             # world dirs
    out = np.full((uv.shape[0], 3), np.nan)
    denom = d_world[:, 2]
    ok = np.abs(denom) > 1e-9
    t = np.where(ok, (ground_z - C[2]) / np.where(ok, denom, 1.0), np.nan)
    fwd = ok & (t > 0)
    out[fwd] = C[None, :] + t[fwd, None] * d_world[fwd]
    return out


__all__ = ["Intrinsics", "bridge_extrinsics", "project", "pixel_to_ground"]

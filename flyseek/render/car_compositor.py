# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Composite a 3D target car onto a rendered GS background (Phase C).

The static 3DGS scene has no movable car, so we draw one ourselves: an
oriented 3D **car mesh** (an extruded side-profile body with a glass
greenhouse + wheels) placed on the street at the target world pose, projected
into the exact pixel frame with the validated ``gs_camera`` model, rasterised
with painter's-algorithm depth sorting + Lambert shading, plus a soft contact
shadow on the ground so the car reads as *grounded* (not floating). Exact 2D
bbox + instance mask come for free.

Pure CPU/numpy/PIL -> no GPU, no offscreen GL. The profile-extruded mesh gives
a recognisable car silhouette (hood / windshield / roof / trunk / wheels) good
enough to validate placement/scale/occlusion and to train a detector; swap in
a textured OBJ later by replacing ``car_mesh`` (keep the (verts, faces)
contract: faces = list of (vertex-index-list, label, rgb)).

Occlusion note (v1): the car is drawn on top of the GS background (we have no
per-pixel GS depth). LOS gating upstream (``car_in_view``) means the car is
only composited when the road point is visible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from flyseek.render.gs_camera import Intrinsics, bridge_extrinsics, project


@dataclass(frozen=True)
class CarModel:
    length: float = 4.6        # along heading (x-body), metres BEFORE scale
    width: float = 1.85        # cross (y-body)
    height: float = 1.45       # total height (ground -> roof)
    body_rgb: tuple = (180, 35, 35)
    cabin_rgb: tuple = (40, 48, 60)     # glass / greenhouse
    wheel_rgb: tuple = (20, 20, 22)
    style: str = "car"         # "car" (profile body+glass+wheels) | "box" | "sprite"
    scale: float = 1.0         # uniform size multiplier (the 3DGS world scale is
                               # not truly metric, so shrink the car to match the
                               # rendered ground features, e.g. 0.1 = 1/10)
    sprite_path: str = ""      # if set + style=="sprite": photo billboard imposter
    sprite_scale: float = 1.0  # extra size multiplier for the sprite billboard

    @property
    def L(self) -> float:
        return self.length * self.scale

    @property
    def W(self) -> float:
        return self.width * self.scale

    @property
    def H(self) -> float:
        return self.height * self.scale


_LIGHT = np.array([0.35, 0.2, 1.0]); _LIGHT = _LIGHT / np.linalg.norm(_LIGHT)

# Car side silhouette in body frame: (x as fraction of L in [-0.5,0.5], +x=front;
# z as fraction of H in [0,1]). Closed loop; each edge carries a material so the
# greenhouse (windshield/roof/rear window) renders as glass.
_PROFILE = [
    # (x_frac, z_frac, material_of_edge_to_next)
    (0.50, 0.16, "body"),   # 0 front bumper bottom
    (0.50, 0.34, "body"),   # 1 front bumper top   (edge 1->2 hood)
    (0.33, 0.44, "glass"),  # 2 hood / windshield base (edge 2->3 windshield)
    (0.12, 0.93, "body"),   # 3 windshield top / roof front (edge 3->4 roof)
    (-0.17, 0.99, "glass"), # 4 roof rear (edge 4->5 rear window)
    (-0.33, 0.50, "body"),  # 5 rear window base / trunk (edge 5->6 trunk)
    (-0.50, 0.40, "body"),  # 6 trunk top (edge 6->7 rear face)
    (-0.50, 0.16, "body"),  # 7 rear bumper bottom (edge 7->0 underbody)
]


def _box(xr, yr, zr, rgb):
    corners = np.array([
        [xr[0], yr[0], zr[0]], [xr[1], yr[0], zr[0]],
        [xr[1], yr[1], zr[0]], [xr[0], yr[1], zr[0]],
        [xr[0], yr[0], zr[1]], [xr[1], yr[0], zr[1]],
        [xr[1], yr[1], zr[1]], [xr[0], yr[1], zr[1]],
    ], float)
    faces = [([4, 5, 6, 7], "roof", rgb), ([0, 1, 5, 4], "side", rgb),
             ([2, 3, 7, 6], "side", rgb), ([1, 2, 6, 5], "front", rgb),
             ([3, 0, 4, 7], "back", rgb), ([0, 1, 2, 3], "bottom", rgb)]
    return corners, faces


def _car_parts(model: CarModel):
    L, W, H = model.L, model.W, model.H
    if model.style == "box":
        v, f = _box((-L / 2, L / 2), (-W / 2, W / 2), (0.0, H), model.body_rgb)
        return v, f

    body_rgb, glass_rgb, wheel_rgb = model.body_rgb, model.cabin_rgb, model.wheel_rgb
    hw = 0.5 * W
    P = len(_PROFILE)
    verts = []
    for (xf, zf, _m) in _PROFILE:           # left side (y = -hw): 0..P-1
        verts.append([xf * L, -hw, zf * H])
    for (xf, zf, _m) in _PROFILE:           # right side (y = +hw): P..2P-1
        verts.append([xf * L, +hw, zf * H])
    verts = np.array(verts, float)

    faces = []
    faces.append((list(range(P)), "side", body_rgb))            # left silhouette
    faces.append((list(range(2 * P - 1, P - 1, -1)), "side", body_rgb))  # right
    for i in range(P):                                          # surface strips
        j = (i + 1) % P
        mat = _PROFILE[i][2]
        rgb = glass_rgb if mat == "glass" else body_rgb
        lbl = "roof" if mat != "glass" else "glass"
        faces.append(([i, j, j + P, i + P], lbl, rgb))

    parts_v = [verts]
    parts_f = [faces]
    # four wheels (short dark boxes peeking below the body sills)
    wl, ww, wr = 0.20 * L, 0.10 * W, 0.22 * H
    for sx in (-0.30 * L, 0.32 * L):
        for sy in (-hw - 0.01 * W, hw - ww + 0.01 * W):
            v, f = _box((sx - wl / 2, sx + wl / 2), (sy, sy + ww),
                        (0.0, wr), wheel_rgb)
            parts_v.append(v)
            parts_f.append(f)

    out_v, out_f, off = [], [], 0
    for v, fs in zip(parts_v, parts_f):
        out_v.append(v)
        for idx, label, rgb in fs:
            out_f.append(([i + off for i in idx], label, rgb))
        off += len(v)
    return np.concatenate(out_v, 0), out_f


def car_mesh(pos_xyz, heading_rad: float, model: CarModel = CarModel()):
    """Return (verts[N,3] world, faces) for the car mesh at a world pose.

    pos_xyz = (x, y, ground_z); the mesh base sits on ground_z.
    """
    v, faces = _car_parts(model)
    c, s = np.cos(heading_rad), np.sin(heading_rad)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    v = (Rz @ v.T).T + np.asarray(pos_xyz, dtype=np.float64)
    return v, faces


def car_box_mesh(pos_xyz, heading_rad: float, model: CarModel = CarModel()):
    v, faces = car_mesh(pos_xyz, heading_rad, model)
    return v, [(idx, label) for idx, label, _rgb in faces]


def _shade(base_rgb, world_normal, label):
    lam = max(0.32, float(np.dot(world_normal, _LIGHT)))
    tint = {"roof": 1.10, "glass": 0.85, "front": 0.95, "back": 0.85,
            "side": 0.92, "bottom": 0.4}.get(label, 1.0)
    col = np.clip(np.array(base_rgb) * lam * tint, 0, 255).astype(int)
    return tuple(int(x) for x in col)


def _ground_footprint(pos_xyz, heading_rad, model: CarModel):
    L, W = model.L * 0.55, model.W * 0.62   # shadow a touch tighter
    c, s = np.cos(heading_rad), np.sin(heading_rad)
    d = np.array([c, s, 0.0]); p = np.array([-s, c, 0.0])
    base = np.array([pos_xyz[0], pos_xyz[1], pos_xyz[2]])
    return np.array([base + sl * d + sw * p
                     for sl, sw in ((L, W), (L, -W), (-L, -W), (-L, W))])


def composite_mesh(bg: Image.Image, cam_xyz, yaw_input_deg, pitch_deg,
                   verts: np.ndarray, faces, intr: Intrinsics,
                   shadow_uv: np.ndarray | None = None):
    """Rasterise a world-space mesh onto bg for the given camera pose.

    Returns (image, bbox|None, mask[H,W] uint8).
    """
    R, t = bridge_extrinsics(cam_xyz, yaw_input_deg, pitch_deg)
    uv, _depth, _ = project(verts, R, t, intr)
    Xc = (R @ verts.T).T + t
    cam_center = np.asarray(cam_xyz, float)
    mesh_center = verts.mean(0)

    img = bg.convert("RGB")
    # soft contact shadow first (grounds the car)
    if shadow_uv is not None and len(shadow_uv) >= 3:
        sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).polygon([tuple(map(float, p)) for p in shadow_uv],
                                   fill=(0, 0, 0, 110))
        sh = sh.filter(ImageFilter.GaussianBlur(7))
        img = Image.alpha_composite(img.convert("RGBA"), sh).convert("RGB")
    img = img.copy()
    dr = ImageDraw.Draw(img)
    mask = Image.new("L", img.size, 0)
    mdr = ImageDraw.Draw(mask)

    face_list = []
    for idx, label, rgb in faces:
        zc = Xc[idx, 2]
        if np.any(zc <= 1e-3):
            continue
        p = verts[idx]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        if np.dot(n, p.mean(0) - mesh_center) < 0:
            n = -n
        if np.dot(n, cam_center - p.mean(0)) <= 0:
            continue
        face_list.append((float(zc.mean()), idx, label, rgb, n))
    face_list.sort(key=lambda f: -f[0])

    drew = False
    for _z, idx, label, rgb, n in face_list:
        if label == "bottom":
            continue
        poly = [(float(uv[i, 0]), float(uv[i, 1])) for i in idx]
        dr.polygon(poly, fill=_shade(rgb, n, label))
        mdr.polygon(poly, fill=255)
        drew = True

    bbox = None
    if drew:
        front = (Xc[:, 2] > 1e-3)
        if np.any(front):
            us = uv[front, 0]; vs = uv[front, 1]
            umin = max(0, float(us.min())); vmin = max(0, float(vs.min()))
            umax = min(intr.w - 1, float(us.max())); vmax = min(intr.h - 1, float(vs.max()))
            if umax > umin and vmax > vmin:
                bbox = (umin, vmin, umax, vmax)
    return img, bbox, np.array(mask, dtype=np.uint8)


def composite_car(bg: Image.Image, cam_xyz, yaw_input_deg, pitch_deg,
                  car_pos, heading_rad, intr: Intrinsics,
                  model: CarModel = CarModel()):
    """Draw the car mesh (+ contact shadow) onto bg for the camera + target pose.

    Returns (image, bbox|None, mask[H,W] uint8).
    """
    verts, faces = car_mesh(car_pos, heading_rad, model)
    R, t = bridge_extrinsics(cam_xyz, yaw_input_deg, pitch_deg)
    foot = _ground_footprint(car_pos, heading_rad, model)
    fuv, fz, _ = project(foot, R, t, intr)
    shadow_uv = fuv if np.all(fz > 1e-6) else None
    return composite_mesh(bg, cam_xyz, yaw_input_deg, pitch_deg, verts, faces,
                          intr, shadow_uv=shadow_uv)


__all__ = ["CarModel", "car_mesh", "car_box_mesh", "composite_mesh", "composite_car"]

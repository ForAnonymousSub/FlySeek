# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Draw the (virtual) target-car marker onto a rendered GS frame.

Until the 3D car mesh is composited (Phase C), the target is invisible in the
static GS render and indistinguishable from the scene's many baked-in parked
cars. This overlays a 3D wireframe box + crosshair + label at the projected
target pose so the target is unambiguous, and doubles as a projection sanity
check on real rendered frames.

Uses the validated flyseek.render.gs_camera projection.
"""
from __future__ import annotations

import numpy as np
from PIL import ImageDraw

from flyseek.render.gs_camera import Intrinsics, project

# car footprint (m): length along heading, width across, height up
CAR_L, CAR_W, CAR_H = 4.6, 2.0, 1.6


def _box_corners(center, heading, gz):
    d = np.array([np.cos(heading), np.sin(heading), 0.0])
    p = np.array([-np.sin(heading), np.cos(heading), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    c = np.array([center[0], center[1], gz])
    corners = []
    for sl in (+CAR_L / 2, -CAR_L / 2):
        for sw in (+CAR_W / 2, -CAR_W / 2):
            for sh in (0.0, CAR_H):
                corners.append(c + sl * d + sw * p + sh * up)
    return np.array(corners)  # [8,3]; index bit2=length,bit1=width,bit0=height


def draw_target_marker(img, car_xyz, heading, R, tvec, intr: Intrinsics,
                       label="TARGET", trail_world=None, color=(0, 255, 60)):
    """Draw target box + crosshair + label on a PIL image (in place).

    Returns dict with 'center_uv', 'bbox' (xyxy or None), 'in_view'.
    """
    dr = ImageDraw.Draw(img, "RGBA")
    W, H = img.size

    # motion trail (past/future ground positions)
    if trail_world is not None and len(trail_world):
        uv, z, _ = project(np.asarray(trail_world, float), R, tvec, intr)
        pts = [(float(u), float(v)) for (u, v), zz in zip(uv, z) if zz > 1e-6]
        if len(pts) >= 2:
            dr.line(pts, fill=(255, 200, 0, 180), width=3)

    # 3D box
    corners = _box_corners(car_xyz, heading, car_xyz[2])
    uv, z, _ = project(corners, R, tvec, intr)
    infront = z > 1e-6
    edges = [(0, 1), (2, 3), (4, 5), (6, 7),         # vertical-ish (height)
             (0, 2), (1, 3), (4, 6), (5, 7),         # width
             (0, 4), (1, 5), (2, 6), (3, 7)]         # length
    for a, b in edges:
        if infront[a] and infront[b]:
            dr.line([tuple(uv[a]), tuple(uv[b])], fill=color + (230,), width=3)

    # 2D bbox from visible corners
    vis = uv[infront]
    bbox = None
    if len(vis):
        x0, y0 = vis[:, 0].min(), vis[:, 1].min()
        x1, y1 = vis[:, 0].max(), vis[:, 1].max()
        bbox = [float(x0), float(y0), float(x1), float(y1)]

    # crosshair at projected center + label
    cen = project(np.array([[car_xyz[0], car_xyz[1], car_xyz[2]]]), R, tvec, intr)
    cuv, cz, _ = cen
    in_view = bool(cz[0] > 1e-6 and 0 <= cuv[0, 0] < W and 0 <= cuv[0, 1] < H)
    center_uv = (float(cuv[0, 0]), float(cuv[0, 1])) if cz[0] > 1e-6 else None
    if center_uv:
        u, v = center_uv
        # outer attention ring (the target is small at tracking range)
        for rr, a in ((40, 130), (33, 200)):
            dr.ellipse([u - rr, v - rr, u + rr, v + rr], outline=(255, 230, 0, a), width=2)
        # red crosshair
        r = 20
        dr.line([(u - r, v), (u + r, v)], fill=(255, 0, 0, 255), width=3)
        dr.line([(u, v - r), (u, v + r)], fill=(255, 0, 0, 255), width=3)
        dr.ellipse([u - r, v - r, u + r, v + r], outline=(255, 0, 0, 255), width=2)
        # label above the box
        ly = max(0, (bbox[1] if bbox else v) - 26)
        lx = min(max(0, u - 36), W - 130)
        dr.rectangle([lx, ly, lx + 126, ly + 20], fill=(0, 0, 0, 190))
        dr.text((lx + 5, ly + 5), label, fill=color + (255,))
    return {"center_uv": center_uv, "bbox": bbox, "in_view": in_view}


__all__ = ["draw_target_marker", "CAR_L", "CAR_W", "CAR_H"]

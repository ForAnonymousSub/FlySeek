# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Smooth-path + speed-profile builder for a route of waypoints.

Given an ordered set of NED waypoints, produces:

* A C¹-continuous parametric curve  ``s ↦ (x(s), y(s))``  with ``s`` =
  cumulative arclength.
* A speed profile  ``s ↦ v(s)``  that respects three constraints:
    - lateral acceleration:    v² · |κ(s)| ≤ a_lat_max
    - longitudinal accel:      v² ≤ v_prev² + 2 · a_long_max · ds
    - braking:                 v² ≤ v_next² + 2 · brake_max  · ds
  (Standard "trapezoidal" profile, used in motion planning textbooks.)

The intent is to make a teleport-driven car *look* like it's being driven by
a real driver: it slows before a corner, accelerates out, never abruptly
changes speed, and follows a smooth curve through waypoints instead of a
polyline with visible kinks.

Pure offline math — no AirSim, no rendering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline


@dataclass
class SmoothPath:
    """Resampled, smoothed path with arclength parameterisation."""
    s_samples: np.ndarray            # (N,) cumulative arclength [m]
    xy_samples: np.ndarray           # (N, 2) NED-XY positions
    heading_samples: np.ndarray      # (N,) tangent heading [rad]
    curvature_samples: np.ndarray    # (N,) signed curvature [1/m]
    v_profile: np.ndarray            # (N,) feasible speed [m/s]
    total_length_m: float

    def project_s(
        self,
        xy: np.ndarray,
        *,
        hint_s: float | None = None,
        search_window_m: float = 25.0,
    ) -> float:
        """Project an XY point onto the path → return its arclength.

        Returns *interpolated* arclength (not just the nearest sample). The
        car typically moves 0.05–0.3 m per tick, well below the path's
        sample spacing (~0.5 m), so snapping to the nearest sample would
        glue ``_s`` to a single value and the car would never advance.
        """
        xy = np.asarray(xy, dtype=np.float64).reshape(2)
        if hint_s is None:
            lo, hi = 0, len(self.s_samples)
        else:
            s_lo = max(0.0, hint_s - search_window_m)
            s_hi = min(self.total_length_m, hint_s + search_window_m)
            lo = int(np.searchsorted(self.s_samples, s_lo))
            hi = int(np.searchsorted(self.s_samples, s_hi)) + 1
            lo = max(0, lo)
            hi = min(len(self.s_samples), max(lo + 2, hi))
        block = self.xy_samples[lo:hi]
        diffs = block - xy
        d2 = (diffs * diffs).sum(axis=1)
        local_idx = int(np.argmin(d2))
        idx = lo + local_idx
        # Sub-sample refinement: project xy onto the segment connecting the
        # nearest sample with whichever neighbour is closer. Returns a
        # fractional arclength between the two sample s-values.
        if 0 < idx < len(self.s_samples) - 1:
            d_prev = float(np.linalg.norm(self.xy_samples[idx - 1] - xy))
            d_next = float(np.linalg.norm(self.xy_samples[idx + 1] - xy))
            j = idx - 1 if d_prev < d_next else idx + 1
        elif idx == 0:
            j = 1
        else:
            j = idx - 1
        a = self.xy_samples[min(idx, j)]
        b = self.xy_samples[max(idx, j)]
        ab = b - a
        seg_len_sq = float(ab @ ab)
        if seg_len_sq < 1e-12:
            return float(self.s_samples[idx])
        u = float(np.clip(((xy - a) @ ab) / seg_len_sq, 0.0, 1.0))
        s_a = float(self.s_samples[min(idx, j)])
        s_b = float(self.s_samples[max(idx, j)])
        return s_a + u * (s_b - s_a)

    def pose_at(self, s: float) -> tuple[np.ndarray, float]:
        """Interpolate (xy, heading) at arclength ``s``."""
        s_clamped = float(np.clip(s, 0.0, self.total_length_m))
        idx = float(np.interp(s_clamped, self.s_samples,
                              np.arange(len(self.s_samples))))
        lo = int(math.floor(idx))
        hi = min(lo + 1, len(self.s_samples) - 1)
        frac = idx - lo
        xy = self.xy_samples[lo] * (1.0 - frac) + self.xy_samples[hi] * frac
        # Heading: angle-wrap-safe interpolation.
        h_lo = float(self.heading_samples[lo])
        h_hi = float(self.heading_samples[hi])
        dh = math.atan2(math.sin(h_hi - h_lo), math.cos(h_hi - h_lo))
        heading = h_lo + dh * frac
        return xy, heading

    def speed_at(self, s: float) -> float:
        s_clamped = float(np.clip(s, 0.0, self.total_length_m))
        return float(np.interp(s_clamped, self.s_samples, self.v_profile))


def build_smooth_path(
    waypoints_xy: np.ndarray,
    *,
    resample_step_m: float = 0.5,
    v_cap: float = 8.0,
    a_lat_max: float = 2.5,      # m/s² (≈ comfortable cornering for a car)
    a_long_max: float = 1.5,     # m/s² (accel)
    brake_max: float = 2.5,      # m/s² (braking)
    v_start: float = 0.0,
    v_end: float = 0.0,
    min_v: float = 0.6,          # m/s — never crawl to zero except at endpoints
) -> SmoothPath:
    """Build a smooth path + speed profile through ``waypoints_xy`` (Nx2)."""
    pts = np.asarray(waypoints_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
        raise ValueError(
            f"waypoints_xy must be (N>=2, 2); got shape {pts.shape}"
        )

    if pts.shape[0] == 2:
        # Linear segment — no spline needed.
        seg = pts[1] - pts[0]
        L = float(np.linalg.norm(seg))
        n = max(2, int(math.ceil(L / max(resample_step_m, 0.1))) + 1)
        s = np.linspace(0.0, L, n)
        xy = np.outer(1.0 - s / L, pts[0]) + np.outer(s / L, pts[1])
        heading = np.full(n, math.atan2(seg[1], seg[0]))
        curvature = np.zeros(n)
    else:
        # Param the spline by chord length (de-facto arclength approximation).
        chord = np.concatenate([
            [0.0],
            np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1)),
        ])
        if chord[-1] < 1e-6:
            raise ValueError("waypoints have zero total length")
        # CubicSpline with natural BC keeps endpoints smooth (no oscillation).
        cs_x = CubicSpline(chord, pts[:, 0], bc_type="natural")
        cs_y = CubicSpline(chord, pts[:, 1], bc_type="natural")

        # Resample finely → then recompute true arclength → reparametrise.
        fine_t = np.linspace(0.0, chord[-1], max(64, int(chord[-1] * 4)))
        fine_xy = np.column_stack((cs_x(fine_t), cs_y(fine_t)))
        fine_ds = np.linalg.norm(np.diff(fine_xy, axis=0), axis=1)
        fine_s = np.concatenate([[0.0], np.cumsum(fine_ds)])
        L = float(fine_s[-1])

        # Resample at constant arclength spacing.
        n = max(8, int(math.ceil(L / max(resample_step_m, 0.1))) + 1)
        s = np.linspace(0.0, L, n)
        # Map even-s back to t (since spline is in chord-param), then sample.
        t_at_s = np.interp(s, fine_s, fine_t)
        xy = np.column_stack((cs_x(t_at_s), cs_y(t_at_s)))

        # First / second derivative wrt chord-param t (chain rule cancels in
        # curvature κ = (x'y'' − y'x'') / (x'² + y'²)^(3/2) ).
        dxdt = cs_x(t_at_s, 1)
        dydt = cs_y(t_at_s, 1)
        d2xdt2 = cs_x(t_at_s, 2)
        d2ydt2 = cs_y(t_at_s, 2)
        denom = (dxdt * dxdt + dydt * dydt) ** 1.5
        curvature_raw = np.where(
            denom > 1e-9,
            (dxdt * d2ydt2 - dydt * d2xdt2) / np.where(denom > 1e-9, denom, 1.0),
            0.0,
        )
        # Clip pathological spikes at folded waypoints: a real car can't turn
        # sharper than ~6 m radius (κ ≈ 0.17 /m). Without this clip a tiny
        # ~25 cm spline anomaly would peg the speed profile at the floor.
        curvature = np.clip(curvature_raw, -1.0 / 6.0, 1.0 / 6.0)
        # Light low-pass to smooth single-sample curvature artifacts.
        if len(curvature) >= 5:
            k = curvature.copy()
            for _ in range(2):
                k[1:-1] = (k[:-2] + 2.0 * k[1:-1] + k[2:]) / 4.0
            curvature = k
        heading = np.arctan2(dydt, dxdt)

    # Speed profile.
    # 1. Per-sample curvature limit:  v² · |κ| ≤ a_lat_max  →  v ≤ √(a/|κ|)
    abs_k = np.abs(curvature)
    v_lat = np.where(
        abs_k > 1e-4,
        np.sqrt(a_lat_max / np.clip(abs_k, 1e-4, None)),
        v_cap,
    )
    v = np.minimum(v_cap, v_lat).astype(np.float64)

    # 2. Forward pass — bound rate of speed-UP by a_long_max.
    v[0] = min(v[0], max(v_start, min_v))
    ds = np.diff(s)
    for i in range(1, len(v)):
        v_acc = math.sqrt(v[i - 1] * v[i - 1] + 2.0 * a_long_max * ds[i - 1])
        v[i] = min(v[i], v_acc)

    # 3. Backward pass — bound rate of speed-DOWN by brake_max.
    v[-1] = min(v[-1], max(v_end, min_v))
    for i in range(len(v) - 2, -1, -1):
        v_brake = math.sqrt(v[i + 1] * v[i + 1] + 2.0 * brake_max * ds[i])
        v[i] = min(v[i], v_brake)

    # Floor — never let v collapse to zero mid-route (looks like the car died).
    v = np.maximum(v, min_v)
    v[0] = max(v_start, min_v)
    v[-1] = max(v_end, min_v)

    return SmoothPath(
        s_samples=s,
        xy_samples=xy,
        heading_samples=heading,
        curvature_samples=curvature,
        v_profile=v,
        total_length_m=float(s[-1]),
    )


__all__ = ["SmoothPath", "build_smooth_path"]

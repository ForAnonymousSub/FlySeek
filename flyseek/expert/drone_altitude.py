# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Drone altitude control — OpenFly traj_gen style (stable AGL + slow roof ceiling).

OpenFly keeps a segment ``flyheight_`` and only lifts starts with
``getMaxZinP(x, y, 2) + 6``. Per-tick ``min_safe_map_z`` probing at every
waypoint causes visible vertical jitter in flyseek demos; this module uses:

  1. **Target-ground AGL** — ``local_ground_map_z(target) + follow_altitude``
  2. **EMA roof ceiling** — slow tracking of ``min_safe_map_z`` at drone XY
  3. **Rate-limited vertical EMA** — same idea as ``TrackingDroneController``
"""

from __future__ import annotations

import math

import numpy as np

from flyseek.adapters.pcd_occupancy import PcdOccupancyMap
from flyseek.adversary.base import DroneState, TargetState
from flyseek.utils.coords import airsim_ned_to_map


class OpenFlyDroneAltitude:
    """Compute smoothed NED ``z`` for a tracking drone."""

    def __init__(
        self,
        follow_altitude_agl_m: float,
        occupancy: PcdOccupancyMap | None = None,
        *,
        roof_smooth_tau_s: float = 6.0,
        alt_smooth_tau_s: float = 4.0,
        max_climb_mps: float = 1.5,
        max_drop_mps: float = 2.0,
        roof_probe_range_m: float = 2.0,
    ) -> None:
        self._follow = abs(float(follow_altitude_agl_m))
        self._occupancy = occupancy
        self._roof_tau = max(0.5, float(roof_smooth_tau_s))
        self._alt_tau = max(0.3, float(alt_smooth_tau_s))
        self._max_climb = float(max_climb_mps)
        self._max_drop = float(max_drop_mps)
        self._roof_range = float(roof_probe_range_m)
        self._roof_ceiling_ned: float | None = None
        self._alt_ema: float | None = None

    def reset(self, drone: DroneState, target: TargetState) -> None:
        z0 = self._raw_desired_ned_z(target.position, drone.position)
        self._roof_ceiling_ned = z0
        self._alt_ema = float(drone.position[2])

    def step(
        self,
        drone: DroneState,
        target: TargetState,
        dt: float,
    ) -> tuple[float, dict[str, float]]:
        """Return smoothed NED z and debug scalars."""
        desired = self._raw_desired_ned_z(target.position, drone.position)
        if self._alt_ema is None:
            self._alt_ema = float(drone.position[2])

        alpha = 1.0 - math.exp(-dt / self._alt_tau)
        self._alt_ema += alpha * (desired - self._alt_ema)

        dz = self._alt_ema - float(drone.position[2])
        dz = float(np.clip(dz, -self._max_climb * dt, self._max_drop * dt))
        out_z = float(drone.position[2]) + dz
        return out_z, {
            "desired_z_ned": round(desired, 2),
            "roof_ceiling_ned": round(float(self._roof_ceiling_ned or desired), 2),
        }

    def _raw_desired_ned_z(
        self,
        target_ned: np.ndarray,
        drone_ned: np.ndarray,
    ) -> float:
        if self._occupancy is None:
            return -self._follow

        target_map = airsim_ned_to_map(np.asarray(target_ned, dtype=np.float64))
        ground_z = self._occupancy.local_ground_map_z(target_map)
        desired = -(ground_z + self._follow)

        ceiling = self._roof_ceiling_ned_at(drone_ned)
        if desired > ceiling:
            desired = ceiling
        return float(desired)

    def _roof_ceiling_ned_at(self, drone_ned: np.ndarray) -> float:
        """OpenFly ``getMaxZinP`` + clearance, heavily low-passed (NED z)."""
        if self._occupancy is None:
            return -self._follow

        probe = airsim_ned_to_map(np.asarray(drone_ned, dtype=np.float64))
        roof_z = self._occupancy.local_roof_map_z_window(
            probe, range_m=self._roof_range
        )
        ceiling = -(roof_z + self._occupancy.cfg.min_drone_clearance)

        if self._roof_ceiling_ned is None:
            self._roof_ceiling_ned = ceiling
            return ceiling

        alpha = 0.15
        self._roof_ceiling_ned = (
            (1.0 - alpha) * self._roof_ceiling_ned + alpha * ceiling
        )
        return float(self._roof_ceiling_ned)


__all__ = ["OpenFlyDroneAltitude"]

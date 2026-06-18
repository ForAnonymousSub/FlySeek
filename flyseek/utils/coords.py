# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Coordinate transforms between AirSim NED and OpenFly map frame.

OpenFly's traj_gen / PCD / seg_map share one **map** frame. The AirSim bridge
(``scripts/sim/airsim_bridge.py``) converts planner poses to AirSim as::

    airsim_x =  map_x
    airsim_y = -map_y
    airsim_z = -map_z

We invert that here so flyseek modules can query ``scene_data/pcd_map`` using
poses read from ``simGetVehiclePose`` / ``simGetObjectPose``.
"""

from __future__ import annotations

import numpy as np


def airsim_ned_to_map(pos_ned: np.ndarray) -> np.ndarray:
    """AirSim NED (x,y,z) → OpenFly map (x,y,z)."""
    p = np.asarray(pos_ned, dtype=np.float64).reshape(3)
    return np.array([p[0], -p[1], -p[2]], dtype=np.float64)


def map_to_airsim_ned(pos_map: np.ndarray) -> np.ndarray:
    """OpenFly map (x,y,z) → AirSim NED (x,y,z)."""
    p = np.asarray(pos_map, dtype=np.float64).reshape(3)
    return np.array([p[0], -p[1], -p[2]], dtype=np.float64)


def airsim_altitude_m(pos_ned: np.ndarray) -> float:
    """Altitude above the NED origin plane (positive up)."""
    return float(-np.asarray(pos_ned, dtype=np.float64).reshape(3)[2])


def map_agl_m(pos_map: np.ndarray, ground_elevation: float = 0.0) -> float:
    """Height above configured ground elevation in map frame."""
    return float(np.asarray(pos_map, dtype=np.float64).reshape(3)[2] - ground_elevation)

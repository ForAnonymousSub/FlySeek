# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Drone tracking controller with SEARCH when target is occluded."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from flyseek.adapters.pcd_occupancy import PcdOccupancyMap
from flyseek.adversary.base import DroneState, TargetState, wrap_to_pi
from flyseek.utils.coords import airsim_ned_to_map
from flyseek.expert.drone_altitude import OpenFlyDroneAltitude
from flyseek.utils.visibility import fov_centering_offset_xy, target_visible


@dataclass
class TrackerConfig:
    follow_distance: float = 12.0
    follow_altitude: float = 12.0
    drone_smoothing: float = 3.0
    hfov_deg: float = 50.0
    max_range_m: float = 80.0
    lost_after_s: float = 0.6
    predict_after_s: float = 0.25
    fov_center_gain_m_per_rad: float = 10.0
    search_orbit_radius: float = 14.0
    search_orbit_speed: float = 0.35
    search_climb_extra_m: float = 3.0
    search_scan_yaw_rate: float = 0.45


@dataclass
class TrackerState:
    mode: str = "track"
    last_visible_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    last_visible_heading: float = 0.0
    lost_since: float = -1.0
    search_angle: float = 0.0
    search_expansion: float = 1.0
    alt_ned_ema: float = 0.0
    alt_ema_initialized: bool = False


class TrackingDroneController:
    """TRACK when target visible; SEARCH orbit when lost; REACQUIRE on sight."""

    def __init__(
        self,
        args: Any,
        occupancy: PcdOccupancyMap | None = None,
        cfg: TrackerConfig | None = None,
    ) -> None:
        self.args = args
        self.occupancy = occupancy
        self.cfg = cfg or TrackerConfig(
            follow_distance=float(getattr(args, "follow_distance", 12.0)),
            follow_altitude=float(getattr(args, "follow_altitude", 12.0)),
            drone_smoothing=float(getattr(args, "drone_smoothing", 3.0)),
            hfov_deg=float(getattr(args, "camera_hfov_deg", 50.0)),
            lost_after_s=float(getattr(args, "lost_after_s", 0.6)),
            predict_after_s=float(getattr(args, "predict_after_s", 0.8)),
            fov_center_gain_m_per_rad=float(
                getattr(args, "tracker_fov_center_gain", 10.0)
            ),
            search_orbit_radius=float(getattr(args, "search_orbit_radius", 14.0)),
        )
        self.state = TrackerState()
        self._altitude = OpenFlyDroneAltitude(
            self.cfg.follow_altitude,
            occupancy,
            roof_smooth_tau_s=float(getattr(args, "roof_smooth_tau", 6.0)),
            alt_smooth_tau_s=float(getattr(args, "altitude_smooth_tau", 4.0)),
            max_climb_mps=float(getattr(args, "max_climb_mps", 1.5)),
            max_drop_mps=float(getattr(args, "max_drop_mps", 2.0)),
            roof_probe_range_m=float(getattr(args, "roof_probe_range_m", 2.0)),
        )

    def reset(self, drone: DroneState, target: TargetState) -> None:
        self.state = TrackerState(
            last_visible_pos=target.position.copy(),
            last_visible_heading=target.heading,
            alt_ned_ema=float(drone.position[2]),
            alt_ema_initialized=True,
        )
        self._altitude.reset(drone, target)

    def step(
        self,
        drone: DroneState,
        target: TargetState,
        dt: float,
    ) -> tuple[DroneState, dict[str, Any]]:
        visible = target_visible(
            self.occupancy,
            drone,
            target,
            hfov_deg=self.cfg.hfov_deg,
            max_range_m=self.cfg.max_range_m,
            drone_eye_agl_m=float(self.cfg.follow_altitude),
        )

        if visible:
            self.state.last_visible_pos = target.position.copy()
            self.state.last_visible_heading = target.heading
            self.state.lost_since = -1.0
            if self.state.mode in ("search", "predict", "reacquire"):
                self.state.mode = "reacquire"
            else:
                self.state.mode = "track"
        else:
            if self.state.lost_since < 0:
                self.state.lost_since = target.timestamp
            else:
                lost_dur = target.timestamp - self.state.lost_since
                if lost_dur >= self.cfg.lost_after_s:
                    self.state.mode = "search"
                elif lost_dur >= self.cfg.predict_after_s:
                    self.state.mode = "predict"

        if self.state.mode == "track":
            new_drone, log = self._track(drone, target, dt)
        elif self.state.mode == "reacquire":
            new_drone, log = self._track(drone, target, dt)
            self.state.mode = "track"
        elif self.state.mode == "predict":
            pred_target = target.copy_with(
                position=self.state.last_visible_pos
                + target.velocity * max(dt, self.cfg.predict_after_s),
            )
            new_drone, log = self._track(drone, pred_target, dt)
            log["predict"] = True
        else:
            new_drone, log = self._search(drone, target, dt)

        log["tracker_mode"] = self.state.mode
        log["target_visible"] = visible
        return new_drone, log

    def _smooth_altitude_ned(
        self,
        drone: DroneState,
        desired_z: float,
        dt: float,
    ) -> float:
        """Low-pass + rate-limit vertical NED z to avoid roof-probe jitter."""
        if not self.state.alt_ema_initialized:
            self.state.alt_ned_ema = float(drone.position[2])
            self.state.alt_ema_initialized = True

        tau = float(getattr(self.args, "altitude_smooth_tau", 3.0))
        tau = max(tau, 0.2)
        alpha = 1.0 - math.exp(-dt / tau)
        self.state.alt_ned_ema += alpha * (desired_z - self.state.alt_ned_ema)

        max_climb = float(getattr(self.args, "max_climb_mps", 1.5))
        max_drop = float(getattr(self.args, "max_drop_mps", 2.0))
        dz = self.state.alt_ned_ema - float(drone.position[2])
        dz = float(np.clip(dz, -max_drop * dt, max_climb * dt))
        return float(drone.position[2]) + dz

    def _track(
        self,
        drone: DroneState,
        target: TargetState,
        dt: float,
    ) -> tuple[DroneState, dict[str, Any]]:
        v_xy = float(np.linalg.norm(target.velocity[:2]))
        if v_xy > 0.2:
            motion_dir = target.velocity[:2] / v_xy
        else:
            motion_dir = np.array([
                math.cos(target.heading),
                math.sin(target.heading),
            ])

        back = -motion_dir
        desired_xy = target.position[:2] + back * self.cfg.follow_distance
        desired_xy = desired_xy + fov_centering_offset_xy(
            drone.position,
            drone.heading,
            target.position,
            gain_m_per_rad=self.cfg.fov_center_gain_m_per_rad,
        )
        desired_z, _alt_log = self._altitude.step(drone, target, dt)

        yaw = math.atan2(
            target.position[1] - desired_xy[1],
            target.position[0] - desired_xy[0],
        )

        alpha = float(np.clip(self.cfg.drone_smoothing * dt, 0.0, 1.0))
        proposed = np.array([
            drone.position[0] + alpha * (desired_xy[0] - drone.position[0]),
            drone.position[1] + alpha * (desired_xy[1] - drone.position[1]),
            drone.position[2] + alpha * (desired_z - drone.position[2]),
        ])
        new_yaw = wrap_to_pi(
            drone.heading + alpha * wrap_to_pi(yaw - drone.heading)
        )

        if self.occupancy is not None and not getattr(self.args, "no_collision", False):
            proposed = self.occupancy.resolve_drone_ned(drone.position, proposed)

        new_state = DroneState(
            position=proposed,
            velocity=(proposed - drone.position) / max(dt, 1e-6),
            heading=new_yaw,
            timestamp=drone.timestamp + dt,
        )
        return new_state, {"follow": True}

    def _search(
        self,
        drone: DroneState,
        target: TargetState,
        dt: float,
    ) -> tuple[DroneState, dict[str, Any]]:
        self.state.search_angle += self.cfg.search_orbit_speed * dt
        self.state.search_expansion = min(
            2.0,
            self.state.search_expansion + 0.02 * dt,
        )

        centre = self.state.last_visible_pos
        r = self.cfg.search_orbit_radius * self.state.search_expansion
        desired_xy = centre[:2] + r * np.array([
            math.cos(self.state.search_angle),
            math.sin(self.state.search_angle),
        ])
        search_target = target.copy_with(
            position=np.array([
                desired_xy[0], desired_xy[1], centre[2],
            ], dtype=np.float64),
        )
        desired_z, _alt_log = self._altitude.step(drone, search_target, dt)

        yaw = wrap_to_pi(self.state.search_angle + math.pi / 2)

        alpha = float(np.clip(self.cfg.drone_smoothing * 1.5 * dt, 0.0, 1.0))
        proposed = np.array([
            drone.position[0] + alpha * (desired_xy[0] - drone.position[0]),
            drone.position[1] + alpha * (desired_xy[1] - drone.position[1]),
            drone.position[2] + alpha * (desired_z - drone.position[2]),
        ])
        new_yaw = wrap_to_pi(
            drone.heading + alpha * wrap_to_pi(yaw - drone.heading)
        )

        if self.occupancy is not None and not getattr(self.args, "no_collision", False):
            proposed = self.occupancy.resolve_drone_ned(drone.position, proposed)

        new_state = DroneState(
            position=proposed,
            velocity=(proposed - drone.position) / max(dt, 1e-6),
            heading=new_yaw,
            timestamp=drone.timestamp + dt,
        )
        return new_state, {
            "search_orbit_r": round(r, 1),
            "last_known": [round(float(v), 1) for v in centre[:3]],
        }


__all__ = ["TrackingDroneController", "TrackerConfig", "TrackerState"]

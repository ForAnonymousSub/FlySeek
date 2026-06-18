# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Adaptive UAV tracker — decides whether & how to chase a moving target.

Replaces the simple TRACK/SEARCH split in ``_InlineTracker`` with an explicit
state machine that reacts to (a) target motion and (b) occlusion:

    TRACK     — target visible: lead-pursuit follow at adaptive distance.
    PREDICT   — just lost (< t_predict): keep following the extrapolated pose.
    REACQUIRE — lost > t_predict: fly to predicted position with widened
                yaw-scan to recover the target.
    PEEK      — lost specifically due to PCD line-of-sight: side-step
                perpendicular to the LOS to peek around the occluder.
    SEARCH    — lost > t_search: expanding spiral around the *current*
                predicted position (NOT the stale anchor).
    HOLD      — target visible and stationary for hold_dwell_s: stop
                chasing, hover at a safe vantage point, keep camera centered.

All inputs and outputs use the same NED conventions as ``base.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from flyseek.adapters.pcd_occupancy import PcdOccupancyMap
from flyseek.adversary.base import (
    DroneState,
    TargetState,
    bearing_xy,
    horizontal_distance,
    wrap_to_pi,
)
from flyseek.expert.drone_altitude import OpenFlyDroneAltitude
from flyseek.utils.visibility import (
    find_clear_vantage_xy,
    fov_centering_offset_xy,
    visibility_status,
)


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class AdaptiveTrackerConfig:
    # Follow geometry
    follow_distance: float = 12.0
    follow_altitude: float = 12.0
    follow_distance_speed_gain: float = 0.8  # m per (m/s) of target speed
    follow_distance_max: float = 22.0
    hfov_deg: float = 50.0
    max_range_m: float = 80.0

    # Motion dynamics
    drone_smoothing: float = 3.0           # 1/s for xy
    yaw_gain: float = 2.0                  # 1/s
    max_yaw_rate_dps: float = 120.0
    motion_dir_tau_s: float = 1.5
    lead_s: float = 0.7
    fov_center_gain_m_per_rad: float = 10.0

    # State transition thresholds (seconds)
    predict_after_s: float = 0.4
    reacquire_after_s: float = 1.2
    peek_after_s: float = 0.8
    search_after_s: float = 3.0
    hold_speed_thresh: float = 0.3
    hold_dwell_s: float = 4.0
    hold_resume_speed: float = 1.0

    # PEEK
    peek_lateral_offsets_m: tuple[float, ...] = (6.0, -6.0, 10.0, -10.0, 14.0, -14.0)
    peek_forward_offsets_m: tuple[float, ...] = (-4.0, 0.0, 4.0)

    # REACQUIRE / SEARCH
    search_orbit_radius: float = 14.0
    search_orbit_speed_dps: float = 40.0
    search_radius_growth_mps: float = 1.5
    search_radius_max: float = 32.0
    reacquire_yaw_scan_amp_deg: float = 35.0
    reacquire_yaw_scan_period_s: float = 1.8

    # HOLD vantage
    hold_distance: float = 14.0
    hold_lateral_drift_m: float = 0.0

    # Altitude EMA passthroughs to OpenFlyDroneAltitude
    altitude_smooth_tau_s: float = 3.0
    roof_smooth_tau_s: float = 6.0
    max_climb_mps: float = 1.5
    max_drop_mps: float = 2.0
    roof_probe_range_m: float = 2.0


# --------------------------------------------------------------------------- #
# State                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class _AdaptiveState:
    mode: str = "track"
    # Last-confirmed target observation
    last_seen_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    last_seen_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    last_seen_heading: float = 0.0
    last_seen_t: float = 0.0
    # Filtered direction the target is moving in
    motion_dir: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    # Filtered target speed (for HOLD / adaptive distance)
    target_speed_filt: float = 0.0
    # Bookkeeping
    lost_since: float = -1.0
    stopped_since: float = -1.0
    last_loss_reason: str = ""
    orbit_phase: float = 0.0
    orbit_radius: float = 0.0
    peek_target_xy: np.ndarray | None = None
    peek_age_s: float = 0.0
    t_now: float = 0.0


# --------------------------------------------------------------------------- #
# Tracker                                                                     #
# --------------------------------------------------------------------------- #
class AdaptiveTracker:
    """Adaptive FSM tracker. See module docstring for state semantics."""

    MODES = ("track", "predict", "reacquire", "peek", "search", "hold")

    def __init__(
        self,
        cfg: AdaptiveTrackerConfig | None = None,
        occupancy: PcdOccupancyMap | None = None,
        *,
        no_collision: bool = False,
    ) -> None:
        self.cfg = cfg or AdaptiveTrackerConfig()
        self.occupancy = occupancy
        self._no_collision = bool(no_collision)
        self._max_yaw_rate = math.radians(self.cfg.max_yaw_rate_dps)
        self._altitude = OpenFlyDroneAltitude(
            self.cfg.follow_altitude,
            occupancy,
            roof_smooth_tau_s=self.cfg.roof_smooth_tau_s,
            alt_smooth_tau_s=self.cfg.altitude_smooth_tau_s,
            max_climb_mps=self.cfg.max_climb_mps,
            max_drop_mps=self.cfg.max_drop_mps,
            roof_probe_range_m=self.cfg.roof_probe_range_m,
        )
        self._state = _AdaptiveState()

    @classmethod
    def from_args(
        cls,
        args: Any,
        occupancy: PcdOccupancyMap | None = None,
    ) -> "AdaptiveTracker":
        cfg = AdaptiveTrackerConfig(
            follow_distance=float(getattr(args, "follow_distance", 12.0)),
            follow_altitude=float(getattr(args, "follow_altitude", 12.0)),
            hfov_deg=float(getattr(args, "camera_hfov_deg", 50.0)),
            drone_smoothing=float(getattr(args, "drone_smoothing", 3.0)),
            yaw_gain=float(getattr(args, "tracker_yaw_gain", 2.0)),
            motion_dir_tau_s=float(getattr(args, "tracker_motion_dir_tau", 1.5)),
            lead_s=float(getattr(args, "tracker_lead_s", 0.7)),
            fov_center_gain_m_per_rad=float(
                getattr(args, "tracker_fov_center_gain", 10.0)
            ),
            predict_after_s=float(getattr(args, "tracker_predict_after_s", 0.4)),
            reacquire_after_s=float(getattr(args, "tracker_reacquire_after_s", 1.2)),
            peek_after_s=float(getattr(args, "tracker_peek_after_s", 0.8)),
            search_after_s=float(getattr(args, "tracker_search_after_s", 3.0)),
            hold_speed_thresh=float(getattr(args, "tracker_hold_speed", 0.3)),
            hold_dwell_s=float(getattr(args, "tracker_hold_dwell_s", 4.0)),
            hold_resume_speed=float(getattr(args, "tracker_hold_resume_speed", 1.0)),
            search_orbit_radius=float(getattr(args, "search_orbit_radius", 14.0)),
            search_orbit_speed_dps=float(getattr(args, "search_orbit_speed_dps", 40.0)),
            altitude_smooth_tau_s=float(getattr(args, "altitude_smooth_tau", 3.0)),
            roof_smooth_tau_s=float(getattr(args, "roof_smooth_tau", 6.0)),
            max_climb_mps=float(getattr(args, "max_climb_mps", 1.5)),
            max_drop_mps=float(getattr(args, "max_drop_mps", 2.0)),
            roof_probe_range_m=float(getattr(args, "roof_probe_range_m", 2.0)),
        )
        return cls(
            cfg=cfg,
            occupancy=occupancy,
            no_collision=bool(getattr(args, "no_collision", False)),
        )

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #
    def reset(self, drone: DroneState, target: TargetState) -> None:
        st = _AdaptiveState()
        st.last_seen_pos = target.position.copy()
        st.last_seen_vel = target.velocity.copy()
        st.last_seen_heading = float(target.heading)
        st.last_seen_t = float(target.timestamp)
        h = float(target.heading)
        st.motion_dir = np.array([math.cos(h), math.sin(h)], dtype=np.float64)
        st.target_speed_filt = float(np.linalg.norm(target.velocity[:2]))
        st.orbit_radius = self.cfg.search_orbit_radius
        st.t_now = float(target.timestamp)
        self._state = st
        self._altitude.reset(drone, target)

    # ------------------------------------------------------------------ #
    # Main entry                                                         #
    # ------------------------------------------------------------------ #
    def step(
        self,
        drone: DroneState,
        target: TargetState,
        dt: float,
    ) -> tuple[DroneState, dict[str, Any]]:
        st = self._state
        st.t_now = float(target.timestamp)

        # Use the drone's ACTUAL altitude for the LOS check — a tracker that
        # climbs above buildings should be rewarded with a clear sightline.
        # cfg.follow_altitude acts as a floor only.
        eye_agl = max(self.cfg.follow_altitude, float(-drone.position[2]))
        visible, vis_reason = visibility_status(
            self.occupancy, drone, target,
            hfov_deg=self.cfg.hfov_deg,
            max_range_m=self.cfg.max_range_m,
            drone_eye_agl_m=eye_agl,
        )

        speed = float(np.linalg.norm(target.velocity[:2]))
        st.target_speed_filt = _ema(
            st.target_speed_filt, speed, dt, tau=max(0.4, self.cfg.motion_dir_tau_s)
        )
        if speed > 0.5:
            raw = target.velocity[:2] / max(speed, 1e-9)
            a = float(np.clip(dt / max(0.2, self.cfg.motion_dir_tau_s), 0.0, 1.0))
            blended = (1.0 - a) * st.motion_dir + a * raw
            n = float(np.linalg.norm(blended))
            if n > 1e-6:
                st.motion_dir = blended / n

        if visible:
            st.last_seen_pos = target.position.copy()
            st.last_seen_vel = target.velocity.copy()
            st.last_seen_heading = float(target.heading)
            st.last_seen_t = st.t_now
            st.lost_since = -1.0
            st.last_loss_reason = ""
            st.peek_target_xy = None
            st.peek_age_s = 0.0
            st.orbit_radius = self.cfg.search_orbit_radius
        else:
            if st.lost_since < 0.0:
                st.lost_since = st.t_now
            st.last_loss_reason = vis_reason

        if st.target_speed_filt < self.cfg.hold_speed_thresh:
            if st.stopped_since < 0.0:
                st.stopped_since = st.t_now
        else:
            st.stopped_since = -1.0

        st.mode = self._next_mode(visible, vis_reason, drone)

        predicted = self._predicted_target_pos(target)
        if st.mode == "track":
            new_drone, log = self._do_track(drone, target, predicted, dt, visible)
        elif st.mode == "predict":
            new_drone, log = self._do_predict(drone, target, predicted, dt)
        elif st.mode == "reacquire":
            new_drone, log = self._do_reacquire(drone, target, predicted, dt)
        elif st.mode == "peek":
            new_drone, log = self._do_peek(drone, target, predicted, dt)
        elif st.mode == "search":
            new_drone, log = self._do_search(drone, target, predicted, dt)
        else:  # hold
            new_drone, log = self._do_hold(drone, target, dt)

        log.update({
            "tracker_mode": st.mode,
            "visible": visible,
            "vis_reason": vis_reason,
            "lost_s": round(
                (st.t_now - st.lost_since) if st.lost_since >= 0 else 0.0, 2
            ),
            "target_speed_mps": round(speed, 2),
            "target_speed_filt": round(st.target_speed_filt, 2),
            "follow_dist_m": round(
                horizontal_distance(target.position, new_drone.position), 2
            ),
            "predicted_xy": [round(float(predicted[0]), 1),
                             round(float(predicted[1]), 1)],
        })
        return new_drone, log

    # ------------------------------------------------------------------ #
    # Transition logic                                                   #
    # ------------------------------------------------------------------ #
    def _next_mode(
        self,
        visible: bool,
        vis_reason: str,
        drone: DroneState,
    ) -> str:
        st = self._state
        cfg = self.cfg
        cur = st.mode

        # Visible: collapse all "lost" states back to track/hold.
        if visible:
            if (st.stopped_since >= 0
                    and (st.t_now - st.stopped_since) >= cfg.hold_dwell_s):
                return "hold"
            return "track"

        # Not visible. Decide degradation level by lost duration.
        lost = (st.t_now - st.lost_since) if st.lost_since >= 0 else 0.0
        if cur == "hold":
            # Target disappeared while we were holding — react.
            return "predict"
        if lost < cfg.predict_after_s:
            return "predict"
        if (cfg.peek_after_s <= lost < cfg.search_after_s
                and vis_reason == "los_blocked"
                and self.occupancy is not None):
            return "peek"
        if lost < cfg.reacquire_after_s:
            return "reacquire"
        if lost < cfg.search_after_s:
            return "reacquire"
        return "search"

    # ------------------------------------------------------------------ #
    # Predicted target pose                                              #
    # ------------------------------------------------------------------ #
    def _predicted_target_pos(self, target: TargetState) -> np.ndarray:
        st = self._state
        if st.lost_since < 0.0:
            return target.position.copy()
        dt_since = max(0.0, st.t_now - st.last_seen_t)
        # Clamp extrapolation distance to avoid wild drifts.
        v = st.last_seen_vel[:2]
        v_speed = float(np.linalg.norm(v))
        max_drift = 60.0
        if v_speed * dt_since > max_drift:
            v = (v / max(v_speed, 1e-9)) * (max_drift / max(dt_since, 1e-6))
        out = st.last_seen_pos.copy()
        out[0] += float(v[0]) * dt_since
        out[1] += float(v[1]) * dt_since
        return out

    # ------------------------------------------------------------------ #
    # Adaptive follow geometry                                           #
    # ------------------------------------------------------------------ #
    def _adaptive_follow_distance(self) -> float:
        cfg = self.cfg
        base = cfg.follow_distance
        extra = cfg.follow_distance_speed_gain * max(0.0, self._state.target_speed_filt)
        return float(min(cfg.follow_distance_max, base + extra))

    # ------------------------------------------------------------------ #
    # State controllers                                                  #
    # ------------------------------------------------------------------ #
    def _do_track(
        self,
        drone: DroneState,
        target: TargetState,
        predicted: np.ndarray,
        dt: float,
        visible: bool,
    ) -> tuple[DroneState, dict[str, Any]]:
        cfg = self.cfg
        st = self._state
        # Adaptive follow distance — gets SHORTER during sharp turns so the
        # camera doesn't trail behind the car on cornering. Detect "turning"
        # via the high-frequency component of the target's heading rate.
        target_turn_rate = float(np.linalg.norm(target.velocity[:2])) * (
            abs(wrap_to_pi(target.heading - st.last_seen_heading))
            / max(dt, 1e-6)
        )
        turn_factor = float(np.clip(target_turn_rate / 4.0, 0.0, 1.0))
        fd_base = self._adaptive_follow_distance()
        fd = fd_base * (1.0 - 0.30 * turn_factor)
        lead_xy = (target.position[:2].astype(np.float64)
                   + target.velocity[:2] * cfg.lead_s)
        back = -st.motion_dir
        desired_xy = lead_xy + back * fd
        if visible:
            desired_xy = desired_xy + fov_centering_offset_xy(
                drone.position,
                drone.heading,
                target.position,
                gain_m_per_rad=cfg.fov_center_gain_m_per_rad,
            )
        # face_xy is what the drone YAWS toward. Use the lead point — the
        # drone aims at where the car WILL be 0.5s from now, not where it is.
        # Camera is body-mounted, so yaw == camera direction; this lets the
        # camera "anticipate" instead of trailing on corners.
        face_xy = lead_xy
        return self._apply_motion(drone, target, desired_xy, face_xy, dt,
                                  log={"follow_distance_m": round(fd, 2),
                                       "turn_factor": round(turn_factor, 2)})

    def _do_predict(
        self,
        drone: DroneState,
        target: TargetState,
        predicted: np.ndarray,
        dt: float,
    ) -> tuple[DroneState, dict[str, Any]]:
        cfg = self.cfg
        st = self._state
        fd = self._adaptive_follow_distance()
        back = -st.motion_dir
        desired_xy = predicted[:2] + back * fd
        face_xy = predicted[:2]
        return self._apply_motion(drone, target, desired_xy, face_xy, dt,
                                  log={"predict": True})

    def _do_reacquire(
        self,
        drone: DroneState,
        target: TargetState,
        predicted: np.ndarray,
        dt: float,
    ) -> tuple[DroneState, dict[str, Any]]:
        cfg = self.cfg
        st = self._state
        fd = max(8.0, self._adaptive_follow_distance() * 0.85)
        back = -st.motion_dir
        desired_xy = predicted[:2] + back * fd
        # Sweep yaw around predicted bearing to widen the FOV cone.
        nominal_bearing = math.atan2(
            float(predicted[1] - drone.position[1]),
            float(predicted[0] - drone.position[0]),
        )
        phase = (st.t_now / max(0.1, cfg.reacquire_yaw_scan_period_s)) * 2.0 * math.pi
        offset = math.radians(cfg.reacquire_yaw_scan_amp_deg) * math.sin(phase)
        face_yaw = wrap_to_pi(nominal_bearing + offset)
        return self._apply_motion(drone, target, desired_xy, None, dt,
                                  override_face_yaw=face_yaw,
                                  log={"reacquire_scan_deg": round(math.degrees(offset), 1)})

    def _do_peek(
        self,
        drone: DroneState,
        target: TargetState,
        predicted: np.ndarray,
        dt: float,
    ) -> tuple[DroneState, dict[str, Any]]:
        cfg = self.cfg
        st = self._state
        # Recompute peek target periodically (every ~0.4s) or when stale.
        st.peek_age_s += dt
        if st.peek_target_xy is None or st.peek_age_s > 0.4:
            cand = find_clear_vantage_xy(
                self.occupancy,
                np.array([predicted[0], predicted[1],
                          float(st.last_seen_pos[2])]),
                drone.position,
                follow_distance=self._adaptive_follow_distance(),
                lateral_offsets_m=cfg.peek_lateral_offsets_m,
                forward_offsets_m=cfg.peek_forward_offsets_m,
                drone_eye_agl_m=cfg.follow_altitude,
                target_agl_m=max(0.5, -float(st.last_seen_pos[2])),
                keep_z_ned=float(drone.position[2]),
            )
            if cand is None:
                # No clear vantage nearby — degrade to reacquire path.
                return self._do_reacquire(drone, target, predicted, dt)
            st.peek_target_xy = cand[:2].copy()
            st.peek_age_s = 0.0
        desired_xy = st.peek_target_xy
        face_xy = predicted[:2]
        return self._apply_motion(drone, target, desired_xy, face_xy, dt,
                                  log={"peek": True,
                                       "peek_xy": [round(float(desired_xy[0]), 1),
                                                   round(float(desired_xy[1]), 1)]})

    def _do_search(
        self,
        drone: DroneState,
        target: TargetState,
        predicted: np.ndarray,
        dt: float,
    ) -> tuple[DroneState, dict[str, Any]]:
        cfg = self.cfg
        st = self._state
        st.orbit_radius = min(
            cfg.search_radius_max,
            st.orbit_radius + cfg.search_radius_growth_mps * dt,
        )
        omega = math.radians(cfg.search_orbit_speed_dps)
        st.orbit_phase = wrap_to_pi(st.orbit_phase + omega * dt)
        anchor = predicted[:2]
        desired_xy = np.array([
            anchor[0] + st.orbit_radius * math.cos(st.orbit_phase),
            anchor[1] + st.orbit_radius * math.sin(st.orbit_phase),
        ], dtype=np.float64)
        face_xy = anchor
        return self._apply_motion(drone, target, desired_xy, face_xy, dt,
                                  log={"search_orbit_r": round(st.orbit_radius, 1)})

    def _do_hold(
        self,
        drone: DroneState,
        target: TargetState,
        dt: float,
    ) -> tuple[DroneState, dict[str, Any]]:
        # Stay at a fixed vantage behind target along the last motion direction.
        st = self._state
        cfg = self.cfg
        back = -st.motion_dir
        desired_xy = target.position[:2] + back * cfg.hold_distance
        face_xy = target.position[:2].astype(np.float64)
        return self._apply_motion(drone, target, desired_xy, face_xy, dt,
                                  log={"hold": True,
                                       "hold_dist_m": round(cfg.hold_distance, 2)})

    # ------------------------------------------------------------------ #
    # Motion primitive                                                   #
    # ------------------------------------------------------------------ #
    def _apply_motion(
        self,
        drone: DroneState,
        target: TargetState,
        desired_xy: np.ndarray,
        face_xy: np.ndarray | None,
        dt: float,
        *,
        override_face_yaw: float | None = None,
        log: dict[str, Any] | None = None,
    ) -> tuple[DroneState, dict[str, Any]]:
        cfg = self.cfg
        alt_target = target.copy_with(
            position=np.array([
                float(desired_xy[0]), float(desired_xy[1]),
                float(target.position[2]),
            ], dtype=np.float64),
        )
        new_z, _alt_log = self._altitude.step(drone, alt_target, dt)

        alpha_xy = float(np.clip(cfg.drone_smoothing * dt, 0.0, 1.0))
        new_x = drone.position[0] + alpha_xy * (float(desired_xy[0]) - drone.position[0])
        new_y = drone.position[1] + alpha_xy * (float(desired_xy[1]) - drone.position[1])

        if override_face_yaw is not None:
            yaw_des = float(override_face_yaw)
        elif face_xy is not None:
            yaw_des = math.atan2(
                float(face_xy[1] - new_y),
                float(face_xy[0] - new_x),
            )
        else:
            yaw_des = drone.heading

        alpha_yaw = float(np.clip(cfg.yaw_gain * dt, 0.0, 1.0))
        d_yaw = alpha_yaw * wrap_to_pi(yaw_des - drone.heading)
        cap = self._max_yaw_rate * dt
        d_yaw = max(-cap, min(cap, d_yaw))
        new_yaw = wrap_to_pi(drone.heading + d_yaw)

        proposed = np.array([new_x, new_y, new_z], dtype=np.float64)
        if self.occupancy is not None and not self._no_collision:
            proposed = self.occupancy.resolve_drone_ned(drone.position, proposed)

        new_state = DroneState(
            position=proposed,
            velocity=np.array([
                (proposed[0] - drone.position[0]) / max(dt, 1e-6),
                (proposed[1] - drone.position[1]) / max(dt, 1e-6),
                (proposed[2] - drone.position[2]) / max(dt, 1e-6),
            ]),
            heading=new_yaw,
            timestamp=drone.timestamp + dt,
        )
        out_log: dict[str, Any] = dict(log) if log else {}
        return new_state, out_log


def _ema(prev: float, new: float, dt: float, *, tau: float) -> float:
    a = float(np.clip(dt / max(tau, 1e-3), 0.0, 1.0))
    return (1.0 - a) * float(prev) + a * float(new)


__all__ = ["AdaptiveTracker", "AdaptiveTrackerConfig"]

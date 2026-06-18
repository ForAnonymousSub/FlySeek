# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Adversarial agent — base contracts (pure numpy, no AirSim).

Per SKILL.md §2.1-§2.2: adversary state is updated offline at 10 Hz, decoupled
from the 30 Hz render loop. **The agent never touches AirSim** — it ingests
``DroneState`` + ``TargetState`` and emits an ``AgentAction``. The caller is
responsible for integrating the action into a new target pose and feeding it
back to the renderer (``simSetObjectPose``).

All coordinates are AirSim NED:
    +X = north (forward of drone at yaw=0)
    +Y = east
    +Z = down  (so altitudes increase as z decreases)
Headings are in radians, world frame, with 0 = +X axis and positive = CCW from
above (right-hand rule with +Z up convention)... but because z is down, "yaw"
in NED is actually clockwise from above. We just follow AirSim's
``to_quaternion(pitch, roll, yaw)`` convention end-to-end and document this
explicitly.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# --------------------------------------------------------------------------- #
# Coordinate helpers (pure functions, no class state)                         #
# --------------------------------------------------------------------------- #
def wrap_to_pi(angle: float) -> float:
    """Wrap an angle in radians into [-pi, +pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def bearing_xy(from_pos: np.ndarray, to_pos: np.ndarray) -> float:
    """Yaw angle (rad) pointing from ``from_pos`` toward ``to_pos`` in the XY
    plane (NED). 0 = pointing +X (north)."""
    dx = float(to_pos[0] - from_pos[0])
    dy = float(to_pos[1] - from_pos[1])
    return math.atan2(dy, dx)


def horizontal_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance in the XY plane (NED), ignoring altitude."""
    return float(np.linalg.norm(np.asarray(a[:2]) - np.asarray(b[:2])))


# --------------------------------------------------------------------------- #
# State dataclasses                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class DroneState:
    """Snapshot of the drone in NED world frame."""

    position: np.ndarray            # (3,) [x, y, z]
    velocity: np.ndarray            # (3,) [vx, vy, vz]
    heading: float = 0.0            # yaw (rad)
    timestamp: float = 0.0          # simulation time (s) since episode start

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        self.velocity = np.asarray(self.velocity, dtype=np.float64).reshape(3)

    @property
    def speed_xy(self) -> float:
        return float(np.linalg.norm(self.velocity[:2]))

    def copy_with(self, **overrides) -> "DroneState":
        return DroneState(
            position=overrides.get("position", self.position.copy()),
            velocity=overrides.get("velocity", self.velocity.copy()),
            heading=overrides.get("heading", self.heading),
            timestamp=overrides.get("timestamp", self.timestamp),
        )


@dataclass
class TargetState:
    """Snapshot of the adversarial target (car/cart/etc.) in NED world frame."""

    position: np.ndarray
    velocity: np.ndarray
    heading: float = 0.0
    timestamp: float = 0.0

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        self.velocity = np.asarray(self.velocity, dtype=np.float64).reshape(3)

    @property
    def speed_xy(self) -> float:
        return float(np.linalg.norm(self.velocity[:2]))

    def copy_with(self, **overrides) -> "TargetState":
        return TargetState(
            position=overrides.get("position", self.position.copy()),
            velocity=overrides.get("velocity", self.velocity.copy()),
            heading=overrides.get("heading", self.heading),
            timestamp=overrides.get("timestamp", self.timestamp),
        )


@dataclass
class AgentAction:
    """One adversary decision — the *desired* kinematics for the next dt."""

    desired_velocity: np.ndarray    # (3,) NED, m/s
    desired_heading: float          # rad (world yaw)
    behavior_state: str = "idle"    # human-readable mode label
    decision_log: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.desired_velocity = np.asarray(
            self.desired_velocity, dtype=np.float64
        ).reshape(3)


# --------------------------------------------------------------------------- #
# Episode-level box (optional safety net so target doesn't drive off scene)   #
# --------------------------------------------------------------------------- #
@dataclass
class PlayBox:
    """Axis-aligned XY play area. Z is *not* constrained; targets stay on the
    ground via ``z = ground_z`` upstream."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, pos: np.ndarray) -> bool:
        return (self.x_min <= pos[0] <= self.x_max
                and self.y_min <= pos[1] <= self.y_max)

    def clamp(self, pos: np.ndarray) -> np.ndarray:
        out = pos.copy()
        out[0] = float(np.clip(out[0], self.x_min, self.x_max))
        out[1] = float(np.clip(out[1], self.y_min, self.y_max))
        return out

    def reflect_velocity_at_boundary(
        self, pos: np.ndarray, vel: np.ndarray, restitution: float = 0.8
    ) -> np.ndarray:
        """If ``pos`` is at/over a boundary, flip the corresponding component."""
        v = vel.copy()
        if pos[0] <= self.x_min and v[0] < 0:
            v[0] = -v[0] * restitution
        if pos[0] >= self.x_max and v[0] > 0:
            v[0] = -v[0] * restitution
        if pos[1] <= self.y_min and v[1] < 0:
            v[1] = -v[1] * restitution
        if pos[1] >= self.y_max and v[1] > 0:
            v[1] = -v[1] * restitution
        return v


# --------------------------------------------------------------------------- #
# Abstract adversarial agent                                                  #
# --------------------------------------------------------------------------- #
class AdversarialAgent(ABC):
    """Per SKILL.md §2.2.

    Lifecycle::

        agent = create_adversarial_agent(...)
        agent.reset(initial_target_state)
        for t in time_steps:                # 10 Hz typical
            action = agent.step(drone, target, dt)
            target = integrate(target, action, dt)   # caller's job

    Subclasses must implement ``_decide(drone, target, dt) -> AgentAction``.
    The base class handles boundary reflection (if ``play_box`` set) and
    book-keeping (``self._time``, ``self._tick``).
    """

    def __init__(self, config: dict | None = None,
                 play_box: PlayBox | None = None,
                 rng: np.random.Generator | None = None) -> None:
        self.config: dict = config or {}
        self.play_box = play_box
        self.rng = rng if rng is not None else np.random.default_rng()

        self._time: float = 0.0     # accumulated simulation time (s)
        self._tick: int = 0         # number of step() calls
        self._initial_target: TargetState | None = None

    # -- Public API --------------------------------------------------------
    def reset(self, target_state: TargetState) -> None:
        self._time = float(target_state.timestamp)
        self._tick = 0
        self._initial_target = target_state.copy_with()

    def step(self, drone: DroneState, target: TargetState, dt: float) -> AgentAction:
        """Advance the agent by ``dt`` seconds and produce an action."""
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        action = self._decide(drone, target, dt)
        # Boundary reflection — only modulates the velocity, not the heading
        if self.play_box is not None:
            action.desired_velocity = self.play_box.reflect_velocity_at_boundary(
                target.position, action.desired_velocity
            )
        self._time += dt
        self._tick += 1
        return action

    # -- Subclass hook -----------------------------------------------------
    @abstractmethod
    def _decide(self, drone: DroneState, target: TargetState,
                dt: float) -> AgentAction:
        """Compute the next action. Pure function of inputs + self state."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Integration helper (caller uses this between step() calls)                  #
# --------------------------------------------------------------------------- #
def integrate_target(
    target: TargetState,
    action: AgentAction,
    dt: float,
    *,
    keep_z: float | None = None,
    max_speed: float | None = None,
    max_turn_rate_rad_s: float | None = None,
) -> TargetState:
    """Apply an ``AgentAction`` to a ``TargetState`` and return the new state.

    Parameters
    ----------
    keep_z
        If set, override the integrated z with this constant (ground-locked
        vehicles).
    max_speed
        Cap the magnitude of horizontal velocity.
    max_turn_rate_rad_s
        Cap the change in heading per second (kinematic-ish constraint).
    """
    vel = action.desired_velocity.copy()
    if max_speed is not None:
        vxy = np.linalg.norm(vel[:2])
        if vxy > max_speed and vxy > 1e-9:
            scale = max_speed / vxy
            vel[0] *= scale
            vel[1] *= scale

    new_pos = target.position + vel * dt
    if keep_z is not None:
        new_pos[2] = keep_z

    # Heading transition with optional rate-limit
    desired_h = action.desired_heading
    if max_turn_rate_rad_s is not None:
        delta = wrap_to_pi(desired_h - target.heading)
        max_step = max_turn_rate_rad_s * dt
        delta = float(np.clip(delta, -max_step, max_step))
        new_heading = wrap_to_pi(target.heading + delta)
    else:
        new_heading = wrap_to_pi(desired_h)

    return TargetState(
        position=new_pos,
        velocity=vel,
        heading=new_heading,
        timestamp=target.timestamp + dt,
    )

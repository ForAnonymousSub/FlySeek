# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Hide-and-seek car agent: open-road drive → hide behind building → peek.

Phase order (when PCD occupancy is available):
    open_road  — follow a major-street route (``open_then_hide``)
    goto_hide  — drive to an alley / building-occluded point
    hiding     — creep behind cover
    peek       — partial re-emerge for reacquire

Without PCD, falls back to S-curve evade then hide.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from flyseek.adapters.pcd_occupancy import PcdOccupancyMap
from flyseek.scenarios.road_scenarios import RoadScenarioConfig, RoadScenarioController

from .base import (
    AdversarialAgent,
    AgentAction,
    DroneState,
    PlayBox,
    TargetState,
    bearing_xy,
    horizontal_distance,
    wrap_to_pi,
)
from .medium import SCurveEvasionAgent, DEFAULTS as MEDIUM_DEFAULTS


DEFAULTS: dict[str, Any] = {
    **MEDIUM_DEFAULTS,
    "use_open_road_phase": True,
    "open_road_duration_s": 14.0,
    "open_road_route_len_m": 140.0,
    "open_road_min_route_frac": 0.55,
    "open_road_speed_mps": 4.5,
    "evade_before_hide_s": 10.0,
    "hide_trigger_range_m": 35.0,
    "hide_speed": 4.0,
    "hide_arrive_m": 3.0,
    "hide_stuck_timeout_s": 6.0,
    "hide_search_radius_m": 32.0,
    "hide_duration_s": 25.0,
    "peek_after_hide_s": 8.0,
    "peek_speed": 2.5,
    "peek_distance_m": 18.0,
    "hide_creep_speed": 0.2,
}


class HideSeekCarAgent(AdversarialAgent):
    """Car drives open roads, hides in alleys/buildings, then peeks."""

    def __init__(
        self,
        config: dict | None = None,
        play_box: PlayBox | None = None,
        rng: np.random.Generator | None = None,
        occupancy: PcdOccupancyMap | None = None,
    ) -> None:
        cfg = {**DEFAULTS, **(config or {})}
        super().__init__(config=cfg, play_box=play_box, rng=rng)
        self._occupancy = occupancy
        self._evade = SCurveEvasionAgent(config=cfg, play_box=play_box, rng=rng)
        self._road_ctrl: RoadScenarioController | None = None

        self._phase: str = "evade"
        self._hide_goal: np.ndarray | None = None
        self._hide_entered_at: float = -1.0
        self._hiding_started_at: float = -1.0
        self._peek_goal: np.ndarray | None = None
        self._creep_heading: float | None = None

    def reset(self, target_state: TargetState) -> None:
        super().reset(target_state)
        self._evade.reset(target_state)
        self._hide_goal = None
        self._hide_entered_at = -1.0
        self._hiding_started_at = -1.0
        self._peek_goal = None
        self._creep_heading = None
        self._road_ctrl = None

        if (self._occupancy is not None
                and bool(self.config.get("use_open_road_phase", True))):
            road_cfg = RoadScenarioConfig(
                name="open_then_hide",
                route_len_m=float(self.config["open_road_route_len_m"]),
                speed_mps=float(self.config["open_road_speed_mps"]),
                waypoint_step_m=6.0,
                search_radius_m=180.0,
            )
            self._road_ctrl = RoadScenarioController(
                self._occupancy,
                target_state,
                self.rng,
                road_cfg,
            )
            self._phase = "open_road"
        else:
            self._phase = "evade"

    def _route_progress_frac(self) -> float:
        if self._road_ctrl is None:
            return 0.0
        return self._road_ctrl.route_progress_frac

    def _should_begin_hide(self, drone: DroneState, target: TargetState) -> bool:
        if self._time < float(self.config.get("open_road_duration_s", 14.0)):
            if self._route_progress_frac() < float(
                self.config.get("open_road_min_route_frac", 0.55)
            ):
                return False
        r = horizontal_distance(target.position, drone.position)
        return r < float(self.config["hide_trigger_range_m"])

    def _decide(self, drone: DroneState, target: TargetState,
                dt: float) -> AgentAction:
        r = horizontal_distance(target.position, drone.position)

        if self._phase == "open_road":
            assert self._road_ctrl is not None
            _, action = self._road_ctrl.step(drone, self._time, dt)
            action.behavior_state = "open_road"
            action.decision_log = {
                **action.decision_log,
                "route_progress": round(self._route_progress_frac(), 2),
            }
            if self._should_begin_hide(drone, target):
                self._begin_hide(drone, target)
                return self._goto_hide(drone, target, r)
            return self._annotate(action, r, drone, target)

        if self._phase == "evade":
            action = self._evade.step(drone, target, dt)
            action.behavior_state = "evade"
            if (self._time >= float(self.config["evade_before_hide_s"])
                    and r < float(self.config["hide_trigger_range_m"])):
                self._begin_hide(drone, target)
            return self._annotate(action, r, drone, target)

        if self._phase == "goto_hide":
            return self._goto_hide(drone, target, r)

        if self._phase == "hiding":
            return self._hiding(drone, target, r)

        return self._peek(drone, target, r)

    def _begin_hide(self, drone: DroneState, target: TargetState) -> None:
        self._phase = "goto_hide"
        self._hide_entered_at = self._time
        keep_z = float(target.position[2])
        if self._occupancy is not None:
            goal = self._occupancy.find_hide_goal_ned(
                target.position,
                drone.position,
                keep_z=keep_z,
                search_radius_m=float(self.config["hide_search_radius_m"]),
            )
            if goal is not None:
                self._hide_goal = goal
            else:
                away = bearing_xy(target.position, drone.position) + math.pi
                dist = float(self.config["peek_distance_m"])
                self._hide_goal = target.position + np.array([
                    math.cos(away) * dist,
                    math.sin(away) * dist,
                    0.0,
                ])
                self._hide_goal[2] = keep_z
        else:
            away = bearing_xy(target.position, drone.position) + math.pi
            self._hide_goal = target.position + np.array([
                math.cos(away) * 15.0,
                math.sin(away) * 15.0,
                0.0,
            ])
            self._hide_goal[2] = keep_z

    def _goto_hide(self, drone: DroneState, target: TargetState,
                   r: float) -> AgentAction:
        assert self._hide_goal is not None
        to_goal = self._hide_goal[:2] - target.position[:2]
        dist = float(np.linalg.norm(to_goal))
        stuck_s = self._time - self._hide_entered_at
        if (dist < float(self.config["hide_arrive_m"])
                or stuck_s >= float(self.config["hide_stuck_timeout_s"])):
            self._phase = "hiding"
            self._hiding_started_at = self._time
            return self._hiding(drone, target, r)

        heading = math.atan2(to_goal[1], to_goal[0])
        speed = float(self.config["hide_speed"])
        return AgentAction(
            desired_velocity=np.array([
                speed * math.cos(heading),
                speed * math.sin(heading),
                0.0,
            ]),
            desired_heading=heading,
            behavior_state="goto_hide",
            decision_log={
                "dist_to_hide_m": round(dist, 2),
                "goto_hide_s": round(stuck_s, 1),
            },
        )

    def _hiding(self, drone: DroneState, target: TargetState,
                r: float) -> AgentAction:
        hide_dur = float(self.config["hide_duration_s"])
        if self._hiding_started_at < 0:
            self._hiding_started_at = self._time

        if self._creep_heading is None:
            self._creep_heading = wrap_to_pi(
                bearing_xy(target.position, drone.position) + math.pi / 2
            )

        if (self._time - self._hiding_started_at) >= hide_dur:
            self._phase = "peek"
            self._plan_peek(target, drone)

        creep_heading = float(self._creep_heading)
        speed = float(self.config.get("hide_creep_speed", 0.2))
        return AgentAction(
            desired_velocity=np.array([
                speed * math.cos(creep_heading),
                speed * math.sin(creep_heading),
                0.0,
            ]),
            desired_heading=creep_heading,
            behavior_state="hiding",
            decision_log={
                "hidden_s": round(self._time - self._hiding_started_at, 1),
                "creep_heading_deg": round(math.degrees(creep_heading), 1),
            },
        )

    def _plan_peek(self, target: TargetState, drone: DroneState) -> None:
        if self._hide_goal is not None:
            direction = self._hide_goal[:2] - target.position[:2]
            if np.linalg.norm(direction) < 1e-3:
                direction = np.array([1.0, 0.0])
            direction = direction / np.linalg.norm(direction)
            peek_xy = target.position[:2] + direction * float(self.config["peek_distance_m"])
            self._peek_goal = np.array([peek_xy[0], peek_xy[1], target.position[2]])
        else:
            bearing = bearing_xy(target.position, drone.position)
            d = float(self.config["peek_distance_m"])
            self._peek_goal = target.position + np.array([
                math.cos(bearing) * d * 0.5,
                math.sin(bearing) * d * 0.5,
                0.0,
            ])

    def _peek(self, drone: DroneState, target: TargetState,
              r: float) -> AgentAction:
        if self._peek_goal is None:
            self._plan_peek(target, drone)
        assert self._peek_goal is not None
        to_goal = self._peek_goal[:2] - target.position[:2]
        dist = float(np.linalg.norm(to_goal))
        heading = math.atan2(to_goal[1], to_goal[0]) if dist > 0.5 else bearing_xy(
            target.position, drone.position
        )
        speed = float(self.config["peek_speed"]) if dist > 2.0 else 0.0
        return AgentAction(
            desired_velocity=np.array([
                speed * math.cos(heading),
                speed * math.sin(heading),
                0.0,
            ]),
            desired_heading=heading,
            behavior_state="peek_reemerge",
            decision_log={"dist_to_peek_m": round(dist, 2)},
        )

    def _annotate(
        self,
        action: AgentAction,
        r: float,
        drone: DroneState,
        target: TargetState,
    ) -> AgentAction:
        action.decision_log = {
            **action.decision_log,
            "phase": self._phase,
            "drone_distance_m": round(r, 2),
        }
        if self._occupancy is not None:
            action.decision_log["los_blocked"] = self._occupancy.los_blocked_ned(
                drone.position, target.position
            )
        return action


__all__ = ["HideSeekCarAgent", "DEFAULTS"]

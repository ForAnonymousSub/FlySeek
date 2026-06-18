# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Visibility-aware expert UAV viewpoint annotation (paper §3).

Produces a *reference* (expert) UAV viewpoint sequence for an episode. It does
not control the UAV — it is saved as benchmark annotation in ``trajectories.json``.

The expert is **visibility-aware and preemptive**, not shortest-path: at each
planning step it samples candidate UAV viewpoints around the target and scores
them over a short horizon of the target's *upcoming* positions (an offline oracle
lookahead), so the chosen viewpoint anticipates where the target is going and
where it might be occluded.

Scoring objective (higher is better):

    score = alpha * expected_visibility
          - beta  * occlusion_risk
          - gamma * distance_cost
          - eta   * collision_risk
          - mu    * smoothness_cost

Geometry uses the PCD occupancy map when available (``los_blocked_ned`` for LoS /
occlusion, ``is_3d_occupied_map`` + roof for collision). When occupancy is absent,
documented fallbacks are used:
  - occlusion_risk -> 0.0 (cannot be measured without a scene),
  - collision_risk -> altitude floor check only,
  - visibility    -> range + FOV only (no LoS).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from flyseek.adversary.base import bearing_xy, wrap_to_pi
from flyseek.utils.coords import airsim_ned_to_map


@dataclass
class ExpertTrajectoryConfig:
    # Objective weights.
    alpha_visibility: float = 1.0
    beta_occlusion: float = 0.8
    gamma_distance: float = 0.3
    eta_collision: float = 1.0
    mu_smoothness: float = 0.2
    # Viewpoint sampling.
    follow_distance_m: float = 12.0
    follow_altitude_m: float = 18.0
    n_angles: int = 12
    radius_factors: tuple[float, ...] = (0.8, 1.0, 1.2)
    altitude_factors: tuple[float, ...] = (0.85, 1.0, 1.15)
    # Visibility / preemption.
    hfov_deg: float = 70.0
    max_range_m: float = 100.0
    horizon_steps: int = 4          # how many future target steps to anticipate
    min_altitude_m: float = 6.0     # fallback collision floor when no occupancy
    plan_stride: int = 1            # plan every Nth trajectory sample


def _extract_traj(traj: Any) -> tuple[list[float], np.ndarray, np.ndarray]:
    """Return ``(times, positions[N,3], velocities[N,3])`` from flexible input.

    Accepts: list of dicts (``pos``/``position`` + optional ``vel``/``velocity``
    + ``t``/``timestamp``), list of ``[x,y,z]``, or an ``(N,3)`` array. Missing
    velocities are finite-differenced; missing times use a 0.05 s default step.
    """
    positions: list[list[float]] = []
    velocities: list[list[float] | None] = []
    times: list[float | None] = []

    if isinstance(traj, np.ndarray):
        traj = traj.tolist()
    for i, item in enumerate(traj):
        if isinstance(item, dict):
            p = item.get("pos", item.get("position"))
            if p is None and "target_state" in item:  # flyseek_meta record
                p = item["target_state"].get("pos")
                v = item["target_state"].get("vel")
            else:
                v = item.get("vel", item.get("velocity"))
            t = item.get("t", item.get("timestamp"))
        else:
            p, v, t = item, None, None
        positions.append([float(x) for x in np.asarray(p, dtype=np.float64).reshape(3)])
        velocities.append([float(x) for x in v] if v is not None else None)
        times.append(float(t) if t is not None else None)

    n = len(positions)
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)

    # Fill times.
    if any(t is None for t in times):
        dt = 0.05
        known = [(i, t) for i, t in enumerate(times) if t is not None]
        if len(known) >= 2:
            dt = (known[-1][1] - known[0][1]) / max(known[-1][0] - known[0][0], 1)
            dt = dt if dt > 1e-6 else 0.05
        times = [i * dt for i in range(n)]
    times = [float(t) for t in times]

    # Fill velocities by finite difference where missing.
    vel = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        if velocities[i] is not None:
            vel[i] = np.asarray(velocities[i], dtype=np.float64).reshape(3)
        elif n >= 2:
            j0 = max(0, i - 1)
            j1 = min(n - 1, i + 1)
            dt = max(times[j1] - times[j0], 1e-6)
            vel[i] = (pos[j1] - pos[j0]) / dt
    return times, pos, vel


class ExpertViewpointPlanner:
    """Visibility-aware, preemptive expert viewpoint planner."""

    def __init__(
        self,
        config: ExpertTrajectoryConfig | None = None,
        scene_context: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        self.cfg = config or ExpertTrajectoryConfig()
        scene_context = dict(scene_context or {})
        self.occupancy = scene_context.get("occupancy")
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    def plan(
        self,
        target_trajectory: Any,
        uav_trajectory: Any | None = None,
    ) -> dict[str, Any]:
        """Return the expert annotation dict (see module docstring)."""
        times, tpos, tvel = _extract_traj(target_trajectory)
        n = len(times)
        cfg = self.cfg

        expert_viewpoints: list[dict[str, Any]] = []
        selected_scores: list[dict[str, Any]] = []
        prev_vp: np.ndarray | None = None
        prev_heading: float | None = None

        for i in range(0, n, max(1, cfg.plan_stride)):
            t_now = tpos[i]
            # Preemptive horizon: the target's upcoming positions (oracle).
            horizon_idx = [min(n - 1, i + k) for k in range(cfg.horizon_steps + 1)]
            future = tpos[horizon_idx]

            best = None
            for cand_pos, cand_heading in self._candidates(t_now):
                comp = self._score_components(
                    cand_pos, cand_heading, future, prev_vp, prev_heading,
                )
                total = (
                    cfg.alpha_visibility * comp["expected_visibility"]
                    - cfg.beta_occlusion * comp["occlusion_risk"]
                    - cfg.gamma_distance * comp["distance_cost"]
                    - cfg.eta_collision * comp["collision_risk"]
                    - cfg.mu_smoothness * comp["smoothness_cost"]
                )
                if best is None or total > best[0]:
                    best = (total, cand_pos, cand_heading, comp)

            total, vp, heading, comp = best
            expert_viewpoints.append({
                "t": float(times[i]),
                "frame_idx": int(i),
                "position": [float(v) for v in vp],
                "heading": float(heading),
            })
            selected_scores.append({
                "t": float(times[i]),
                "frame_idx": int(i),
                "score": round(float(total), 4),
                "components": {k: round(float(v), 4) for k, v in comp.items()},
            })
            prev_vp = vp
            prev_heading = heading

        target_out = [
            {"t": float(times[i]), "pos": [float(x) for x in tpos[i]]}
            for i in range(n)
        ]
        uav_out = self._normalize_uav(uav_trajectory)

        return {
            "uav_trajectory": uav_out,
            "target_trajectory": target_out,
            "expert_viewpoints": expert_viewpoints,
            "selected_scores": selected_scores,
            "config": asdict(cfg),
            "occupancy_available": self.occupancy is not None,
            "preemptive_horizon_steps": cfg.horizon_steps,
        }

    # ------------------------------------------------------------------ #
    def _candidates(self, target_pos: np.ndarray):
        cfg = self.cfg
        out = []
        for k in range(cfg.n_angles):
            ang = -math.pi + 2.0 * math.pi * k / cfg.n_angles
            for rf in cfg.radius_factors:
                r = cfg.follow_distance_m * rf
                xy = target_pos[:2] + r * np.array([math.cos(ang), math.sin(ang)])
                for af in cfg.altitude_factors:
                    z = -cfg.follow_altitude_m * af
                    pos = np.array([xy[0], xy[1], z], dtype=np.float64)
                    heading = bearing_xy(pos, target_pos)
                    out.append((pos, heading))
        return out

    def _score_components(
        self, cand_pos, cand_heading, future, prev_vp, prev_heading,
    ) -> dict[str, float]:
        cfg = self.cfg
        eye_agl = max(cfg.follow_altitude_m, float(-cand_pos[2]))

        vis = []
        occ = []
        for f in future:
            vis.append(self._visibility_one(cand_pos, cand_heading, f, eye_agl))
            occ.append(self._occlusion_one(cand_pos, f, eye_agl))
        expected_visibility = float(np.mean(vis)) if vis else 0.0
        # Fallback (documented): no occupancy -> occlusion cannot be measured.
        occlusion_risk = (float(np.mean(occ)) if (self.occupancy is not None and occ)
                          else 0.0)

        dist = float(np.hypot(*(future[0][:2] - cand_pos[:2])))
        distance_cost = min(
            1.0, abs(dist - cfg.follow_distance_m) / max(cfg.follow_distance_m, 1e-6)
        )

        collision_risk = self._collision_risk(cand_pos)

        if prev_vp is None:
            smoothness_cost = 0.0
        else:
            move = float(np.linalg.norm(cand_pos - prev_vp))
            pos_term = min(1.0, move / max(cfg.follow_distance_m, 1e-6))
            yaw_term = abs(wrap_to_pi(cand_heading - (prev_heading or 0.0))) / math.pi
            smoothness_cost = 0.7 * pos_term + 0.3 * yaw_term

        return {
            "expected_visibility": expected_visibility,
            "occlusion_risk": occlusion_risk,
            "distance_cost": distance_cost,
            "collision_risk": collision_risk,
            "smoothness_cost": smoothness_cost,
        }

    def _visibility_one(self, cand_pos, cand_heading, target_pos, eye_agl) -> float:
        cfg = self.cfg
        r = float(np.hypot(*(target_pos[:2] - cand_pos[:2])))
        if r > cfg.max_range_m:
            return 0.0
        off = abs(wrap_to_pi(bearing_xy(cand_pos, target_pos) - cand_heading))
        if off > math.radians(cfg.hfov_deg / 2.0):
            return 0.0
        if self.occupancy is not None:
            try:
                if self.occupancy.los_blocked_ned(
                    cand_pos, target_pos,
                    drone_eye_agl_m=eye_agl,
                    target_agl_m=max(0.5, -float(target_pos[2])),
                ):
                    return 0.0
            except Exception:
                pass
        return 1.0

    def _occlusion_one(self, cand_pos, target_pos, eye_agl) -> float:
        if self.occupancy is None:
            return 0.0
        try:
            return 1.0 if self.occupancy.los_blocked_ned(
                cand_pos, target_pos,
                drone_eye_agl_m=eye_agl,
                target_agl_m=max(0.5, -float(target_pos[2])),
            ) else 0.0
        except Exception:
            return 0.0

    def _collision_risk(self, cand_pos) -> float:
        cfg = self.cfg
        if self.occupancy is None:
            # Fallback: only an altitude floor can be checked without a scene.
            return 1.0 if (-float(cand_pos[2])) < cfg.min_altitude_m else 0.0
        try:
            mp = airsim_ned_to_map(cand_pos)
            if self.occupancy.is_3d_occupied_map(mp):
                return 1.0
            roof = float(self.occupancy.local_roof_map_z(mp))
            altitude_up = -float(cand_pos[2])
            # Soft penalty for skimming just above / below the local roofline.
            if altitude_up < roof + 1.0:
                return 0.7
        except Exception:
            return 0.0
        return 0.0

    def _normalize_uav(self, uav_trajectory):
        if uav_trajectory is None:
            return None
        times, pos, _ = _extract_traj(uav_trajectory)
        headings = []
        for item in (uav_trajectory if not isinstance(uav_trajectory, np.ndarray)
                     else uav_trajectory.tolist()):
            if isinstance(item, dict):
                headings.append(float(item.get("yaw", item.get("heading", 0.0)) or 0.0))
            else:
                arr = np.asarray(item, dtype=np.float64).reshape(-1)
                headings.append(float(arr[3]) if arr.size >= 4 else 0.0)
        return [
            {"t": float(times[i]), "pos": [float(x) for x in pos[i]],
             "heading": headings[i] if i < len(headings) else 0.0}
            for i in range(len(times))
        ]


def save_trajectories(record: dict[str, Any], path: str | Path) -> Path:
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def build_expert_trajectory_for_episode(
    episode_dir: str | Path,
    *,
    config: ExpertTrajectoryConfig | None = None,
    scene_context: dict[str, Any] | None = None,
    seed: int | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Read ``flyseek_meta.jsonl`` from an episode dir, plan, write trajectories.json."""
    import json
    episode_dir = Path(episode_dir)
    meta_path = episode_dir / "flyseek_meta.jsonl"
    records: list[dict] = []
    if meta_path.exists():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    target_traj = [
        {"t": r.get("timestamp"), "pos": r["target_state"]["pos"],
         "vel": r["target_state"].get("vel")}
        for r in records if "target_state" in r
    ]
    uav_traj = [
        {"t": r.get("timestamp"), "pos": r["drone_state"]["pos"],
         "heading": r["drone_state"].get("heading", 0.0)}
        for r in records if "drone_state" in r
    ]

    planner = ExpertViewpointPlanner(config=config, scene_context=scene_context, seed=seed)
    out = planner.plan(target_traj, uav_trajectory=uav_traj or None)
    if write:
        save_trajectories(out, episode_dir / "trajectories.json")
    return out


__all__ = [
    "ExpertTrajectoryConfig",
    "ExpertViewpointPlanner",
    "build_expert_trajectory_for_episode",
    "save_trajectories",
]

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Place any scene target on a valid drivable road pose (PCD offline).

Filters out water, building footprints, rooftops, and narrow elevated
structures such as guardrails (护栏) by requiring:

  - BEV-free + ground PCD support (``is_drivable_ned``)
  - Minimum road corridor width (lateral free space)
  - Minimum forward ray along heading
  - Local ground height variation below a threshold
  - Sufficient vertical clearance (roof − ground) — rejects thin rails
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from flyseek.adapters.pcd_occupancy import PcdOccupancyMap
from flyseek.adversary.base import wrap_to_pi
from flyseek.utils.coords import airsim_ned_to_map
from flyseek.utils.road_graph import (
    _corridor_width,
    _ray_free,
    find_major_road_seed,
    road_score,
)

@dataclass(frozen=True)
class TargetInitConfig:
  min_corridor_width_m: float = 10.0
  min_forward_ray_m: float = 18.0
  max_ground_slope_m: float = 1.2
  min_vertical_clearance_m: float = 5.5
  search_radius_m: float = 120.0
  sample_step_m: float = 4.0
  max_samples: int = 1200
  min_accept_score: float = 35.0
  prefer_near_anchor_m: float = 80.0
  # New hard gates (added to defeat 'island' false-positives where a single
  # drivable cell is surrounded by non-drivable ones).
  min_open_ray_sum_m: float = 30.0      # forward + backward must be a real road
  min_drive_feasibility_m: float = 8.0  # car must be able to roll forward N m
  # Hard cap on candidate-to-anchor distance (metres). 0 disables the cap.
  # Defends against pathological seeds that locate "drivable" water cells
  # far from the spawn point: the PCD ground band can pick up sparse water-
  # surface points and pass `is_drivable_ned`, but only at large shifts.
  # Capping the radius forces the spiral / road-seed search to stay near
  # the actor's scene spawn, which is by construction on a real road.
  max_shift_from_anchor_m: float = 0.0


@dataclass
class TargetInitResult:
  position_ned: np.ndarray
  heading_rad: float
  score: float
  ok: bool
  reason: str = ""
  tried: int = 0
  samples_tried: int = 0
  init_method: str = ""
  profile: str = ""
  anchor_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))

  def __post_init__(self) -> None:
    self.position_ned = np.asarray(self.position_ned, dtype=np.float64).reshape(3)
    self.anchor_ned = np.asarray(self.anchor_ned, dtype=np.float64).reshape(3)


def _heading_from_quaternion(qw: float, qx: float, qy: float, qz: float) -> float:
  """Yaw (rad) from AirSim quaternion (x,y,z,w ordering in to_quaternion)."""
  siny_cosp = 2.0 * (qw * qz + qx * qy)
  cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
  return math.atan2(siny_cosp, cosy_cosp)


def drive_feasibility_distance_m(
    occupancy: PcdOccupancyMap,
    pos_ned: np.ndarray,
    heading_rad: float,
    *,
    step_m: float = 1.0,
    max_dist_m: float = 15.0,
) -> float:
  """How many contiguous metres a car can roll forward along ``heading``
  before hitting a non-drivable cell. Single-cell drivable islands return 0.
  """
  p = np.asarray(pos_ned, dtype=np.float64).reshape(3).copy()
  dx = math.cos(float(heading_rad)) * step_m
  dy = math.sin(float(heading_rad)) * step_m
  travelled = 0.0
  while travelled + step_m <= max_dist_m:
    p[0] += dx
    p[1] += dy
    if not occupancy.is_drivable_ned(p):
      return travelled
    travelled += step_m
  return travelled


def _local_ground_slope_m(occupancy: PcdOccupancyMap, pos_ned: np.ndarray) -> float:
  """Max |Δground| across a 3×3 BEV neighborhood (map Z, metres)."""
  pos_map = airsim_ned_to_map(pos_ned)
  ix0, iy0, _ = occupancy._map_to_voxel(pos_map)  # noqa: SLF001
  vw = occupancy._vw  # noqa: SLF001
  zs: list[float] = []
  for dx in (-1, 0, 1):
    for dy in (-1, 0, 1):
      trial = pos_map.copy()
      trial[0] = pos_map[0] + dx * vw
      trial[1] = pos_map[1] + dy * vw
      zs.append(occupancy.local_ground_map_z(trial))
  return float(max(zs) - min(zs)) if zs else 0.0


def score_init_pose_ned(
    occupancy: PcdOccupancyMap,
    pos_ned: np.ndarray,
    heading_rad: float,
    *,
    cfg: TargetInitConfig | None = None,
    anchor_ned: np.ndarray | None = None,
) -> tuple[float, str]:
  """Higher is better; negative means hard reject."""
  cfg = cfg or TargetInitConfig()
  p = occupancy.snap_car_to_ground_ned(pos_ned)
  if not np.isfinite(p).all():
    return -1e9, "nan_pose"

  if anchor_ned is not None and cfg.max_shift_from_anchor_m > 0.0:
    dist = float(np.linalg.norm(p[:2] - np.asarray(anchor_ned)[:2]))
    if dist > cfg.max_shift_from_anchor_m:
      return -1e9, "shift_too_far"

  if not occupancy.is_drivable_ned(p):
    return -1e9, "not_drivable"

  keep_z = float(p[2])
  h = float(heading_rad)
  width = _corridor_width(occupancy, p, h, keep_z=keep_z, max_width=24.0)
  if width < cfg.min_corridor_width_m:
    return -1e9, "too_narrow"

  fwd = _ray_free(occupancy, p, h, keep_z=keep_z, max_dist=60.0)
  if fwd < cfg.min_forward_ray_m:
    return -1e9, "short_forward"

  back = _ray_free(occupancy, p, h + math.pi, keep_z=keep_z, max_dist=60.0)
  if (fwd + back) < cfg.min_open_ray_sum_m:
    return -1e9, "closed_island"

  drive_dist = drive_feasibility_distance_m(
      occupancy, p, h, max_dist_m=max(8.0, cfg.min_drive_feasibility_m + 4.0),
  )
  if drive_dist < cfg.min_drive_feasibility_m:
    return -1e9, "not_drivable_forward"

  slope = _local_ground_slope_m(occupancy, p)
  if slope > cfg.max_ground_slope_m:
    return -1e9, "steep_ground"

  pos_map = airsim_ned_to_map(p)
  ground_z = occupancy.local_ground_map_z(pos_map)
  roof_z = occupancy.local_roof_map_z(pos_map)
  v_clear = roof_z - ground_z
  # Reject only when an overhead structure clearly exists at low height
  # (guardrail / curb / canopy). When ``v_clear ≈ 0`` the cell has no overhead
  # voxels at all = open sky = perfectly fine for a car.
  if 0.5 < v_clear < cfg.min_vertical_clearance_m:
    return -1e9, "on_rail_or_curb"

  score = road_score(occupancy, p, h, keep_z=keep_z)
  score += 0.4 * width
  score += 0.15 * min(fwd, 60.0)
  score += 0.05 * min(drive_dist, 12.0)
  score -= 3.0 * slope

  if anchor_ned is not None:
    dist = float(np.linalg.norm(p[:2] - np.asarray(anchor_ned)[:2]))
    score -= 0.04 * max(0.0, dist - cfg.prefer_near_anchor_m)

  return float(score), "ok"


def _road_seed_relaxed_score(
    occupancy: PcdOccupancyMap,
    pos_ned: np.ndarray,
    heading_rad: float,
    *,
    min_corridor_m: float,
    min_road_score: float,
    min_open_ray_sum_m: float = 24.0,
    min_drive_feasibility_m: float = 6.0,
    min_vertical_clearance_m: float = 3.0,
) -> tuple[float, str]:
  """Lighter (but still ANTI-island) gates for major-road seed fallback.

  Even the relaxed path must reject:
    - drivable-cell islands (no forward+backward continuity)
    - guardrail-like tall narrow strips (vertical clearance too small)
    - cells that cannot actually be driven 6 m forward
  """
  p = occupancy.snap_car_to_ground_ned(pos_ned)
  if not np.isfinite(p).all():
    return -1e9, "nan_pose"
  if not occupancy.is_drivable_ned(p):
    return -1e9, "not_drivable"
  keep_z = float(p[2])
  h = float(heading_rad)
  width = _corridor_width(occupancy, p, h, keep_z=keep_z, max_width=24.0)
  if width < min_corridor_m:
    return -1e9, "too_narrow"
  fwd = _ray_free(occupancy, p, h, keep_z=keep_z, max_dist=40.0)
  back = _ray_free(occupancy, p, h + math.pi, keep_z=keep_z, max_dist=40.0)
  if (fwd + back) < min_open_ray_sum_m:
    return -1e9, "closed_island"
  drive_dist = drive_feasibility_distance_m(
      occupancy, p, h, max_dist_m=max(8.0, min_drive_feasibility_m + 4.0),
  )
  if drive_dist < min_drive_feasibility_m:
    return -1e9, "not_drivable_forward"
  pos_map = airsim_ned_to_map(p)
  v_clear = occupancy.local_roof_map_z(pos_map) - occupancy.local_ground_map_z(pos_map)
  # Same "open-sky tolerant" guardrail check as strict path.
  if 0.5 < v_clear < min_vertical_clearance_m:
    return -1e9, "on_rail_or_curb"
  rs = road_score(occupancy, p, h, keep_z=keep_z)
  if rs < min_road_score:
    return -1e9, "low_road_score"
  return float(rs), "road_seed_ok"


def _refine_seed_via_local_spiral(
    occupancy: PcdOccupancyMap,
    raw_seed: np.ndarray,
    anchor_ned: np.ndarray,
    cfg: TargetInitConfig,
    *,
    rng: np.random.Generator,
    hint_heading: float | None,
    max_radius_m: float = 32.0,
    step_m: float = 4.0,
) -> tuple[np.ndarray, int]:
  """Search a small spiral around ``raw_seed`` for the first STRICT-passing
  pose. Falls back to the raw seed if no strict candidate exists within
  ``max_radius_m``."""
  best_pos = raw_seed
  best_score = -1e9
  tried = 0
  offsets: list[tuple[float, float]] = [(0.0, 0.0)]
  for r in np.arange(step_m, max_radius_m + 1e-6, step_m):
    n = max(8, int(2 * math.pi * r / step_m))
    jitter = float(rng.uniform(0.0, 2.0 * math.pi / n))
    for k in range(n):
      a = jitter + 2.0 * math.pi * k / n
      offsets.append((r * math.cos(a), r * math.sin(a)))
  for dx, dy in offsets[:200]:
    p = raw_seed.copy()
    p[0] += dx
    p[1] += dy
    s, _h, _r, n_h = _best_heading_at(
        occupancy, p, anchor_ned, cfg, hint_heading=hint_heading,
    )
    tried += n_h
    if s > best_score:
      best_score = s
      best_pos = p.copy()
      if s > -1e8:
        break  # first strict-passing candidate is good enough
  return best_pos, tried


def _best_heading_at(
    occupancy: PcdOccupancyMap,
    pos_ned: np.ndarray,
    anchor_ned: np.ndarray,
    cfg: TargetInitConfig,
    *,
    hint_heading: float | None,
    n_dirs: int = 16,
) -> tuple[float, float, str, int]:
  """Pick heading with highest ``score_init_pose_ned`` at a fixed XY."""
  headings = np.linspace(0.0, 2.0 * math.pi, n_dirs, endpoint=False)
  if hint_heading is not None:
    headings = np.array([
      wrap_to_pi(float(hint_heading) + d)
      for d in (0.0, math.pi / 8, -math.pi / 8, math.pi / 4, -math.pi / 4,
                math.pi / 2, -math.pi / 2)
    ])
  best_score = -1e9
  best_h = float(headings[0])
  best_reason = "no_candidate"
  tried = 0
  for h in headings:
    tried += 1
    score, reason = score_init_pose_ned(
        occupancy, pos_ned, float(h), cfg=cfg, anchor_ned=anchor_ned,
    )
    if score > best_score:
      best_score = score
      best_h = float(h)
      best_reason = reason
  return best_score, best_h, best_reason, tried


def _best_heading_road_relaxed(
    occupancy: PcdOccupancyMap,
    pos_ned: np.ndarray,
    *,
    hint_heading: float | None,
    min_corridor_m: float,
    min_road_score: float,
    n_dirs: int = 16,
    min_vertical_clearance_m: float = 3.0,
) -> tuple[float, float, str, int]:
  headings = np.linspace(0.0, 2.0 * math.pi, n_dirs, endpoint=False)
  if hint_heading is not None:
    headings = np.array([
      wrap_to_pi(float(hint_heading) + d)
      for d in (0.0, math.pi / 8, -math.pi / 8, math.pi / 4, -math.pi / 4,
                math.pi / 2, -math.pi / 2)
    ])
  best_score = -1e9
  best_h = float(headings[0])
  best_reason = "no_candidate"
  tried = 0
  for h in headings:
    tried += 1
    score, reason = _road_seed_relaxed_score(
        occupancy, pos_ned, float(h),
        min_corridor_m=min_corridor_m,
        min_road_score=min_road_score,
        min_vertical_clearance_m=min_vertical_clearance_m,
    )
    if score > best_score:
      best_score = score
      best_h = float(h)
      best_reason = reason
  return best_score, best_h, best_reason, tried


def _result_from_candidate(
    occupancy: PcdOccupancyMap,
    pos_ned: np.ndarray,
    heading_rad: float,
    anchor_ned: np.ndarray,
    cfg: TargetInitConfig,
    *,
    init_method: str,
    samples_tried: int,
    profile: str = "",
) -> TargetInitResult:
  snapped = occupancy.snap_car_to_ground_ned(pos_ned)
  score, reason = score_init_pose_ned(
      occupancy, snapped, float(heading_rad), cfg=cfg, anchor_ned=anchor_ned,
  )
  ok = score >= cfg.min_accept_score
  if not ok and score > -1e8:
    ok = score >= cfg.min_accept_score * 0.65
    if ok:
      reason = "relaxed_threshold"
  return TargetInitResult(
      position_ned=snapped,
      heading_rad=float(heading_rad),
      score=score,
      ok=ok,
      reason=reason,
      tried=samples_tried,
      samples_tried=samples_tried,
      init_method=init_method,
      profile=profile,
      anchor_ned=anchor_ned,
  )


def _road_seed_init(
    occupancy: PcdOccupancyMap,
    anchor_ned: np.ndarray,
    rng: np.random.Generator,
    cfg: TargetInitConfig,
    *,
    hint_heading: float | None,
    search_radius_m: float,
    sample_step_m: float,
    profile: str,
    relaxed_accept: bool = True,
    min_road_score: float = 14.0,
    min_corridor_m: float = 6.0,
) -> TargetInitResult | None:
  keep_z = float(occupancy.snap_car_to_ground_ned(anchor_ned)[2])
  # Clamp to the anchor-shift cap so the road-seed fallback can't reach a
  # far-away "drivable water" cell that the strict spiral correctly rejected.
  if cfg.max_shift_from_anchor_m > 0.0:
    search_radius_m = min(float(search_radius_m), float(cfg.max_shift_from_anchor_m))
  raw_seed_pos, seed_h, _ = find_major_road_seed(
      occupancy,
      anchor_ned,
      rng,
      keep_z=keep_z,
      search_radius_m=search_radius_m,
      sample_step_m=sample_step_m,
  )
  # Reject up-front if the major-road seed already exceeds the shift cap
  # (find_major_road_seed only enforces a soft preference, not a hard gate).
  if cfg.max_shift_from_anchor_m > 0.0:
    seed_shift = float(np.linalg.norm(raw_seed_pos[:2] - np.asarray(anchor_ned)[:2]))
    if seed_shift > cfg.max_shift_from_anchor_m:
      return None
  # The "major road seed" may land on a parking-lot / square / island where the
  # BEV cell is free but no forward driving is possible. Refine it with a small
  # local spiral until a STRICT-passing pose is found, falling back to relaxed
  # only if no strict pose is reachable nearby.
  seed_pos, n_refine = _refine_seed_via_local_spiral(
      occupancy, raw_seed_pos, anchor_ned, cfg,
      rng=rng,
      hint_heading=hint_heading if hint_heading is not None else seed_h,
      max_radius_m=min(40.0, max(20.0, search_radius_m * 0.25)),
  )
  hint = hint_heading if hint_heading is not None else seed_h
  score, heading, reason, n_h = _best_heading_at(
      occupancy, seed_pos, anchor_ned, cfg, hint_heading=hint,
  )
  n_h += n_refine
  init_method = "road_seed"
  if score <= -1e8 and relaxed_accept:
    rs, heading, reason, n_h2 = _best_heading_road_relaxed(
        occupancy, seed_pos,
        hint_heading=hint,
        min_corridor_m=min_corridor_m,
        min_road_score=min_road_score,
        min_vertical_clearance_m=max(2.5, cfg.min_vertical_clearance_m * 0.6),
    )
    n_h += n_h2
    if rs > -1e8:
      score, heading, reason = rs, heading, reason
      init_method = "road_seed_relaxed"
  if score <= -1e8:
    return None
  snapped = occupancy.snap_car_to_ground_ned(seed_pos)
  if cfg.max_shift_from_anchor_m > 0.0:
    final_shift = float(np.linalg.norm(snapped[:2] - np.asarray(anchor_ned)[:2]))
    if final_shift > cfg.max_shift_from_anchor_m:
      return None
  ok = score >= cfg.min_accept_score
  if not ok and init_method == "road_seed_relaxed":
    ok = True
    reason = reason if reason != "ok" else "road_seed_ok"
  elif not ok and score > -1e8:
    ok = score >= cfg.min_accept_score * 0.65
    if ok:
      reason = "relaxed_threshold"
  return TargetInitResult(
      position_ned=snapped,
      heading_rad=float(heading),
      score=float(score),
      ok=ok,
      reason=reason,
      tried=n_h,
      samples_tried=n_h,
      init_method=init_method,
      profile=profile,
      anchor_ned=anchor_ned,
  )


def find_valid_init_pose(
    occupancy: PcdOccupancyMap,
    anchor_ned: np.ndarray,
    rng: np.random.Generator,
    *,
    hint_heading: float | None = None,
    cfg: TargetInitConfig | None = None,
    profile: str = "",
) -> TargetInitResult:
  """Search spiral offsets from ``anchor`` for the best init pose."""
  cfg = cfg or TargetInitConfig()
  anchor = np.asarray(anchor_ned, dtype=np.float64).reshape(3).copy()
  anchor = occupancy.snap_car_to_ground_ned(anchor)

  headings = np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)
  if hint_heading is not None:
    headings = np.array([
      wrap_to_pi(float(hint_heading) + d)
      for d in (0.0, math.pi / 4, -math.pi / 4, math.pi / 2, -math.pi / 2)
    ])

  best = TargetInitResult(
      position_ned=anchor.copy(),
      heading_rad=float(headings[0]),
      score=-1e9,
      ok=False,
      reason="no_candidate",
      init_method="spiral",
      profile=profile,
      anchor_ned=anchor,
  )
  samples_tried = 0

  offsets: list[tuple[float, float]] = [(0.0, 0.0)]
  for r in np.arange(cfg.sample_step_m, cfg.search_radius_m + 1e-6, cfg.sample_step_m):
    n = max(10, int(2 * math.pi * r / cfg.sample_step_m))
    jitter = float(rng.uniform(0.0, 2.0 * math.pi / n))
    for k in range(n):
      a = jitter + 2.0 * math.pi * k / n
      offsets.append((r * math.cos(a), r * math.sin(a)))

  rng.shuffle(offsets)

  for dx, dy in offsets:
    if samples_tried >= cfg.max_samples:
      break
    if (cfg.max_shift_from_anchor_m > 0.0
        and (dx * dx + dy * dy) > cfg.max_shift_from_anchor_m ** 2):
      # Anchor-shift cap is the primary defence against
      # "drivable water cell"-style false positives at large radii.
      continue
    p = anchor.copy()
    p[0] += dx
    p[1] += dy
    for h in headings:
      samples_tried += 1
      score, reason = score_init_pose_ned(
        occupancy, p, float(h), cfg=cfg, anchor_ned=anchor,
      )
      if score <= best.score:
        continue
      snapped = occupancy.snap_car_to_ground_ned(p)
      best = TargetInitResult(
        position_ned=snapped,
        heading_rad=float(h),
        score=score,
        ok=score >= cfg.min_accept_score,
        reason=reason,
        tried=samples_tried,
        samples_tried=samples_tried,
        init_method="spiral",
        profile=profile,
        anchor_ned=anchor,
      )

  if not best.ok and best.score > -1e8:
    best.ok = best.score >= cfg.min_accept_score * 0.65
    best.reason = "relaxed_threshold" if best.ok else best.reason

  best.samples_tried = samples_tried
  if best.tried == 0:
    best.tried = samples_tried
  return best


def resolve_target_init_pose(
    occupancy: PcdOccupancyMap,
    anchor_ned: np.ndarray,
    rng: np.random.Generator,
    profile,
    *,
    hint_heading: float | None = None,
) -> TargetInitResult:
  """Run env-specific init strategy (spiral / road-seed / combined)."""
  from flyseek.utils.target_init_presets import TargetInitProfile  # noqa: PLC0415
  assert isinstance(profile, TargetInitProfile)
  cfg = profile.config
  anchor = occupancy.snap_car_to_ground_ned(
      np.asarray(anchor_ned, dtype=np.float64).reshape(3),
  )
  strategy = profile.strategy
  spiral_res: TargetInitResult | None = None
  road_res: TargetInitResult | None = None

  def _spiral() -> TargetInitResult:
    return find_valid_init_pose(
        occupancy, anchor, rng,
        hint_heading=hint_heading, cfg=cfg, profile=profile.name,
    )

  def _road() -> TargetInitResult | None:
    if not profile.use_road_seed_fallback:
      return None
    return _road_seed_init(
        occupancy, anchor, rng, cfg,
        hint_heading=hint_heading,
        search_radius_m=profile.road_seed_search_radius_m,
        sample_step_m=profile.road_seed_sample_step_m,
        profile=profile.name,
        relaxed_accept=profile.road_seed_relaxed_accept,
        min_road_score=profile.road_seed_min_road_score,
        min_corridor_m=profile.road_seed_min_corridor_m,
    )

  if strategy == "spiral_only":
    return _spiral()

  if strategy == "road_seed_then_spiral":
    road_res = _road()
    if road_res is not None and road_res.ok:
      road_res.init_method = "road_seed"
      return road_res
    spiral_res = _spiral()
    if spiral_res.ok:
      return spiral_res
    if road_res is not None and road_res.score > spiral_res.score:
      return road_res
    return spiral_res

  # spiral_then_road_seed (default)
  spiral_res = _spiral()
  if spiral_res.ok:
    return spiral_res
  road_res = _road()
  if road_res is None:
    spiral_res.samples_tried = max(
        spiral_res.samples_tried, spiral_res.tried,
    )
    return spiral_res
  road_res.samples_tried += spiral_res.samples_tried
  if road_res.ok or road_res.score > spiral_res.score:
    road_res.init_method = "spiral+road_seed" if spiral_res.score > -1e8 else "road_seed"
    return road_res
  spiral_res.init_method = "spiral"
  return spiral_res


def apply_init_pose_to_sim(
    client,
    target_name: str,
    result: TargetInitResult,
    *,
    make_pose_fn,
) -> bool:
  """Teleport target in AirSim; returns False on RPC failure."""
  if not result.ok:
    return False
  pose = make_pose_fn(
    float(result.position_ned[0]),
    float(result.position_ned[1]),
    float(result.position_ned[2]),
    float(result.heading_rad),
  )
  try:
    try:
      client.simSetObjectPose(target_name, pose, True)
    except TypeError:
      client.simSetObjectPose(target_name, pose)
    return True
  except Exception:
    return False


__all__ = [
  "TargetInitConfig",
  "TargetInitResult",
  "score_init_pose_ned",
  "find_valid_init_pose",
  "resolve_target_init_pose",
  "apply_init_pose_to_sim",
  "drive_feasibility_distance_m",
  "_heading_from_quaternion",
]

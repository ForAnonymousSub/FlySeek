# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Paper-level comparison visualization for the alley-chase demo.

Reuses the existing trajectory pipeline (no framework rewrite): it consumes the
standard episode artifacts written by ``demo_adversary_chase.run_demo``
(``frames.jsonl``, ``trajectories.json``, ``metadata.json``, ``metrics.json``)
for two runs of the *same* seeded scene —

  * the **FlySeek** run (adaptive predictive FSM tracker), and
  * the **reactive baseline** run (chase-current-pose follower),

and renders publication-quality figures contrasting the two policies.

What is extracted / drawn (all from already-recorded data + the offline PCD
occupancy map, so nothing new is simulated):

  * UAV (drone) trajectory, coloured by line-of-sight state,
  * target-car trajectory,
  * camera view frustum wedges sampled along each trajectory,
  * an **occlusion / track-loss risk field** over the street network whose
    high-risk regions concentrate at intersections, narrow alleys and building
    edges (the geometry the FlySeek policy is designed to anticipate),
  * candidate hiding zones (high-risk street cells + the detected hutong),
  * selected observation viewpoints (the expert-annotated UAV viewpoints from
    ``trajectories.json``),
  * per-frame occlusion risk along the path (real PCD line-of-sight to the
    target's upcoming positions, via ``bench.visibility.VisibilityEvaluator``).

The risk field is a *geometric proxy* (documented, not a learned model): it
combines (a) proximity to building walls, (b) corridor narrowness — buildings
flanking a drivable cell, and (c) road branching — drivable exits around a
cell, which captures the sudden-turn loss risk at intersections.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from flyseek.adapters.pcd_occupancy import PcdOccupancyMap
from flyseek.bench.visibility import VisibilityEvaluator
from flyseek.utils.coords import airsim_ned_to_map


# --------------------------------------------------------------------------- #
# Episode loading                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeData:
    """Parsed artifacts for one episode (one tracking policy)."""

    name: str
    label: str
    out_dir: Path
    times: np.ndarray                 # [N]
    drone_xy: np.ndarray              # [N, 2] NED
    drone_z: np.ndarray               # [N]
    drone_yaw: np.ndarray             # [N]
    target_xy: np.ndarray             # [N, 2] NED
    target_z: np.ndarray              # [N]
    visible: np.ndarray               # [N] bool
    in_frustum: np.ndarray            # [N] bool (None -> False)
    los_clear: np.ndarray             # [N] bool (None -> True)
    distance: np.ndarray              # [N]
    hfov_deg: float
    cam_pitch_deg: float
    metrics: dict[str, Any]
    instruction: str
    expert_viewpoints: np.ndarray     # [M, 2] NED (selected observation points)
    occlusion_risk_path: np.ndarray   # [N] per-frame occlusion risk (may be NaN)
    vis_reason: np.ndarray            # [N] str: ok / out_of_fov / los_blocked /
    #                                   los_blocked_occluder (tree/foliage) / ...


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _pose_xy_z_yaw(pose: Any) -> tuple[float, float, float, float]:
    if isinstance(pose, dict):
        return (float(pose.get("x", 0.0)), float(pose.get("y", 0.0)),
                float(pose.get("z", 0.0)),
                float(pose.get("yaw", pose.get("heading", 0.0)) or 0.0))
    arr = np.asarray(pose, dtype=np.float64).reshape(-1)
    x = float(arr[0]) if arr.size > 0 else 0.0
    y = float(arr[1]) if arr.size > 1 else 0.0
    z = float(arr[2]) if arr.size > 2 else 0.0
    yaw = float(arr[3]) if arr.size > 3 else 0.0
    return x, y, z, yaw


def load_episode(
    out_dir: Path | str,
    *,
    name: str,
    label: str,
) -> EpisodeData | None:
    """Load one episode's recorded artifacts into an :class:`EpisodeData`."""
    out_dir = Path(out_dir)
    frames = _read_jsonl(out_dir / "frames.jsonl")
    if not frames:
        return None

    times, dxy, dz, dyaw, txy, tz = [], [], [], [], [], []
    vis, infr, los, dist, occ, reasons = [], [], [], [], [], []
    for fr in frames:
        dx, dy, dzz, dyw = _pose_xy_z_yaw(fr.get("uav_pose"))
        tx, ty, tzz, _ = _pose_xy_z_yaw(fr.get("target_pose"))
        times.append(float(fr.get("timestamp", len(times))))
        dxy.append([dx, dy]); dz.append(dzz); dyaw.append(dyw)
        txy.append([tx, ty]); tz.append(tzz)
        vis.append(bool(fr.get("target_visible", False)))
        ic = fr.get("in_camera_frustum")
        infr.append(bool(ic) if ic is not None else bool(fr.get("target_visible", False)))
        lc = fr.get("line_of_sight_clear")
        los.append(bool(lc) if lc is not None else True)
        dist.append(float(fr.get("distance_to_target", np.nan)))
        orisk = fr.get("occlusion_risk")
        occ.append(float(orisk) if orisk is not None else np.nan)
        extra = fr.get("extra") or {}
        reasons.append(str(extra.get("vis_reason", "")))

    meta = {}
    meta_path = out_dir / "metadata.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
    cam = meta.get("camera_config", {}) or {}
    hfov = float(cam.get("hfov_deg", 50.0))
    pitch = float(cam.get("pitch_deg", 55.0))
    instruction = str(meta.get("instruction", "") or "")

    metrics = {}
    metrics_path = out_dir / "metrics.json"
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metrics = {}

    expert_vp: list[list[float]] = []
    traj_path = out_dir / "trajectories.json"
    if traj_path.is_file():
        try:
            tj = json.loads(traj_path.read_text(encoding="utf-8"))
            for vp in tj.get("expert_viewpoints", []) or []:
                p = vp.get("position")
                if p is not None and len(p) >= 2:
                    expert_vp.append([float(p[0]), float(p[1])])
        except json.JSONDecodeError:
            pass

    return EpisodeData(
        name=name,
        label=label,
        out_dir=out_dir,
        times=np.asarray(times, dtype=np.float64),
        drone_xy=np.asarray(dxy, dtype=np.float64).reshape(-1, 2),
        drone_z=np.asarray(dz, dtype=np.float64),
        drone_yaw=np.asarray(dyaw, dtype=np.float64),
        target_xy=np.asarray(txy, dtype=np.float64).reshape(-1, 2),
        target_z=np.asarray(tz, dtype=np.float64),
        visible=np.asarray(vis, dtype=bool),
        in_frustum=np.asarray(infr, dtype=bool),
        los_clear=np.asarray(los, dtype=bool),
        distance=np.asarray(dist, dtype=np.float64),
        hfov_deg=hfov,
        cam_pitch_deg=pitch,
        metrics=metrics,
        instruction=instruction,
        expert_viewpoints=np.asarray(expert_vp, dtype=np.float64).reshape(-1, 2),
        occlusion_risk_path=np.asarray(occ, dtype=np.float64),
        vis_reason=np.asarray(reasons, dtype=object),
    )


# --------------------------------------------------------------------------- #
# Occlusion / track-loss risk field                                           #
# --------------------------------------------------------------------------- #
@dataclass
class RiskField:
    x_edges: np.ndarray               # [W+1] NED x cell edges
    y_edges: np.ndarray               # [H+1] NED y cell edges
    extent: tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax)
    risk: np.ndarray                  # [H, W] in [0,1], NaN off-street
    building: np.ndarray              # [H, W] bool building footprint
    drivable: np.ndarray              # [H, W] bool street cell
    edge: np.ndarray                  # [H, W] building-edge proximity term
    alley: np.ndarray                 # [H, W] corridor-narrowness term
    intersection: np.ndarray          # [H, W] road-branching term
    intersection_pts: np.ndarray      # [K, 2] NED intersection markers
    step_m: float


def _ned_grid_masks(
    occupancy: PcdOccupancyMap,
    xs: np.ndarray,
    ys: np.ndarray,
    keep_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(building[H,W], drivable[H,W])`` boolean grids (y-major)."""
    H, W = ys.size, xs.size
    building = np.zeros((H, W), dtype=bool)
    drivable = np.zeros((H, W), dtype=bool)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            ned = np.array([x, y, keep_z], dtype=np.float64)
            mp = airsim_ned_to_map(ned)
            is_b = occupancy.is_building_bev_map(mp)
            building[j, i] = bool(is_b)
            drivable[j, i] = bool(occupancy.is_drivable_ned(ned))
    return building, drivable


def _ring_count(mask: np.ndarray, offsets: list[tuple[int, int]]) -> np.ndarray:
    """For each cell, count how many of the shifted ``offsets`` land on True."""
    acc = np.zeros(mask.shape, dtype=np.float64)
    for dj, di in offsets:
        acc += np.roll(np.roll(mask, dj, axis=0), di, axis=1).astype(np.float64)
    return acc


def compute_occlusion_risk_field(
    occupancy: PcdOccupancyMap,
    *,
    bounds: tuple[float, float, float, float],
    keep_z: float,
    seg_map: Any | None = None,
    step_m: float = 2.0,
    edge_decay_m: float = 9.0,
    alley_probe_m: float = 9.0,
    branch_probe_m: float = 13.0,
    near_building_m: float = 26.0,
    weights: tuple[float, float, float] = (0.78, 0.70, 0.58),
) -> RiskField:
    """Build the geometric occlusion / track-loss risk field over a street region.

    High risk concentrates where the target can break the drone's line of sight
    or escape the frame:

      * **edge** — proximity to building walls (distance transform off buildings),
      * **alley** — drivable cells flanked by buildings on opposite sides
        (narrow corridors / hutongs),
      * **intersection** — drivable cells with many drivable exits (road
        branching) that are near buildings (sudden-turn loss at junctions).
    """
    from scipy import ndimage

    xmin, xmax, ymin, ymax = bounds
    xs = np.arange(xmin, xmax + step_m, step_m, dtype=np.float64)
    ys = np.arange(ymin, ymax + step_m, step_m, dtype=np.float64)
    building, drivable = _ned_grid_masks(occupancy, xs, ys, keep_z)
    H, W = ys.size, xs.size

    # --- edge term: distance (m) from each cell to the nearest building wall.
    if building.any():
        dist_cells = ndimage.distance_transform_edt(~building)
    else:
        dist_cells = np.full((H, W), 1e6, dtype=np.float64)
    dist_m = dist_cells * step_m
    edge = np.clip(1.0 - dist_m / max(edge_decay_m, 1e-6), 0.0, 1.0)
    edge[building] = 0.0

    # --- alley term: opposite-side building flanking at a probe distance.
    p = max(1, int(round(alley_probe_m / step_m)))
    axis_pairs = [
        [(p, 0), (-p, 0)],          # north/south
        [(0, p), (0, -p)],          # east/west
        [(p, p), (-p, -p)],         # diagonal
        [(p, -p), (-p, p)],         # anti-diagonal
    ]
    flank = np.zeros((H, W), dtype=np.float64)
    bf = building.astype(np.float64)
    for a, b in axis_pairs:
        sa = np.roll(np.roll(bf, a[0], axis=0), a[1], axis=1)
        sb = np.roll(np.roll(bf, b[0], axis=0), b[1], axis=1)
        flank += (sa > 0) & (sb > 0)
    alley = np.clip(flank / 2.0, 0.0, 1.0)   # >=2 flanking pairs -> full
    alley[~drivable] = 0.0

    # --- intersection term: count drivable exits on an 8-neighbour ring.
    pb = max(1, int(round(branch_probe_m / step_m)))
    ring = [(pb, 0), (-pb, 0), (0, pb), (0, -pb),
            (pb, pb), (pb, -pb), (-pb, pb), (-pb, -pb)]
    branches = _ring_count(drivable, ring)
    # Near-building gate so open plazas / car parks don't light up.
    nb = max(1, int(round(near_building_m / step_m)))
    near_b = ndimage.uniform_filter(bf, size=2 * nb + 1, mode="nearest") > 1e-3
    inter = np.clip((branches - 3.0) / 4.0, 0.0, 1.0)
    inter = inter * near_b.astype(np.float64)
    inter[~drivable] = 0.0

    w_e, w_a, w_i = weights
    risk = np.clip(np.maximum(w_a * alley, w_e * edge) + w_i * inter, 0.0, 1.0)
    # Show risk only on the street network; buildings/off-road -> NaN.
    risk = np.where(drivable, risk, np.nan)

    # --- discrete intersection markers: peaks of the branching term,
    # greedily de-duplicated so junctions are marked once (not per-cell).
    inter_pts: list[list[float]] = []
    if inter.any():
        local_max = (inter == ndimage.maximum_filter(inter, size=2 * pb + 1))
        cand = np.argwhere(local_max & (inter > 0.5))
        scored = sorted(
            ([float(xs[i]), float(ys[j]), float(inter[j, i])] for j, i in cand),
            key=lambda c: -c[2],
        )
        min_sep = max(branch_probe_m, 12.0)
        for x, y, _s in scored:
            if all((x - px) ** 2 + (y - py) ** 2 >= min_sep ** 2
                   for px, py in inter_pts):
                inter_pts.append([x, y])

    return RiskField(
        x_edges=np.concatenate([xs - step_m / 2.0, [xs[-1] + step_m / 2.0]]),
        y_edges=np.concatenate([ys - step_m / 2.0, [ys[-1] + step_m / 2.0]]),
        extent=(float(xs[0] - step_m / 2.0), float(xs[-1] + step_m / 2.0),
                float(ys[0] - step_m / 2.0), float(ys[-1] + step_m / 2.0)),
        risk=risk,
        building=building,
        drivable=drivable,
        edge=edge,
        alley=alley,
        intersection=inter,
        intersection_pts=np.asarray(inter_pts, dtype=np.float64).reshape(-1, 2),
        step_m=step_m,
    )


# --------------------------------------------------------------------------- #
# Per-frame occlusion risk along the trajectory (real PCD line of sight)      #
# --------------------------------------------------------------------------- #
def compute_path_occlusion_risk(
    ep: EpisodeData,
    occupancy: PcdOccupancyMap | None,
    *,
    horizon_steps: int = 6,
    drone_eye_agl_m: float = 12.0,
) -> np.ndarray:
    """Per-frame occlusion risk = fraction of the target's upcoming positions
    whose line of sight from the current drone pose is blocked (PCD ray-cast)."""
    n = ep.drone_xy.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if occupancy is None:
        return out
    evaluator = VisibilityEvaluator(drone_eye_agl_m=drone_eye_agl_m)
    for i in range(n):
        future = []
        for k in range(1, horizon_steps + 1):
            j = min(n - 1, i + k)
            future.append([ep.target_xy[j, 0], ep.target_xy[j, 1], ep.target_z[j]])
        risk = evaluator.estimate_occlusion_risk(
            np.array([ep.drone_xy[i, 0], ep.drone_xy[i, 1], ep.drone_z[i]]),
            scene_context={
                "occupancy": occupancy,
                "future_target_positions": future,
                "drone_eye_agl_m": drone_eye_agl_m,
            },
        )
        if risk is not None:
            out[i] = float(risk)
    return out


# --------------------------------------------------------------------------- #
# Drawing helpers                                                             #
# --------------------------------------------------------------------------- #
def _tree_occlusion_mask(ep: EpisodeData) -> np.ndarray:
    """Per-frame mask: target lost specifically to tree / foliage occlusion.

    Uses the recorded ``vis_reason == 'los_blocked_occluder'`` tag (emitted when
    ``--los-include-trees`` counts a tall non-building PCD column). Falls back to
    all-False when the episode predates the tag.
    """
    n = ep.drone_xy.shape[0]
    if ep.vis_reason is None or ep.vis_reason.size != n:
        return np.zeros(n, dtype=bool)
    return np.array([str(r) == "los_blocked_occluder" for r in ep.vis_reason],
                    dtype=bool)


def _frustum_polygon(
    x: float, y: float, yaw: float, hfov_deg: float, length: float,
) -> np.ndarray:
    half = math.radians(hfov_deg) / 2.0
    a0, a1 = yaw - half, yaw + half
    return np.array([
        [x, y],
        [x + length * math.cos(a0), y + length * math.sin(a0)],
        [x + length * math.cos(a1), y + length * math.sin(a1)],
    ], dtype=np.float64)


def _draw_scene(
    ax,
    ep: EpisodeData,
    rf: RiskField,
    *,
    title: str,
    show_risk: bool = True,
    show_expert: bool = True,
    frustum_every: int = 0,
    cmap: str = "inferno",
):
    """Draw one policy's BEV panel over the shared risk field."""
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.lines import Line2D

    xmin, xmax, ymin, ymax = rf.extent

    # Building footprints (light raster backdrop).
    bld = np.where(rf.building, 1.0, np.nan)
    ax.imshow(bld, extent=rf.extent, origin="lower", cmap="Greys",
              vmin=0.0, vmax=1.4, alpha=0.55, interpolation="nearest", zorder=1)

    # Occlusion / track-loss risk heatmap (street network only).
    if show_risk:
        im = ax.imshow(rf.risk, extent=rf.extent, origin="lower", cmap=cmap,
                       vmin=0.0, vmax=1.0, alpha=0.82, interpolation="bilinear",
                       zorder=2)
    else:
        im = None

    # Candidate hiding zones: outline only the strongest-risk pockets so the
    # panel stays readable on dense street grids.
    try:
        risk_filled = np.nan_to_num(rf.risk, nan=0.0)
        ax.contour(
            np.linspace(xmin, xmax, rf.risk.shape[1]),
            np.linspace(ymin, ymax, rf.risk.shape[0]),
            risk_filled, levels=[0.8], colors="#19f0c8", linewidths=1.0,
            alpha=0.7, zorder=3,
        )
    except Exception:
        pass

    # Intersection markers.
    if rf.intersection_pts.size:
        ax.scatter(rf.intersection_pts[:, 0], rf.intersection_pts[:, 1],
                   marker="P", s=60, facecolor="none", edgecolor="#ffe066",
                   linewidths=1.4, zorder=6, label="intersection (high-risk)")

    # Target-car trajectory.
    ax.plot(ep.target_xy[:, 0], ep.target_xy[:, 1], "-", color="#ff5c8a",
            lw=2.4, zorder=7, label="target car")
    ax.scatter(ep.target_xy[0, 0], ep.target_xy[0, 1], marker="o", s=45,
               color="#ff5c8a", edgecolor="white", zorder=8)
    ax.scatter(ep.target_xy[-1, 0], ep.target_xy[-1, 1], marker="X", s=70,
               color="#ff5c8a", edgecolor="white", zorder=8)

    # Drone trajectory coloured by LoS state (clear=cyan, blocked=red).
    los = ep.los_clear & ep.visible
    dxy = ep.drone_xy
    ax.plot(dxy[:, 0], dxy[:, 1], "-", color="#7fd4ff", lw=1.6, alpha=0.5,
            zorder=7)
    ax.scatter(dxy[los, 0], dxy[los, 1], s=9, color="#39ff88", zorder=8,
               label="drone (LoS held)")
    lost = ~los
    # Separate tree / foliage occlusion losses from other losses so the figure
    # can call out "lost behind a tree".
    tree = _tree_occlusion_mask(ep)
    lost_tree = lost & tree
    lost_other = lost & ~tree
    if lost_other.any():
        ax.scatter(dxy[lost_other, 0], dxy[lost_other, 1], s=16, color="#ff3b3b",
                   marker="x", zorder=9, label="drone (target lost)")
    if lost_tree.any():
        ax.scatter(dxy[lost_tree, 0], dxy[lost_tree, 1], s=42, color="#ffb000",
                   marker="*", edgecolor="#7a3b00", linewidths=0.5, zorder=11,
                   label="lost behind tree/foliage")
    ax.scatter(dxy[0, 0], dxy[0, 1], marker="^", s=55, color="#7fd4ff",
               edgecolor="white", zorder=10)

    # Camera frustum wedges sampled along the path.
    if frustum_every and frustum_every > 0:
        n = dxy.shape[0]
        for i in range(0, n, frustum_every):
            length = float(ep.distance[i]) if np.isfinite(ep.distance[i]) else 14.0
            length = float(np.clip(length, 6.0, 28.0))
            poly = _frustum_polygon(dxy[i, 0], dxy[i, 1], ep.drone_yaw[i],
                                    ep.hfov_deg, length)
            held = bool(los[i])
            ax.add_patch(MplPolygon(
                poly, closed=True,
                facecolor=("#39ff88" if held else "#ff3b3b"),
                edgecolor=("#39ff88" if held else "#ff3b3b"),
                alpha=0.12 if held else 0.16, lw=0.8, zorder=5,
            ))

    # Selected observation viewpoints (expert annotation).
    if show_expert and ep.expert_viewpoints.size:
        vp = ep.expert_viewpoints
        stride = max(1, vp.shape[0] // 24)
        ax.scatter(vp[::stride, 0], vp[::stride, 1], marker="*", s=55,
                   facecolor="#ffd54a", edgecolor="#7a5a00", linewidths=0.5,
                   zorder=8, label="selected observation pt")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("NED x (m)")
    ax.set_ylabel("NED y (m)")
    ax.tick_params(labelsize=8)
    return im


def _metrics_caption(ep: EpisodeData) -> str:
    m = ep.metrics or {}
    vis = ep.visible.mean() if ep.visible.size else 0.0
    success = m.get("tracking_success")
    vr = m.get("target_visibility_ratio", round(float(vis), 3))
    los_c = m.get("line_of_sight_continuity", "-")
    lost = m.get("target_lost_ratio", round(float(1.0 - vis), 3))
    reacq = m.get("re_acquisition_events", "-")
    tag = ("SUCCESS" if success else "FAIL") if success is not None else "-"
    tree = _tree_occlusion_mask(ep)
    tree_line = ""
    if tree.any():
        tree_line = f"\nlost behind tree/foliage: {int(tree.sum())} frames"
    return (f"{ep.label}  [{tag}]\n"
            f"visibility={vr}  LoS-continuity={los_c}\n"
            f"lost-ratio={lost}  re-acq events={reacq}{tree_line}")


# --------------------------------------------------------------------------- #
# Top-level figure builders                                                   #
# --------------------------------------------------------------------------- #
def render_occlusion_risk_map(
    rf: RiskField,
    out_path: Path | str,
    *,
    target_xy: np.ndarray | None = None,
    alley_markers: list[tuple[float, float, str]] | None = None,
    title: str = "Occlusion / track-loss risk map",
) -> Path:
    """Standalone annotated occlusion-risk heatmap (with component panels)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax_main = fig.add_subplot(gs[:, :2])
    ax_e = fig.add_subplot(gs[0, 2])
    ax_a = fig.add_subplot(gs[1, 2])

    bld = np.where(rf.building, 1.0, np.nan)
    ax_main.imshow(bld, extent=rf.extent, origin="lower", cmap="Greys",
                   vmin=0.0, vmax=1.4, alpha=0.6, interpolation="nearest")
    im = ax_main.imshow(rf.risk, extent=rf.extent, origin="lower",
                        cmap="inferno", vmin=0.0, vmax=1.0, alpha=0.85,
                        interpolation="bilinear")
    if rf.intersection_pts.size:
        ax_main.scatter(rf.intersection_pts[:, 0], rf.intersection_pts[:, 1],
                        marker="P", s=80, facecolor="none", edgecolor="#ffe066",
                        linewidths=1.6, label="intersection")
    if target_xy is not None and target_xy.size:
        ax_main.plot(target_xy[:, 0], target_xy[:, 1], "-", color="#5cffd6",
                     lw=2.0, label="target route")
    if alley_markers:
        for x, y, txt in alley_markers:
            ax_main.scatter([x], [y], marker="s", s=70, facecolor="none",
                            edgecolor="#19f0c8", linewidths=1.8)
            ax_main.annotate(txt, (x, y), color="#19f0c8", fontsize=9,
                             xytext=(6, 6), textcoords="offset points")
    ax_main.set_aspect("equal")
    ax_main.set_title(title, fontsize=13, fontweight="bold")
    ax_main.set_xlabel("NED x (m)"); ax_main.set_ylabel("NED y (m)")
    ax_main.legend(loc="upper right", fontsize=8, framealpha=0.85)
    fig.colorbar(im, ax=ax_main, shrink=0.7, label="occlusion / loss risk")

    edge_show = np.where(rf.drivable, rf.edge, np.nan)
    alley_show = np.where(rf.drivable, np.maximum(rf.alley, rf.intersection), np.nan)
    for ax, data, ttl in (
        (ax_e, edge_show, "building-edge proximity"),
        (ax_a, alley_show, "alley + intersection"),
    ):
        ax.imshow(bld, extent=rf.extent, origin="lower", cmap="Greys",
                  vmin=0.0, vmax=1.4, alpha=0.6, interpolation="nearest")
        ax.imshow(data, extent=rf.extent, origin="lower", cmap="inferno",
                  vmin=0.0, vmax=1.0, alpha=0.85, interpolation="bilinear")
        ax.set_aspect("equal"); ax.set_title(ttl, fontsize=10)
        ax.tick_params(labelsize=7)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_comparison_figure(
    flyseek: EpisodeData,
    baseline: EpisodeData,
    rf: RiskField,
    out_path: Path | str,
    *,
    frustum_every: int = 18,
) -> Path:
    """Two-panel BEV comparison + a visibility / occlusion-risk timeline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[3.0, 0.9, 0.9])
    ax_f = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_t1 = fig.add_subplot(gs[1, :])
    ax_t2 = fig.add_subplot(gs[2, :], sharex=ax_t1)

    im = _draw_scene(ax_f, flyseek, rf,
                     title="FlySeek (adaptive predictive policy)",
                     frustum_every=frustum_every)
    _draw_scene(ax_b, baseline, rf,
                title="Reactive baseline (chase current pose)",
                frustum_every=frustum_every)

    ax_f.legend(loc="upper left", fontsize=7.5, framealpha=0.85, ncol=2)
    ax_f.text(0.02, -0.13, _metrics_caption(flyseek), transform=ax_f.transAxes,
              fontsize=9, va="top", family="monospace",
              bbox=dict(boxstyle="round", fc="#102a18", ec="#39ff88", alpha=0.6))
    ax_b.text(0.02, -0.13, _metrics_caption(baseline), transform=ax_b.transAxes,
              fontsize=9, va="top", family="monospace",
              bbox=dict(boxstyle="round", fc="#2a1010", ec="#ff3b3b", alpha=0.6))
    if im is not None:
        fig.colorbar(im, ax=[ax_f, ax_b], shrink=0.6,
                     label="occlusion / track-loss risk")

    # --- timeline 1: target-visible state for both policies.
    tree_marked = False
    for ep, color, yv in ((flyseek, "#39ff88", 1.0), (baseline, "#ff3b3b", 0.0)):
        t = ep.times
        v = ep.visible.astype(float)
        ax_t1.fill_between(t, yv, yv + 0.8 * v, step="pre", color=color,
                           alpha=0.55, label=f"{ep.label} visible")
        tree = _tree_occlusion_mask(ep)
        if tree.any():
            ax_t1.scatter(t[tree], np.full(int(tree.sum()), yv + 0.1),
                          marker="*", s=36, color="#ffb000",
                          edgecolor="#7a3b00", linewidths=0.4, zorder=5,
                          label=("tree/foliage occlusion" if not tree_marked
                                 else None))
            tree_marked = True
    ax_t1.set_yticks([0.4, 1.4])
    ax_t1.set_yticklabels(["baseline", "FlySeek"], fontsize=8)
    ax_t1.set_ylabel("target in view")
    ax_t1.set_title("Target visibility over time (filled = visible)", fontsize=11)
    ax_t1.legend(loc="upper right", fontsize=8, ncol=2)
    ax_t1.grid(True, alpha=0.2)

    # --- timeline 2: per-frame occlusion risk along each path.
    for ep, color in ((flyseek, "#39ff88"), (baseline, "#ff3b3b")):
        r = ep.occlusion_risk_path
        if np.isfinite(r).any():
            ax_t2.plot(ep.times, r, "-", color=color, lw=1.8,
                       label=f"{ep.label}")
    ax_t2.set_ylim(-0.02, 1.02)
    ax_t2.set_ylabel("occlusion risk")
    ax_t2.set_xlabel("time (s)")
    ax_t2.set_title("Per-frame occlusion risk (PCD line-of-sight to upcoming "
                    "target positions)", fontsize=11)
    ax_t2.legend(loc="upper right", fontsize=8)
    ax_t2.grid(True, alpha=0.2)

    fig.suptitle("Active viewpoint adjustment vs. reactive baseline — "
                 "alley-chase (env_airsim_16)", fontsize=15, fontweight="bold")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def region_bounds_from_episodes(
    episodes: list[EpisodeData],
    *,
    margin_m: float = 35.0,
) -> tuple[float, float, float, float]:
    """Bounding box (NED x/y) over all trajectories + a margin."""
    xs, ys = [], []
    for ep in episodes:
        if ep.drone_xy.size:
            xs.append(ep.drone_xy[:, 0]); ys.append(ep.drone_xy[:, 1])
        if ep.target_xy.size:
            xs.append(ep.target_xy[:, 0]); ys.append(ep.target_xy[:, 1])
    if not xs:
        return (-50.0, 50.0, -50.0, 50.0)
    xcat = np.concatenate(xs); ycat = np.concatenate(ys)
    return (float(xcat.min() - margin_m), float(xcat.max() + margin_m),
            float(ycat.min() - margin_m), float(ycat.max() + margin_m))


__all__ = [
    "EpisodeData",
    "RiskField",
    "load_episode",
    "compute_occlusion_risk_field",
    "compute_path_occlusion_risk",
    "render_occlusion_risk_map",
    "render_comparison_figure",
    "region_bounds_from_episodes",
]

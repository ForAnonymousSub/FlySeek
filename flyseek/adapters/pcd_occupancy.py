# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Offline PCD occupancy map — mirrors OpenFly traj_gen collision checks.

Loads ``scene_data/pcd_map/<env>.pcd``, voxelises at ``VoxelWidth`` resolution,
dilates by ``DilateRadius``, and exposes:

* ``is_bev_occupied`` — 2-D building footprint (for ground vehicles)
* ``is_3d_occupied``  — full 3-D voxel query (for drone body)
* ``resolve_bev_move`` — slide / shorten step so cars don't enter buildings
* ``is_drivable_ned`` — BEV-free cells with PCD ground support (not water/roof)
* ``snap_car_to_ground_ned`` — lock vehicle Z to local ground surface
* ``resolve_drone_ned`` — lift drone above local roofline + clearance

This is the Python-side equivalent of ``TrajGen::globalMapBulid()`` +
``bevMapBulid()`` + ``VoxelMap::query()``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flyseek.utils.coords import airsim_ned_to_map, map_to_airsim_ned


@dataclass(frozen=True)
class OccupancyConfig:
    voxel_width: float = 1.5
    dilate_radius: float = 3.0
    # Multiplier applied to raw PCD coordinates so they land in the same
    # "unified" metric frame the MapBound/voxel grid assumes. OpenFly's
    # ``pcd_scale_ratio`` (env_airsim_*: 1, env_ue_*: 100). The UE City Sample
    # PCDs are stored at ~1/100 scale (raw range ≈ ±12), so without this the
    # whole map collapses into a handful of voxels.
    coord_scale: float = 1.0
    map_bound: tuple[float, float, float, float, float, float] = (
        -1300.0, 1000.0, -600.0, 1000.0, -200.0, 200.0,
    )
    map_elevation: float = 0.0
    min_height_thresh: float = 6.0
    min_drone_clearance: float = 8.0
    min_ground_points_per_cell: int = 2
    car_agl_m: float = 0.35
    # --- ground-surface detection (added v3) --------------------------------
    # The OpenFly env PCDs store the drivable road surface slightly BELOW
    # ``map_elevation`` (env_airsim_16 roads sit at z≈-0.6 m), while a separate
    # deep plane (≈-11 m, surrounding void/water) must stay non-drivable. The
    # old ``[map_elevation, +6m)`` band missed the real road entirely (it has
    # no points ≥0) and instead snapped cars onto roadside curbs/medians. We
    # now build the ground band as ``[map_elevation - ground_below_m, +6m)``
    # and take the per-column LOW as the road surface.
    ground_below_m: float = 2.0
    # Points this far above the local road surface (and below the building
    # threshold) are treated as a CAR obstacle: curbs, guardrails (护栏),
    # medians, low walls, parked vehicles. Cars cannot drive through them; the
    # drone (3-D + roof clearance) is unaffected.
    car_obstacle_min_h_m: float = 0.6
    car_obstacle_min_points: int = 2
    # Curbs/rails are often only 1 voxel wide between two open lanes — do NOT
    # dilate the car-obstacle layer or we close the narrow roads themselves.
    car_obstacle_dilate_radius: float = 0.0
    # Thickness of the road-surface stratum used for ground-support counting.
    ground_surface_band_m: float = 0.8

    @classmethod
    def from_yaml_section(cls, traj_map: dict) -> "OccupancyConfig":
        mb = traj_map.get("MapBound", [-1300, 1000, -600, 1000, -200, 200])
        return cls(
            voxel_width=float(traj_map.get("VoxelWidth", 1.5)),
            dilate_radius=float(traj_map.get("DilateRadius", 3.0)),
            coord_scale=float(traj_map.get("pcd_scale_ratio", 1.0)),
            map_bound=tuple(float(v) for v in mb),  # type: ignore[arg-type]
            map_elevation=float(traj_map.get("map_elevation", 0.0)),
            min_height_thresh=float(traj_map.get("min_height_thresh", 6.0)),
            min_drone_clearance=float(traj_map.get("min_height_thresh", 6.0)) + 2.0,
        )


def _lzf_decompress(data: bytes, expected: int) -> bytearray:
    """Pure-Python LZF (liblzf) decompressor — no external deps.

    PCD ``binary_compressed`` stores ``[u32 compressed][u32 uncompressed][LZF]``.
    Slow on huge clouds (env_ue_bigcity ≈ 2.5 GB uncompressed) but runs only on
    the FIRST occupancy build; the result is cached as an .npz afterwards.
    """
    out = bytearray(expected)
    op = 0
    ip = 0
    n = len(data)
    while ip < n:
        ctrl = data[ip]
        ip += 1
        if ctrl < 32:                       # literal run of ctrl+1 bytes
            run = ctrl + 1
            out[op:op + run] = data[ip:ip + run]
            ip += run
            op += run
        else:                               # back-reference
            length = ctrl >> 5
            if length == 7:
                length += data[ip]
                ip += 1
            ref = op - ((ctrl & 0x1f) << 8) - data[ip] - 1
            ip += 1
            length += 2
            if ref < 0:
                raise ValueError("corrupt LZF stream (negative back-ref)")
            dist = op - ref
            if length <= dist:              # non-overlapping → fast slice copy
                out[op:op + length] = out[ref:ref + length]
            else:                           # overlapping → byte-wise
                for i in range(length):
                    out[op + i] = out[ref + i]
            op += length
    return out[:op] if op != expected else out


def _parse_pcd_binary_xyz(path: Path) -> np.ndarray:
    """Binary PCD reader extracting x/y/z from arbitrary FIELDS layouts.

    Supports ``DATA binary`` (interleaved AoS — env_airsim_*, env_ue_smallcity)
    and ``DATA binary_compressed`` (LZF + per-field SoA — env_ue_bigcity). x/y/z
    must be 4-byte floats (holds for every OpenFly PCD).
    """
    header_lines: list[str] = []
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"truncated PCD header: {path}")
            text = line.decode("ascii", errors="replace").strip()
            header_lines.append(text)
            if text.upper().startswith("DATA"):
                break
        data_tag = header_lines[-1].split()[1].lower()
        if data_tag not in ("binary", "binary_compressed"):
            raise ValueError(f"unsupported PCD DATA {data_tag}")

        hdr: dict[str, list[str]] = {}
        n_points = 0
        for ln in header_lines:
            parts = ln.split()
            if not parts:
                continue
            key = parts[0].upper()
            if key in ("FIELDS", "SIZE", "TYPE", "COUNT"):
                hdr[key] = parts[1:]
            elif key == "POINTS":
                n_points = int(parts[1])

        fields = hdr.get("FIELDS", [])
        sizes = [int(s) for s in hdr.get("SIZE", [])]
        types = [t.upper() for t in hdr.get("TYPE", ["F"] * len(fields))]
        counts = [int(c) for c in hdr.get("COUNT", ["1"] * len(fields))]
        if not fields or len(sizes) != len(fields):
            raise ValueError(f"bad PCD header FIELDS/SIZE: {path}")
        if n_points <= 0:
            raise ValueError("PCD POINTS count missing")
        for axis in ("x", "y", "z"):
            i = fields.index(axis) if axis in fields else -1
            if i < 0 or types[i] != "F" or sizes[i] != 4:
                raise ValueError(
                    f"PCD x/y/z must be float32; got FIELDS={fields} "
                    f"SIZE={sizes} TYPE={types}")
        widths = [sizes[i] * counts[i] for i in range(len(fields))]

        if data_tag == "binary":
            point_step = sum(widths)
            off = {name: o for name, o in zip(
                fields, np.cumsum([0] + widths[:-1]))}
            raw = f.read(n_points * point_step)
            if len(raw) != n_points * point_step:
                raise ValueError("PCD data size mismatch (binary)")
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(n_points, point_step)
            pts = np.stack([
                arr[:, off[a]:off[a] + 4].copy().view("<f4").reshape(-1)
                for a in ("x", "y", "z")], axis=1)
            return np.asarray(pts, dtype=np.float64)

        # ---- binary_compressed: [u32 comp][u32 uncomp][LZF], SoA layout ----
        import struct
        comp_size, uncomp_size = struct.unpack("<II", f.read(8))
        comp = f.read(comp_size)
        if len(comp) != comp_size:
            raise ValueError("PCD compressed block truncated")
        print(f"[occupancy] LZF-decompressing {comp_size/1e6:.0f}MB → "
              f"{uncomp_size/1e6:.0f}MB (one-time; slow for big clouds)...",
              flush=True)
        buf = _lzf_decompress(comp, uncomp_size)
        # SoA: each field occupies n_points*width contiguous bytes, in FIELDS order.
        soa_off: dict[str, int] = {}
        acc = 0
        for name, w in zip(fields, widths):
            soa_off[name] = acc
            acc += n_points * w
        cols = [np.frombuffer(buf, dtype="<f4", count=n_points,
                              offset=soa_off[a]) for a in ("x", "y", "z")]
        return np.asarray(np.stack(cols, axis=1), dtype=np.float64)


def _voxel_indices(points: np.ndarray, cfg: OccupancyConfig) -> np.ndarray:
    x0, x1, y0, y1, z0, z1 = cfg.map_bound
    vw = cfg.voxel_width
    ix = np.floor((points[:, 0] - x0) / vw).astype(np.int32)
    iy = np.floor((points[:, 1] - y0) / vw).astype(np.int32)
    iz = np.floor((points[:, 2] - z0) / vw).astype(np.int32)
    nx = int(np.floor((x1 - x0) / vw)) + 1
    ny = int(np.floor((y1 - y0) / vw)) + 1
    nz = int(np.floor((z1 - z0) / vw)) + 1
    valid = (
        (ix >= 0) & (ix < nx) &
        (iy >= 0) & (iy < ny) &
        (iz >= 0) & (iz < nz) &
        np.isfinite(points).all(axis=1)
    )
    return np.stack([ix[valid], iy[valid], iz[valid]], axis=1)


def _dilate_voxels(voxels: set[tuple[int, int, int]], radius_cells: int) -> set[tuple[int, int, int]]:
    if radius_cells <= 0:
        return set(voxels)
    if not voxels:
        return set()
    r = radius_cells
    offsets = np.array([
        (dx, dy, dz)
        for dx in range(-r, r + 1)
        for dy in range(-r, r + 1)
        for dz in range(-r, r + 1)
        if dx * dx + dy * dy + dz * dz <= r * r
    ], dtype=np.int64)
    base = np.fromiter((v for tup in voxels for v in tup), dtype=np.int64,
                       count=len(voxels) * 3).reshape(-1, 3)
    # Vectorised dilation in memory-bounded chunks (UE maps have millions of
    # occupied voxels; the old per-voxel Python loop was O(M*K) and hung).
    out: set[tuple[int, int, int]] = set()
    chunk = 200_000
    for s in range(0, base.shape[0], chunk):
        blk = base[s:s + chunk]
        grown = (blk[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
        grown = np.unique(grown, axis=0)
        out.update(map(tuple, grown.tolist()))
    return out


class PcdOccupancyMap:
    """Voxelised scene occupancy built from a PCD file."""

    def __init__(self, cfg: OccupancyConfig,
                 occ3d: set[tuple[int, int, int]],
                 bev2d: set[tuple[int, int]],
                 roof_z: dict[tuple[int, int], float],
                 ground_z: dict[tuple[int, int], float] | None = None,
                 ground_count: dict[tuple[int, int], int] | None = None,
                 car_obs2d: set[tuple[int, int]] | None = None,
                 *,
                 ground_enabled: bool = False) -> None:
        self.cfg = cfg
        self._occ3d = occ3d
        self._bev2d = bev2d
        self._roof_z = roof_z
        self._ground_z = ground_z or {}
        self._ground_count = ground_count or {}
        # Low (sub-building) obstacles a CAR cannot cross: curbs, guardrails,
        # medians, low walls, parked cars. The drone path ignores this layer.
        self._car_obs2d = car_obs2d or set()
        # Precomputed union used by every car-facing query (buildings + low
        # obstacles). Buildings alone (``_bev2d``) remain available for the
        # drone roof/3-D clearance path.
        self._car_blocked2d = self._bev2d | self._car_obs2d
        self._ground_enabled = bool(ground_enabled)
        self._x0, self._x1, self._y0, self._y1, self._z0, self._z1 = cfg.map_bound
        self._vw = cfg.voxel_width
        self._dilate_cells = max(0, int(np.ceil(cfg.dilate_radius / cfg.voxel_width)))

    @classmethod
    def from_pcd(cls, pcd_path: Path, cfg: OccupancyConfig | None = None) -> "PcdOccupancyMap":
        cfg = cfg or OccupancyConfig()
        print(f"[occupancy] loading PCD: {pcd_path}")
        pts = _parse_pcd_binary_xyz(pcd_path)
        if float(cfg.coord_scale) != 1.0:
            pts = pts * float(cfg.coord_scale)
            print(f"[occupancy] applied pcd_scale_ratio={cfg.coord_scale} "
                  f"→ coords now in unified metric frame")

        # Auto-expand the XY MapBound to enclose the (scaled) cloud — env_ue_bigcity
        # spans well beyond its yaml ±2000 bound (city is ~6 km across). The
        # expanded bound is persisted in the cache, so queries stay consistent.
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        x0, x1, y0, y1, z0, z1 = cfg.map_bound
        margin = 5.0 * cfg.voxel_width
        if mn[0] < x0 or mx[0] > x1 or mn[1] < y0 or mx[1] > y1:
            from dataclasses import replace
            nb = (min(x0, float(mn[0]) - margin), max(x1, float(mx[0]) + margin),
                  min(y0, float(mn[1]) - margin), max(y1, float(mx[1]) + margin),
                  z0, z1)
            cfg = replace(cfg, map_bound=tuple(float(v) for v in nb))
            print(f"[occupancy] expanded MapBound → "
                  f"{tuple(round(v) for v in cfg.map_bound)} to enclose cloud")

        # Lightly subsample huge clouds (voxelisation at 1.5-2 m doesn't need
        # 150M points; keeps parse/voxelise arrays manageable). stride≤2 keeps
        # per-cell ground-support counts well above the drivability threshold.
        max_points = 80_000_000
        if pts.shape[0] > max_points:
            stride = int(np.ceil(pts.shape[0] / max_points))
            pts = pts[::stride]
            print(f"[occupancy] subsampled to {pts.shape[0]:,} points "
                  f"(stride {stride})")

        print(f"[occupancy] {pts.shape[0]:,} points → voxelising "
              f"(voxel={cfg.voxel_width}m, dilate={cfg.dilate_radius}m)")

        vox = _voxel_indices(pts, cfg)
        occ3d_raw = set(map(tuple, np.unique(vox, axis=0)))

        x0, _, y0, _, _, _ = cfg.map_bound
        vw = cfg.voxel_width
        nx = int(np.floor((cfg.map_bound[1] - x0) / vw)) + 1
        ny = int(np.floor((cfg.map_bound[3] - y0) / vw)) + 1
        z_all = pts[:, 2]
        ix_all = np.floor((pts[:, 0] - x0) / vw).astype(np.int64)
        iy_all = np.floor((pts[:, 1] - y0) / vw).astype(np.int64)
        in_bounds = (
            (ix_all >= 0) & (ix_all < nx) &
            (iy_all >= 0) & (iy_all < ny) &
            np.isfinite(z_all)
        )
        flat_all = ix_all * ny + iy_all

        # ---- roof (max Z over ALL in-bounds points) — drone clearance ------
        # Store for EVERY populated column (not just those above map_elevation)
        # so an open road cell reports roof == surface (≈ground), keeping the
        # init "rail/curb" clearance check from firing on flat lanes whose
        # surface sits slightly below map_elevation.
        roof_z: dict[tuple[int, int], float] = {}
        flat_ib = flat_all[in_bounds]
        roof_flat = np.full(nx * ny, -np.inf, dtype=np.float64)
        np.maximum.at(roof_flat, flat_ib, z_all[in_bounds])
        for idx in np.where(np.isfinite(roof_flat))[0]:
            roof_z[(int(idx // ny), int(idx % ny))] = float(roof_flat[idx])

        # ---- per-column road SURFACE (low Z within the ground band) --------
        # Band reaches ``ground_below_m`` BELOW map_elevation so the slightly
        # negative road surface is captured, while the deep void plane
        # (well below the band) is excluded → stays non-drivable.
        g_lo = cfg.map_elevation - cfg.ground_below_m
        g_hi = cfg.map_elevation + cfg.min_height_thresh  # building threshold
        band = in_bounds & (z_all >= g_lo) & (z_all < g_hi)
        surf_flat = np.full(nx * ny, np.inf, dtype=np.float64)
        if np.any(band):
            np.minimum.at(surf_flat, flat_all[band], z_all[band])
        has_surf = np.isfinite(surf_flat)

        # Per-point local surface (inf where the column has no road band).
        psurf = np.where(has_surf[flat_all], surf_flat[flat_all], np.inf)
        # Ground-support stratum: points within a thin slab above the surface.
        support_mask = (
            in_bounds & (z_all >= psurf)
            & (z_all <= psurf + cfg.ground_surface_band_m)
        )
        # Car-obstacle stratum: vertical structure standing between the road
        # surface and the building threshold (curbs / guardrails / medians).
        obstacle_mask = (
            in_bounds
            & (z_all >= psurf + cfg.car_obstacle_min_h_m)
            & (z_all < g_hi)
        )
        gc_flat = np.zeros(nx * ny, dtype=np.int32)
        oc_flat = np.zeros(nx * ny, dtype=np.int32)
        np.add.at(gc_flat, flat_all[support_mask], 1)
        np.add.at(oc_flat, flat_all[obstacle_mask], 1)

        ground_z: dict[tuple[int, int], float] = {}
        ground_count: dict[tuple[int, int], int] = {}
        for idx in np.where(gc_flat >= cfg.min_ground_points_per_cell)[0]:
            z_val = float(surf_flat[idx])
            if np.isfinite(z_val):
                ground_z[(int(idx // ny), int(idx % ny))] = z_val
                ground_count[(int(idx // ny), int(idx % ny))] = int(gc_flat[idx])

        # ---- building footprint (≥ min_height_thresh ABOVE map_elevation) --
        bev_mask = in_bounds & (z_all >= g_hi)
        bev2d_raw = set(
            (int(a), int(b))
            for a, b in np.unique(
                np.stack([ix_all[bev_mask], iy_all[bev_mask]], axis=1), axis=0
            )
        ) if np.any(bev_mask) else set()

        # ---- car-obstacle footprint ---------------------------------------
        car_obs_raw = set(
            (int(idx // ny), int(idx % ny))
            for idx in np.where(oc_flat >= cfg.car_obstacle_min_points)[0]
        )

        dilate = max(0, int(np.ceil(cfg.dilate_radius / cfg.voxel_width)))
        occ3d = _dilate_voxels(occ3d_raw, dilate)
        bev2d = cls._dilate_bev(bev2d_raw, dilate)
        car_dilate = max(
            0, int(np.ceil(cfg.car_obstacle_dilate_radius / cfg.voxel_width))
        )
        car_obs2d = cls._dilate_bev(car_obs_raw, car_dilate)

        print(f"[occupancy] 3-D voxels: {len(occ3d_raw):,} raw → {len(occ3d):,} dilated")
        print(f"[occupancy] building cells: {len(bev2d_raw):,} raw → {len(bev2d):,} dilated")
        print(f"[occupancy] car-obstacle cells (curb/rail/median): "
              f"{len(car_obs_raw):,} raw → {len(car_obs2d):,} dilated")
        print(f"[occupancy] ground cells: {len(ground_z):,} "
              f"(band [{g_lo:.1f},{g_hi:.1f}) m; deep void = no support)")
        return cls(cfg, occ3d, bev2d, roof_z, ground_z, ground_count, car_obs2d,
                   ground_enabled=True)

    @staticmethod
    def _dilate_bev(cells: set[tuple[int, int]], dilate: int) -> set[tuple[int, int]]:
        if dilate <= 0:
            return set(cells)
        out: set[tuple[int, int]] = set()
        for ix, iy in cells:
            for dx in range(-dilate, dilate + 1):
                for dy in range(-dilate, dilate + 1):
                    if dx * dx + dy * dy <= dilate * dilate:
                        out.add((ix + dx, iy + dy))
        return out

    @classmethod
    def from_env(cls, repo_root: Path, env_name: str = "env_airsim_16",
                 cfg: OccupancyConfig | None = None) -> "PcdOccupancyMap":
        pcd = repo_root / "scene_data" / "pcd_map" / f"{env_name}.pcd"
        if not pcd.exists():
            raise FileNotFoundError(f"PCD not found: {pcd}")
        return cls.from_pcd(pcd, cfg)

    @property
    def has_ground_layer(self) -> bool:
        """True when cache/PCD build included ground-support grids."""
        return self._ground_enabled

    def _sanitize_ned(self, pos_ned: np.ndarray) -> np.ndarray:
        p = np.asarray(pos_ned, dtype=np.float64).reshape(3).copy()
        if not np.isfinite(p).all():
            p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        return p

    def _map_to_voxel(self, pos_map: np.ndarray) -> tuple[int, int, int]:
        p = np.asarray(pos_map, dtype=np.float64).reshape(3)
        if not np.isfinite(p).all():
            p = np.nan_to_num(p, nan=self.cfg.map_elevation,
                              posinf=self.cfg.map_elevation,
                              neginf=self.cfg.map_elevation)
        ix = int(np.floor((p[0] - self._x0) / self._vw))
        iy = int(np.floor((p[1] - self._y0) / self._vw))
        iz = int(np.floor((p[2] - self._z0) / self._vw))
        return ix, iy, iz

    def is_3d_occupied_map(self, pos_map: np.ndarray) -> bool:
        return self._map_to_voxel(pos_map) in self._occ3d

    def is_bev_occupied_map(self, pos_map: np.ndarray) -> bool:
        """Car-facing BEV obstacle: buildings ∪ low obstacles (curb/rail).

        Used by every ground-vehicle query (drivability, move resolution,
        street motion). The drone path uses 3-D occupancy + roof clearance and
        is unaffected — see ``is_building_bev_map`` for building-only lookups.
        """
        ix, iy, _ = self._map_to_voxel(pos_map)
        return (ix, iy) in self._car_blocked2d

    def is_building_bev_map(self, pos_map: np.ndarray) -> bool:
        """Building footprint only (≥ min_height_thresh tall), no low obstacles."""
        ix, iy, _ = self._map_to_voxel(pos_map)
        return (ix, iy) in self._bev2d

    def is_car_obstacle_map(self, pos_map: np.ndarray) -> bool:
        """Low (sub-building) car obstacle: curb / guardrail / median / wall."""
        ix, iy, _ = self._map_to_voxel(pos_map)
        return (ix, iy) in self._car_obs2d

    def is_bev_occupied_ned(self, pos_ned: np.ndarray) -> bool:
        return self.is_bev_occupied_map(airsim_ned_to_map(self._sanitize_ned(pos_ned)))

    def local_roof_map_z(self, pos_map: np.ndarray) -> float:
        ix, iy, _ = self._map_to_voxel(pos_map)
        return float(self._roof_z.get((ix, iy), self.cfg.map_elevation))

    def local_roof_map_z_window(
        self,
        pos_map: np.ndarray,
        *,
        range_m: float = 2.0,
    ) -> float:
        """OpenFly ``getMaxZinP(x, y, range)`` — max roof Z in a local window."""
        p = np.asarray(pos_map, dtype=np.float64).reshape(3)
        r = max(self._vw, float(range_m))
        best = self.cfg.map_elevation
        steps = int(np.ceil(r / self._vw))
        for dx in range(-steps, steps + 1):
            for dy in range(-steps, steps + 1):
                if dx * dx + dy * dy > steps * steps:
                    continue
                trial = p.copy()
                trial[0] += dx * self._vw
                trial[1] += dy * self._vw
                best = max(best, self.local_roof_map_z(trial))
        return float(best)

    def local_ground_map_z(self, pos_map: np.ndarray) -> float:
        """Highest PCD point in the ground band (map Z up)."""
        ix, iy, _ = self._map_to_voxel(pos_map)
        return float(self._ground_z.get((ix, iy), self.cfg.map_elevation))

    def has_ground_support_map(self, pos_map: np.ndarray) -> bool:
        if not self.has_ground_layer:
            return True
        ix, iy, _ = self._map_to_voxel(pos_map)
        return int(self._ground_count.get((ix, iy), 0)) >= self.cfg.min_ground_points_per_cell

    def is_drivable_ned(self, pos_ned: np.ndarray) -> bool:
        """Drivable street cell: not inside a building, has ground, not on a roof."""
        pos_ned = self._sanitize_ned(pos_ned)
        pos_map = airsim_ned_to_map(pos_ned)
        if self.is_bev_occupied_map(pos_map):
            return False
        if not self.has_ground_support_map(pos_map):
            return False
        return True

    def is_bev_free_ned(self, pos_ned: np.ndarray) -> bool:
        """Looser ``drivable`` check: only requires the cell to NOT be a
        BEV obstacle (building / wall / guardrail). Used for lateral-clearance
        checks where a missing ground stratum is acceptable (off-pavement
        shoulder is fine, hitting a wall is not).
        """
        pos_ned = self._sanitize_ned(pos_ned)
        return not self.is_bev_occupied_map(airsim_ned_to_map(pos_ned))

    def snap_car_to_ground_ned(self, pos_ned: np.ndarray) -> np.ndarray:
        """Place the vehicle on the local ground surface (map frame → NED)."""
        p = self._sanitize_ned(pos_ned)
        pos_map = airsim_ned_to_map(p)
        ground_z = self.local_ground_map_z(pos_map)
        if not np.isfinite(ground_z):
            ground_z = self.cfg.map_elevation
        p[2] = -(ground_z + self.cfg.car_agl_m)
        return p

    def min_safe_map_z(self, pos_map_xy: np.ndarray) -> float:
        """Minimum map-frame Z (up) for drone to clear local roof + margin."""
        probe = np.array([pos_map_xy[0], pos_map_xy[1], 0.0], dtype=np.float64)
        roof = self.local_roof_map_z(probe)
        return roof + self.cfg.min_drone_clearance

    def resolve_bev_move_ned(
        self,
        prev_ned: np.ndarray,
        proposed_ned: np.ndarray,
        *,
        keep_z: float | None = None,
    ) -> np.ndarray:
        """Shorten / cancel a ground move that would cross a non-drivable cell.

        Both the endpoint AND ~``ceil(step / voxel_width)`` evenly spaced
        midpoints are tested: a car moving 0.6 m/tick across a 1.5 m wide
        guardrail voxel must not be allowed to "tunnel" just because its
        endpoint lands on the far side of the guardrail.
        """
        prev = np.asarray(prev_ned, dtype=np.float64).reshape(3).copy()
        prop = np.asarray(proposed_ned, dtype=np.float64).reshape(3).copy()
        if keep_z is not None:
            prop[2] = keep_z

        # Two-stage check:
        # 1) WALL test (BEV-occupied) — buildings / guardrails are hard
        #    barriers. NEVER cross them, regardless of route plan.
        # 2) GROUND test (drivable centerline) — preferred, but if the entire
        #    segment is wall-free we let the car cross brief no-ground
        #    stretches (parking-lot, grass shoulder). This avoids freezing
        #    the car at every off-road voxel along a mostly-clean route.
        if (self._segment_wall_free(prev, prop, keep_z=keep_z)
                and not self.is_bev_occupied_ned(prop)):
            return self.snap_car_to_ground_ned(prop)

        for alpha in (0.75, 0.5, 0.25, 0.1, 0.05):
            trial = prev + alpha * (prop - prev)
            if keep_z is not None:
                trial[2] = keep_z
            if (self._segment_wall_free(prev, trial, keep_z=keep_z)
                    and not self.is_bev_occupied_ned(trial)):
                return self.snap_car_to_ground_ned(trial)

        # Wall ahead and we can't slip past — hold position. RoadScenario's
        # stuck-detector will advance arclength to escape the obstacle.
        return self.snap_car_to_ground_ned(prev)

    def _segment_wall_free(
        self,
        a_ned: np.ndarray,
        b_ned: np.ndarray,
        *,
        keep_z: float | None = None,
    ) -> bool:
        """``True`` iff segment a→b never crosses a BEV-occupied cell.

        Looser than ``_segment_drivable``: doesn't require ground support at
        every midpoint, only requires no walls. Used by
        ``resolve_bev_move_ned`` so a car can briefly graze a no-ground voxel
        (parking lot / shoulder) without being frozen, while still being
        hard-blocked by guardrails / building corners.
        """
        a = np.asarray(a_ned, dtype=np.float64).reshape(3).copy()
        b = np.asarray(b_ned, dtype=np.float64).reshape(3).copy()
        if keep_z is not None:
            a[2] = keep_z
            b[2] = keep_z
        dist = float(np.linalg.norm(b[:2] - a[:2]))
        n = max(1, int(math.ceil(dist / max(0.5 * self._vw, 0.25))))
        for i in range(1, n + 1):
            t = i / n
            mid = a + t * (b - a)
            if keep_z is not None:
                mid[2] = keep_z
            if not self.is_bev_free_ned(mid):
                return False
        return True

    def _segment_drivable(
        self,
        a_ned: np.ndarray,
        b_ned: np.ndarray,
        *,
        keep_z: float | None = None,
    ) -> bool:
        """``True`` iff every voxel-spaced sample on segment a→b is drivable
        (BEV empty + ground supported) at the centerline.

        Lateral (car-half-width) safety used to be enforced here but is too
        strict for env_airsim_16's 2-voxel-wide PCD roads — walls / guardrails
        within ±1 m of the centerline are the COMMON case, so a hard
        rejection would freeze the car. Lateral safety is now handled as a
        *soft* score penalty during ``build_route`` (see road_graph.py).
        """
        a = np.asarray(a_ned, dtype=np.float64).reshape(3).copy()
        b = np.asarray(b_ned, dtype=np.float64).reshape(3).copy()
        if keep_z is not None:
            a[2] = keep_z
            b[2] = keep_z
        if not self.is_drivable_ned(b):
            return False
        dist = float(np.linalg.norm(b[:2] - a[:2]))
        n = max(1, int(math.ceil(dist / max(0.5 * self._vw, 0.25))))
        for i in range(1, n):
            t = i / n
            mid = a + t * (b - a)
            if keep_z is not None:
                mid[2] = keep_z
            if not self.is_drivable_ned(mid):
                return False
        return True

    def los_blocked_ned(
        self,
        observer_ned: np.ndarray,
        target_ned: np.ndarray,
        *,
        drone_eye_agl_m: float = 12.0,
        target_agl_m: float = 1.0,
    ) -> bool:
        """Ray-march in map frame; True if 3-D voxels block the segment.

        The drone's *actual* altitude is preferred over ``drone_eye_agl_m`` so a
        tracker that climbs above buildings actually wins LOS. ``target_agl_m``
        is measured **above the local ground**, not the world floor — this is
        essential because the PCD stores the road surface as ~4 m of stacked
        voxels and naïvely placing the ray endpoint at ``target_agl_m`` would
        always intersect that ground stratum.

        The first and last march steps are skipped (one voxel padding at each
        endpoint) so the test never reports a self-occlusion on the very voxel
        containing the drone camera or the target's roof.
        """
        a_map = airsim_ned_to_map(observer_ned).copy()
        b_map = airsim_ned_to_map(target_ned).copy()
        # Drone eye: respect the actual altitude when it's safely higher than
        # the requested floor; otherwise lift to the floor.
        a_map[2] = max(float(a_map[2]), float(drone_eye_agl_m))
        # Target eye: place above the *local* ground (the road may sit several
        # metres above world Z=0).
        ground_z = self.local_ground_map_z(b_map)
        if not np.isfinite(ground_z):
            ground_z = self.cfg.map_elevation
        b_map[2] = max(float(ground_z + target_agl_m), self.cfg.map_elevation + 0.5)

        seg = b_map - a_map
        length = float(np.linalg.norm(seg))
        if length < self._vw * 0.5:
            return False
        steps = max(4, int(length / self._vw))
        # PCD ground / road surfaces are stored as several stacked voxels
        # below the actual road surface (typical thickness 2-5 m). A drone
        # descending toward a low target sweeps through this stratum at
        # intermediate columns, and would otherwise be erroneously declared
        # "occluded by the road itself". To distinguish a road-only column
        # from a real obstacle (building / guardrail), look at the column
        # ``vertical extent`` = ``local_roof_map_z - local_ground_map_z``:
        # below ~``road_only_thickness`` we treat the column as ground-only.
        road_only_thickness_m = 5.0
        for i in range(1, steps):
            t = i / steps
            p = a_map + t * seg
            if not self.is_3d_occupied_map(p):
                continue
            col_ground = self.local_ground_map_z(p)
            col_roof = self.local_roof_map_z(p)
            if not (np.isfinite(col_ground) and np.isfinite(col_roof)):
                return True
            if (col_roof - col_ground) <= road_only_thickness_m:
                # No tall obstacle in this column — must be road / curb
                # stratum; never count it as a blocker.
                continue
            return True
        return False

    def _column_vertical_extent_map(self, pos_map: np.ndarray) -> float:
        col_ground = self.local_ground_map_z(pos_map)
        col_roof = self.local_roof_map_z(pos_map)
        if not (np.isfinite(col_ground) and np.isfinite(col_roof)):
            return 0.0
        return float(col_roof - col_ground)

    def _building_footprint_cells_map(
        self,
        pos_map: np.ndarray,
        *,
        radius_cells: int = 2,
    ) -> int:
        """Count dilated building BEV cells in a (2r+1)² window."""
        ix0, iy0, _ = self._map_to_voxel(pos_map)
        count = 0
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                if (ix0 + dx, iy0 + dy) in self._bev2d:
                    count += 1
        return count

    def _is_large_building_occluder_map(
        self,
        pos_map: np.ndarray,
        *,
        min_building_height_m: float | None = None,
        min_footprint_cells: int = 9,
        road_only_thickness_m: float = 5.0,
    ) -> bool:
        """True when a map column is a *large building*, not a pole / rail / tree."""
        if not self.is_3d_occupied_map(pos_map):
            return False
        extent = self._column_vertical_extent_map(pos_map)
        if extent <= road_only_thickness_m:
            return False
        min_h = float(
            min_building_height_m
            if min_building_height_m is not None
            else max(self.cfg.min_height_thresh, 12.0)
        )
        if extent < min_h:
            return False
        if not self.is_building_bev_map(pos_map):
            return False
        if self._building_footprint_cells_map(
            pos_map, radius_cells=2,
        ) < int(min_footprint_cells):
            return False
        return True

    def has_adjacent_building_wall_ned(
        self,
        pos_ned: np.ndarray,
        *,
        keep_z: float,
        min_footprint_cells: int = 9,
        probe_dist_m: float = 7.5,
        footprint_radius_cells: int = 2,
    ) -> bool:
        """True if a wide building footprint sits within ``probe_dist_m`` of ``pos_ned``."""
        pos_map = airsim_ned_to_map(
            np.asarray(pos_ned, dtype=np.float64).reshape(3).copy())
        pos_map[2] = keep_z if keep_z is not None else pos_map[2]
        d = float(probe_dist_m)
        probes = (
            (d, 0.0), (-d, 0.0), (0.0, d), (0.0, -d),
            (0.7 * d, 0.7 * d), (-0.7 * d, 0.7 * d),
            (0.7 * d, -0.7 * d), (-0.7 * d, -0.7 * d),
        )
        need = int(min_footprint_cells)
        for dx, dy in probes:
            probe = pos_map.copy()
            probe[0] += dx
            probe[1] += dy
            if self._building_footprint_cells_map(
                probe, radius_cells=int(footprint_radius_cells),
            ) >= need:
                return True
        return False

    def los_blocked_by_building_ned(
        self,
        observer_ned: np.ndarray,
        target_ned: np.ndarray,
        *,
        drone_eye_agl_m: float = 12.0,
        target_agl_m: float = 1.0,
        min_building_height_m: float | None = None,
        min_footprint_cells: int = 9,
    ) -> bool:
        """Like ``los_blocked_ned`` but only counts large *building* occluders.

        Filters out street lamps, sign poles, and thin guardrails that still
        exceed the generic 5 m road-stratum threshold but occupy only 1–2 BEV
        cells and lack a wide building footprint.
        """
        a_map = airsim_ned_to_map(observer_ned).copy()
        b_map = airsim_ned_to_map(target_ned).copy()
        a_map[2] = max(float(a_map[2]), float(drone_eye_agl_m))
        ground_z = self.local_ground_map_z(b_map)
        if not np.isfinite(ground_z):
            ground_z = self.cfg.map_elevation
        b_map[2] = max(float(ground_z + target_agl_m), self.cfg.map_elevation + 0.5)

        seg = b_map - a_map
        length = float(np.linalg.norm(seg))
        if length < self._vw * 0.5:
            return False
        steps = max(4, int(length / self._vw))
        for i in range(1, steps):
            t = i / steps
            p = a_map + t * seg
            if self._is_large_building_occluder_map(
                p,
                min_building_height_m=min_building_height_m,
                min_footprint_cells=min_footprint_cells,
            ):
                return True
        return False

    def first_building_occluder_on_ray_ned(
        self,
        observer_ned: np.ndarray,
        target_ned: np.ndarray,
        *,
        drone_eye_agl_m: float = 12.0,
        target_agl_m: float = 1.0,
        min_building_height_m: float | None = None,
        min_footprint_cells: int = 9,
    ) -> tuple[float, np.ndarray] | None:
        """Return ``(t, map_xyz)`` of the first large-building hit on the ray, or None."""
        a_map = airsim_ned_to_map(observer_ned).copy()
        b_map = airsim_ned_to_map(target_ned).copy()
        a_map[2] = max(float(a_map[2]), float(drone_eye_agl_m))
        ground_z = self.local_ground_map_z(b_map)
        if not np.isfinite(ground_z):
            ground_z = self.cfg.map_elevation
        b_map[2] = max(float(ground_z + target_agl_m), self.cfg.map_elevation + 0.5)

        seg = b_map - a_map
        length = float(np.linalg.norm(seg))
        if length < self._vw * 0.5:
            return None
        steps = max(4, int(length / self._vw))
        for i in range(1, steps):
            t = i / steps
            p = a_map + t * seg
            if self._is_large_building_occluder_map(
                p,
                min_building_height_m=min_building_height_m,
                min_footprint_cells=min_footprint_cells,
            ):
                return float(t), np.asarray(p, dtype=np.float64)
        return None

    def building_occludes_between_ned(
        self,
        observer_ned: np.ndarray,
        target_ned: np.ndarray,
        *,
        drone_eye_agl_m: float = 12.0,
        target_agl_m: float = 1.0,
        min_building_height_m: float | None = None,
        min_footprint_cells: int = 9,
        near_target_m: float = 12.0,
        min_t: float = 0.08,
        max_t: float = 0.92,
    ) -> bool:
        """P1: a large building on the ray lies between drone and target, near the car."""
        hit = self.first_building_occluder_on_ray_ned(
            observer_ned, target_ned,
            drone_eye_agl_m=drone_eye_agl_m,
            target_agl_m=target_agl_m,
            min_building_height_m=min_building_height_m,
            min_footprint_cells=min_footprint_cells,
        )
        if hit is None:
            return False
        t, p_map = hit
        if t < float(min_t) or t > float(max_t):
            return False
        tgt_map = airsim_ned_to_map(
            np.asarray(target_ned, dtype=np.float64).reshape(3))
        dist_xy = float(np.linalg.norm(p_map[:2] - tgt_map[:2]))
        return dist_xy <= float(near_target_m)

    def _alley_hide_bonus_ned(
        self,
        pos_ned: np.ndarray,
        *,
        keep_z: float,
        max_width: float = 16.0,
        step: float = 2.0,
    ) -> float:
        """Score bonus for narrow BEV-free corridors (hutong / alley pockets)."""
        p = np.asarray(pos_ned, dtype=np.float64).reshape(3).copy()
        p[2] = keep_z
        side_h = 0.0
        for sign in (-1.0, 1.0):
            cur = p.copy()
            walked = 0.0
            while walked + step <= max_width:
                cur = cur.copy()
                cur[0] += sign * step
                cur[2] = keep_z
                if self.is_bev_occupied_ned(cur):
                    break
                walked += step
            side_h += walked
        return max(0.0, 12.0 - side_h) * 1.5

    def find_hide_goal_ned(
        self,
        target_ned: np.ndarray,
        drone_ned: np.ndarray,
        *,
        keep_z: float,
        search_radius_m: float = 28.0,
        min_hide_dist_m: float = 6.0,
        building_only: bool = False,
        min_building_height_m: float | None = None,
        min_footprint_cells: int = 9,
        require_adjacent_building: bool = True,
        building_probe_dist_m: float = 7.5,
        use_hide_visibility: bool = True,
        hide_vis_config: Any | None = None,
        chase_drone_poses: list[Any] | None = None,
        require_occluder_between: bool = True,
        occluder_near_target_m: float = 12.0,
    ) -> np.ndarray | None:
        """Find a street point near ``target`` where the car is hidden from ``drone``.

        The point must be BEV-free (not inside a building footprint) and the
        drone→point ray must pass through 3-D occupancy (building occludes LoS).

        When ``building_only`` is True, only large building footprints count as
        occluders (street lamps / thin poles are ignored).

        When ``hide_vis_config`` is supplied (P0), candidates must be hidden from
        all sampled chase drones using frustum + building LoS (same as demo).
        """
        from flyseek.adversary.base import DroneState, TargetState
        from flyseek.utils.hide_visibility import (
            HideVisibilityConfig,
            target_hidden_from_drone,
        )

        target_ned = np.asarray(target_ned, dtype=np.float64).reshape(3)
        drone_ned = np.asarray(drone_ned, dtype=np.float64).reshape(3)
        target_map = airsim_ned_to_map(target_ned)

        best: np.ndarray | None = None
        best_score = -1.0

        away = target_map[:2] - airsim_ned_to_map(drone_ned)[:2]
        away_norm = float(np.linalg.norm(away))
        if away_norm > 1e-3:
            away = away / away_norm
        else:
            away = np.array([1.0, 0.0])

        los_kw = dict(
            drone_eye_agl_m=12.0,
            target_agl_m=1.0,
            min_building_height_m=min_building_height_m,
            min_footprint_cells=min_footprint_cells,
        )

        vis_cfg: HideVisibilityConfig | None = None
        if hide_vis_config is not None:
            vis_cfg = (
                hide_vis_config if isinstance(hide_vis_config, HideVisibilityConfig)
                else HideVisibilityConfig(**dict(hide_vis_config))
            )
        elif use_hide_visibility:
            vis_cfg = HideVisibilityConfig(
                building_only_los=bool(building_only),
                min_building_height_m=min_building_height_m,
                min_footprint_cells=int(min_footprint_cells),
                occluder_between_required=bool(require_occluder_between),
                occluder_near_target_m=float(occluder_near_target_m),
            )

        drone_list: list[DroneState] = []
        if chase_drone_poses:
            for d in chase_drone_poses:
                if isinstance(d, DroneState):
                    drone_list.append(d)
                else:
                    pos = np.asarray(d, dtype=np.float64).reshape(3)
                    yaw = math.atan2(
                        float(target_ned[1] - pos[1]),
                        float(target_ned[0] - pos[0]),
                    )
                    drone_list.append(
                        DroneState(position=pos, velocity=np.zeros(3), heading=yaw)
                    )
        else:
            yaw = math.atan2(
                float(target_ned[1] - drone_ned[1]),
                float(target_ned[0] - drone_ned[0]),
            )
            drone_list.append(
                DroneState(position=drone_ned.copy(), velocity=np.zeros(3), heading=yaw)
            )

        def _passes_hide_checks(c: np.ndarray) -> bool:
            if vis_cfg is not None:
                tgt = TargetState(position=c.copy(), velocity=np.zeros(3), heading=0.0)
                for drone in drone_list:
                    ok, _ = target_hidden_from_drone(self, drone, tgt, vis_cfg)
                    if not ok:
                        return False
                    if require_occluder_between:
                        if not self.building_occludes_between_ned(
                            drone.position, c,
                            drone_eye_agl_m=vis_cfg.drone_eye_agl_m,
                            min_building_height_m=min_building_height_m,
                            min_footprint_cells=min_footprint_cells,
                            near_target_m=occluder_near_target_m,
                        ):
                            return False
                return True
            if building_only:
                return self.los_blocked_by_building_ned(drone_ned, c, **los_kw)
            return self.los_blocked_ned(drone_ned, c)

        radii = np.arange(min_hide_dist_m, search_radius_m + 1e-6, 3.0)
        angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)

        for dist in radii:
            for ang in angles:
                offset = np.array([np.cos(ang), np.sin(ang)]) * dist
                cand_map = np.array([
                    target_map[0] + offset[0],
                    target_map[1] + offset[1],
                    target_map[2],
                ])
                cand_ned = map_to_airsim_ned(cand_map)
                cand_ned[2] = keep_z

                if not self.is_drivable_ned(cand_ned):
                    continue
                if not _passes_hide_checks(cand_ned):
                    continue
                if building_only and require_adjacent_building:
                    if not self.has_adjacent_building_wall_ned(
                        cand_ned, keep_z=keep_z,
                        min_footprint_cells=int(min_footprint_cells),
                        probe_dist_m=float(building_probe_dist_m),
                    ):
                        continue

                toward_hide = float(np.dot(offset / (dist + 1e-9), away))
                # Prefer hutong / alley pockets (narrow but BEV-free).
                alley_bonus = self._alley_hide_bonus_ned(cand_ned, keep_z=keep_z)
                score = toward_hide * 2.0 + dist + alley_bonus
                # Prefer spots with a larger building wall nearby.
                if building_only:
                    cand_map_xy = airsim_ned_to_map(cand_ned)
                    for dx, dy in ((7.5, 0.0), (-7.5, 0.0), (0.0, 7.5), (0.0, -7.5)):
                        probe = cand_map_xy.copy()
                        probe[0] += dx
                        probe[1] += dy
                        fp = self._building_footprint_cells_map(probe, radius_cells=2)
                        if fp >= min_footprint_cells:
                            score += fp * 1.2
                            break
                if score > best_score:
                    best_score = score
                    best = cand_ned.copy()

        return best

    def resolve_drone_ned(
        self,
        prev_ned: np.ndarray,
        proposed_ned: np.ndarray,
        *,
        body_radius: float = 1.0,
    ) -> np.ndarray:
        """Lift / reject drone moves that intersect dilated 3-D occupancy."""
        prev = np.asarray(prev_ned, dtype=np.float64).reshape(3)
        prop = np.asarray(proposed_ned, dtype=np.float64).reshape(3).copy()

        prop_map = airsim_ned_to_map(prop)
        min_z = self.local_roof_map_z_window(prop_map) + self.cfg.min_drone_clearance
        if prop_map[2] < min_z:
            prop_map[2] = min_z

        if self.is_3d_occupied_map(prop_map):
            for lift in (self._vw, 2 * self._vw, 3 * self._vw, 5 * self._vw):
                trial_map = prop_map.copy()
                trial_map[2] += lift
                if not self.is_3d_occupied_map(trial_map):
                    prop_map = trial_map
                    break
            else:
                prop_map = airsim_ned_to_map(prev)

        prop = map_to_airsim_ned(prop_map)

        if body_radius > 0:
            for dx, dy in ((body_radius, 0), (-body_radius, 0),
                           (0, body_radius), (0, -body_radius)):
                probe = prop.copy()
                probe[0] += dx
                probe[1] += dy
                if self.is_3d_occupied_map(airsim_ned_to_map(probe)):
                    return prev.copy()
        return prop

    def cache_path(self, env_name: str, cache_dir: Path) -> Path:
        return cache_dir / f"{env_name}_occ3d_v{self._vw}_d{self.cfg.dilate_radius}.npz"

    # Cache schema version — bump when the build changes shape so stale caches
    # are ignored (v3 = ground-band fix + car-obstacle layer).
    CACHE_SCHEMA_VERSION = 3

    def save_cache(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        occ = np.array(list(self._occ3d), dtype=np.int32)
        bev = np.array(list(self._bev2d), dtype=np.int32)
        car_obs = (np.array(list(self._car_obs2d), dtype=np.int32)
                   if self._car_obs2d else np.empty((0, 2), dtype=np.int32))
        roof_keys = np.array(list(self._roof_z.keys()), dtype=np.int32)
        roof_vals = np.array(list(self._roof_z.values()), dtype=np.float64)
        g_keys = np.array(list(self._ground_z.keys()), dtype=np.int32)
        g_vals = np.array(list(self._ground_z.values()), dtype=np.float64)
        gc_keys = np.array(list(self._ground_count.keys()), dtype=np.int32)
        gc_vals = np.array(list(self._ground_count.values()), dtype=np.int32)
        np.savez_compressed(
            path,
            occ3d=occ,
            bev2d=bev,
            car_obs2d=car_obs,
            roof_keys=roof_keys,
            roof_vals=roof_vals,
            ground_keys=g_keys,
            ground_vals=g_vals,
            ground_count_keys=gc_keys,
            ground_count_vals=gc_vals,
            map_bound=np.array(self.cfg.map_bound),
            voxel_width=self.cfg.voxel_width,
            dilate_radius=self.cfg.dilate_radius,
            map_elevation=self.cfg.map_elevation,
            min_height_thresh=self.cfg.min_height_thresh,
            min_drone_clearance=self.cfg.min_drone_clearance,
            min_ground_points_per_cell=self.cfg.min_ground_points_per_cell,
            car_agl_m=self.cfg.car_agl_m,
            ground_below_m=self.cfg.ground_below_m,
            car_obstacle_min_h_m=self.cfg.car_obstacle_min_h_m,
            ground_enabled=np.array(self._ground_enabled),
            schema_version=np.array(self.CACHE_SCHEMA_VERSION),
        )

    @classmethod
    def load_cache(cls, path: Path) -> "PcdOccupancyMap":
        data = np.load(path, allow_pickle=False)
        schema = int(data["schema_version"]) if "schema_version" in data else 1
        if schema < cls.CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"stale occupancy cache schema v{schema} "
                f"(< v{cls.CACHE_SCHEMA_VERSION}); rebuild required"
            )
        mb = tuple(float(v) for v in data["map_bound"])
        cfg = OccupancyConfig(
            voxel_width=float(data["voxel_width"]),
            dilate_radius=float(data["dilate_radius"]),
            map_bound=mb,  # type: ignore[arg-type]
            map_elevation=float(data["map_elevation"]),
            min_height_thresh=float(data["min_height_thresh"]),
            min_drone_clearance=float(data["min_drone_clearance"]),
            min_ground_points_per_cell=int(
                data.get("min_ground_points_per_cell", 2)
            ),
            car_agl_m=float(data.get("car_agl_m", 0.35)),
            ground_below_m=float(data.get("ground_below_m", 2.0)),
            car_obstacle_min_h_m=float(data.get("car_obstacle_min_h_m", 0.6)),
        )
        occ3d = set(map(tuple, data["occ3d"].tolist()))
        bev2d = set(map(tuple, data["bev2d"].tolist()))
        car_obs2d = (set(map(tuple, data["car_obs2d"].tolist()))
                     if "car_obs2d" in data else set())
        roof_z = {
            (int(k[0]), int(k[1])): float(v)
            for k, v in zip(data["roof_keys"], data["roof_vals"])
        }
        ground_z: dict[tuple[int, int], float] = {}
        ground_count: dict[tuple[int, int], int] = {}
        ground_enabled = bool(data.get("ground_enabled", False))
        if "ground_keys" in data:
            ground_enabled = True
            ground_z = {
                (int(k[0]), int(k[1])): float(v)
                for k, v in zip(data["ground_keys"], data["ground_vals"])
            }
            ground_count = {
                (int(k[0]), int(k[1])): int(v)
                for k, v in zip(data["ground_count_keys"], data["ground_count_vals"])
            }
        print(f"[occupancy] loaded cache: {path.name} (schema v{schema}; "
              f"{len(occ3d):,} 3-D, {len(bev2d):,} building, "
              f"{len(car_obs2d):,} car-obstacle, {len(ground_z):,} ground, "
              f"enabled={ground_enabled})")
        return cls(cfg, occ3d, bev2d, roof_z, ground_z, ground_count, car_obs2d,
                   ground_enabled=ground_enabled)

    @classmethod
    def load_or_build(
        cls,
        repo_root: Path,
        env_name: str = "env_airsim_16",
        cfg: OccupancyConfig | None = None,
        cache_dir: Path | None = None,
        rebuild: bool = False,
        *,
        min_height_thresh: float | None = None,
        disable_car_obstacles: bool = False,
    ) -> "PcdOccupancyMap":
        cache_dir = cache_dir or (repo_root / "flyseek_extend" / "output" / "cache")
        if cfg is None:
            import yaml  # type: ignore
            yaml_path = repo_root / "configs" / f"{env_name}.yaml"
            traj_map = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["traj_map"]
            cfg = OccupancyConfig.from_yaml_section(traj_map)

        # Per-call overrides (UE scaled PCDs: the yaml's 6 m building threshold
        # flags curbs/low clutter as buildings and boxes ground vehicles in).
        if min_height_thresh is not None or disable_car_obstacles:
            from dataclasses import replace
            kw: dict = {}
            if min_height_thresh is not None:
                kw["min_height_thresh"] = float(min_height_thresh)
            if disable_car_obstacles:
                kw["car_obstacle_min_h_m"] = 1.0e9  # empties the curb/rail band
            cfg = replace(cfg, **kw)

        tag = ""
        if min_height_thresh is not None or disable_car_obstacles:
            tag = f"_h{int(cfg.min_height_thresh)}"
            if disable_car_obstacles:
                tag += "_nco"
        cache_path = (
            cache_dir
            / f"{env_name}_occ_v{cfg.voxel_width}_d{int(cfg.dilate_radius)}"
              f"{tag}_s{cls.CACHE_SCHEMA_VERSION}.npz"
        )
        if cache_path.exists() and not rebuild:
            try:
                return cls.load_cache(cache_path)
            except Exception as e:
                print(f"[occupancy] cache load failed ({e}), rebuilding…")

        built = cls.from_env(repo_root, env_name, cfg)
        try:
            built.save_cache(cache_path)
            print(f"[occupancy] cache saved → {cache_path}")
        except Exception as e:
            print(f"[occupancy] cache save skipped: {e}")
        return built

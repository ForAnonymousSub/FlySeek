# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""FlySeek UE-city chase demo (UnrealCV render backend).

The env_ue_smallcity / env_ue_bigcity counterpart of ``demo_adversary_chase.py``.
Instead of AirSim RPC, it drives ONE real City Sample car along a
collision-aware (road-constrained, building-avoiding) adversarial trajectory via
UnrealCV ``vset /object/.../location`` and renders an **angled-down** UAV chase
view via the ``UnrealCVRenderer`` adapter.

Reused, simulator-agnostic FlySeek machinery:
  * ``PcdOccupancyMap`` (now UE-aware: extended PCD fields + pcd_scale_ratio) —
    keeps the car on drivable road and off buildings (cached after first build).
  * ``TargetPolicy`` — the FlySeek-Bench adversarial target (direct_escape /
    sharp_turn / detour_feint / occlusion_seeking); integrates + ground/collision
    stabilises internally against the occupancy.

Prereqs (sim launched STANDALONE so the probe/this demo gets the UnrealCV slot):

    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    ./City_UE52/Binaries/Linux/CitySample City_UE52 -RenderOffScreen -ResX=1280 -ResY=720

    python flyseek_extend/scripts/demo_unrealcv_chase.py --env env_ue_smallcity \
        --target-behavior detour_feint --target-policy-difficulty medium --duration 30

NOTE: camera framing (follow distance/altitude/pitch) and the car-yaw sign are
exposed as flags; expect one quick live tuning pass on the first run.
"""

from __future__ import annotations

import argparse
import math
import re
import signal
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for sub in ("flyseek_extend", "flyseek_extend/scripts"):
    p = str(REPO_ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from types import SimpleNamespace  # noqa: E402

from flyseek.adapters.pcd_occupancy import PcdOccupancyMap  # noqa: E402
from flyseek.adapters.unrealcv_render import (  # noqa: E402
    UnrealCVRenderer,
    ue_cm_to_map_m,
)
from flyseek.adversary import DroneState, TargetState, horizontal_distance  # noqa: E402
from flyseek.bench.target_policy import BEHAVIOR_TYPES  # noqa: E402
from flyseek.bench.expert_trajectory import (  # noqa: E402
    ExpertTrajectoryConfig,
    ExpertViewpointPlanner,
    save_trajectories,
)
from flyseek.scenarios.road_scenarios import (  # noqa: E402
    RoadScenarioController,
    RoadScenarioConfig,
)
from flyseek.utils.road_graph import find_major_road_seed  # noqa: E402
from flyseek.utils.coords import airsim_ned_to_map, map_to_airsim_ned  # noqa: E402

# Standardized FlySeek-Bench exporters (same bundle as env_airsim_16 episodes).
from flyseek.bench.schema import (  # noqa: E402
    CameraConfig,
    EpisodeMetadata,
    FrameMetadata,
)
from flyseek.bench.export import append_frame_jsonl, save_metadata_json  # noqa: E402
from flyseek.bench.visibility import VisibilityEvaluator  # noqa: E402
from flyseek.bench.instruction_generator import (  # noqa: E402
    InstructionGenerator,
    attributes_from_label,
    write_instruction_json,
)
from flyseek.bench.metrics import evaluate_episode_dir  # noqa: E402


def _car_label(name: str) -> str:
    n = name.lower()
    if "van" in n:
        return "a delivery van"
    if "truck" in n:
        return "a truck"
    if "bus" in n:
        return "a city bus"
    return "a small car"

from flyseek.utils.occlusion_route import (  # noqa: E402
    BEHAVIOR_ROUTE_MANEUVER,
    build_occlusion_seeking_route,
)

# Closed-loop UAV trackers — the SAME implementations used by the AirSim
# (env_airsim_16) pipeline, reused here so env_ue_smallcity gets identical
# FlySeek (adaptive FSM) vs reactive / reactive_lost baseline behaviour. They
# are simulator-agnostic (pure numpy over the PCD occupancy + visibility), so
# only the render backend differs.
from flyseek.expert.adaptive_tracker import AdaptiveTracker  # noqa: E402
from demo_adversary_chase import (  # noqa: E402
    _InlineTracker,
    _ReactiveTracker,
    _ReactiveLostTracker,
)

CLOSED_LOOP_TRACKERS = ("adaptive", "inline", "reactive", "reactive_lost")

_BEHAVIOR_TO_MANEUVER = BEHAVIOR_ROUTE_MANEUVER


def _tracker_args(args, seed: int) -> SimpleNamespace:
    """Build the attribute bag the trackers read (UE args + sane defaults).

    The AirSim demo passes its full argparse namespace; here we synthesize an
    equivalent one so the reused tracker classes find every field they touch.
    """
    g = lambda n, d: getattr(args, n, d)  # noqa: E731
    return SimpleNamespace(
        camera_hfov_deg=float(args.camera_hfov_deg),
        follow_distance=float(args.follow_distance),
        follow_altitude=float(args.follow_altitude),
        drone_smoothing=float(g("drone_smoothing", 4.0)),
        tracker_yaw_gain=float(g("tracker_yaw_gain", 3.0)),
        tracker_motion_dir_tau=float(g("tracker_motion_dir_tau", 1.5)),
        tracker_lead_s=float(g("tracker_lead_s", 0.7)),
        tracker_fov_center_gain=float(g("tracker_fov_center_gain", 10.0)),
        lost_after_s=float(g("lost_after_s", 0.6)),
        lost_wander_radius_m=float(g("lost_wander_radius_m", 6.0)),
        lost_wander_scan_dps=float(g("lost_wander_scan_dps", 35.0)),
        search_orbit_radius=float(g("search_orbit_radius", 14.0)),
        search_orbit_speed_dps=float(g("search_orbit_speed_dps", 40.0)),
        tracker_predict_after_s=float(g("tracker_predict_after_s", 0.4)),
        tracker_reacquire_after_s=float(g("tracker_reacquire_after_s", 1.2)),
        tracker_peek_after_s=float(g("tracker_peek_after_s", 0.8)),
        tracker_search_after_s=float(g("tracker_search_after_s", 3.0)),
        tracker_hold_speed=float(g("tracker_hold_speed", 0.3)),
        tracker_hold_dwell_s=float(g("tracker_hold_dwell_s", 4.0)),
        tracker_hold_resume_speed=float(g("tracker_hold_resume_speed", 1.0)),
        altitude_smooth_tau=float(g("altitude_smooth_tau", 3.0)),
        roof_smooth_tau=float(g("roof_smooth_tau", 6.0)),
        max_climb_mps=float(g("max_climb_mps", 1.5)),
        max_drop_mps=float(g("max_drop_mps", 2.0)),
        roof_probe_range_m=float(g("roof_probe_range_m", 2.0)),
        vis_max_range_m=float(args.vis_max_range_m),
        no_collision=bool(g("no_collision", False)),
        seed=int(seed),
    )


def _build_chase_tracker(mode: str, targs: SimpleNamespace, occ):
    """Instantiate one of the reused closed-loop UAV trackers."""
    if mode == "adaptive":
        return AdaptiveTracker.from_args(targs, occupancy=occ)
    if mode == "inline":
        return _InlineTracker(targs, occupancy=occ)
    if mode == "reactive":
        return _ReactiveTracker(targs, occupancy=occ)
    if mode == "reactive_lost":
        return _ReactiveLostTracker(targs, occupancy=occ)
    raise ValueError(f"not a closed-loop tracker mode: {mode!r}")
_DIFFICULTY_SPEED = {  # (cruise speed, high/burst speed) m/s
    "easy": (3.0, 5.0),
    "medium": (4.5, 7.0),
    "hard": (6.0, 9.0),
}


def _heading_map_from_ned_vel(v: np.ndarray, fallback_heading: float) -> tuple[np.ndarray, float]:
    """Return (unit motion dir in NED xy, heading in MAP frame).

    map = (nx,-ny,-nz), so a NED in-plane heading maps to atan2(-vy, vx).
    """
    mdir = np.array([float(v[0]), float(v[1])], dtype=np.float64)
    if np.linalg.norm(mdir) < 1e-3:
        mdir = np.array([math.cos(fallback_heading), math.sin(fallback_heading)])
    mdir /= max(np.linalg.norm(mdir), 1e-9)
    return mdir, math.atan2(-mdir[1], mdir[0])


def _chase_estimate_ned(target_pos: np.ndarray, mdir_ned: np.ndarray,
                        fd: float, fa: float) -> np.ndarray:
    back = -mdir_ned
    return np.array([target_pos[0] + back[0] * fd,
                     target_pos[1] + back[1] * fd,
                     target_pos[2] - abs(fa)], dtype=np.float64)

CAR_INCLUDE = ("BP_vehCar", "BP_vehVan", "BP_vehTruck")
CAR_EXCLUDE = ("Spawner", "Initialize", "MassTraffic")
# Broader vehicle matcher used for HIDING traffic (City Sample actorizes distant
# Mass traffic into BP_veh* / *Vehicle* actors as the camera nears).
_VEH_RE = re.compile(r"veh|car|truck|bus|van|suv|sedan|taxi", re.IGNORECASE)


def _select_controllable_car(r: UnrealCVRenderer, args) -> tuple[str, np.ndarray] | None:
    """Return (name, start_ue_cm) of a car whose teleport sticks (not Mass/AI)."""
    objs = r.list_objects()
    cands = [o for o in objs
             if any(k in o for k in CAR_INCLUDE) and not any(k in o for k in CAR_EXCLUDE)]
    if args.target:
        cands = [args.target] + [c for c in cands if c != args.target]
    print(f"[info] {len(cands)} car candidates; testing controllability...")
    tested = 0
    for name in cands:
        if tested >= args.max_try:
            break
        loc = r.get_object_location_ue_cm(name)
        if loc is None or np.all(np.abs(loc) < 1e-6):
            continue
        tested += 1
        probe = loc.copy(); probe[0] += 500.0
        r.set_object_location_ue_cm(name, *probe)
        time.sleep(args.settle_s)
        after = r.get_object_location_ue_cm(name)
        r.set_object_location_ue_cm(name, *loc)  # restore
        if after is not None and abs(after[0] - probe[0]) < 150.0:
            print(f"[ok] controllable car: {name} @ {np.round(loc,0)} cm")
            return name, loc
        print(f"  [skip] {name}: teleport overridden (Mass/AI)")
    return None


def _hide_vehicles(r: UnrealCVRenderer, car: str, hidden: set, args,
                   near_ue: np.ndarray | None = None) -> int:
    """Hide addressable vehicles (except the target) not already hidden.

    Re-callable: City Sample LODs distant Mass traffic as ISM instances and
    *actorizes* them into BP_veh*/`*Vehicle*` actors only when near the camera.
    So new vehicle actors appear as the chase camera moves — calling this every
    few ticks catches the newly-actorized traffic the target would clip through.
    (The ISM-only far traffic has no actor name and cannot be hidden.)
    """
    if not (args.hide_others or args.hide_radius_m > 0):
        return 0
    radius_cm = args.hide_radius_m * 100.0
    newly = 0
    for o in r.list_objects():
        if o == car or o in hidden:
            continue
        if not _VEH_RE.search(o) or any(k in o for k in CAR_EXCLUDE):
            continue
        if args.hide_radius_m > 0 and near_ue is not None:
            loc = r.get_object_location_ue_cm(o)
            if loc is None or np.linalg.norm(loc[:2] - near_ue[:2]) > radius_cm:
                continue
        if r.hide(o):
            hidden.add(o)
            newly += 1
    return newly


def generate_episode(r, occ, car: str, start_ue: np.ndarray, args,
                     behavior: str, difficulty: str, seed: int, idx: int,
                     hidden: set | None = None) -> dict:
    """Generate ONE episode and write the full FlySeek-Bench bundle.

    Two-pass: simulate the road-constrained target, plan visibility-aware expert
    UAV viewpoints, render the car + expert-follow camera, and write the same
    bundle as env_airsim_16 episodes (metadata.json, frames.jsonl, instruction.json,
    trajectories.json, metrics.json, config.yaml, images/) for train.json prep.
    """
    dt = 1.0 / args.tick_hz
    total = int(args.duration * args.tick_hz)
    episode_id = f"{args.env}_{behavior}_{difficulty}_seed{seed}_{idx:03d}"
    # Tag the episode dir with the UAV tracker so a multi-tracker comparison
    # (e.g. adaptive vs reactive_lost) writes distinct dirs in one connection.
    _tmode = getattr(args, "tracker_mode", None)
    if _tmode:
        episode_id = f"{episode_id}_{_tmode}"

    # ---- PASS 1: roll out a ROAD-CONSTRAINED target trajectory (no render) --
    # Use the route follower (build_route on wide BEV-free corridors + spline +
    # car dynamics) instead of the reactive evasion policy: it stays strictly on
    # drivable streets (never enters buildings / alleys narrower than the car),
    # locks the road-plane Z (no climbing), and drives like a car (slows into
    # corners). The 4 adversarial behaviors map onto route maneuvers.
    rng = np.random.default_rng(seed)
    spawn_ned = map_to_airsim_ned(ue_cm_to_map_m(start_ue))
    kz = float(spawn_ned[2])
    seed_p, seed_h, seed_sc = find_major_road_seed(
        occ, spawn_ned, rng, keep_z=kz, search_radius_m=args.road_search_m)
    maneuver = _BEHAVIOR_TO_MANEUVER.get(behavior, "normal_drive")
    spd, hspd = _DIFFICULTY_SPEED.get(difficulty, (4.5, 7.0))
    print(f"[pass1] {behavior}/{difficulty} → maneuver={maneuver} "
          f"(seed road score={seed_sc:.0f}, {np.linalg.norm(seed_p[:2]-spawn_ned[:2]):.0f}m "
          f"from spawn); simulating {total} ticks...", flush=True)
    scen_cfg = RoadScenarioConfig(name=maneuver, route_len_m=args.route_len_m,
                                  speed_mps=spd, high_speed_mps=hspd)
    init = TargetState(position=seed_p.copy(), velocity=np.zeros(3),
                       heading=seed_h, timestamp=0.0)
    sim_drone0 = DroneState(
        position=seed_p + np.array([-args.follow_distance, 0.0,
                                    -args.follow_altitude]),
        velocity=np.zeros(3), heading=seed_h, timestamp=0.0)
    prebuilt_route = None
    route_meta: dict = {}
    if behavior == "alley_hutong":
        # Drive the car into a narrow gap *between annotated buildings* and park
        # deep inside — the occlusion-rich pursuit episode from the paper (the
        # car reaches a building corner / alley, a reactive baseline follows
        # from behind and loses it, FlySeek pre-positions to a side/elevated
        # viewpoint to keep line-of-sight). Reuses the SAME route planner as the
        # AirSim alley demo (build_alley_hutong_route + seg buildings).
        from flyseek.utils.seg_buildings import SegBuildingMap
        from flyseek.utils.alley_route import build_alley_hutong_route
        seg = SegBuildingMap.from_jsonl(
            args.seg_building_jsonl, footprint_radius_m=10.0)
        print(f"[pass1] seg buildings: {len(seg)} from {args.seg_building_jsonl}",
              flush=True)
        alley_kw: dict = {}
        if args.open_approach_m:
            alley_kw["open_approach_m"] = float(args.open_approach_m)
        if args.max_corridor_width_m:
            alley_kw["max_corridor_width_m"] = float(args.max_corridor_width_m)
        prebuilt_route, route_meta = build_alley_hutong_route(
            occ, seg, seed_p, rng,
            keep_z=kz,
            search_radius_m=float(args.road_search_m),
            **alley_kw,
        )
        if prebuilt_route is None:
            print(f"[FAIL] alley_hutong route planning failed: "
                  f"{route_meta.get('error', 'unknown')}. Try a larger "
                  f"--road-search-m, larger --max-corridor-width-m, or another "
                  f"--seed near a hutong.", flush=True)
            raise RuntimeError("alley_hutong route planning failed")
        wp0 = np.asarray(prebuilt_route.waypoints, dtype=np.float64)
        split_idx = int(route_meta.get(
            "split_idx", max(0, wp0.shape[0] - 3)))
        open_frac = max(0.15, min(0.85, split_idx / max(wp0.shape[0] - 1, 1)))
        h0 = math.atan2(float(wp0[1, 1] - wp0[0, 1]),
                        float(wp0[1, 0] - wp0[0, 0])) if wp0.shape[0] > 1 else seed_h
        init = TargetState(position=wp0[0].copy(), velocity=np.zeros(3),
                           heading=h0, timestamp=0.0)
        maneuver = "alley_hutong"
        scen_cfg = RoadScenarioConfig(
            name="alley_hutong", route_len_m=float(args.route_len_m),
            speed_mps=spd * 0.85, high_speed_mps=spd * 1.1,
            open_road_frac=open_frac)
        print(f"[pass1] alley route: corridor="
              f"{route_meta.get('corridor_width_m')}m depth="
              f"{route_meta.get('alley_depth_m')}m wp={wp0.shape[0]} "
              f"split={split_idx} (alley_starts≈"
              f"{route_meta.get('est_alley_start_s')}s)", flush=True)
    elif behavior == "occlusion_seeking":
        prebuilt_route, route_meta = build_occlusion_seeking_route(
            occ, seed_p, rng,
            keep_z=kz,
            drone_ned=sim_drone0.position,
            route_len_m=float(args.route_len_m),
            search_radius_m=float(args.road_search_m),
            anchor_heading_rad=float(seed_h),
            drone_eye_agl_m=float(args.follow_altitude),
        )
        print(f"[pass1] hide route: building_occluded="
              f"{route_meta.get('building_occluded_frac', 0):.0%} "
              f"hide_goal_building={route_meta.get('hide_goal_building_occluded')} "
              f"wp={route_meta.get('route_waypoints')}", flush=True)
    ctl = RoadScenarioController(
        occ, init, rng, scen_cfg,
        route=prebuilt_route,
    )

    target_records: list[dict] = []
    sim_drone = sim_drone0
    st = ctl.initial_state()
    for tick in range(total):
        t = tick * dt
        st, _act = ctl.step(sim_drone, t, dt)
        mdir_ned, _ = _heading_map_from_ned_vel(st.velocity, st.heading)
        target_records.append({
            "t": t, "pos": st.position.copy(), "vel": st.velocity.copy(),
            "heading": float(st.heading), "mdir_ned": mdir_ned,
        })
        est = _chase_estimate_ned(st.position, mdir_ned,
                                  args.follow_distance, args.follow_altitude)
        sim_drone = sim_drone.copy_with(position=est, timestamp=t)

    # ---- PLAN: visibility-aware expert UAV viewpoints ----------------------
    expert_cfg = ExpertTrajectoryConfig(
        follow_distance_m=args.follow_distance,
        follow_altitude_m=args.follow_altitude,
        hfov_deg=args.camera_hfov_deg, max_range_m=args.vis_max_range_m,
        plan_stride=1)
    planner = ExpertViewpointPlanner(
        config=expert_cfg, scene_context={"occupancy": occ}, seed=seed)
    expert_out = planner.plan(
        [{"t": rr["t"], "pos": rr["pos"].tolist(), "vel": rr["vel"].tolist()}
         for rr in target_records])
    vp_by_idx = {int(vp["frame_idx"]): np.array(vp["position"], dtype=np.float64)
                 for vp in expert_out["expert_viewpoints"]}

    # ---- PASS 2: render + write FlySeek-Bench bundle -----------------------
    out_dir = args.out / episode_id
    frames_dir = out_dir / "images"
    frames_jsonl = out_dir / "frames.jsonl"
    if frames_jsonl.exists():
        frames_jsonl.unlink()
    target_class = _car_label(car)
    cam_cfg = CameraConfig(name="front_custom", hfov_deg=float(args.camera_hfov_deg),
                           pitch_deg=float(args.camera_pitch_deg),
                           width=int(args.width), height=int(args.height))
    instr = InstructionGenerator(seed=seed).generate(
        target_class=target_class,
        target_attributes=attributes_from_label(target_class),
        initial_context={"motion": "the street"},
        behavior_type=behavior, difficulty_level=difficulty)
    vis_eval = VisibilityEvaluator(max_range_m=float(args.vis_max_range_m),
                                   drone_eye_agl_m=float(args.follow_altitude))
    print(f"[pass2] {episode_id}: rendering ({args.camera_mode}) → {out_dir}",
          flush=True)
    uav_records: list[dict] = []
    _fallback_cam0 = _chase_estimate_ned(
        target_records[0]["pos"], target_records[0]["mdir_ned"],
        args.follow_distance, args.follow_altitude)
    prev_cam_ned = vp_by_idx.get(0, _fallback_cam0).copy()
    smooth_cam = prev_cam_ned.copy()      # EMA state for smooth UAV motion
    alpha = float(np.clip(dt / max(args.camera_smooth_tau, 1e-3), 0.0, 1.0))
    in_frustum = 0
    half_hfov = math.radians(args.camera_hfov_deg * 0.5)
    if hidden is None:
        hidden = set()

    # ---- closed-loop UAV tracker (FlySeek adaptive / reactive_lost baseline) -
    # When --tracker-mode names a closed-loop tracker, the UAV camera is driven
    # by the SAME controller as the AirSim pipeline instead of the expert/chase
    # viewpoint placement. The drone is seeded at a CANONICAL pose behind the
    # target (identical regardless of tracker, for a fair comparison) and steps
    # against the pre-rolled target trajectory; the camera takes the tracker's
    # own heading (so a lost baseline scans/wanders rather than magically facing
    # the hidden target).
    tracker_mode = getattr(args, "tracker_mode", None)
    chase_tracker = None
    tracker_drone = None
    if tracker_mode in CLOSED_LOOP_TRACKERS:
        targs = _tracker_args(args, seed)
        chase_tracker = _build_chase_tracker(tracker_mode, targs, occ)
        t0 = target_records[0]
        h0 = float(t0["heading"])
        back0 = h0 + math.pi
        cam0 = np.array([
            float(t0["pos"][0] + math.cos(back0) * args.follow_distance),
            float(t0["pos"][1] + math.sin(back0) * args.follow_distance),
            -abs(float(args.follow_altitude)),
        ], dtype=np.float64)
        tracker_drone = DroneState(position=cam0, velocity=np.zeros(3),
                                   heading=h0, timestamp=0.0)
        tgt0 = TargetState(position=t0["pos"].copy(), velocity=t0["vel"].copy(),
                           heading=h0, timestamp=0.0)
        chase_tracker.reset(tracker_drone, tgt0)
        prev_cam_ned = cam0.copy()
        print(f"[pass2] UAV tracker = {tracker_mode} (closed-loop, "
              "simulator-agnostic — same as env_airsim_16)", flush=True)
    for tick in range(total):
        rec = target_records[tick]
        tmap = airsim_ned_to_map(rec["pos"])
        _, heading_map = _heading_map_from_ned_vel(rec["vel"], rec["heading"])

        # Re-hide newly-actorized traffic near the moving target/camera.
        if args.rehide_every > 0 and tick % args.rehide_every == 0:
            near_ue = np.array([tmap[0] * 100.0, -tmap[1] * 100.0,
                                tmap[2] * 100.0])
            nh = _hide_vehicles(r, car, hidden, args, near_ue=near_ue)
            if nh:
                print(f"  [rehide] +{nh} vehicles at tick {tick}", flush=True)

        r.set_object_pose_map(car, tmap, heading_map)

        if chase_tracker is not None:
            # Closed-loop tracker: step against the pre-rolled target state.
            tgt = TargetState(position=rec["pos"].copy(),
                              velocity=rec["vel"].copy(),
                              heading=float(rec["heading"]),
                              timestamp=float(rec["t"]))
            tracker_drone, _tlog = chase_tracker.step(tracker_drone, tgt, dt)
            cam_ned = tracker_drone.position.copy()
            prev_cam_ned = cam_ned.copy()
            uav_yaw_ned = float(tracker_drone.heading)
            # Aim the UE camera along the tracker's OWN heading (when lost it
            # scans / wanders instead of facing the hidden target).
            look_ned = cam_ned + np.array([
                math.cos(uav_yaw_ned) * args.follow_distance,
                math.sin(uav_yaw_ned) * args.follow_distance,
                0.0,
            ], dtype=np.float64)
            cam_map = airsim_ned_to_map(cam_ned)
            look_map = airsim_ned_to_map(look_ned)
            log = r.place_camera_map(cam_map, look_map, args.camera_pitch_deg)
            uav_records.append({"t": rec["t"], "pos": [float(x) for x in cam_ned],
                                "heading": uav_yaw_ned})
        else:
            if args.camera_mode == "expert" and tick in vp_by_idx:
                desired_cam_ned = vp_by_idx[tick]
            else:
                desired_cam_ned = _chase_estimate_ned(
                    rec["pos"], rec["mdir_ned"], args.follow_distance,
                    args.follow_altitude)
            # (a) EMA low-pass → smooth, non-jittery UAV path.
            smooth_cam = smooth_cam + alpha * (desired_cam_ned - smooth_cam)
            cam_target = smooth_cam.copy()
            # (b) optional grid quantisation → per-step moves are integer
            #     multiples of a fixed quantum (discrete-action friendly).
            if args.camera_quant_m > 0:
                q = args.camera_quant_m
                cam_target = np.round(cam_target / q) * q
            # (c) collision-safe final (lift above roof / out of buildings).
            cam_ned = occ.resolve_drone_ned(prev_cam_ned, cam_target)
            prev_cam_ned = cam_ned.copy()
            cam_map = airsim_ned_to_map(cam_ned)
            log = r.place_camera_map(cam_map, tmap, args.camera_pitch_deg)
            # UAV heading in NED facing the target (for bench pose record).
            uav_yaw_ned = math.atan2(rec["pos"][1] - cam_ned[1],
                                     rec["pos"][0] - cam_ned[0])
            uav_records.append({"t": rec["t"], "pos": [float(x) for x in cam_ned],
                                "heading": float(uav_yaw_ned)})

        img_path = frames_dir / f"frame_{tick:04d}_rgb.png"
        r.capture(img_path, kind="lit")

        # ---- standardized per-frame visibility + FrameMetadata -------------
        eye_agl = max(float(args.follow_altitude), float(-cam_ned[2]))
        uav_pose = [float(cam_ned[0]), float(cam_ned[1]), float(cam_ned[2]),
                    float(uav_yaw_ned)]
        target_pose = [float(rec["pos"][0]), float(rec["pos"][1]),
                       float(rec["pos"][2]), float(rec["heading"])]
        vstd = vis_eval.evaluate_frame(
            uav_pose=uav_pose, target_pose=target_pose,
            camera_config=cam_cfg.to_dict(),
            scene_context={"occupancy": occ,
                           "max_range_m": float(args.vis_max_range_m),
                           "drone_eye_agl_m": eye_agl})
        if vstd.get("in_camera_frustum"):
            in_frustum += 1
        fm = FrameMetadata(
            frame_id=tick, image_path=str(img_path), step_index=tick,
            timestamp=float(rec["t"]), uav_pose=uav_pose, target_pose=target_pose,
            uav_velocity=[0.0, 0.0, 0.0],
            target_velocity=[float(v) for v in rec["vel"]],
            target_visible=bool(vstd["target_visible"]),
            in_camera_frustum=vstd["in_camera_frustum"],
            line_of_sight_clear=vstd["line_of_sight_clear"],
            visibility_score=vstd["visibility_score"],
            distance_to_target=float(vstd["distance_to_target"]),
            relative_bearing=float(vstd["relative_bearing"]),
            occlusion_risk=vstd.get("occlusion_risk"),
            selected_viewpoint=[float(x) for x in cam_ned],
            collision=False, target_behavior_type=behavior,
            difficulty_level=difficulty,
            extra={"maneuver": maneuver,
                   "tracker_mode": getattr(args, "tracker_mode", None),
                   "visibility_source": vstd.get("visibility_source")})
        append_frame_jsonl(fm, frames_jsonl)

    frustum_frac = in_frustum / max(1, total)

    # ---- write the rest of the bench bundle --------------------------------
    expert_out["uav_trajectory"] = uav_records
    expert_out["episode"] = {
        "episode_id": episode_id, "env": args.env, "car": car,
        "target_behavior": behavior, "difficulty": difficulty,
        "tick_hz": args.tick_hz, "duration_s": args.duration,
        "camera_mode": args.camera_mode,
        "target_in_frustum_frac": round(frustum_frac, 3)}
    save_trajectories(expert_out, out_dir / "trajectories.json")
    write_instruction_json(instr, out_dir / "instruction.json")

    t0 = target_records[0]
    meta = EpisodeMetadata(
        episode_id=episode_id, scene_id=args.env, difficulty_level=difficulty,
        target_behavior_type=behavior, target_class=target_class,
        instruction=str(instr["instruction"]), random_seed=int(seed),
        max_steps=int(total), camera_config=cam_cfg,
        uav_initial_pose=uav_records[0]["pos"] + [uav_records[0]["heading"]],
        target_initial_pose=[float(t0["pos"][0]), float(t0["pos"][1]),
                             float(t0["pos"][2]), float(t0["heading"])],
        environment_summary={"env": args.env, "maneuver": maneuver},
        occluder_summary={"source": "pcd_occupancy"})
    save_metadata_json(meta, out_dir / "metadata.json")

    _write_config_yaml(out_dir / "config.yaml", args, behavior, difficulty,
                       seed, idx, episode_id)

    metrics = {}
    try:
        metrics = evaluate_episode_dir(out_dir, difficulty=difficulty, write=True)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] metrics failed: {e}", flush=True)

    if args.make_mp4:
        _render_mp4(frames_dir, out_dir / "chase.mp4", args.tick_hz)
    flag = "OK" if frustum_frac >= 0.5 else "LOW(<50%)"
    print(f"[done] {episode_id}  frustum={frustum_frac*100:.0f}%[{flag}] "
          f"success={metrics.get('tracking_success')} → {out_dir}", flush=True)
    return {"episode_dir": str(out_dir), "episode_id": episode_id, "seed": int(seed),
            "behavior": behavior, "difficulty": difficulty,
            "n_frames": total, "frustum_frac": round(frustum_frac, 3),
            "tracking_success": metrics.get("tracking_success")}


def run(args) -> int:
    beh = args.behaviors or args.target_behavior
    dif = args.difficulties or args.target_policy_difficulty
    behaviors = (list(BEHAVIOR_TYPES) if beh == "all"
                 else [b.strip() for b in beh.split(",") if b.strip()])
    difficulties = [d.strip() for d in dif.split(",") if d.strip()]
    for b in behaviors:
        if b not in BEHAVIOR_TYPES:
            print(f"[ERR] unknown behavior {b!r}; choose from {BEHAVIOR_TYPES}")
            return 2

    print("[..] loading occupancy (UE-aware; first build ~3 min, then cached)...",
          flush=True)
    # Strict occupancy (real buildings block; curb layer kept) so the route
    # follower stays on genuine wide streets and never enters buildings.
    # --building-min-h can still override the yaml threshold if needed.
    occ = PcdOccupancyMap.load_or_build(
        REPO_ROOT, env_name=args.env, min_height_thresh=args.building_min_h)
    print(f"[ok] occupancy ready (min_h={args.building_min_h or 'yaml'})",
          flush=True)

    # Connect ONCE for the whole batch (robust connect handles a held slot).
    r = UnrealCVRenderer(args.ue_ip, args.ue_port, args.connect_timeout)
    r.connect()
    r.setup_camera(args.width, args.height)
    print(f"[ok] connected + camera {args.width}x{args.height}", flush=True)

    try:
        picked = _select_controllable_car(r, args)
        if picked is None:
            print("[FAIL] no controllable car found (all Mass/AI). Try --target a "
                  "parked car, or spawn a dedicated actor.")
            return 2
        car, start_ue = picked

        # Hide addressable vehicles (re-hidden during render to catch traffic
        # that City Sample actorizes as the camera moves).
        hidden: set[str] = set()
        n0 = _hide_vehicles(r, car, hidden, args, near_ue=start_ue)
        print(f"[ok] hid {n0} vehicles initially", flush=True)

        # Optional multi-tracker comparison in ONE connection (UnrealCV only
        # allows a single client slot, so we must NOT spawn a second process).
        # Each tracker mode renders the same seeded target route as its own
        # episode dir (…_<mode>), guaranteeing an identical car trajectory.
        if getattr(args, "tracker_modes", None):
            tracker_modes = [m.strip() for m in args.tracker_modes.split(",")
                             if m.strip()]
        else:
            tracker_modes = [getattr(args, "tracker_mode", None)]

        combos = [(b, d) for b in behaviors for d in difficulties]
        n_total = len(combos) * args.num_episodes * len(tracker_modes)
        print(f"[batch] {len(combos)} combo(s) × {args.num_episodes} seed(s) × "
              f"{len(tracker_modes)} tracker(s) = {n_total} episodes", flush=True)
        manifest: list[dict] = []
        try:
            for b, d in combos:
                for ep in range(args.num_episodes):
                    seed = int(args.seed) + ep
                    for tmode in tracker_modes:
                        args.tracker_mode = tmode
                        info = generate_episode(r, occ, car, start_ue, args,
                                                b, d, seed, ep, hidden)
                        manifest.append(info)
        finally:
            r.set_object_location_ue_cm(car, *start_ue)
            for name in hidden:
                r.show(name)

        import json
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "batch_manifest.json").write_text(json.dumps({
            "scene_id": args.env, "backend": "unrealcv",
            "behaviors": behaviors, "difficulties": difficulties,
            "base_seed": int(args.seed), "num_episodes": int(args.num_episodes),
            "output_dir": str(args.out), "episodes": manifest,
            "total": len(manifest)}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print("\n" + "=" * 60)
        print(f"BATCH DONE — {len(manifest)} episodes under {args.out}")
        print(f"  manifest: {args.out / 'batch_manifest.json'}")
        print("=" * 60)
        return 0
    finally:
        r.close()


def _write_config_yaml(path: Path, args, behavior: str, difficulty: str,
                       seed: int, idx: int, episode_id: str) -> None:
    import yaml
    cfg = {
        "scene_id": args.env, "difficulty": difficulty, "behavior": behavior,
        "seed": int(seed), "episode_index": int(idx), "episode_id": episode_id,
        "output_dir": str(args.out), "backend": "unrealcv",
        "render": {
            "tick_hz": float(args.tick_hz), "duration_s": float(args.duration),
            "num_frames": int(args.duration * args.tick_hz),
            "follow_distance_m": float(args.follow_distance),
            "follow_altitude_m": float(args.follow_altitude),
            "camera_hfov_deg": float(args.camera_hfov_deg),
            "camera_pitch_deg": float(args.camera_pitch_deg),
            "camera_mode": args.camera_mode,
            "image_width": int(args.width), "image_height": int(args.height),
            "route_len_m": float(args.route_len_m),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _render_mp4(frames_dir: Path, out_path: Path, fps: float) -> None:
    import shutil
    import subprocess
    if shutil.which("ffmpeg") is None:
        print("[warn] ffmpeg not found; skipping mp4")
        return
    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-pattern_type", "glob",
           "-i", str(frames_dir / "frame_*.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[ok] mp4 → {out_path}")
    except subprocess.CalledProcessError as e:
        print(f"[warn] ffmpeg failed: {e.stderr[:300] if e.stderr else e}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="env_ue_smallcity")
    p.add_argument("--ue-ip", default="127.0.0.1")
    p.add_argument("--ue-port", type=int, default=9000)
    p.add_argument("--connect-timeout", type=float, default=10.0)
    p.add_argument("--target", default=None, help="Exact car actor name.")
    p.add_argument("--max-try", type=int, default=15)
    p.add_argument("--settle-s", type=float, default=0.3)
    p.add_argument("--target-behavior", choices=list(BEHAVIOR_TYPES),
                   default="detour_feint")
    p.add_argument("--target-policy-difficulty",
                   choices=["easy", "medium", "hard"], default="medium")
    p.add_argument("--behaviors", default=None,
                   help="Batch: comma list of behaviors or 'all' (connects once, "
                        "loops). Default: just --target-behavior.")
    p.add_argument("--difficulties", default=None,
                   help="Batch: comma list of difficulties. "
                        "Default: just --target-policy-difficulty.")
    p.add_argument("--max-speed-mps", type=float, default=6.0,
                   help="(unused with route follower; speed set by difficulty)")
    p.add_argument("--route-len-m", type=float, default=240.0,
                   help="Planned road route length (m).")
    p.add_argument("--road-search-m", type=float, default=180.0,
                   help="Radius to search for a major-road seed near spawn.")
    p.add_argument("--seg-building-jsonl", type=Path, default=None,
                   help="Annotated buildings JSONL (for --target-behavior "
                        "alley_hutong). Defaults to "
                        "scene_data/seg_map/<env>.jsonl.")
    p.add_argument("--open-approach-m", type=float, default=35.0,
                   help="alley_hutong: open road before the hutong entry (m).")
    p.add_argument("--max-corridor-width-m", type=float, default=12.0,
                   help="alley_hutong: max corridor width of the gap (m).")
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--tick-hz", type=float, default=5.0)
    p.add_argument("--follow-distance", type=float, default=12.0)
    p.add_argument("--follow-altitude", type=float, default=14.0)
    p.add_argument("--camera-pitch-deg", type=float, default=55.0)
    p.add_argument("--camera-mode", choices=["expert", "chase"], default="expert",
                   help="expert=render UAV along the visibility-aware expert "
                        "viewpoints (follows target, anticipates occlusion); "
                        "chase=simple behind+above follower. Default expert.")
    p.add_argument("--camera-hfov-deg", type=float, default=70.0,
                   help="Camera HFOV used for expert visibility scoring.")
    p.add_argument("--vis-max-range-m", type=float, default=100.0,
                   help="Max range for expert visibility scoring.")
    p.add_argument("--tracker-mode", choices=list(CLOSED_LOOP_TRACKERS),
                   default=None,
                   help="Drive the UAV with a closed-loop tracker (the SAME "
                        "implementations as the env_airsim_16 pipeline) instead "
                        "of expert/chase viewpoint placement: adaptive=FlySeek "
                        "FSM (PREDICT/PEEK/REACQUIRE), reactive_lost=baseline "
                        "that stalls + wanders aimlessly on occlusion, "
                        "reactive=plain follower, inline=TRACK+SEARCH. Overrides "
                        "--camera-mode when set.")
    p.add_argument("--tracker-modes", default=None,
                   help="Comma list of closed-loop trackers to render in ONE "
                        "connection over the SAME seeded route, e.g. "
                        "'adaptive,reactive_lost' (FlySeek vs baseline). Each "
                        "writes its own episode dir (…_<mode>). Avoids the "
                        "UnrealCV single-client-slot reconnect failure.")
    p.add_argument("--drone-smoothing", type=float, default=4.0,
                   help="Closed-loop tracker: position controller gain (1/s).")
    p.add_argument("--tracker-yaw-gain", type=float, default=3.0,
                   help="Closed-loop tracker: yaw controller gain (1/s).")
    p.add_argument("--lost-after-s", type=float, default=0.6,
                   help="Closed-loop tracker: seconds w/o LoS before 'lost'.")
    p.add_argument("--lost-wander-radius-m", type=float, default=6.0,
                   help="reactive_lost baseline: loiter radius (m) when lost.")
    p.add_argument("--lost-wander-scan-dps", type=float, default=35.0,
                   help="reactive_lost baseline: yaw sweep rate (deg/s) wandering.")
    p.add_argument("--no-collision", action="store_true",
                   help="Disable PCD collision clamping of the UAV pose.")
    p.add_argument("--camera-smooth-tau", type=float, default=0.6,
                   help="EMA time constant (s) for smooth UAV motion. "
                        "Larger = smoother/laggier. Default 0.6.")
    p.add_argument("--camera-quant-m", type=float, default=0.0,
                   help="If >0, snap UAV position to this grid (m) so each "
                        "step is a discrete quantum (discrete-action friendly).")
    p.add_argument("--building-min-h", type=float, default=None,
                   help="Building height threshold (m). UE default 30 (the "
                        "yaml's 6 over-blocks curbs). airsim: keep yaml.")
    p.add_argument("--keep-car-obstacles", action="store_true",
                   help="Keep the curb/rail layer (UE drops it by default).")
    p.add_argument("--lock-ground-z", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Lock the car to its spawn road-plane Z (no climbing "
                        "onto rooftops). Default on.")
    p.add_argument("--hide-others", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Hide all other addressable vehicles (default on). "
                        "Use --no-hide-others to keep them.")
    p.add_argument("--hide-radius-m", type=float, default=0.0,
                   help="If >0, only hide vehicles within this radius of the "
                        "current target position (else hide all addressable).")
    p.add_argument("--rehide-every", type=int, default=10,
                   help="Re-hide newly-actorized traffic every N render ticks "
                        "(0=only once at start). Catches City Sample LOD pop-in.")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--seed", type=int, default=42,
                   help="Base seed; episode i uses seed+i (varied routes).")
    p.add_argument("--num-episodes", type=int, default=1,
                   help="Episodes per (behavior×difficulty), each a new seed.")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "flyseek_extend" / "output" / "bench_ue",
                   help="Bench output dir (episode subdirs + batch_manifest.json).")
    p.add_argument("--make-mp4", action="store_true", default=True)
    p.add_argument("--no-mp4", action="store_true")
    p.add_argument("--hard-timeout", type=float, default=7200.0,
                   help="Global watchdog (s). Raise for large batches.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.no_mp4:
        args.make_mp4 = False
    if getattr(args, "seg_building_jsonl", None) is None:
        args.seg_building_jsonl = (
            REPO_ROOT / "scene_data" / "seg_map" / f"{args.env}.jsonl")

    def _on_timeout(_s, _f):
        print(f"\n[FATAL] demo exceeded {args.hard_timeout}s.", flush=True)
        sys.exit(124)
    signal.signal(signal.SIGALRM, _on_timeout)
    signal.alarm(int(args.hard_timeout))

    print("=" * 60)
    print(f"FlySeek UnrealCV chase demo — {args.env}")
    print("=" * 60)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())

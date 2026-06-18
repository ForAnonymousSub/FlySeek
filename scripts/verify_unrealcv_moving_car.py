# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Visual verification: teleport a REAL City Sample car under a top-down camera.

The capability probe (``probe_unrealcv_objects.py``) confirmed UnrealCV can list /
read / move / hide / spawn actors on env_ue_smallcity (verdict GO). But that only
proves the *API* works — it moved an ``InitializeVehicles_C`` manager actor, not a
visible car, and it did not render anything. Before building the full chase demo
we must confirm three things *visually*:

    1. A real ``BP_vehCar_*`` actor is actually visible when rendered.
    2. Our ``vset /object/<name>/location`` teleport STICKS (i.e. the car is not a
       Mass/AI-driven traffic car that immediately overrides our pose).
    3. A drone-style top-down camera placed above the car renders it centered.

This script auto-selects a *controllable* car (teleports a candidate and checks
the pose sticks), then drives it in a straight line over N steps, snapping a
top-down ``lit`` frame each step. Eyeball ``frames/`` (or the mp4) to confirm a
real car visibly moves across the city under our control.

Everything here is in **UE world centimetres** (UnrealCV's native frame) — we do
NOT involve the PCD/NED frame yet; that conversion comes in the render adapter.

Run it against a STANDALONE sim (not env_bridge — single client slot):

    # terminal 1
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    bash envs/ue/env_ue_smallcity/CitySample.sh   # or with -RenderOffScreen

    # terminal 2
    conda activate openfly-latest
    python flyseek_extend/scripts/verify_unrealcv_moving_car.py --env env_ue_smallcity
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT = REPO_ROOT / "flyseek_extend" / "output" / "unrealcv_verify"

# Real, individually-addressable vehicle bodies (exclude spawners/managers).
CAR_INCLUDE = ("BP_vehCar", "BP_vehVan", "BP_vehTruck")
CAR_EXCLUDE = ("Spawner", "Initialize", "MassTraffic")


def _as_text(resp) -> str:
    if resp is None:
        return ""
    if isinstance(resp, bytes):
        return resp.decode("utf-8", errors="replace")
    return str(resp)


def _is_err(text: str) -> bool:
    t = text.strip().lower()
    return (t == "") or t.startswith("error")


def _floats(text: str) -> list[float]:
    out: list[float] = []
    for tok in text.replace(",", " ").split():
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def _get_location(client, name: str) -> list[float] | None:
    vals = _floats(_as_text(client.request(f"vget /object/{name}/location")))
    return vals[:3] if len(vals) >= 3 else None


def _set_location(client, name: str, x: float, y: float, z: float) -> bool:
    r = _as_text(client.request(
        f"vset /object/{name}/location {x:.1f} {y:.1f} {z:.1f}"))
    return not _is_err(r)


def _setup_camera(client, width: int, height: int) -> str:
    """Spawn a capture camera; return the camera id string to use."""
    client.request("vset /cameras/spawn")
    time.sleep(0.5)
    for cam in ("1", "0"):
        r = _as_text(client.request(f"vset /camera/{cam}/size {width} {height}"))
        if not _is_err(r):
            return cam
    return "1"


def _place_topdown(client, cam: str, x: float, y: float, z_cm: float,
                   pitch_deg: float, yaw_deg: float) -> None:
    client.request(f"vset /camera/{cam}/location {x:.1f} {y:.1f} {z_cm:.1f}")
    client.request(
        f"vset /camera/{cam}/rotation {-abs(pitch_deg):.1f} {yaw_deg:.1f} 0")


def _capture(client, cam: str, out_path: Path) -> bool:
    data = client.request(f"vget /camera/{cam}/lit png")
    if not isinstance(data, (bytes, bytearray)) or len(data) < 100:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return True


def _find_controllable_car(client, args) -> tuple[str, list[float]] | None:
    objs = _as_text(client.request("vget /objects")).split()
    cands = [
        o for o in objs
        if any(k in o for k in CAR_INCLUDE) and not any(k in o for k in CAR_EXCLUDE)
    ]
    if args.target:
        cands = [args.target] + [c for c in cands if c != args.target]
    print(f"[info] {len(cands)} candidate car bodies; testing controllability...")
    tested = 0
    for name in cands:
        if tested >= args.max_try:
            break
        loc = _get_location(client, name)
        if loc is None or all(abs(v) < 1e-6 for v in loc):
            continue
        tested += 1
        probe_x = loc[0] + args.step_cm
        if not _set_location(client, name, probe_x, loc[1], loc[2]):
            continue
        time.sleep(args.settle_s)
        after = _get_location(client, name)
        if after is None:
            continue
        drift = math.hypot(after[0] - probe_x, after[1] - loc[1])
        if drift < args.stick_tol_cm:
            # Restore and accept this car.
            _set_location(client, name, loc[0], loc[1], loc[2])
            time.sleep(args.settle_s)
            print(f"[ok] controllable car: {name} @ "
                  f"({loc[0]:.0f},{loc[1]:.0f},{loc[2]:.0f}) cm "
                  f"(teleport stuck, drift={drift:.0f}cm)")
            return name, loc
        else:
            print(f"  [skip] {name}: teleport overridden (drift={drift:.0f}cm) "
                  "— likely Mass/AI traffic")
            _set_location(client, name, loc[0], loc[1], loc[2])
    return None


def run(args) -> int:
    try:
        from unrealcv import Client  # type: ignore
    except ImportError as e:
        print(f"[ERR] unrealcv not importable: {e} (activate openfly-latest)")
        return 1

    print(f"[..] connecting to UnrealCV {args.ue_ip}:{args.ue_port} ...", flush=True)
    client = Client((args.ue_ip, args.ue_port))
    if not client.connect(timeout=args.connect_timeout):
        print(f"[ERR] cannot connect to UnrealCV {args.ue_ip}:{args.ue_port}. "
              "Sim up? No other client holding the slot?")
        return 1
    print("[ok] connected", flush=True)

    try:
        cam = _setup_camera(client, args.width, args.height)
        print(f"[ok] capture camera = '{cam}' @ {args.width}x{args.height}")

        found = _find_controllable_car(client, args)
        if found is None:
            print("\n[FAIL] no controllable BP_vehCar found (all candidates were "
                  "Mass/AI-overridden or unreadable). Options: try --target with a "
                  "parked-car name, or we spawn our own actor instead.")
            return 2
        car, base = found

        out_dir = args.out / time.strftime("%Y%m%d_%H%M%S")
        frames_dir = out_dir / "frames"
        alt_cm = args.altitude_m * 100.0

        print(f"\n[loop] driving {car} +X over {args.steps} steps "
              f"({args.step_cm/100:.1f} m/step), top-down @ {args.altitude_m} m\n"
              f"       output → {out_dir}")
        overridden_steps = 0
        for i in range(args.steps):
            tx = base[0] + i * args.step_cm
            ty = base[1]
            tz = base[2]
            _set_location(client, car, tx, ty, tz)
            time.sleep(args.settle_s)
            cur = _get_location(client, car) or [tx, ty, tz]
            drift = math.hypot(cur[0] - tx, cur[1] - ty)
            if drift > args.stick_tol_cm:
                overridden_steps += 1
            # Top-down camera centered above the car's *commanded* position.
            _place_topdown(client, cam, tx, ty, tz + alt_cm,
                           args.pitch_deg, args.yaw_deg)
            ok = _capture(client, cam, frames_dir / f"frame_{i:03d}.png")
            print(f"  step {i:02d}: car_xy=({cur[0]:.0f},{cur[1]:.0f}) "
                  f"drift={drift:.0f}cm  frame={'ok' if ok else 'FAIL'}")

        # Restore original pose.
        _set_location(client, car, base[0], base[1], base[2])

        print("\n" + "=" * 60)
        print(f"VERIFY DONE — frames in {frames_dir}")
        print(f"  car                 : {car}")
        print(f"  steps overridden    : {overridden_steps}/{args.steps} "
              f"(0 = teleport fully controlled it)")
        if overridden_steps > args.steps // 2:
            print("  ⚠ this car fights our teleport (Mass/AI). Try --target a "
                  "parked car, or switch to spawning our own actor.")
        else:
            print("  ✓ teleport controls this car — good to build the chase demo.")
        if args.make_mp4:
            _render_mp4(frames_dir, out_dir / "verify.mp4", args.fps)
        print("=" * 60)
        return 0
    finally:
        try:
            client.disconnect()
        except Exception:
            pass


def _render_mp4(frames_dir: Path, out_path: Path, fps: float) -> None:
    import shutil
    import subprocess
    if shutil.which("ffmpeg") is None:
        print("  [warn] ffmpeg not found; skipping mp4")
        return
    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-pattern_type", "glob",
           "-i", str(frames_dir / "frame_*.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  [ok] mp4 → {out_path}")
    except subprocess.CalledProcessError as e:
        print(f"  [warn] ffmpeg failed: {e.stderr[:300] if e.stderr else e}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="env_ue_smallcity", help="(label only)")
    p.add_argument("--ue-ip", default="127.0.0.1")
    p.add_argument("--ue-port", type=int, default=9000)
    p.add_argument("--connect-timeout", type=float, default=10.0)
    p.add_argument("--target", default=None,
                   help="Exact car actor name to use (skips auto-select).")
    p.add_argument("--max-try", type=int, default=15,
                   help="How many candidate cars to test for controllability.")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--step-cm", type=float, default=500.0,
                   help="Per-step +X teleport distance (UE cm). 500 = 5 m.")
    p.add_argument("--settle-s", type=float, default=0.4,
                   help="Wait after teleport before readback/capture.")
    p.add_argument("--stick-tol-cm", type=float, default=150.0,
                   help="Max drift to consider the teleport 'stuck' (controllable).")
    p.add_argument("--altitude-m", type=float, default=25.0,
                   help="Top-down camera height above the car (m).")
    p.add_argument("--pitch-deg", type=float, default=90.0,
                   help="Camera downward pitch (90 = straight down).")
    p.add_argument("--yaw-deg", type=float, default=0.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--make-mp4", action="store_true", default=True)
    p.add_argument("--no-mp4", action="store_true")
    p.add_argument("--fps", type=float, default=4.0)
    p.add_argument("--hard-timeout", type=float, default=180.0)
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.no_mp4:
        args.make_mp4 = False

    def _on_timeout(_s, _f):
        print(f"\n[FATAL] exceeded {args.hard_timeout}s — UnrealCV not responding.",
              flush=True)
        sys.exit(124)
    signal.signal(signal.SIGALRM, _on_timeout)
    signal.alarm(int(args.hard_timeout))

    print("=" * 60)
    print(f"UnrealCV moving-car visual verification — {args.env}")
    print("=" * 60)
    rc = run(args)
    signal.alarm(0)
    return rc


if __name__ == "__main__":
    sys.exit(main())

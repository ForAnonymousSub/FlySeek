# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Probe a running UnrealCV (City Sample) instance for object-control capability.

This is the **go/no-go feasibility test for FlySeek path B, plan 2**: rendering a
*moving* target car on the UE City Sample maps (env_ue_smallcity / env_ue_bigcity).

OpenFly's UE bridge (``scripts/sim/ue_bridge.py``) only ever moves the *camera*
(``vset /camera/...``). To make a chased car appear and move in the rendered RGB
we need UnrealCV's *object* commands. Whether this packaged City Sample build
exposes them — and whether any car-like actor is addressable / movable — can only
be answered against the live sim. That is exactly what this script checks.

It connects to UnrealCV directly (NOT through env_bridge.py, which would occupy
the single client slot), enumerates scene objects, classifies car-like names, and
empirically tests: read pose / move+restore / hide+show / mask color / spawn.

Run it like this (launch the sim standalone first, no env_bridge):

    # terminal 1 — launch the sim (headless Xvfb, UnrealCV on port 9000)
    bash envs/ue/env_ue_smallcity/CitySample.sh

    # terminal 2 — once it is up (~15 s), probe it
    conda activate openfly-latest
    python flyseek_extend/scripts/probe_unrealcv_objects.py \
        --env env_ue_smallcity --move-test

A capability report is printed and saved to
``flyseek_extend/output/assets/unrealcv_probe_<env>_<ts>.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "flyseek_extend" / "output" / "assets"

# Name heuristics for "is this a car-like actor?". City Sample uses BP_/SM_
# prefixes; we cast a wide net and let the report show what actually exists.
CAR_NAME_PATTERNS = [
    r"[Vv]ehicle",
    r"[Cc]ar(?![a-z])",
    r"[Ss]edan",
    r"[Ss]UV",
    r"[Tt]ruck",
    r"[Bb]us(?![a-z])",
    r"[Tt]axi",
    r"[Vv]an(?![a-z])",
    r"[Cc]oupe",
    r"[Hh]atchback",
    r"BP_.*Veh",
]
_CAR_RE = re.compile("|".join(CAR_NAME_PATTERNS))


@dataclass
class CapabilityResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class ProbeReport:
    timestamp: str
    env: str
    ue_ip: str
    ue_port: int
    connected: bool = False
    status: str = ""
    objects_total: int = 0
    car_candidates_total: int = 0
    car_candidates_sample: list[str] = field(default_factory=list)
    capabilities: list[dict[str, Any]] = field(default_factory=list)
    moved_object: str | None = None
    move_delta_cm: list[float] | None = None
    verdict: str = ""
    errors: list[str] = field(default_factory=list)


def _as_text(resp: Any) -> str:
    if resp is None:
        return ""
    if isinstance(resp, bytes):
        try:
            return resp.decode("utf-8", errors="replace")
        except Exception:
            return ""
    return str(resp)


def _is_err(text: str) -> bool:
    t = text.strip().lower()
    return (t == "") or t.startswith("error")


def _parse_floats(text: str) -> list[float]:
    out: list[float] = []
    for tok in text.replace(",", " ").split():
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def _add_cap(report: ProbeReport, name: str, ok: bool, detail: str = "") -> None:
    report.capabilities.append(asdict(CapabilityResult(name=name, ok=ok, detail=detail)))
    flag = "OK " if ok else "FAIL"
    print(f"  [{flag}] {name}: {detail}")


def run_probe(args: argparse.Namespace) -> ProbeReport:
    report = ProbeReport(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        env=args.env,
        ue_ip=args.ue_ip,
        ue_port=args.ue_port,
    )

    try:
        from unrealcv import Client  # type: ignore
    except ImportError as e:
        report.errors.append(f"unrealcv not importable: {e} "
                             "(activate the openfly-latest conda env)")
        return report

    # NOTE: do NOT pre-open a raw TCP socket to 9000 — UnrealCV only serves one
    # client and a connect+close cycle wedges its accept loop, which then hangs
    # the real handshake below. We connect exactly like OpenFly's ue_bridge.py.
    print(f"[..] connecting to UnrealCV {args.ue_ip}:{args.ue_port} ...",
          flush=True)
    client = Client((args.ue_ip, args.ue_port))
    if not client.connect(timeout=args.connect_timeout):
        report.errors.append(
            f"could not connect to UnrealCV at {args.ue_ip}:{args.ue_port}. "
            "Is the sim up and serving on this port (check unrealcv.ini Port=)? "
            "Make sure no other client (env_bridge / a previous probe) holds the "
            "single UnrealCV client slot."
        )
        return report
    report.connected = True
    print(f"[ok] connected to UnrealCV {args.ue_ip}:{args.ue_port}", flush=True)

    try:
        report.status = _as_text(client.request("vget /unrealcv/status"))

        # ---- 1. enumerate objects -----------------------------------------
        objs_text = _as_text(client.request("vget /objects"))
        objects = objs_text.split()
        report.objects_total = len(objects)
        _add_cap(report, "list_objects (vget /objects)",
                 report.objects_total > 0,
                 f"{report.objects_total} objects")

        cars = [o for o in objects if _CAR_RE.search(o)]
        report.car_candidates_total = len(cars)
        report.car_candidates_sample = cars[: args.max_sample]
        _add_cap(report, "car-like actors found", len(cars) > 0,
                 f"{len(cars)} matched; sample={cars[:8]}")

        # Probe targets: prefer car candidates, else fall back to first objects
        probe_names = (cars[: args.max_sample]
                       or objects[: args.max_sample])

        # ---- 2. read pose of candidates -----------------------------------
        readable: list[tuple[str, list[float]]] = []
        for name in probe_names:
            loc = _as_text(client.request(f"vget /object/{name}/location"))
            if not _is_err(loc):
                vals = _parse_floats(loc)
                if len(vals) >= 3:
                    readable.append((name, vals[:3]))
        _add_cap(report, "read object location", len(readable) > 0,
                 f"{len(readable)}/{len(probe_names)} readable")

        # ---- 3. mask color (for segmentation GT) --------------------------
        if readable:
            cname = readable[0][0]
            col = _as_text(client.request(f"vget /object/{cname}/color"))
            _add_cap(report, "read object mask color", not _is_err(col),
                     f"{cname} -> {col!r}")

        # ---- 4. hide / show ------------------------------------------------
        if readable and args.visibility_test:
            cname = readable[0][0]
            h = _as_text(client.request(f"vset /object/{cname}/hide"))
            time.sleep(0.2)
            s = _as_text(client.request(f"vset /object/{cname}/show"))
            ok = (not _is_err(h)) and (not _is_err(s))
            _add_cap(report, "hide/show object", ok,
                     f"{cname} hide={h!r} show={s!r}")

        # ---- 5. MOVE + restore (the decisive test) ------------------------
        if readable and args.move_test:
            cname, loc0 = readable[0]
            target = [loc0[0] + args.move_cm, loc0[1], loc0[2]]
            set_resp = _as_text(client.request(
                f"vset /object/{cname}/location {target[0]} {target[1]} {target[2]}"
            ))
            time.sleep(0.3)
            loc1 = _parse_floats(_as_text(
                client.request(f"vget /object/{cname}/location")))
            moved = False
            delta = None
            if len(loc1) >= 3:
                delta = [round(loc1[i] - loc0[i], 1) for i in range(3)]
                moved = abs(delta[0]) > 0.5 * abs(args.move_cm)
            # restore
            client.request(
                f"vset /object/{cname}/location {loc0[0]} {loc0[1]} {loc0[2]}"
            )
            report.moved_object = cname if moved else None
            report.move_delta_cm = delta
            _add_cap(report, "MOVE object (vset location) + readback", moved,
                     f"{cname} set={set_resp!r} delta_cm={delta}")

        # ---- 6. spawn (optional, best-effort) -----------------------------
        if args.spawn_test:
            spawn_resp = _as_text(client.request(
                "vset /objects/spawn StaticMeshActor flyseek_probe_spawn"))
            ok = not _is_err(spawn_resp)
            _add_cap(report, "spawn object (vset /objects/spawn)", ok,
                     f"resp={spawn_resp!r}")
            if ok:
                client.request("vset /object/flyseek_probe_spawn/destroy")

    except Exception as e:
        report.errors.append(f"probe raised: {e}")
    finally:
        try:
            client.disconnect()
        except Exception:
            pass

    # ---- verdict ----------------------------------------------------------
    caps = {c["name"]: c["ok"] for c in report.capabilities}
    can_move = bool(report.moved_object) if args.move_test else None
    if can_move:
        report.verdict = (
            "GO (plan 2 feasible): a car-like actor is addressable AND movable "
            "via UnrealCV — we can teleport it along the adversary trajectory "
            "so the chased car renders for real."
        )
    elif args.move_test and caps.get("car-like actors found"):
        report.verdict = (
            "PARTIAL: car actors exist but the move test did not change the "
            "pose (likely Mass/AI-driven traffic, not a settable Actor). Try a "
            "parked/static car name, or fall back to plan 1/3."
        )
    elif args.move_test:
        report.verdict = (
            "NO-GO for plan 2 as-is: no movable car actor found. Consider "
            "spawning our own mesh (--spawn-test) or fall back to plan 1/3."
        )
    else:
        report.verdict = ("inventory-only run (pass --move-test for the "
                          "decisive movability check).")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="env_ue_smallcity",
                   help="Scene env (for the report/output filename only).")
    p.add_argument("--ue-ip", default="127.0.0.1")
    p.add_argument("--ue-port", type=int, default=9000,
                   help="UnrealCV port (unrealcv.ini default 9000).")
    p.add_argument("--connect-timeout", type=float, default=10.0)
    p.add_argument("--hard-timeout", type=float, default=45.0,
                   help="Abort the whole probe after N seconds (watchdog).")
    p.add_argument("--max-sample", type=int, default=30,
                   help="How many candidate objects to pose-probe.")
    p.add_argument("--move-cm", type=float, default=500.0,
                   help="X offset (UE cm) for the move test. Default 500 (=5 m).")
    p.add_argument("--move-test", action="store_true",
                   help="Run the decisive move+restore test (recommended).")
    p.add_argument("--visibility-test", action="store_true",
                   help="Test hide/show on a candidate.")
    p.add_argument("--spawn-test", action="store_true",
                   help="Best-effort: try spawning a StaticMeshActor.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p


def main() -> int:
    args = build_parser().parse_args()
    print("=" * 64)
    print(f"UnrealCV object-capability probe — {args.env}")
    print("=" * 64)

    # Hard watchdog: the unrealcv handshake / a wedged server can block the
    # connect indefinitely. Never let the probe hang silently.
    import signal

    def _on_timeout(_sig, _frm):
        print(f"\n[FATAL] probe exceeded {args.hard_timeout}s — UnrealCV likely "
              "not responding (handshake stuck / slot held / sim busy). "
              "Restart the sim standalone and retry.", flush=True)
        sys.exit(124)

    signal.signal(signal.SIGALRM, _on_timeout)
    signal.alarm(int(args.hard_timeout))

    report = run_probe(args)
    signal.alarm(0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output_dir / f"unrealcv_probe_{args.env}_{ts}.json"
    out_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False),
                        encoding="utf-8")

    print("\n" + "=" * 64)
    print("PROBE SUMMARY")
    print("=" * 64)
    print(f"Connected            : {report.connected}")
    print(f"Objects total        : {report.objects_total}")
    print(f"Car candidates       : {report.car_candidates_total}")
    if report.moved_object:
        print(f"Moved actor          : {report.moved_object} "
              f"(delta_cm={report.move_delta_cm})")
    print(f"\nVERDICT: {report.verdict}")
    if report.errors:
        print(f"\nErrors ({len(report.errors)}):")
        for e in report.errors:
            print(f"  - {e}")
    print(f"\nReport saved → {out_path}")
    print("=" * 64)
    return 0 if (report.connected and not report.errors) else 1


if __name__ == "__main__":
    sys.exit(main())

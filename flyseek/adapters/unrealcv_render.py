# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""UnrealCV render adapter for the FlySeek UE-city chase pipeline.

This is the UE/UnrealCV counterpart of the AirSim RPC calls used in
``demo_adversary_chase.py``. It lets the chase demo:

  * teleport a single target car along a planned trajectory (``set_object_pose``),
  * place an *angled-down* chase camera behind/above the car (``place_chase_camera``),
  * capture the drone-view RGB / object-mask / depth (``capture``),
  * hide nuisance entities along the route (``hide`` / ``show``).

Frames
------
FlySeek plans in AirSim-style **NED** (x-fwd, y-right, z-down). The PCD occupancy
lives in the OpenFly **map** frame: ``map = (nx, -ny, -nz)`` (see ``utils.coords``),
built from ``raw_pcd * pcd_scale_ratio`` (metres). UnrealCV's **UE world** is in
**centimetres** and mirrors OpenFly's bridge convention
(``scripts/sim/ue_bridge.py``)::

    ue_cm = (map_x * 100, -map_y * 100, map_z * 100)

So the full NED → UE-cm chain is ``ue = (nx*100, ny*100, -nz*100)`` — but callers
should pass **map-frame metres** (what the occupancy uses) and let this adapter do
the ``×100 + Y-flip``. Camera framing math is done directly in UE-cm (proven by
``verify_unrealcv_moving_car.py``): negative pitch looks down, yaw in UE degrees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

UE_SCALE = 100.0  # metres → UE centimetres (OpenFly pcd_scale_ratio for UE envs)


# --------------------------------------------------------------------------- #
# Frame conversions                                                           #
# --------------------------------------------------------------------------- #
def map_m_to_ue_cm(p_map: np.ndarray) -> tuple[float, float, float]:
    """OpenFly map frame (metres) → UE world (centimetres)."""
    p = np.asarray(p_map, dtype=np.float64).reshape(3)
    return (float(p[0] * UE_SCALE), float(-p[1] * UE_SCALE), float(p[2] * UE_SCALE))


def ue_cm_to_map_m(p_ue: np.ndarray) -> np.ndarray:
    """UE world (centimetres) → OpenFly map frame (metres)."""
    p = np.asarray(p_ue, dtype=np.float64).reshape(3)
    return np.array([p[0] / UE_SCALE, -p[1] / UE_SCALE, p[2] / UE_SCALE],
                    dtype=np.float64)


def map_yaw_to_ue_deg(yaw_map_rad: float) -> float:
    """Heading in the map XY plane (rad) → UE yaw (deg).

    map→UE flips Y, which negates the in-plane angle.
    """
    return float(-math.degrees(yaw_map_rad))


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


@dataclass
class ChaseCamConfig:
    follow_distance_m: float = 12.0    # behind the car, in map metres
    follow_altitude_m: float = 14.0    # above the car
    pitch_deg: float = 55.0            # downward tilt (sent as -pitch to UE)
    width: int = 1280
    height: int = 720


class UnrealCVRenderer:
    """Thin, dependency-light wrapper around an UnrealCV ``Client``."""

    def __init__(self, ip: str = "127.0.0.1", port: int = 9000,
                 connect_timeout: float = 10.0) -> None:
        self.ip = ip
        self.port = port
        self.connect_timeout = connect_timeout
        self._client = None
        self._cam = "1"

    # ----- lifecycle --------------------------------------------------------
    def connect(self) -> None:
        """Connect with a hard per-attempt timeout + retries.

        unrealcv's ``Client.connect`` ignores its ``timeout`` arg and blocks
        forever in ``ReceivePayload`` waiting for the server's ``connected``
        confirm — which never arrives if the single client slot is still held
        by a just-closed session. We run each attempt in a daemon thread and
        abandon it if it stalls, so back-to-back episodes can't hang.
        """
        import threading
        import time as _time
        from unrealcv import Client  # type: ignore

        deadline = _time.monotonic() + max(self.connect_timeout, 5.0)
        attempt = 0
        while _time.monotonic() < deadline:
            attempt += 1
            client = Client((self.ip, self.port))
            result: dict = {}

            def _do(c=client):
                try:
                    result["ok"] = c.connect()
                except Exception as e:  # noqa: BLE001
                    result["err"] = e

            th = threading.Thread(target=_do, daemon=True)
            th.start()
            th.join(timeout=4.0)
            if not th.is_alive() and result.get("ok") and client.isconnected():
                self._client = client
                return
            # Stuck (slot not yet released) or failed → abandon and retry.
            try:
                client.disconnect()
            except Exception:
                pass
            _time.sleep(1.0)
        raise ConnectionError(
            f"UnrealCV connect failed/stuck at {self.ip}:{self.port} after "
            f"{attempt} attempts (single client slot likely still held; "
            "if this persists, restart the sim)."
        )

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.disconnect()
            finally:
                self._client = None

    def __enter__(self) -> "UnrealCVRenderer":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _req(self, cmd: str):
        if self._client is None:
            raise RuntimeError("renderer not connected")
        return self._client.request(cmd)

    # ----- camera setup -----------------------------------------------------
    def setup_camera(self, width: int, height: int) -> str:
        self._req("vset /cameras/spawn")
        for cam in ("1", "0"):
            if not _is_err(_as_text(self._req(f"vset /camera/{cam}/size {width} {height}"))):
                self._cam = cam
                break
        return self._cam

    # ----- scene objects ----------------------------------------------------
    def list_objects(self) -> list[str]:
        return _as_text(self._req("vget /objects")).split()

    def get_object_location_ue_cm(self, name: str) -> np.ndarray | None:
        v = _floats(_as_text(self._req(f"vget /object/{name}/location")))
        return np.array(v[:3], dtype=np.float64) if len(v) >= 3 else None

    def set_object_pose_map(self, name: str, pos_map_m: np.ndarray,
                            yaw_map_rad: float) -> bool:
        """Teleport an actor given a map-frame (metres) pose + heading."""
        x, y, z = map_m_to_ue_cm(pos_map_m)
        yaw = map_yaw_to_ue_deg(yaw_map_rad)
        ok1 = not _is_err(_as_text(
            self._req(f"vset /object/{name}/location {x:.1f} {y:.1f} {z:.1f}")))
        ok2 = not _is_err(_as_text(
            self._req(f"vset /object/{name}/rotation 0 {yaw:.1f} 0")))
        return ok1 and ok2

    def set_object_location_ue_cm(self, name: str, x: float, y: float,
                                  z: float) -> bool:
        return not _is_err(_as_text(
            self._req(f"vset /object/{name}/location {x:.1f} {y:.1f} {z:.1f}")))

    def hide(self, name: str) -> bool:
        return not _is_err(_as_text(self._req(f"vset /object/{name}/hide")))

    def show(self, name: str) -> bool:
        return not _is_err(_as_text(self._req(f"vset /object/{name}/show")))

    # ----- chase camera -----------------------------------------------------
    def place_chase_camera(
        self,
        target_map_m: np.ndarray,
        motion_dir_map_xy: np.ndarray,
        cfg: ChaseCamConfig,
    ) -> dict:
        """Place an angled-down camera behind+above the target, facing it.

        All framing is computed in UE cm (the frame the camera lives in).
        ``motion_dir_map_xy`` is the target's unit heading in map XY; the camera
        sits ``follow_distance`` *behind* it and ``follow_altitude`` above.
        Returns a small log dict (cam UE pose).
        """
        tx, ty, tz = map_m_to_ue_cm(target_map_m)
        # Motion direction in UE XY (map→UE flips Y).
        d = np.asarray(motion_dir_map_xy, dtype=np.float64).reshape(2)
        n = float(np.linalg.norm(d))
        d = (d / n) if n > 1e-6 else np.array([1.0, 0.0])
        dir_ue = np.array([d[0], -d[1]], dtype=np.float64)  # Y flip
        back = -dir_ue
        fd_cm = cfg.follow_distance_m * UE_SCALE
        fa_cm = cfg.follow_altitude_m * UE_SCALE
        cx = tx + back[0] * fd_cm
        cy = ty + back[1] * fd_cm
        cz = tz + fa_cm
        # Yaw faces from camera toward target in UE XY.
        yaw = math.degrees(math.atan2(ty - cy, tx - cx))
        self._req(f"vset /camera/{self._cam}/location {cx:.1f} {cy:.1f} {cz:.1f}")
        self._req(
            f"vset /camera/{self._cam}/rotation {-abs(cfg.pitch_deg):.1f} {yaw:.1f} 0")
        return {"cam_ue_cm": [cx, cy, cz], "yaw_deg": yaw,
                "pitch_deg": -abs(cfg.pitch_deg)}

    def place_camera_ue(self, x: float, y: float, z: float,
                        pitch_deg: float, yaw_deg: float) -> None:
        self._req(f"vset /camera/{self._cam}/location {x:.1f} {y:.1f} {z:.1f}")
        self._req(f"vset /camera/{self._cam}/rotation {pitch_deg:.1f} {yaw_deg:.1f} 0")

    def place_camera_map(
        self,
        cam_map_m: np.ndarray,
        look_at_map_m: np.ndarray,
        pitch_deg: float,
    ) -> dict:
        """Place the camera at a map-frame point, yawed to face ``look_at``.

        The caller is responsible for collision-free ``cam_map_m`` (e.g. via
        ``PcdOccupancyMap.resolve_drone_ned``). Pitch is the downward tilt
        (sent as -|pitch| so the camera looks down).
        """
        cx, cy, cz = map_m_to_ue_cm(cam_map_m)
        lx, ly, _ = map_m_to_ue_cm(look_at_map_m)
        yaw = math.degrees(math.atan2(ly - cy, lx - cx))
        self._req(f"vset /camera/{self._cam}/location {cx:.1f} {cy:.1f} {cz:.1f}")
        self._req(
            f"vset /camera/{self._cam}/rotation {-abs(pitch_deg):.1f} {yaw:.1f} 0")
        return {"cam_ue_cm": [cx, cy, cz], "yaw_deg": yaw}

    # ----- capture ----------------------------------------------------------
    def capture(self, out_path: Path, kind: str = "lit") -> bool:
        """Capture a frame. kind ∈ {lit, object_mask, depth}."""
        if kind == "depth":
            data = self._req(f"vget /camera/{self._cam}/depth npy")
            if not isinstance(data, (bytes, bytearray)) or len(data) < 64:
                return False
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.with_suffix(".npy").write_bytes(data)
            return True
        suffix = "object_mask" if kind == "object_mask" else "lit"
        data = self._req(f"vget /camera/{self._cam}/{suffix} png")
        if not isinstance(data, (bytes, bytearray)) or len(data) < 100:
            return False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        return True


__all__ = [
    "UE_SCALE",
    "ChaseCamConfig",
    "UnrealCVRenderer",
    "map_m_to_ue_cm",
    "ue_cm_to_map_m",
    "map_yaw_to_ue_deg",
]

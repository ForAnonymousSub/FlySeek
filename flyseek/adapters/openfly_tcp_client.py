# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""OpenFly TCP client.

Connects to OpenFly's `scripts/sim/env_bridge.py` (or `airsim_bridge.py`) as a
client. Sends pose strings using the exact same wire format as the C++ planner
in `tool_ws/src/traj_gen/include/base/tcpserver.hpp`.

Wire format (bytes over TCP):
    "path:<save_dir>"                  → bridge sets the image save directory
    "x,y,z,pitch,yaw,roll"             → 6 floats, comma-separated, ASCII

The bridge treats every newline-terminated message of 1024 bytes max as one
record (see `scripts/sim/env_bridge.py:30`). Our encoding mirrors
`sendCameraPose` in `tcpserver.hpp` lines 30-66:

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << x << "," << y << "," << z << "," << pitch << "," << yaw << "," << roll;
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------- #
# Pure-function codec (testable without sockets, mirrors C++ tcpserver.hpp)    #
# ---------------------------------------------------------------------------- #


def encode_pose(
    x: float,
    y: float,
    z: float,
    pitch: float,
    yaw: float,
    roll: float,
) -> bytes:
    """Encode a 6-DoF pose into the on-wire byte string.

    Format must be byte-identical to the C++ `sendCameraPose` in
    `tool_ws/src/traj_gen/include/base/tcpserver.hpp:30-66` so that OpenFly's
    Python bridge (`receive_pose` in `airsim_bridge.py`) decodes correctly.
    """
    return (
        f"{x:.6f},{y:.6f},{z:.6f},{pitch:.6f},{yaw:.6f},{roll:.6f}"
    ).encode("utf-8")


def encode_path_prefix(save_dir: str) -> bytes:
    """Encode the `path:<dir>` control message.

    OpenFly's bridge parses this with: `data.split('path:')[1]` (see
    `scripts/sim/env_bridge.py:71-74`). No trailing newline, no surrounding
    whitespace, save_dir kept verbatim.
    """
    if "path:" in save_dir:
        raise ValueError("save_dir must not contain literal 'path:'")
    return f"path:{save_dir}".encode("utf-8")


def decode_pose(raw: bytes | str) -> tuple[float, float, float, float, float, float]:
    """Inverse of `encode_pose`.

    Used in tests and by anyone replaying recorded traffic. Mirrors the bridge
    logic in `airsim_bridge.py:140`:
        pose = list(map(float, data.split(',')))
    """
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    parts = text.strip().split(",")
    if len(parts) != 6:
        raise ValueError(
            f"Expected 6 comma-separated floats, got {len(parts)}: {text!r}"
        )
    x, y, z, pitch, yaw, roll = (float(p) for p in parts)
    return x, y, z, pitch, yaw, roll


def is_path_prefix(raw: bytes | str) -> bool:
    """Return True if message is a `path:...` control record."""
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    return "path:" in text


def decode_path_prefix(raw: bytes | str) -> str:
    """Extract the directory from a `path:<dir>` record."""
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    if "path:" not in text:
        raise ValueError(f"Not a path prefix: {text!r}")
    return text.split("path:", 1)[1]


# ---------------------------------------------------------------------------- #
# Optional socket client (used at runtime, not in unit tests)                  #
# ---------------------------------------------------------------------------- #


@dataclass
class TCPClientConfig:
    sim_ip: str = "127.0.0.1"
    aim_port: int = 9999
    connect_timeout_s: float = 30.0
    retry_interval_s: float = 1.0


class OpenFlyTCPClient:
    """Persistent TCP connection to one OpenFly bridge listener.

    Mirrors the role of `TCPServer::sendString` in the C++ planner: opens a
    socket to the bridge and pushes messages one at a time.
    """

    def __init__(self, config: Optional[TCPClientConfig] = None):
        self.config = config or TCPClientConfig()
        self._sock: Optional[socket.socket] = None

    def connect(self) -> None:
        deadline = time.monotonic() + self.config.connect_timeout_s
        last_err: Optional[Exception] = None
        while time.monotonic() < deadline:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self.config.sim_ip, self.config.aim_port))
                self._sock = sock
                return
            except OSError as e:
                last_err = e
                time.sleep(self.config.retry_interval_s)
        raise ConnectionError(
            f"Failed to connect to {self.config.sim_ip}:{self.config.aim_port} "
            f"after {self.config.connect_timeout_s}s: {last_err}"
        )

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def __enter__(self) -> "OpenFlyTCPClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ----- High-level message helpers ---------------------------------------

    def send_path_prefix(self, save_dir: str) -> None:
        if self._sock is None:
            raise RuntimeError("client not connected; call connect() first")
        self._sock.sendall(encode_path_prefix(save_dir))

    def send_pose(
        self,
        x: float,
        y: float,
        z: float,
        pitch: float,
        yaw: float,
        roll: float,
    ) -> None:
        if self._sock is None:
            raise RuntimeError("client not connected; call connect() first")
        self._sock.sendall(encode_pose(x, y, z, pitch, yaw, roll))


__all__ = [
    "encode_pose",
    "encode_path_prefix",
    "decode_pose",
    "decode_path_prefix",
    "is_path_prefix",
    "TCPClientConfig",
    "OpenFlyTCPClient",
]

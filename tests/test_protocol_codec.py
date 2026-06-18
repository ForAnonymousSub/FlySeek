# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Test that flyseek's TCP codec is byte-identical to OpenFly's expectations.

OpenFly's Python bridge decodes via `list(map(float, data.split(',')))` and
the C++ planner encodes via `std::fixed std::setprecision(6)`. We must match
both sides exactly so a flyseek client can drive an unmodified OpenFly bridge.

References:
- scripts/sim/airsim_bridge.py:140  → pose = list(map(float, data.split(',')))
- scripts/sim/env_bridge.py:71-74  → 'path:' parsing
- tool_ws/src/traj_gen/include/base/tcpserver.hpp:30-66  → sendCameraPose
"""

from __future__ import annotations

import re

import pytest

from flyseek.adapters.openfly_tcp_client import (
    decode_path_prefix,
    decode_pose,
    encode_path_prefix,
    encode_pose,
    is_path_prefix,
)


# --------------------------------------------------------------------------- #
# Pose encoding                                                                #
# --------------------------------------------------------------------------- #


def test_encode_pose_byte_identical_to_cpp_format():
    """C++ uses `std::fixed << std::setprecision(6)` → exactly 6 decimal places."""
    encoded = encode_pose(1.0, -2.0, 3.5, 0.123456, 1.570796, 0.0)
    assert encoded == b"1.000000,-2.000000,3.500000,0.123456,1.570796,0.000000"


def test_encode_pose_has_no_trailing_whitespace():
    encoded = encode_pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert encoded == b"0.000000,0.000000,0.000000,0.000000,0.000000,0.000000"
    assert not encoded.endswith(b"\n")
    assert not encoded.endswith(b" ")


def test_encode_pose_uses_six_decimal_places_for_every_field():
    encoded = encode_pose(123.456789012, 0.1, 0.0, 0.0, 0.0, 0.0).decode()
    fields = encoded.split(",")
    assert len(fields) == 6
    for field in fields:
        # Each field must match `<int>.<6 digits>` per C++ setprecision(6)
        assert re.match(r"^-?\d+\.\d{6}$", field), f"bad field: {field!r}"


def test_encode_pose_handles_negative_values():
    encoded = encode_pose(-1.5, -2.5, -3.5, -0.5, -0.5, -0.5)
    assert encoded == b"-1.500000,-2.500000,-3.500000,-0.500000,-0.500000,-0.500000"


# --------------------------------------------------------------------------- #
# Pose decoding                                                                #
# --------------------------------------------------------------------------- #


def test_decode_pose_round_trip_bytes():
    original = (10.0, -20.5, 30.25, 0.78539816, 1.57079632, 0.0)
    decoded = decode_pose(encode_pose(*original))
    for a, b in zip(decoded, original):
        assert abs(a - b) < 1e-6


def test_decode_pose_accepts_str_and_bytes_both():
    raw_str = "1.0,2.0,3.0,4.0,5.0,6.0"
    raw_bytes = raw_str.encode()
    assert decode_pose(raw_str) == decode_pose(raw_bytes)


def test_decode_pose_matches_openfly_bridge_logic_exactly():
    """OpenFly bridge: `pose = list(map(float, data.split(',')))`."""
    raw = b"100.123456,200.654321,50.000000,0.500000,1.500000,0.000000"
    decoded = decode_pose(raw)
    expected_via_bridge_logic = tuple(
        float(v) for v in raw.decode().split(",")
    )
    assert decoded == expected_via_bridge_logic


def test_decode_pose_rejects_wrong_field_count():
    with pytest.raises(ValueError, match="Expected 6"):
        decode_pose(b"1,2,3,4,5")


# --------------------------------------------------------------------------- #
# Path prefix                                                                  #
# --------------------------------------------------------------------------- #


def test_encode_path_prefix_matches_openfly_split_logic():
    """OpenFly bridge: `file_path = data.split('path:')[1]` (env_bridge.py:72)."""
    encoded = encode_path_prefix("uav_vln_data/test/")
    assert encoded == b"path:uav_vln_data/test/"
    decoded = encoded.decode().split("path:")[1]
    assert decoded == "uav_vln_data/test/"


def test_is_path_prefix_detection():
    assert is_path_prefix(b"path:uav_vln_data/")
    assert is_path_prefix("path:foo")
    assert not is_path_prefix(b"1.0,2.0,3.0,4.0,5.0,6.0")


def test_decode_path_prefix_extracts_dir():
    assert decode_path_prefix(b"path:flyseek/output/easy/000001/") == (
        "flyseek/output/easy/000001/"
    )


def test_decode_path_prefix_rejects_non_path():
    with pytest.raises(ValueError, match="Not a path prefix"):
        decode_path_prefix(b"1.0,2.0,3.0,4.0,5.0,6.0")


def test_encode_path_prefix_rejects_embedded_path_literal():
    """Defense against the malformed `path:path:foo` ambiguity."""
    with pytest.raises(ValueError):
        encode_path_prefix("path:nested/")


# --------------------------------------------------------------------------- #
# 1024-byte recv() budget                                                      #
# --------------------------------------------------------------------------- #


def test_encoded_messages_fit_in_bridge_recv_buffer():
    """OpenFly bridge calls `conn.recv(1024)` — our messages must fit."""
    pose_msg = encode_pose(1e6, 1e6, 1e6, 3.14, 3.14, 3.14)
    assert len(pose_msg) <= 1024

    path_msg = encode_path_prefix("a" * 800)  # generous save_dir length
    assert len(path_msg) <= 1024

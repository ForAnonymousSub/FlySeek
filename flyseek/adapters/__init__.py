# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Thin adapter layer between flyseek_extend and OpenFly / AirSim.

This package contains the *only* code that talks to:
- the OpenFly TCP bridge protocol (`openfly_tcp_client.py`)
- the AirSim Python API (`airsim_object_api.py`, strict whitelist)
- the OpenFly on-disk scene data (`scene_data_loader.py`)
- the OpenFly on-disk output format (`output_writer.py`)

Everything in `flyseek/adversary/`, `flyseek/expert/`, `flyseek/instruction/`
must stay simulator-agnostic and go through this layer.
"""

from flyseek.adapters.openfly_tcp_client import OpenFlyTCPClient

__all__ = ["OpenFlyTCPClient"]

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Scenario-level target trajectory generators."""

from .road_scenarios import RoadScenarioController, RoadScenarioConfig

__all__ = ["RoadScenarioController", "RoadScenarioConfig"]

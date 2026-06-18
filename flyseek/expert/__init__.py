# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Expert trajectory generator (dual-loop PID, offline math).

Critical: this module does NOT control AirSim. It is a pure-numpy mathematical
function that computes the drone pose sequence given a target pose sequence.

Forbidden APIs (do not import or call):
- airsim.MultirotorClient.moveByVelocityAsync
- airsim.MultirotorClient.moveOnPath
- airsim.MultirotorClient.takeoffAsync / landAsync
- airsim.MultirotorClient.enableApiControl / armDisarm
"""

from .adaptive_tracker import AdaptiveTracker, AdaptiveTrackerConfig
from .tracking_drone import TrackingDroneController, TrackerConfig, TrackerState

__all__ = [
    "AdaptiveTracker",
    "AdaptiveTrackerConfig",
    "TrackingDroneController",
    "TrackerConfig",
    "TrackerState",
]

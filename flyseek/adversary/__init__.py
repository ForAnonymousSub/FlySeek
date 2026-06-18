# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""flyseek.adversary — offline adversarial agents (SKILL.md §2).

Public API::

    from flyseek.adversary import (
        AdversarialAgent, AgentAction, DroneState, TargetState, PlayBox,
        RandomWalkAgent, SCurveEvasionAgent,
        create_adversarial_agent, integrate_target,
        bearing_xy, horizontal_distance, wrap_to_pi,
    )
"""

from .base import (
    AdversarialAgent,
    AgentAction,
    DroneState,
    PlayBox,
    TargetState,
    bearing_xy,
    horizontal_distance,
    integrate_target,
    wrap_to_pi,
)
from .easy import RandomWalkAgent
from .factory import create_adversarial_agent
from .hide_seek import HideSeekCarAgent
from .medium import SCurveEvasionAgent

__all__ = [
    "AdversarialAgent",
    "AgentAction",
    "DroneState",
    "TargetState",
    "PlayBox",
    "RandomWalkAgent",
    "SCurveEvasionAgent",
    "HideSeekCarAgent",
    "create_adversarial_agent",
    "integrate_target",
    "bearing_xy",
    "horizontal_distance",
    "wrap_to_pi",
]

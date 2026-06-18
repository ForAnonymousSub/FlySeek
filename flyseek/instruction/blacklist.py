# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Three iron rules for VLT-style instructions (SKILL §4.3).

A FlySeek instruction *refers to* the tracked target; it must never:
  1. issue a navigation verb to the drone ("fly forward", "turn left", ...),
  2. tell the drone where to look ("look at", "point camera at", ...),
  3. reveal the target's future position ("will end up at", "headed to B", ...).

``check(text)`` is the single gate the quality filter calls; any hit rejects the
candidate. Matching is case-insensitive and word-boundary aware so substrings
inside legitimate words ("landmark" must not trip "land") do not false-positive.
"""

from __future__ import annotations

import re

# Rule 1 — navigation / flight-control verbs aimed at the drone.
NAV_VERBS: tuple[str, ...] = (
    "take off", "takeoff", "land", "fly forward", "fly toward", "fly to",
    "fly straight", "fly up", "fly down", "fly left", "fly right",
    "turn left", "turn right", "turn around", "ascend", "descend",
    "hover", "go straight", "go forward", "go up", "go down",
    "move forward", "move backward", "climb", "bank left", "bank right",
    "yaw left", "yaw right", "rotate", "strafe",
)

# Rule 2 — camera-aiming / gaze directives.
LOOK_AT: tuple[str, ...] = (
    "look at", "look toward", "look towards", "look down", "look up",
    "face toward", "face towards", "point camera", "point the camera",
    "aim the camera", "aim camera", "focus on the right", "focus on the left",
    "center the camera", "keep your eyes", "gaze at",
)

# Rule 3 — exposure of the target's future position.
FUTURE_POS: tuple[str, ...] = (
    "will be at", "will end up", "will arrive", "will reach", "headed to",
    "heading to", "destination is", "destination of", "reach point",
    "go to point", "ends up at", "will stop at", "final position",
    "will go to", "will move to", "will head", "is going to reach",
)

RULES: dict[str, tuple[str, ...]] = {
    "nav_verb": NAV_VERBS,
    "look_at": LOOK_AT,
    "future_position": FUTURE_POS,
}


def _matches(text: str, phrase: str) -> bool:
    pattern = r"\b" + re.escape(phrase) + r"\b"
    return re.search(pattern, text) is not None


def violations(text: str) -> list[str]:
    """Return the list of ``"<rule>:<phrase>"`` matches in ``text`` (may be empty)."""
    low = text.lower()
    hits: list[str] = []
    for rule, phrases in RULES.items():
        for phrase in phrases:
            if _matches(low, phrase):
                hits.append(f"{rule}:{phrase}")
    return hits


def check(text: str) -> tuple[bool, str]:
    """``(ok, reason)`` — ``ok`` is False on the first violating phrase."""
    hits = violations(text)
    if hits:
        return False, hits[0]
    return True, "ok"


def is_clean(text: str) -> bool:
    return not violations(text)


__all__ = ["NAV_VERBS", "LOOK_AT", "FUTURE_POS", "RULES",
           "violations", "check", "is_clean"]

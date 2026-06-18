# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Numeric target-behavior classification (SKILL §4.4 Step 2 — no LLM).

Turns a target trajectory (positions over time, optional occlusion flags) into a
discrete behavior class plus a short natural-language descriptor used to fill the
``medium`` / ``hard`` instruction templates. Pure numpy so it is fully testable
without a simulator.

Behavior classes:
  - ``straight``      : nearly constant heading, little turning.
  - ``zigzag``        : frequent alternating left/right heading swings.
  - ``dodging``       : large, sharp speed/heading changes (evasive bursts).
  - ``cover_using``   : spends a meaningful fraction of frames occluded (LOS
                        blocked), i.e. exploits buildings/objects for cover.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

BEHAVIOR_DESCRIPTORS: dict[str, str] = {
    "straight": "moving steadily in a straight line",
    "zigzag": "weaving back and forth in a zigzag pattern",
    "dodging": "making sharp, sudden evasive maneuvers",
    "cover_using": "ducking behind objects to break line of sight",
}


@dataclass
class BehaviorResult:
    behavior_class: str
    descriptor: str
    turn_rate_std_deg: float
    direction_changes: int
    occluded_fraction: float

    def as_dict(self) -> dict:
        return {
            "behavior_class": self.behavior_class,
            "descriptor": self.descriptor,
            "turn_rate_std_deg": round(self.turn_rate_std_deg, 3),
            "direction_changes": self.direction_changes,
            "occluded_fraction": round(self.occluded_fraction, 4),
        }


def _headings_from_positions(positions: np.ndarray) -> np.ndarray:
    """Per-step XY heading (rad); steps shorter than 1e-3 m reuse the previous."""
    deltas = np.diff(positions[:, :2], axis=0)
    headings: list[float] = []
    last = 0.0
    for d in deltas:
        if float(np.hypot(d[0], d[1])) > 1e-3:
            last = math.atan2(float(d[1]), float(d[0]))
        headings.append(last)
    return np.asarray(headings, dtype=np.float64)


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + math.pi) % (2 * math.pi) - math.pi


def classify(
    positions: np.ndarray,
    *,
    dt: float = 0.05,
    occluded_flags: np.ndarray | None = None,
    cover_fraction_thresh: float = 0.12,
    zigzag_min_changes: int = 4,
    dodge_turn_std_deg: float = 35.0,
) -> BehaviorResult:
    """Classify a target trajectory.

    ``positions`` is an ``(N, 3)`` array of target NED positions; ``dt`` is the
    seconds-per-step; ``occluded_flags`` is an optional length-N boolean array
    (``is_occluded`` per frame).
    """
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    occ_frac = 0.0
    if occluded_flags is not None and len(occluded_flags) > 0:
        occ_frac = float(np.mean(np.asarray(occluded_flags, dtype=bool)))

    if positions.shape[0] < 3:
        cls = "cover_using" if occ_frac >= cover_fraction_thresh else "straight"
        return BehaviorResult(cls, BEHAVIOR_DESCRIPTORS[cls], 0.0, 0, occ_frac)

    headings = _headings_from_positions(positions)
    dheading = _wrap(np.diff(headings))
    turn_rate = dheading / max(dt, 1e-6)  # rad/s
    turn_rate_std_deg = float(math.degrees(np.std(turn_rate)))

    # Direction changes = sign flips of the per-step heading delta beyond a
    # small deadband (filters numerical jitter).
    deadband = math.radians(8.0)
    signs = np.sign(np.where(np.abs(dheading) > deadband, dheading, 0.0))
    nonzero = signs[signs != 0.0]
    direction_changes = int(np.sum(np.abs(np.diff(nonzero)) > 1.0)) if nonzero.size > 1 else 0

    if occ_frac >= cover_fraction_thresh:
        cls = "cover_using"
    elif turn_rate_std_deg >= dodge_turn_std_deg:
        cls = "dodging"
    elif direction_changes >= zigzag_min_changes:
        cls = "zigzag"
    else:
        cls = "straight"

    return BehaviorResult(
        behavior_class=cls,
        descriptor=BEHAVIOR_DESCRIPTORS[cls],
        turn_rate_std_deg=turn_rate_std_deg,
        direction_changes=direction_changes,
        occluded_fraction=occ_frac,
    )


def classify_from_meta(meta_records: list[dict], *, dt: float | None = None) -> BehaviorResult:
    """Classify from parsed ``flyseek_meta.jsonl`` records."""
    if not meta_records:
        return BehaviorResult("straight", BEHAVIOR_DESCRIPTORS["straight"], 0.0, 0, 0.0)
    positions = np.asarray(
        [r["target_state"]["pos"] for r in meta_records], dtype=np.float64
    )
    occ = np.asarray(
        [bool(r.get("target_state", {}).get("is_occluded", False)) for r in meta_records],
        dtype=bool,
    )
    if dt is None:
        ts = [float(r.get("timestamp", 0.0)) for r in meta_records]
        dt = (ts[-1] - ts[0]) / max(len(ts) - 1, 1) if len(ts) > 1 else 0.05
        dt = dt if dt > 1e-6 else 0.05
    return classify(positions, dt=dt, occluded_flags=occ)


__all__ = ["BehaviorResult", "BEHAVIOR_DESCRIPTORS", "classify", "classify_from_meta"]

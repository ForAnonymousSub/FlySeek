# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Language-conditioned target instruction generation (paper §3).

Produces a per-episode, attribute-grounded instruction for the car target. It
**never hallucinates** appearance: a colour/type/size word only appears in the
instruction if it is actually present in the supplied ``target_attributes``
(or extractable from the target label). When no specific context is available it
falls back to a generic-but-valid instruction.

Four template families:
  - ``appearance``      : describes the target only.
  - ``location``        : adds an initial location context phrase.
  - ``motion``          : adds a motion/road context phrase.
  - ``occlusion``       : frames the target moving toward cover (occlusion risk).

The output record (``instruction.json``) carries:
  ``instruction, target_class, target_attributes, initial_context,
    behavior_type, difficulty_level, seed`` (+ ``template_family``).

All instructions are checked against the three iron rules in
``flyseek.instruction.blacklist`` (no navigation verbs, no camera-aiming, no
future-position leakage); a violating candidate is replaced by a safe
appearance-only instruction.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from flyseek.instruction import blacklist

# Vocabulary used ONLY to recognise words already present in a label — never to
# invent new ones.
KNOWN_COLORS = {
    "red", "blue", "green", "yellow", "white", "black", "silver", "grey",
    "gray", "orange", "brown", "dark", "light",
}
KNOWN_SIZES = {"small", "large", "compact", "big", "mini", "tiny"}
KNOWN_TYPES = {
    "car", "sedan", "suv", "truck", "van", "taxi", "hatchback", "cart",
    "vehicle", "bus", "jeep", "pickup",
}

TEMPLATES: dict[str, list[str]] = {
    "appearance": [
        "Track {appearance}.",
        "Keep tracking {appearance}.",
        "Keep observing {appearance}.",
        "Maintain visual contact with {appearance}.",
    ],
    "location": [
        "Track {appearance} near {location}.",
        "Follow {appearance} starting near {location}.",
        "Keep observing {appearance} around {location}.",
    ],
    "motion": [
        "Keep observing {appearance} moving along {motion}.",
        "Track {appearance} as it travels along {motion}.",
        "Follow {appearance} moving through {motion}.",
    ],
    "occlusion": [
        "Maintain visual contact with {appearance} as it moves toward {occlusion}.",
        "Keep tracking {appearance} as it approaches {occlusion}.",
        "Track {appearance} while it moves into {occlusion}.",
    ],
}

# Behavior -> ordered family preference (first available family wins).
_FAMILY_PREFERENCE: dict[str, list[str]] = {
    "occlusion_seeking": ["occlusion", "motion", "location", "appearance"],
    "direct_escape": ["location", "motion", "appearance"],
    "sharp_turn": ["motion", "location", "appearance"],
    "detour_feint": ["motion", "location", "appearance"],
}
_DEFAULT_PREFERENCE = ["location", "motion", "appearance"]

_DEFAULT_OCCLUSION_PHRASE = "an occluded street"


def attributes_from_label(label: str | None) -> dict[str, str]:
    """Extract only the attributes literally present in a free-text label.

    e.g. ``"a small red taxi"`` -> ``{"size": "small", "color": "red",
    "type": "taxi"}``. Unknown/absent attributes are simply omitted (no guessing).
    """
    attrs: dict[str, str] = {}
    if not label:
        return attrs
    tokens = [t.lower() for t in re.findall(r"[A-Za-z]+", label)]
    for tok in tokens:
        if "color" not in attrs and tok in KNOWN_COLORS:
            attrs["color"] = tok
        if "size" not in attrs and tok in KNOWN_SIZES:
            attrs["size"] = tok
        if "type" not in attrs and tok in KNOWN_TYPES and tok != "vehicle":
            attrs["type"] = tok
    return attrs


def _clean_attributes(attrs: dict[str, Any] | None) -> dict[str, str]:
    """Drop empty / unknown attribute values so they never reach the text."""
    out: dict[str, str] = {}
    for key, val in (attrs or {}).items():
        if val is None:
            continue
        s = str(val).strip()
        if not s or s.lower() in ("unknown", "none", "n/a"):
            continue
        out[key] = s
    return out


def _noun(target_class: str | None, attrs: dict[str, str]) -> str:
    if attrs.get("type"):
        return attrs["type"]
    s = str(target_class or "").strip()
    if not s:
        return "vehicle"
    s = s.split(".")[-1]
    words = re.findall(r"[A-Za-z]+", s)
    return words[-1].lower() if words else "vehicle"


def build_appearance_phrase(target_class: str | None, attrs: dict[str, str]) -> str:
    """``the [size] [color] <noun>`` using only present attributes."""
    parts: list[str] = []
    if attrs.get("size"):
        parts.append(attrs["size"])
    if attrs.get("color"):
        parts.append(attrs["color"])
    noun = _noun(target_class, attrs)
    parts.append(noun)
    return "the " + " ".join(parts)


class InstructionGenerator:
    """Deterministic, attribute-grounded instruction generator."""

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed

    def _available_families(self, ctx: dict[str, Any], behavior_type: str) -> list[str]:
        avail = ["appearance"]
        if ctx.get("location"):
            avail.append("location")
        if ctx.get("motion"):
            avail.append("motion")
        # occlusion family is valid if an occlusion phrase exists OR the target's
        # behavior is occlusion-seeking (a generic, behavior-grounded phrase is
        # used in that case — not a hallucinated location).
        if ctx.get("occlusion") or behavior_type == "occlusion_seeking":
            avail.append("occlusion")
        return avail

    def _choose_family(self, ctx, behavior_type, available) -> str:
        pref = _FAMILY_PREFERENCE.get(behavior_type, _DEFAULT_PREFERENCE)
        for fam in pref:
            if fam in available:
                return fam
        return "appearance"

    def generate(
        self,
        *,
        target_class: str | None,
        target_attributes: dict[str, Any] | None = None,
        initial_context: dict[str, Any] | None = None,
        behavior_type: str = "",
        difficulty_level: str = "",
        seed: int | None = None,
        family: str | None = None,
    ) -> dict[str, Any]:
        seed = self.seed if seed is None else seed
        rng = np.random.default_rng(seed)
        attrs = _clean_attributes(target_attributes)
        ctx = dict(initial_context or {})

        available = self._available_families(ctx, behavior_type)
        if family is not None and family in TEMPLATES:
            chosen = family if (family in available or family == "appearance") else \
                self._choose_family(ctx, behavior_type, available)
        else:
            chosen = self._choose_family(ctx, behavior_type, available)

        appearance = build_appearance_phrase(target_class, attrs)
        slots = {
            "appearance": appearance,
            "location": str(ctx.get("location", "")),
            "motion": str(ctx.get("motion", "")),
            "occlusion": str(ctx.get("occlusion") or _DEFAULT_OCCLUSION_PHRASE),
        }

        templates = TEMPLATES[chosen]
        template = templates[int(rng.integers(len(templates)))]
        instruction = " ".join(template.format(**slots).split())

        # Iron-rule safety net: never emit a blacklisted instruction.
        if not blacklist.is_clean(instruction):
            chosen = "appearance"
            instruction = f"Track {appearance}."

        return {
            "instruction": instruction,
            "target_class": target_class,
            "target_attributes": attrs,
            "initial_context": ctx,
            "behavior_type": behavior_type,
            "difficulty_level": difficulty_level,
            "seed": seed,
            "template_family": chosen,
        }


def write_instruction_json(record: dict[str, Any], path: str | Path) -> Path:
    """Write an instruction record to ``path`` as pretty JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def generate_instruction(
    *,
    target_class: str | None,
    target_attributes: dict[str, Any] | None = None,
    initial_context: dict[str, Any] | None = None,
    behavior_type: str = "",
    difficulty_level: str = "",
    seed: int | None = None,
) -> dict[str, Any]:
    """Convenience wrapper around :class:`InstructionGenerator`."""
    return InstructionGenerator(seed=seed).generate(
        target_class=target_class,
        target_attributes=target_attributes,
        initial_context=initial_context,
        behavior_type=behavior_type,
        difficulty_level=difficulty_level,
        seed=seed,
    )


__all__ = [
    "InstructionGenerator",
    "generate_instruction",
    "write_instruction_json",
    "attributes_from_label",
    "build_appearance_phrase",
    "TEMPLATES",
]

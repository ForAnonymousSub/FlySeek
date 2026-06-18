# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Tests for the attribute-grounded language-conditioned instruction generator."""

from __future__ import annotations

import json

from flyseek.bench.instruction_generator import (
    InstructionGenerator,
    attributes_from_label,
    build_appearance_phrase,
    generate_instruction,
    write_instruction_json,
)
from flyseek.instruction import blacklist

REQUIRED_KEYS = {
    "instruction", "target_class", "target_attributes", "initial_context",
    "behavior_type", "difficulty_level", "seed",
}


def test_attributes_from_label_extracts_only_present():
    assert attributes_from_label("a small motorized car") == {"size": "small", "type": "car"}
    assert attributes_from_label("a red taxi") == {"color": "red", "type": "taxi"}
    assert attributes_from_label("") == {}
    assert "color" not in attributes_from_label("the vehicle")


def test_no_color_hallucination():
    rec = generate_instruction(
        target_class="a small motorized car",
        target_attributes=attributes_from_label("a small motorized car"),
        seed=1,
    )
    # No colour word should appear since the label has none.
    for color in ("red", "blue", "green", "white", "black", "silver"):
        assert color not in rec["instruction"].lower()
    assert "the small car" in rec["instruction"]


def test_color_included_when_present():
    rec = generate_instruction(
        target_class="vehicle.taxi",
        target_attributes={"color": "red", "type": "taxi", "size": "small"},
        seed=1,
    )
    assert "red" in rec["instruction"]
    assert "taxi" in rec["instruction"]


def test_unknown_color_is_dropped():
    phrase = build_appearance_phrase("car", {"size": "small", "color": "unknown"})
    # _clean_attributes happens in generate(); build_appearance_phrase receives
    # raw, so test the full path:
    rec = generate_instruction(
        target_class="car",
        target_attributes={"size": "small", "color": "unknown", "type": "car"},
        seed=0,
    )
    assert "unknown" not in rec["instruction"].lower()


def test_required_keys_present():
    rec = generate_instruction(
        target_class="car", target_attributes={"type": "car"},
        behavior_type="direct_escape", difficulty_level="medium", seed=5,
    )
    assert REQUIRED_KEYS <= set(rec)


def test_determinism():
    kw = dict(target_class="car", target_attributes={"type": "car"},
              initial_context={"motion": "the street"},
              behavior_type="sharp_turn", difficulty_level="hard", seed=9)
    assert generate_instruction(**kw) == generate_instruction(**kw)


def test_location_family_when_context_present():
    rec = generate_instruction(
        target_class="car", target_attributes={"size": "small", "type": "car"},
        initial_context={"location": "the intersection"},
        behavior_type="direct_escape", seed=2,
    )
    assert rec["template_family"] == "location"
    assert "the intersection" in rec["instruction"]


def test_motion_family_when_motion_context():
    rec = generate_instruction(
        target_class="car", target_attributes={"type": "car"},
        initial_context={"motion": "the street"},
        behavior_type="sharp_turn", seed=3,
    )
    assert rec["template_family"] == "motion"
    assert "the street" in rec["instruction"]


def test_occlusion_family_for_occlusion_behavior():
    rec = generate_instruction(
        target_class="car", target_attributes={"size": "small", "type": "car"},
        initial_context={"occlusion": "the occluded street"},
        behavior_type="occlusion_seeking", seed=4,
    )
    assert rec["template_family"] == "occlusion"
    assert "occluded" in rec["instruction"]


def test_occlusion_behavior_without_context_uses_default_phrase():
    rec = generate_instruction(
        target_class="car", target_attributes={"type": "car"},
        initial_context={}, behavior_type="occlusion_seeking", seed=4,
    )
    assert rec["template_family"] == "occlusion"
    assert "occluded" in rec["instruction"]


def test_generic_fallback_appearance_only():
    rec = generate_instruction(
        target_class="vehicle", target_attributes={}, initial_context={},
        behavior_type="", seed=0,
    )
    assert rec["template_family"] == "appearance"
    assert rec["instruction"].lower().startswith(("track", "keep", "maintain", "follow"))


def test_all_instructions_blacklist_clean():
    for behavior in ("direct_escape", "sharp_turn", "detour_feint", "occlusion_seeking"):
        for seed in range(6):
            rec = generate_instruction(
                target_class="a small red taxi",
                target_attributes=attributes_from_label("a small red taxi"),
                initial_context={"location": "the intersection",
                                 "motion": "the street",
                                 "occlusion": "the occluded street"},
                behavior_type=behavior, difficulty_level="medium", seed=seed,
            )
            assert blacklist.is_clean(rec["instruction"]), rec["instruction"]


def test_write_instruction_json(tmp_path):
    rec = generate_instruction(target_class="car",
                               target_attributes={"type": "car"}, seed=0)
    p = write_instruction_json(rec, tmp_path / "instruction.json")
    loaded = json.loads(p.read_text())
    assert loaded["instruction"] == rec["instruction"]
    assert REQUIRED_KEYS <= set(loaded)

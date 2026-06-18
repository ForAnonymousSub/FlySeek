# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Iron-rule blacklist tests (SKILL §4.3) + quality filter integration."""

from __future__ import annotations

from flyseek.instruction import blacklist, templates
from flyseek.instruction.llm_backend import MockBackend
from flyseek.instruction.quality_filter import filter_candidates


def test_nav_verb_rejected():
    for bad in [
        "Track the red car and fly forward.",
        "Follow the sedan, then turn left.",
        "Track the car and ascend quickly.",
    ]:
        ok, reason = blacklist.check(bad)
        assert not ok, bad
        assert reason.startswith("nav_verb")


def test_look_at_rejected():
    for bad in [
        "Track the car and look at the building.",
        "Follow the sedan; point the camera at it.",
    ]:
        ok, reason = blacklist.check(bad)
        assert not ok, bad
        assert reason.startswith("look_at")


def test_future_position_rejected():
    for bad in [
        "Track the car that will end up at the bridge.",
        "Follow the sedan headed to the alley.",
    ]:
        ok, reason = blacklist.check(bad)
        assert not ok, bad
        assert reason.startswith("future_position")


def test_clean_instructions_pass():
    for good in [
        "Track the small red car.",
        "Follow the dark sedan that is weaving back and forth.",
        "Keep tracking the compact car using the nearby buildings.",
    ]:
        ok, reason = blacklist.check(good)
        assert ok, f"{good} -> {reason}"


def test_landmark_word_does_not_false_positive_land():
    # "landmark" contains "land" but must not trip the nav-verb rule.
    ok, _ = blacklist.check("Track the car near the famous landmark.")
    assert ok


def test_quality_filter_drops_blacklisted_and_short():
    cands = [
        "Track the small red car.",            # ok
        "Follow it and turn right.",           # blacklist
        "Track car.",                          # too short
        "Track the small red car.",            # duplicate of #1
    ]
    res = filter_candidates(cands)
    assert "Track the small red car." in res.kept
    assert res.blacklist_rejected >= 1
    # Only the first clean unique one survives.
    assert res.kept.count("Track the small red car.") == 1


def test_templates_fill_are_clean():
    backend = MockBackend(seed=0)
    appearance = backend.extract_appearance(label_hint="car")
    for tier in ("easy", "medium", "hard"):
        for text in templates.fill(
            tier, appearance=appearance,
            behavior="weaving back and forth in a zigzag pattern",
            cover="the nearby buildings", n=5,
        ):
            assert blacklist.is_clean(text), f"{tier}: {text}"

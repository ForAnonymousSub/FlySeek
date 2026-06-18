# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""VLT-style adversarial instruction generation.

Pipeline (see SKILL §4.4):
1. extract target appearance (VLM, first frame)
2. analyze target behavior (numeric, no LLM)
3. extract scene covers (VLM, hard only)
4. fill templates (no LLM)
5. optional polish (LLM)
6. quality filter (blacklist + diversity + judge)

Three iron rules (see SKILL §4.3 and `tests/test_instruction_blacklist.py`):
- No navigation verbs ("fly forward", "turn left", ...)
- No "look at" / camera-aiming directives
- No exposure of target future position
"""

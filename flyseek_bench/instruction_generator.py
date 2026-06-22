# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""``flyseek_bench.instruction_generator`` — re-export of
:mod:`flyseek.bench.instruction_generator`.

The canonical language-conditioned instruction generator lives under
``flyseek.bench.instruction_generator`` (consistent with the package layout:
``flyseek.eval``, ``flyseek.instruction``, ``flyseek.pipeline``). This thin shim
exposes it at the documented ``flyseek_bench.instruction_generator`` import path
and is safe to import without a simulator.

It produces a per-episode, attribute-grounded instruction for the car target and
**never hallucinates** appearance: a colour/type/size word only appears if it is
actually present in ``target_attributes`` (or extractable from the label). Four
template families are provided — ``appearance``, ``location``, ``motion`` and
``occlusion`` (occlusion-risk) — and every candidate is checked against the three
iron rules in ``flyseek.instruction.blacklist`` before being emitted.

The record written to ``instruction.json`` carries: ``instruction``,
``target_class``, ``target_attributes``, ``initial_context``, ``behavior_type``,
``difficulty_level``, ``seed`` (+ ``template_family``).

This generator is already integrated into the episode pipeline
(``scripts/demo_adversary_chase.py`` and ``flyseek_bench.run_generate_episodes``):
each episode saves ``instruction.json`` and the instruction text is also stored in
``metadata.json``. This module only provides the documented import alias.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow ``import flyseek_bench.instruction_generator`` / file-path execution from
# any cwd by making ``flyseek_extend`` importable so ``flyseek.bench`` resolves.
_FLYSEEK_EXTEND = Path(__file__).resolve().parents[1]
if str(_FLYSEEK_EXTEND) not in sys.path:
    sys.path.insert(0, str(_FLYSEEK_EXTEND))

from flyseek.bench.instruction_generator import (  # noqa: E402,F401
    TEMPLATES,
    InstructionGenerator,
    attributes_from_label,
    build_appearance_phrase,
    generate_instruction,
    write_instruction_json,
)

__all__ = [
    "InstructionGenerator",
    "generate_instruction",
    "write_instruction_json",
    "attributes_from_label",
    "build_appearance_phrase",
    "TEMPLATES",
]

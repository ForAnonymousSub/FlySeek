# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""``flyseek_bench.export`` — re-export of :mod:`flyseek.bench.export`.

The canonical serialization helpers live under ``flyseek.bench.export``. This
thin shim exposes them at the documented ``flyseek_bench.export`` import path and
is safe to import without a simulator.

Exposed writers (all create parent dirs, coerce to JSON primitives, validate):
    save_metadata_json    -> <dir>/metadata.json
    append_frame_jsonl    -> appends one line to frames.jsonl
    save_trajectories_json-> <dir>/trajectories.json
    save_instruction_json -> <dir>/instruction.json
    save_metrics_json     -> <dir>/metrics.json
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow ``import flyseek_bench.export`` / file-path execution from any cwd by
# making ``flyseek_extend`` importable so ``flyseek.bench`` resolves.
_FLYSEEK_EXTEND = Path(__file__).resolve().parents[1]
if str(_FLYSEEK_EXTEND) not in sys.path:
    sys.path.insert(0, str(_FLYSEEK_EXTEND))

from flyseek.bench.export import (  # noqa: E402,F401
    append_frame_jsonl,
    save_instruction_json,
    save_metadata_json,
    save_metrics_json,
    save_trajectories_json,
)

__all__ = [
    "save_metadata_json",
    "append_frame_jsonl",
    "save_trajectories_json",
    "save_instruction_json",
    "save_metrics_json",
]

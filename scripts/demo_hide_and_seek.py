#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Hide-and-seek demo: car hides behind buildings; drone searches and reacquires.

Thin entry point — same pipeline as ``demo_adversary_chase.py`` with
``--scenario hide_seek``.

Quickstart (AirVLN must be running):

    python flyseek_extend/scripts/scout_scene_targets.py
    python flyseek_extend/scripts/demo_hide_and_seek.py \\
        --auto-from-scout --target-regex '.*Car.*' --frames 200
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT / "flyseek_extend") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "flyseek_extend"))

# Inject --scenario hide_seek before argparse runs in the chase demo module.
if "--scenario" not in sys.argv:
    sys.argv[1:1] = ["--scenario", "hide_seek"]

runpy.run_path(
    str(REPO_ROOT / "flyseek_extend" / "scripts" / "demo_adversary_chase.py"),
    run_name="__main__",
)

#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Generate VLT-style instructions for already-recorded episodes (offline).

Runs the template + mock instruction pipeline over every episode under a root
(or a single episode dir) and writes ``instruction.json`` into each. No GPU /
network / simulator required.

Usage::

    python flyseek_extend/scripts/gen_instructions.py \\
        --root flyseek_extend/output/trajectories/env_airsim_16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT / "flyseek_extend") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "flyseek_extend"))

from flyseek.eval.episode_eval import find_episode_dirs  # noqa: E402
from flyseek.instruction.pipeline import InstructionPipeline  # noqa: E402

DEFAULT_LLM_CONFIG = REPO_ROOT / "flyseek_extend" / "configs" / "llm_backend.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate FlySeek VLT instructions.")
    parser.add_argument("--root", type=Path, required=True,
                        help="Episode dir or batch root.")
    parser.add_argument("--llm-config", type=Path, default=DEFAULT_LLM_CONFIG)
    args = parser.parse_args()

    llm_cfg = {"backend": "mock"}
    if args.llm_config.exists():
        try:
            import yaml  # type: ignore
            llm_cfg = yaml.safe_load(args.llm_config.read_text(encoding="utf-8")) or llm_cfg
        except Exception:
            pass

    pipe = InstructionPipeline.from_config(llm_cfg)
    episode_dirs = find_episode_dirs(args.root)
    if not episode_dirs:
        print(f"[ERR] no episodes (flyseek_meta.jsonl) under {args.root}")
        return 1

    existing: list[str] = []
    total = 0
    for ep in episode_dirs:
        rec = pipe.generate_for_episode(ep, existing=existing)
        existing.extend(rec["instructions"])
        total += len(rec["instructions"])
        print(f"  {ep.name}: {len(rec['instructions'])} instr "
              f"[{rec['tier']}] — \"{rec['instructions'][0] if rec['instructions'] else '<none>'}\"")
    print(f"[ok] {total} instruction(s) across {len(episode_dirs)} episode(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

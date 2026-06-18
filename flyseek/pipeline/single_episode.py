# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Library entry point for generating a single FlySeek-Bench episode.

The end-to-end closed loop (adversary -> target integration -> visibility-aware
expert tracker -> teleport + capture -> OpenFly/FlySeek export) lives in
``scripts/demo_adversary_chase.py``. This module exposes it as an importable
function so the batch runner and CLI do not have to shell out / hack ``sys.argv``.

It also writes episodes into the canonical SKILL §6.3 layout
``output/trajectories/<env>/<difficulty>/<traj_id>/`` when a ``traj_id`` is given.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
_DEMO_PATH = REPO_ROOT / "flyseek_extend" / "scripts" / "demo_adversary_chase.py"
DEFAULT_TRAJ_ROOT = REPO_ROOT / "flyseek_extend" / "output" / "trajectories"

_demo_module: ModuleType | None = None


def _load_demo_module() -> ModuleType:
    """Import ``demo_adversary_chase`` as a module (cached)."""
    global _demo_module
    if _demo_module is not None:
        return _demo_module
    scripts_dir = REPO_ROOT / "flyseek_extend" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(
        "flyseek_demo_adversary_chase", _DEMO_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load demo module from {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses with PEP 563 string annotations can
    # resolve ``cls.__module__`` during class creation.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _demo_module = module
    return module


def build_default_args(overrides: dict[str, Any] | None = None) -> argparse.Namespace:
    """Build a fully-defaulted args namespace, then apply ``overrides``."""
    demo = _load_demo_module()
    parser = demo.build_parser()
    args = parser.parse_args([])
    for key, value in (overrides or {}).items():
        if not hasattr(args, key):
            raise KeyError(f"unknown episode argument: {key!r}")
        setattr(args, key, value)
    demo.finalize_args(args)
    return args


def canonical_episode_dir(
    *,
    env: str,
    difficulty: str,
    traj_id: str,
    traj_root: Path | None = None,
) -> Path:
    """``<traj_root>/<env>/<difficulty>/<traj_id>`` (SKILL §6.3 layout)."""
    root = Path(traj_root) if traj_root is not None else DEFAULT_TRAJ_ROOT
    return root / env / difficulty / traj_id


def run_episode(
    overrides: dict[str, Any] | None = None,
    *,
    traj_id: str | None = None,
    traj_root: Path | None = None,
    write_summary: bool = True,
):
    """Run one episode and return the demo ``DemoReport``.

    When ``traj_id`` is given the episode is written to the canonical
    ``output/trajectories/<env>/<difficulty>/<traj_id>/`` directory. Otherwise
    the demo's own ``--output``/``--episode-tag`` logic decides the location.
    """
    overrides = dict(overrides or {})
    args = build_default_args(overrides)

    if traj_id is not None:
        env = str(getattr(args, "env", "env_airsim_16"))
        difficulty = str(getattr(args, "difficulty", "medium"))
        ep_dir = canonical_episode_dir(
            env=env, difficulty=difficulty, traj_id=traj_id, traj_root=traj_root
        )
        args.output = ep_dir.parent
        args.episode_tag = ep_dir.name
        args.output.mkdir(parents=True, exist_ok=True)

    demo = _load_demo_module()
    report = demo.run_demo(args)

    if write_summary and report.output_dir:
        out_dir = Path(report.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(
            json.dumps(asdict(report), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return report


__all__ = [
    "run_episode",
    "build_default_args",
    "canonical_episode_dir",
    "DEFAULT_TRAJ_ROOT",
]

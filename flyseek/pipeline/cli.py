# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""``flyseek-bench`` CLI — generate / instruct / eval / stats subcommands.

  flyseek-bench gen      --episodes 6 --env env_airsim_16
  flyseek-bench instruct --root output/trajectories/env_airsim_16
  flyseek-bench eval     --root output/trajectories/env_airsim_16
  flyseek-bench stats    --root output/trajectories/env_airsim_16

``gen`` requires a running AirSim/AirVLN instance; ``instruct``/``eval``/``stats``
are offline and operate on already-generated episodes.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STATS_DIR = REPO_ROOT / "flyseek_extend" / "output" / "stats"


def _cmd_gen(args: argparse.Namespace) -> int:
    from flyseek.pipeline.batch_runner import BatchConfig, run_batch

    mix = {"easy": args.easy, "medium": args.medium, "hard": args.hard}
    mix = {k: v for k, v in mix.items() if v > 0} or {"medium": 1}
    cfg = BatchConfig(
        episodes=args.episodes,
        env=args.env,
        difficulty_mix=mix,
        seed=args.seed,
    )
    manifest = run_batch(cfg, batch_id=args.batch_id)
    ok = sum(1 for r in manifest["results"] if r.get("success"))
    print(f"[ok] batch {manifest['batch_id']}: {ok}/{args.episodes} episodes succeeded")
    print(f"  manifest -> {manifest.get('manifest_path')}")
    return 0 if ok > 0 else 1


def _cmd_instruct(args: argparse.Namespace) -> int:
    import yaml  # type: ignore

    from flyseek.eval.episode_eval import find_episode_dirs
    from flyseek.instruction.pipeline import InstructionPipeline

    llm_cfg = None
    if args.llm_config and Path(args.llm_config).exists():
        llm_cfg = yaml.safe_load(Path(args.llm_config).read_text(encoding="utf-8"))
    pipe = InstructionPipeline.from_config(llm_cfg or {"backend": "mock"})

    episode_dirs = find_episode_dirs(args.root)
    existing: list[str] = []
    n = 0
    for ep in episode_dirs:
        rec = pipe.generate_for_episode(ep, existing=existing)
        existing.extend(rec["instructions"])
        n += len(rec["instructions"])
    print(f"[ok] generated {n} instruction(s) across {len(episode_dirs)} episode(s)")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from flyseek.eval.cli import main as eval_main

    argv = [str(args.root)]
    if args.out:
        argv += ["--out", str(args.out)]
    return eval_main(argv)


def _cmd_stats(args: argparse.Namespace) -> int:
    from flyseek.pipeline.stats import build_report

    report = build_report(args.root)
    out = args.out
    if out is None:
        DEFAULT_STATS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        out = DEFAULT_STATS_DIR / f"{ts}_dataset_report.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"[ok] {report['total_trajectories']} trajectories — report -> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flyseek-bench",
                                     description="FlySeek-Bench data generation.")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("gen", help="Generate episodes (needs AirSim).")
    g.add_argument("--episodes", type=int, default=3)
    g.add_argument("--env", default="env_airsim_16")
    g.add_argument("--easy", type=int, default=1)
    g.add_argument("--medium", type=int, default=1)
    g.add_argument("--hard", type=int, default=1)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--batch-id", default=None)
    g.set_defaults(func=_cmd_gen)

    ins = sub.add_parser("instruct", help="Generate VLT instructions offline.")
    ins.add_argument("--root", type=Path, required=True)
    ins.add_argument("--llm-config", default=str(
        REPO_ROOT / "flyseek_extend" / "configs" / "llm_backend.yaml"))
    ins.set_defaults(func=_cmd_instruct)

    ev = sub.add_parser("eval", help="Score episodes with tracking metrics.")
    ev.add_argument("--root", type=Path, required=True)
    ev.add_argument("--out", type=Path, default=None)
    ev.set_defaults(func=_cmd_eval)

    st = sub.add_parser("stats", help="Dataset-level quality report.")
    st.add_argument("--root", type=Path, required=True)
    st.add_argument("--out", type=Path, default=None)
    st.set_defaults(func=_cmd_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

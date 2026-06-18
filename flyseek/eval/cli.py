# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""``flyseek-eval`` — score generated episodes and write a stats report."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from flyseek.eval.episode_eval import evaluate_batch

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STATS_DIR = REPO_ROOT / "flyseek_extend" / "output" / "stats"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate FlySeek episodes (Track-AUC, Lost-Rate, "
                    "Redetection-Time, Collision-Rate, FOV-keep)."
    )
    parser.add_argument("root", type=Path,
                        help="Episode dir or a batch root containing episodes.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Report JSON path (default: output/stats/<ts>_report.json).")
    args = parser.parse_args(argv)

    report = evaluate_batch(args.root)

    out = args.out
    if out is None:
        DEFAULT_STATS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        out = DEFAULT_STATS_DIR / f"{ts}_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = report.get("summary", {})
    print(f"[ok] evaluated {summary.get('episodes', 0)} episode(s)")
    if summary.get("episodes"):
        print(f"  Track-AUC    : {summary['track_auc']['mean']}")
        print(f"  Lost-Rate    : {summary['lost_rate']['mean']}")
        print(f"  Redetect (s) : {summary['redetection_time_s']['mean']}")
        print(f"  Collision    : {summary['collision_rate']['mean']}")
    print(f"  report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

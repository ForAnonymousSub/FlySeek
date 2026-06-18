# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""``flyseek_bench.metrics`` — re-export of :mod:`flyseek.bench.metrics`.

Runnable as ``python -m flyseek_bench.metrics --episode_dir <dir>``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running by file path from any cwd (e.g.
# ``python flyseek_extend/flyseek_bench/metrics.py``): ensure flyseek_extend is
# importable so ``flyseek.bench`` resolves.
_FLYSEEK_EXTEND = Path(__file__).resolve().parents[1]
if str(_FLYSEEK_EXTEND) not in sys.path:
    sys.path.insert(0, str(_FLYSEEK_EXTEND))

from flyseek.bench.metrics import (  # noqa: E402,F401
    MetricsConfig,
    compute_metrics,
    evaluate_episode_dir,
    main,
)

__all__ = ["MetricsConfig", "compute_metrics", "evaluate_episode_dir", "main"]


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""``flyseek_bench.validate_episode`` — re-export of :mod:`flyseek.bench.validate_episode`.

Runnable as ``python -m flyseek_bench.validate_episode --episode_dir <dir>``
or by file path from any cwd.
"""

from __future__ import annotations

import sys
from pathlib import Path

_FLYSEEK_EXTEND = Path(__file__).resolve().parents[1]
if str(_FLYSEEK_EXTEND) not in sys.path:
    sys.path.insert(0, str(_FLYSEEK_EXTEND))

from flyseek.bench.validate_episode import (  # noqa: E402,F401
    build_consistency_report,
    find_episode_dirs,
    main,
    validate_episode,
)

__all__ = ["validate_episode", "find_episode_dirs", "build_consistency_report", "main"]


if __name__ == "__main__":
    sys.exit(main())

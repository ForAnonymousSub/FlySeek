# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Compatibility namespace: ``flyseek_bench`` mirrors ``flyseek.bench``.

The implementation lives under ``flyseek.bench`` (consistent with the rest of the
package layout: ``flyseek.eval``, ``flyseek.instruction``, ...). This thin shim
exists so the documented commands ``python -m flyseek_bench.metrics`` and
``import flyseek_bench`` work as written.
"""

from flyseek.bench import *  # noqa: F401,F403
from flyseek import bench as _bench

__all__ = getattr(_bench, "__all__", [])

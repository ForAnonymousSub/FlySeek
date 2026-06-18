# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Instruction quality filter (SKILL §4.4 Step 6).

Three gates, in order:
  1. length in [min_words, max_words],
  2. blacklist clean (the three iron rules),
  3. near-duplicate rejection vs. already-accepted instructions.

Dedup uses a lightweight token Jaccard similarity (no sentence-transformers
dependency) so the filter runs offline and is unit-testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from flyseek.instruction import blacklist


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass
class FilterResult:
    kept: list[str] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)  # {text, reason}

    @property
    def blacklist_rejected(self) -> int:
        return sum(1 for r in self.rejected if r["reason"].startswith("blacklist"))


def filter_candidates(
    candidates: list[str],
    *,
    existing: list[str] | None = None,
    min_words: int = 5,
    max_words: int = 30,
    dup_threshold: float = 0.85,
) -> FilterResult:
    """Filter ``candidates``; ``existing`` seeds the dedup set (cross-episode)."""
    result = FilterResult()
    accepted = list(existing or [])

    for cand in candidates:
        text = " ".join(cand.split())
        n_words = len(text.split())
        if n_words < min_words or n_words > max_words:
            result.rejected.append({"text": text, "reason": f"length:{n_words}"})
            continue
        ok, reason = blacklist.check(text)
        if not ok:
            result.rejected.append({"text": text, "reason": f"blacklist:{reason}"})
            continue
        if any(jaccard(text, prev) >= dup_threshold for prev in accepted):
            result.rejected.append({"text": text, "reason": "duplicate"})
            continue
        accepted.append(text)
        result.kept.append(text)

    return result


__all__ = ["FilterResult", "filter_candidates", "jaccard"]

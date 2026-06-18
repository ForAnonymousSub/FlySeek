# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""VLT-style instruction templates (SKILL §4.2 Step 4 — no LLM).

Three tiers mirror the three adversary difficulties:
  - ``easy``   : reference only (appearance).
  - ``medium`` : reference + behavior clause.
  - ``hard``   : reference + behavior + scene/cover clause.

Templates are loaded from ``configs/instruction_templates.yaml`` when present,
otherwise the built-in defaults below are used. Placeholders: ``{appearance}``,
``{behavior}``, ``{cover}``.
"""

from __future__ import annotations

import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEMPLATES_YAML = (
    REPO_ROOT / "flyseek_extend" / "configs" / "instruction_templates.yaml"
)

DEFAULT_TEMPLATES: dict[str, list[str]] = {
    "easy": [
        "Track {appearance}.",
        "Follow {appearance}.",
        "Keep tracking {appearance}.",
        "Stay with {appearance}.",
        "Maintain tracking of {appearance}.",
    ],
    "medium": [
        "Track {appearance} that is {behavior}.",
        "Follow {appearance} while it is {behavior}.",
        "Keep tracking {appearance} as it keeps {behavior}.",
        "Stay with {appearance}, which is {behavior}.",
        "Maintain tracking of {appearance} that is {behavior}.",
    ],
    "hard": [
        "Track {appearance} that is {behavior} near {cover}.",
        "Follow {appearance} as it is {behavior} around {cover}.",
        "Keep tracking {appearance}, which is {behavior} while using {cover}.",
        "Stay with {appearance} that is {behavior} amid {cover}.",
        "Maintain tracking of {appearance} that is {behavior} close to {cover}.",
    ],
}

TIER_FOR_DIFFICULTY: dict[str, str] = {
    "easy": "easy",
    "medium": "medium",
    "hide_seek": "hard",
    "hard": "hard",
}


def load_templates(path: Path | None = None) -> dict[str, list[str]]:
    """Load templates from YAML, falling back to built-in defaults."""
    path = path or DEFAULT_TEMPLATES_YAML
    try:
        import yaml  # type: ignore
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            tiers = data.get("tiers", data)
            out: dict[str, list[str]] = {}
            for tier in ("easy", "medium", "hard"):
                vals = tiers.get(tier)
                out[tier] = list(vals) if vals else list(DEFAULT_TEMPLATES[tier])
            return out
    except Exception:
        pass
    return {k: list(v) for k, v in DEFAULT_TEMPLATES.items()}


def tier_for_difficulty(difficulty: str) -> str:
    return TIER_FOR_DIFFICULTY.get(difficulty, "medium")


def fill(
    tier: str,
    *,
    appearance: str,
    behavior: str = "",
    cover: str = "the surrounding buildings",
    n: int = 3,
    templates: dict[str, list[str]] | None = None,
    rng: random.Random | None = None,
) -> list[str]:
    """Return up to ``n`` distinct filled candidate strings for ``tier``."""
    templates = templates or load_templates()
    rng = rng or random.Random()
    pool = list(templates.get(tier, DEFAULT_TEMPLATES.get(tier, DEFAULT_TEMPLATES["easy"])))
    rng.shuffle(pool)
    out: list[str] = []
    for tmpl in pool:
        text = tmpl.format(appearance=appearance, behavior=behavior, cover=cover)
        text = " ".join(text.split())  # collapse whitespace
        if text not in out:
            out.append(text)
        if len(out) >= n:
            break
    return out


__all__ = [
    "DEFAULT_TEMPLATES", "TIER_FOR_DIFFICULTY",
    "load_templates", "tier_for_difficulty", "fill",
]

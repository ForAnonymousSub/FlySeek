# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Pluggable VLM/LLM backends for instruction generation (SKILL §4.6).

Only the offline ``mock`` backend is implemented in this scope; ``qwen_vl_local``
/ ``openai`` / ``claude`` are declared as explicit hooks that raise
``NotImplementedError`` with guidance, so wiring a real backend later is a
drop-in (no pipeline changes).

A backend provides:
  - ``extract_appearance(image_path, bbox)`` -> appearance dict / phrase (Step 1)
  - ``polish(text)``                          -> rewritten text (Step 5, optional)
"""

from __future__ import annotations

import random
from typing import Any

# Canned appearance phrases keyed loosely by the OpenFly aim_landmark label.
_MOCK_APPEARANCE_POOL: dict[str, list[str]] = {
    "car": [
        "the small car",
        "the compact sedan",
        "the dark-colored car",
        "the little hatchback",
    ],
    "vehicle": [
        "the small vehicle",
        "the moving vehicle",
    ],
    "default": [
        "the small motorized car",
        "the tracked vehicle",
    ],
}


class LLMBackend:
    """Backend interface."""

    name = "base"

    def extract_appearance(
        self, image_path: str | None = None, bbox: list[float] | None = None,
        *, label_hint: str = "",
    ) -> str:
        raise NotImplementedError

    def polish(self, text: str) -> str:
        return text


class MockBackend(LLMBackend):
    """Deterministic offline backend (no GPU / no network)."""

    name = "mock"

    def __init__(self, seed: int | None = None, appearance_pool: dict | None = None):
        self._rng = random.Random(seed)
        self._pool = appearance_pool or _MOCK_APPEARANCE_POOL

    def _bucket(self, label_hint: str) -> str:
        low = (label_hint or "").lower()
        if "car" in low or "sedan" in low or "taxi" in low or "hatchback" in low:
            return "car"
        if "vehicle" in low:
            return "vehicle"
        return "default"

    def extract_appearance(
        self, image_path: str | None = None, bbox: list[float] | None = None,
        *, label_hint: str = "",
    ) -> str:
        bucket = self._bucket(label_hint)
        phrases = self._pool.get(bucket, self._pool["default"])
        return self._rng.choice(phrases)

    def polish(self, text: str) -> str:
        return text


def create_backend(config: dict[str, Any] | None = None) -> LLMBackend:
    """Instantiate a backend from a config dict (``{"backend": "mock", ...}``)."""
    config = config or {}
    name = str(config.get("backend", "mock")).lower()
    if name == "mock":
        mock_cfg = config.get("mock", {}) or {}
        return MockBackend(
            seed=mock_cfg.get("seed"),
            appearance_pool=mock_cfg.get("appearance_pool"),
        )
    if name in ("qwen_vl_local", "openai", "claude", "glm", "gemini"):
        raise NotImplementedError(
            f"LLM backend {name!r} is a declared hook but not implemented in this "
            "scope. Implement extract_appearance/polish here and set "
            "configs/llm_backend.yaml backend accordingly. Use 'mock' for now."
        )
    raise ValueError(f"unknown LLM backend: {name!r}")


__all__ = ["LLMBackend", "MockBackend", "create_backend"]

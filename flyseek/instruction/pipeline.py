# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Six-step instruction generation pipeline (SKILL §4.4), mock-driven.

Per episode directory (containing ``flyseek_meta.jsonl`` and ``pose.jsonl``):

  1. appearance   -> backend.extract_appearance(first frame, bbox)
  2. behavior     -> behavior_analyzer.classify_from_meta(...)   (numeric)
  3. cover        -> hard tier only (mock: generic scene phrase)
  4. templates    -> templates.fill(tier, appearance, behavior, cover)
  5. polish       -> backend.polish(...) (no-op for mock)
  6. quality      -> quality_filter.filter_candidates(...)

Writes ``instruction.json`` into the episode directory and returns the record.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flyseek.instruction import behavior_analyzer, quality_filter, templates
from flyseek.instruction.llm_backend import LLMBackend, MockBackend, create_backend


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _aim_landmark_label(pose_records: list[dict]) -> str:
    for rec in reversed(pose_records):
        lm = rec.get("aim_landmark")
        if lm:
            return str(lm.get("feature") or lm.get("shape") or "vehicle")
    return "vehicle"


def _first_rgb_path(episode_dir: Path, meta_records: list[dict]) -> str | None:
    cand = episode_dir / "image_0.png"
    if cand.exists():
        return str(cand)
    for rec in meta_records:
        imgs = rec.get("images") or {}
        if imgs.get("rgb"):
            return str(imgs["rgb"])
    return None


@dataclass
class InstructionPipeline:
    backend: LLMBackend
    templates_map: dict[str, list[str]]
    n_candidates: int = 5
    n_keep: int = 3
    dup_threshold: float = 0.85

    @classmethod
    def from_config(cls, llm_config: dict[str, Any] | None = None,
                    templates_path: Path | None = None,
                    **kwargs) -> "InstructionPipeline":
        return cls(
            backend=create_backend(llm_config),
            templates_map=templates.load_templates(templates_path),
            **kwargs,
        )

    def generate_for_episode(
        self,
        episode_dir: Path,
        *,
        difficulty: str | None = None,
        existing: list[str] | None = None,
        write: bool = True,
    ) -> dict[str, Any]:
        episode_dir = Path(episode_dir)
        meta = _read_jsonl(episode_dir / "flyseek_meta.jsonl")
        pose = _read_jsonl(episode_dir / "pose.jsonl")

        if difficulty is None:
            difficulty = next(
                (str(r.get("difficulty")) for r in meta if r.get("difficulty")),
                "medium",
            )
        tier = templates.tier_for_difficulty(difficulty)

        label = _aim_landmark_label(pose)
        bbox = next(
            (r["target_state"].get("bbox_2d") for r in meta
             if r.get("target_state", {}).get("bbox_2d")),
            None,
        )
        rgb = _first_rgb_path(episode_dir, meta)

        # Step 1 — appearance.
        appearance = self.backend.extract_appearance(rgb, bbox, label_hint=label)
        # Step 2 — behavior (numeric).
        behavior = behavior_analyzer.classify_from_meta(meta)
        # Step 3 — cover (hard only; mock generic).
        cover = "the nearby buildings"
        # Step 4 — templates.
        cands = templates.fill(
            tier,
            appearance=appearance,
            behavior=behavior.descriptor,
            cover=cover,
            n=self.n_candidates,
            templates=self.templates_map,
        )
        # Step 5 — polish (no-op for mock).
        cands = [self.backend.polish(c) for c in cands]
        # Step 6 — quality filter.
        filtered = quality_filter.filter_candidates(
            cands, existing=existing, dup_threshold=self.dup_threshold,
        )
        instructions = filtered.kept[: self.n_keep]

        record = {
            "episode_dir": str(episode_dir),
            "difficulty": difficulty,
            "tier": tier,
            "backend": self.backend.name,
            "appearance": appearance,
            "behavior": behavior.as_dict(),
            "instructions": instructions,
            "rejected": filtered.rejected,
            "frames": len(meta),
        }
        if write:
            (episode_dir / "instruction.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        return record


def generate_for_episode(
    episode_dir: Path,
    *,
    llm_config: dict[str, Any] | None = None,
    difficulty: str | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Convenience wrapper using a mock-or-configured backend."""
    pipe = InstructionPipeline.from_config(llm_config or {"backend": "mock"})
    return pipe.generate_for_episode(episode_dir, difficulty=difficulty, write=write)


__all__ = ["InstructionPipeline", "generate_for_episode", "MockBackend"]

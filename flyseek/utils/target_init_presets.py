# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Load per-environment target init profiles from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flyseek.utils.target_init import TargetInitConfig

_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
_ENV_FILES: dict[str, str] = {
    "env_airsim_16": "target_init_env_airsim_16.yaml",
}


@dataclass(frozen=True)
class TargetInitProfile:
    name: str
    env: str
    description: str
    config: TargetInitConfig
    strategy: str
    use_road_seed_fallback: bool
    road_seed_search_radius_m: float
    road_seed_sample_step_m: float
    road_seed_relaxed_accept: bool = True
    road_seed_min_road_score: float = 14.0
    road_seed_min_corridor_m: float = 6.0


def _repo_configs_dir() -> Path:
    return _CONFIGS_DIR


def preset_path_for_env(env_name: str) -> Path:
    fname = _ENV_FILES.get(env_name)
    if not fname:
        raise KeyError(
            f"No target-init preset file for env '{env_name}'. "
            f"Known: {sorted(_ENV_FILES)}"
        )
    return _repo_configs_dir() / fname


def load_env_preset_doc(env_name: str) -> dict[str, Any]:
    path = preset_path_for_env(env_name)
    if not path.is_file():
        raise FileNotFoundError(f"Preset file missing: {path}")
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise ImportError("PyYAML required: pip install pyyaml") from e
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def list_profiles(env_name: str) -> list[str]:
    doc = load_env_preset_doc(env_name)
    profiles = doc.get("profiles") or {}
    return sorted(profiles.keys())


def default_profile_name(env_name: str) -> str:
    doc = load_env_preset_doc(env_name)
    return str(doc.get("default_profile", "standard"))


def load_target_init_profile(
    env_name: str,
    profile_name: str | None = None,
) -> TargetInitProfile:
    doc = load_env_preset_doc(env_name)
    pname = profile_name or doc.get("default_profile", "standard")
    profiles = doc.get("profiles") or {}
    if pname not in profiles:
        raise KeyError(
            f"Profile '{pname}' not in {env_name} presets. "
            f"Available: {sorted(profiles)}"
        )
    raw = profiles[pname]
    cfg_fields = {
        k: raw[k]
        for k in (
            "min_corridor_width_m",
            "min_forward_ray_m",
            "max_ground_slope_m",
            "min_vertical_clearance_m",
            "min_open_ray_sum_m",
            "min_drive_feasibility_m",
            "search_radius_m",
            "sample_step_m",
            "max_samples",
            "min_accept_score",
            "prefer_near_anchor_m",
            "max_shift_from_anchor_m",
        )
        if k in raw
    }
    return TargetInitProfile(
        name=pname,
        env=str(doc.get("env", env_name)),
        description=str(raw.get("description", "")),
        config=TargetInitConfig(**cfg_fields),
        strategy=str(raw.get("strategy", "spiral_then_road_seed")),
        use_road_seed_fallback=bool(raw.get("use_road_seed_fallback", True)),
        road_seed_search_radius_m=float(raw.get("road_seed_search_radius_m", 200.0)),
        road_seed_sample_step_m=float(raw.get("road_seed_sample_step_m", 10.0)),
        road_seed_relaxed_accept=bool(raw.get("road_seed_relaxed_accept", True)),
        road_seed_min_road_score=float(raw.get("road_seed_min_road_score", 14.0)),
        road_seed_min_corridor_m=float(raw.get("road_seed_min_corridor_m", 6.0)),
    )


__all__ = [
    "TargetInitProfile",
    "default_profile_name",
    "list_profiles",
    "load_env_preset_doc",
    "load_target_init_profile",
    "preset_path_for_env",
]

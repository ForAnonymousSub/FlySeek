# FlySeek

**An Adversarial Aerial Vision-Language Tracking (VLT) Benchmark & Data-Generation Framework**

[![Python](https://img.shields.io/badge/python-3.10-4B8BBE.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

---

## Overview

FlySeek targets a problem that aerial vision-language *navigation* (VLN) does not address:
**continuously keeping a moving, actively evading target in view from a UAV, given a
referring-style language description**. Where navigation ends once a static goal is
reached, tracking only *begins* there — the target drives away, turns sharply, feints,
and hides behind buildings, and success is judged over the **entire process**, not at a
single endpoint.

FlySeek is a **self-contained Python framework** that (1) drives an adversarial target
and a visibility-aware expert observer entirely offline in NumPy, (2) renders the
resulting episodes through pluggable backends, and (3) emits standardized,
language-conditioned tracking episodes with full evaluation metrics.

| | Aerial VLN (prior platforms) | **FlySeek (this work)** |
|---|---|---|
| Task | reach a static goal from a route command | track a dynamic, adversarial target |
| Language | instructs *how to fly* | *refers to who to track* + predicts behavior |
| Target | fixed landmark | actively evading / occlusion-seeking vehicle |
| Horizon | one-shot plan to an endpoint | continuous, process-level |
| Metrics | NE / SR / OSR / SPL (endpoint) | Track-AUC / Lost-Rate / Redetection-Time (process) |

## What this repository contributes

The modules below are the substance of FlySeek. They are implemented from scratch, run
**fully offline**, and are independent of any simulator's flight-control stack:

- **Adversarial target engine** — `flyseek/bench/target_policy.py`, `flyseek/adversary/`.
  Five deterministic, seed-reproducible behaviors (`direct_escape`, `sharp_turn`,
  `detour_feint`, `occlusion_seeking`, `alley_hutong`) over three difficulty tiers, with
  both per-step and waypoint interfaces, road-graph snapping, and kinematic limits.
- **Visibility-aware preemptive expert** — `flyseek/bench/expert_trajectory.py`.
  A viewpoint planner that scores candidate observer poses over a look-ahead horizon of
  the target's *future* motion
  (`α·visibility − β·occlusion − γ·distance − η·collision − μ·smoothness`), so it
  anticipates occlusion rather than trailing the target (not shortest-path).
- **Offline geometric reasoning** — `flyseek/adapters/pcd_occupancy.py`, `flyseek/utils/`.
  Point-cloud occupancy maps, ray-cast line-of-sight, cover/alley route planning, and
  segmentation-derived building maps — all computed without the renderer.
- **Visibility evaluator** — `flyseek/bench/visibility.py`.
  Standardizes raw view judgments into `in_camera_frustum` / `line_of_sight_clear` /
  `visibility_score` / `occlusion_risk`, with documented fallbacks (never silently
  invents geometry).
- **Language-conditioned VLT instructions** — `flyseek/bench/instruction_generator.py`,
  `flyseek/instruction/`. Attribute-grounded referring expressions across four template
  families, a "three iron rules" safety blacklist, quality filtering, and pluggable
  LLM/VLM backends (mock / local Qwen-VL / OpenAI / Claude / GLM).
- **Standardized dataset schema & I/O** — `flyseek/bench/schema.py`, `export.py`.
  Typed episode / frame / instruction / trajectory / metric records with validation.
- **Process-level tracking metrics & validator** — `flyseek/bench/metrics.py`,
  `validate_episode.py`. Episode metrics plus an episode/batch paper-consistency report.
- **Batch generator** — `flyseek_bench/run_generate_episodes.py`. One-command episode
  generation with a fully offline `--dry_run` mode and built-in sanity checks.
- **Multi-backend rendering** — `flyseek/render/`, `flyseek/adapters/`. AirSim teleport,
  UnrealCV (UE5), and 3D Gaussian Splatting, behind one interface.
- **Reproducibility** — an offline unit-test suite (25+ modules) that needs no simulator.

## Public API (`flyseek_bench`)

Every stage of the framework is a runnable module / importable surface:

```
flyseek_bench.schema                # typed records + validation
flyseek_bench.export                # JSON / JSONL writers
flyseek_bench.target_policy         # adversarial behaviors
flyseek_bench.visibility            # per-frame visibility standardization
flyseek_bench.instruction_generator # language-conditioned instructions
flyseek_bench.expert_trajectory     # visibility-aware expert annotation
flyseek_bench.metrics               # python -m flyseek_bench.metrics --episode_dir ...
flyseek_bench.validate_episode      # python -m flyseek_bench.validate_episode --batch_dir ...
flyseek_bench.run_generate_episodes # python -m flyseek_bench.run_generate_episodes ...
```

## Pipeline

```
adversarial target policy (offline, NumPy)
   → visibility-aware preemptive expert viewpoints (offline, NumPy)
   → render backend  ──┬─ AirSim teleport
                       ├─ UnrealCV (UE5)
                       └─ 3D Gaussian Splatting
   → standardized episode: metadata.json + frames.jsonl + trajectories.json
                           + instruction.json + visibility.json + metrics.json
   → language-conditioned VLT instructions (mock / Qwen-VL / OpenAI / Claude / GLM)
   → process-level tracking metrics (Track-AUC, Lost-Rate, Redetection-Time, Collision-Rate)
```

The target policy, expert planner, geometric reasoning, instruction generation, and
evaluation all run **without a simulator** — the renderer is only needed to turn the
planned poses into RGB frames, and it is swappable.

## Installation

```bash
git clone <your-fork-url> FlySeek
cd FlySeek
python -m venv .venv && source .venv/bin/activate   # or use conda
pip install -e .                              # core: offline pipeline + tests
pip install -e ".[llm-local,quality,dev]"     # + local VLM, scoring, dev/test extras
```

The core package and the offline `--dry_run` pipeline run standalone. **Simulator-backed
rendering** additionally needs the corresponding simulator and 3-D scene assets — see
*Runtime backends* below.

## Quick start

```bash
# A) Fully offline — exercise the whole pipeline with placeholder frames (no simulator)
python -m flyseek_bench.run_generate_episodes \
  --scene_id env_airsim_16 --difficulty hard --behavior occlusion_seeking \
  --seed 42 --num_episodes 3 --output_dir output/bench --dry_run

# B) Offline test suite
pytest tests -q

# C) Metrics, then validation + paper-consistency report
python -m flyseek_bench.metrics          --episode_dir output/bench/<episode_id>
python -m flyseek_bench.validate_episode --batch_dir   output/bench
```

## Runtime backends

For photorealistic frames FlySeek plugs into external simulators. The AirSim, UnrealCV
(UE5), and 3D-Gaussian-Splatting backends each need a simulator install plus 3-D scene
assets (point clouds, segmentation maps, scene binaries). In our experiments these are
provided by an [OpenFly-Platform](https://github.com/SHAILAB-IPEC/OpenFly-Platform)
checkout: place this repository inside it as `flyseek_extend/`, start a scene, then run a
demo (full command set in `shell/demo.sh`):

```bash
bash envs/airsim/env_airsim_16/LinuxNoEditor/start.sh
python flyseek_extend/scripts/demo_adversary_chase.py \
  --env env_airsim_16 --auto-from-scout --init-profile standard \
  --target-behavior occlusion_seeking --target-policy-difficulty hard \
  --seed 66 --duration 75
```

FlySeek talks to the simulator through a thin TCP teleport client and **does not import
the simulator's Python code** — the simulator only serves rendered frames, and can be
replaced by UE5/UnrealCV or 3D-GS without touching the FlySeek core.

## Repository layout

```
FlySeek/
├── flyseek/                  # core framework (renderer-agnostic, runs offline)
│   ├── bench/                # ★ schema, target policy, expert planner, instructions, metrics, validator
│   ├── adversary/            # offline adversarial agents + factory
│   ├── expert/               # visibility-aware tracking drone / adaptive tracker
│   ├── instruction/          # VLT templates, safety blacklist, quality filter, LLM backends
│   ├── render/               # GS chase geometry, car compositor, depth / overlay
│   ├── adapters/             # TCP teleport client, AirSim / UnrealCV bridges, PCD occupancy
│   ├── scenarios/  utils/    # road graph, routes, visibility, target init, coords
│   └── pipeline/  eval/      # batch orchestration + evaluation
├── flyseek_bench/            # public runnable API  (python -m flyseek_bench.*)
├── configs/                  # difficulty, templates, LLM backend, camera configs
├── scripts/                  # demo + probe + verify scripts
├── shell/   tests/   assets/ # launchers · offline tests · target sprites
```

## Tracking metrics

- **Track-AUC** — mean fraction of frames the target is visible (∈ [0, 1]).
- **Lost-Rate** — fraction of frames the target is fully lost.
- **Redetection-Time** — mean time from a lost run to the next re-lock.
- **Line-of-sight continuity** — longest / average continuous visible segment.
- **Collision-Rate** — collisions per frame.
- **Path length / efficiency** — UAV travel cost vs. net displacement.

Episode success = visibility ratio ≥ difficulty threshold (easy/medium 0.70, hard 0.60)
and no collision; thresholds are configurable.

## Roadmap

- [ ] Humanoid targets (current targets are in-scene vehicle stand-ins).
- [ ] Large-scale dataset release.

## License & attribution

FlySeek is original software © 2026 **JoshuaWen**, released under the
[MIT License](./LICENSE). It is designed to interoperate with third-party simulators
(AirSim, UnrealCV, 3D Gaussian Splatting) and, in our experiments, with OpenFly-Platform
as a rendering/scene backend. FlySeek does **not** redistribute any simulator source,
binaries, scene data, or model weights; those remain the property of their respective
authors. See [NOTICE](./NOTICE) for details.

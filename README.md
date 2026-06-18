# FlySeek

**Adversarial Aerial Visual-Language Tracking (VLT) Benchmark**.

[![Python](https://img.shields.io/badge/python-3.10-4B8BBE.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

---

## What is FlySeek

VLN answers *"navigate to a static goal from a language command"* (Vision-Language
**Navigation**). FlySeek answers *"keep tracking a moving, hiding target described by
language"* (Vision-Language **Tracking**). A UAV must keep an **adversarial, evasive,
occlusion-seeking** target in view, guided by *referring-style* instructions such as
`"Track the red SUV"`.

| | VLN | **FlySeek** |
|---|---|---|
| Task | navigate to a static goal | continuously track a dynamic adversarial target |
| Instruction | command *how to fly* | refer to *who to track* + behavior prediction |
| Target | fixed landmark | actively evading / hiding vehicle |
| Evaluation | NE / SR / OSR / SPL (endpoint) | Track-AUC / Lost-Rate / Redetection-Time (process) |

## Pipeline (partial content for now, still being updated)

```
adversarial target policy (offline, numpy)
        → visibility-aware expert viewpoint (offline, numpy)
        → render backend ──┬─ AirSim teleport
                           ├─ UnrealCV (UE5)
                           └─ 3D Gaussian Splatting
        → OpenFly-compatible pose.jsonl + PNG + flyseek_meta.jsonl
        → VLT instructions (mock / local Qwen-VL / OpenAI / Claude / GLM)
        → tracking metrics (Track-AUC, Lost-Rate, Redetection-Time, Collision-Rate)
```

## Features

- **Adversarial targets**: 5 deterministic behaviors (`direct_escape`, `sharp_turn`,
  `detour_feint`, `occlusion_seeking`, `alley_hutong`) × 3 difficulty tiers, fully
  seed-reproducible.
- **Visibility-aware expert**: a preemptive viewpoint planner that anticipates
  occlusion (not shortest-path).
- **Offline scene geometry**: PCD occupancy maps, line-of-sight checks, cover/alley
  route planning, segmentation-based building maps.
- **VLT instruction generation**: referring + behavior templates, a "three iron rules"
  blacklist, quality filtering, multi-backend LLM.
- **Tracking metrics**: Track-AUC, Lost-Rate, Redetection-Time, Collision-Rate,
  FOV-keep-rate.
- **Three render backends**: AirSim teleport, UnrealCV (UE5), 3D Gaussian Splatting.
- **CI-friendly**: offline `--dry_run` generation and a unit-test suite that needs no
  simulator.

## Installation

FlySeek is an **overlay** on OpenFly-Platform: its contents become the
`flyseek_extend/` directory of an OpenFly checkout.

```bash
# 1. Get OpenFly-Platform first (simulators, scene_data, conda env)
git clone https://github.com/SHAILAB-IPEC/OpenFly-Platform.git
cd OpenFly-Platform

# 2. Clone FlySeek INTO it, named exactly `flyseek_extend`
git clone <your-fork-url> flyseek_extend

# 3. Install the package (in the OpenFly conda env)
conda activate openfly
cd flyseek_extend
pip install -e .                              # base
pip install -e ".[llm-local,quality,dev]"     # + local VLM, scoring, tests
```

> The package resolves the OpenFly root by walking up from `flyseek_extend/`, so the
> directory **must** be named `flyseek_extend`. Runtime simulator/scene assets
> (`scene_data/`, `envs/`) come from the OpenFly checkout and are not redistributed here.

## Quick start

```bash
# A) Fully offline — exercise the whole pipeline with placeholder frames (no simulator)
python -m flyseek_bench.run_generate_episodes \
  --scene_id env_airsim_16 --difficulty hard --behavior occlusion_seeking \
  --seed 42 --num_episodes 3 --output_dir flyseek_extend/output/bench --dry_run

# B) Run the offline test suite
pytest flyseek_extend/tests -q

# C) Compute metrics / validate a generated episode
python -m flyseek_bench.metrics         --episode_dir flyseek_extend/output/bench/<episode_id>
python -m flyseek_bench.validate_episode --episode_dir flyseek_extend/output/bench/<episode_id>
```

### With a running simulator

Start a scene from the OpenFly checkout, then run a demo (full command examples live in
`shell/demo.sh`):

```bash
# AirSim
bash envs/airsim/env_airsim_16/LinuxNoEditor/start.sh

# Occlusion-seeking chase (target hides behind annotated buildings; drone may lose track)
python flyseek_extend/scripts/demo_adversary_chase.py \
  --env env_airsim_16 --auto-from-scout --init-profile standard \
  --target-behavior occlusion_seeking --target-policy-difficulty hard \
  --seed 66 --duration 75

# Paired success-vs-fail tracking videos
python flyseek_extend/scripts/demo_chase_pair.py \
  --env env_airsim_16 --auto-from-scout --shared-seed 66 --duration 75

# UnrealCV (UE5) backend
bash flyseek_extend/shell/ue5.sh env_ue_smallcity
python flyseek_extend/scripts/demo_unrealcv_chase.py \
  --env env_ue_smallcity --target-behavior occlusion_seeking \
  --target-policy-difficulty hard --duration 30

# 3D Gaussian Splatting backend (geometry only / + render)
bash flyseek_extend/shell/demo_gs_ommo_urban.sh
bash flyseek_extend/shell/demo_gs_ommo_urban.sh --render
```

## Repository layout

```
flyseek_extend/                 # this repo, mounted into OpenFly-Platform/
├── flyseek/                    # main package (does not import OpenFly python code)
│   ├── adapters/               # OpenFly TCP teleport client, AirSim/UnrealCV bridges, PCD occupancy
│   ├── adversary/              # offline adversarial agents (easy / medium / hide-seek + factory)
│   ├── bench/                  # schema, target policy, expert trajectory, instructions, metrics
│   ├── eval/                   # episode evaluation + tracking metrics CLI
│   ├── expert/                 # visibility-aware tracking drone / adaptive tracker
│   ├── instruction/            # VLT instruction templates, blacklist, quality filter, LLM backends
│   ├── pipeline/               # batch orchestration + flyseek-bench CLI
│   ├── render/                 # GS chase geometry, car compositor, depth/overlay
│   ├── scenarios/              # road scenario controller
│   └── utils/                  # routes, road graph, visibility, target init, coords
├── flyseek_bench/              # runnable entry points (python -m flyseek_bench.*)
├── configs/                    # YAML configs (difficulty, templates, LLM backend, camera)
├── scripts/                    # demo + probe + verify scripts
├── shell/                      # launchers (AirSim/UE5/GS) + demo command cheatsheet
├── tests/                      # offline unit tests (no simulator)
└── assets/                     # target sprites
```

## Tracking metrics

- **Track-AUC** — mean fraction of frames the target is visible (∈ [0, 1]).
- **Lost-Rate** — fraction of frames the target is fully lost.
- **Redetection-Time** — mean seconds from a lost run to the next re-lock.
- **Collision-Rate** — collisions per frame.
- **FOV-keep-rate** — alias of Track-AUC for parity with OpenFly reporting.

## Roadmap

- [ ] Humanoid targets (currently uses in-scene vehicle stand-ins via `simSetObjectPose`).
- [ ] Large-scale dataset release.

## License & attribution

Original FlySeek code is © 2026 **JoshuaWen**, released under the [MIT License](./LICENSE).
See [NOTICE](./NOTICE) for the relationship to OpenFly-Platform. FlySeek does not
redistribute OpenFly source, simulator binaries, scene data, or model weights.

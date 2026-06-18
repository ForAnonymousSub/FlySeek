#!/usr/bin/env bash
# Validate env_gs_ommo_urban chase geometry + optional GS render.
#
# Phase 1 (CPU, always): regenerate trajectories.json + BEV overlay.
# Phase 2 (GPU + display): start SIBR viewer, render marked frames.
#
# Usage:
#   bash flyseek_extend/shell/demo_gs_ommo_urban.sh           # geometry only
#   bash flyseek_extend/shell/demo_gs_ommo_urban.sh --render  # + GS frames
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
ENV="env_gs_ommo_urban"
CFG="$REPO/flyseek_extend/configs/gs_chase_env_gs_${ENV#env_gs_}.yaml"
TRAJ="$REPO/flyseek_extend/output/gs_debug/chase_geom/trajectories.json"
DO_RENDER=false
[[ "${1:-}" == "--render" ]] && DO_RENDER=true

cd "$REPO"

echo "==> Phase A2: chase geometry (offline)"
PYTHONPATH=flyseek_extend python flyseek_extend/scripts/gen_gs_chase_geometry.py --config "$CFG"

echo ""
echo "Outputs:"
echo "  trajectories: $TRAJ"
echo "  BEV overlay:  $REPO/flyseek_extend/output/gs_debug/chase_geom/chase_bev.png"
echo "  analysis:     $REPO/flyseek_extend/output/gs_debug/${ENV}_*.png"

if ! $DO_RENDER; then
  echo ""
  echo "Geometry ready. To render GS background frames (needs GPU + X11):"
  echo "  bash flyseek_extend/shell/demo_gs_ommo_urban.sh --render"
  exit 0
fi

echo ""
echo "==> Phase B+C: GS render + 3D car mesh compositing (continuous frames)"
echo "    (render_gs_chase launches its own SIBR viewer; needs GPU + X11)"
python scripts/sim/render_gs_chase.py \
  --traj "$TRAJ" \
  --env "$ENV"

echo ""
echo "Episode outputs under $REPO/flyseek_extend/output/gs_debug/chase_geom/:"
echo "  frames_composited/   training images (GS background + target car mesh)"
echo "  masks/               per-frame instance masks"
echo "  frames_marked/       debug overlay (bbox + trail)"
echo "  annotations.jsonl    per-frame car/uav pose + bbox + mask path"

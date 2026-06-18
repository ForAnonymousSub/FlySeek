#!/usr/bin/env bash
# One-shot setup for env_gs_ommo_urban (mirrors urban_dense workflow).
# Prerequisite: a 3DGS (h3dgs) scene directory. Point SCENE_SRC at it, e.g.
#   SCENE_SRC=/path/to/h3dgs/ommo-urban bash flyseek_extend/shell/setup_gs_ommo_urban.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SCENE_SRC="${SCENE_SRC:-$HOME/Scenes/h3dgs/ommo-urban}"
ENV_NAME="env_gs_ommo_urban"

cd "$REPO"

echo "==> symlink envs/gs/${ENV_NAME} -> ${SCENE_SRC}"
ln -sfn "${SCENE_SRC}" "envs/gs/${ENV_NAME}"

echo "==> PLY -> PCD"
python scripts/toolchain/ply2pcd.py \
  --ply "envs/gs/${ENV_NAME}/camera_calibration/aligned/sparse/0/points3D.ply" \
  --out "scene_data/pcd_map/${ENV_NAME}.pcd" \
  --xmin -40 --xmax 40 --ymin -35 --ymax 35 --zmin -25 --zmax 5

echo "==> auto-seed seg_map landmarks"
python scripts/toolchain/gen_gs_seg_from_pcd.py \
  --pcd "scene_data/pcd_map/${ENV_NAME}.pcd" \
  --out "scene_data/seg_map/${ENV_NAME}.jsonl" \
  --z_min -12 --grid 8 --min_pts 30

echo "==> Phase A analysis"
PYTHONPATH=flyseek_extend python flyseek_extend/scripts/verify_gs_occupancy.py --env "${ENV_NAME}" --rebuild
PYTHONPATH=flyseek_extend python flyseek_extend/scripts/gs_ground_field.py --env "${ENV_NAME}" --xmin -30 --xmax 30 --ymin -25 --ymax 25 --max-road-z -12
PYTHONPATH=flyseek_extend python flyseek_extend/scripts/detect_cars_pcd.py --env "${ENV_NAME}" --xmin -30 --xmax 30 --ymin -25 --ymax 25
PYTHONPATH=flyseek_extend python flyseek_extend/scripts/gs_navmap.py --env "${ENV_NAME}" --xmin -30 --xmax 30 --ymin -25 --ymax 25

echo "==> Phase A2 chase geometry (offline, no render)"
bash flyseek_extend/shell/demo_gs_ommo_urban.sh

echo ""
echo "Done. Debug outputs under flyseek_extend/output/gs_debug/"
echo "Optional viewpoint scan (GPU, visual quality map):"
echo "  python scripts/sim/scan_gs_viewpoints.py --env ${ENV_NAME}"
echo "Render GS chase frames:"
echo "  bash flyseek_extend/shell/demo_gs_ommo_urban.sh --render"

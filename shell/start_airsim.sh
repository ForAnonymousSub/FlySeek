#!/usr/bin/env bash
# Headless launcher for the AirSim (AirVLN/UE4.27) env binary.
#
# Why this exists: the stock envs/.../LinuxNoEditor/start.sh launches with
# "-windowed", which makes UE4.27 create an on-screen window + swapchain and
# SEGV (Signal 11) at startup on headless / RDP / mismatched-display sessions.
# This teleport + simGetImages pipeline renders to offscreen targets only
# (~/Documents/AirSim/settings.json has "ViewMode": "NoDisplay"), so we launch
# with "-RenderOffScreen" instead — no window, GPU still used for capture.
#
# Usage:
#   bash flyseek_extend/shell/start_airsim.sh                 # env_airsim_16
#   bash flyseek_extend/shell/start_airsim.sh env_airsim_18   # other env
#   ENV_AIRSIM_RES=1920x1080 bash flyseek_extend/shell/start_airsim.sh
#
# Then in another terminal run the demo, e.g.:
#   python flyseek_extend/scripts/demo_alley_chase.py \
#     --env env_airsim_16 --auto-from-scout --seed 66 --duration 55
set -euo pipefail

ENV_NAME="${1:-env_airsim_16}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="$REPO_ROOT/envs/airsim/$ENV_NAME/LinuxNoEditor/AirVLN/Binaries/Linux/AirVLN-Linux-Shipping"

if [ ! -f "$BIN" ]; then
  echo "[start_airsim] binary not found: $BIN" >&2
  echo "[start_airsim] check that envs/airsim/$ENV_NAME is unzipped." >&2
  exit 1
fi

RES="${ENV_AIRSIM_RES:-1280x720}"
RESX="${RES%x*}"
RESY="${RES#*x}"

chmod +x "$BIN" 2>/dev/null || true

echo "[start_airsim] env=$ENV_NAME  res=${RESX}x${RESY}  (headless / -RenderOffScreen)"
echo "[start_airsim] binary=$BIN"
echo "[start_airsim] waiting for RPC on 127.0.0.1:41451 — keep this terminal open."

# -RenderOffScreen : offscreen render targets, no window (avoids the -windowed SEGV).
# -ResX/-ResY      : capture resolution backing store.
# Extra args after the env name are forwarded (e.g. -graphicsadapter=0).
exec "$BIN" AirVLN \
  -RenderOffScreen \
  -ResX="$RESX" -ResY="$RESY" \
  "${@:2}"

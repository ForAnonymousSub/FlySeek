#!/bin/bash
# Robust launcher for env_ue_smallcity (UnrealCV) with auto-retry.
#
# Why retry: UE5.2 + NVIDIA 570 intermittently hits VK_ERROR_DEVICE_LOST in the
# startup GPU benchmark (MeasureLongGPUTaskExecutionTime), ~9 s after launch.
# It is nondeterministic — a relaunch usually gets past it. This script keeps
# relaunching until the sim survives the startup window, then hands over.
set -u

# Usage: bash flyseek_extend/shell/ue5.sh [env_ue_smallcity|env_ue_bigcity]
ENV_NAME="${1:-env_ue_smallcity}"
# Repo root is the OpenFly-Platform checkout that hosts flyseek_extend/ (this
# file lives at <openfly>/flyseek_extend/shell/ue5.sh). Override with OPENFLY_ROOT.
OPENFLY_ROOT="${OPENFLY_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
ENV_DIR="${OPENFLY_ROOT}/envs/ue/${ENV_NAME}"
BIN="./City_UE52/Binaries/Linux/CitySample"
ARGS=(City_UE52 -RenderOffScreen -ResX=1280 -ResY=720)

export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json

MAX_TRIES=8
STABLE_S=35          # survive this long past launch => passed the GPU benchmark
cd "$ENV_DIR" || { echo "[ue5] no such env dir: $ENV_DIR"; exit 1; }
echo "[ue5] launching $ENV_NAME (offscreen, NVIDIA ICD, auto-retry)"

for attempt in $(seq 1 "$MAX_TRIES"); do
  echo "[ue5] launch attempt $attempt/$MAX_TRIES ..."
  pkill -9 -f CitySample 2>/dev/null; pkill -9 -f CrashReportClient 2>/dev/null
  sleep 2
  "$BIN" "${ARGS[@]}" &
  PID=$!

  ok=1
  for s in $(seq 1 "$STABLE_S"); do
    if ! kill -0 "$PID" 2>/dev/null; then
      echo "[ue5] process died at ~${s}s (startup device-lost?) — retrying"
      ok=0; break
    fi
    sleep 1
  done

  if [ "$ok" -eq 1 ] && kill -0 "$PID" 2>/dev/null; then
    echo "[ue5] survived ${STABLE_S}s — sim is up (pid $PID). UnrealCV on :9000."
    echo "[ue5] Ctrl-C here to stop. Now run the demo in another terminal."
    wait "$PID"
    exit $?
  fi
  kill -9 "$PID" 2>/dev/null
done

echo "[ue5] FAILED to launch after $MAX_TRIES tries. If this persists, the "
echo "      UE5.2 + NVIDIA 570 startup device-lost may need a driver downgrade "
echo "      (UE5.2-era ~535/545) or -norhithread."
exit 1

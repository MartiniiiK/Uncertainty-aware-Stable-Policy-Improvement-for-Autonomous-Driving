#!/usr/bin/env bash
set -euo pipefail
# Run as a non-root account with GPU device access and writable CARLA installation.
if [ "$(id -u)" = 0 ]; then echo 'Run as a non-root CARLA user' >&2; exit 1; fi
CARLA_ROOT=${CARLA_ROOT:?Set CARLA_ROOT}
ICD=${CARLA_VULKAN_ICD:?Set CARLA_VULKAN_ICD to a verified NVIDIA EGL ICD JSON}
test -x "$CARLA_ROOT/CarlaUE4.sh"; test -r "$ICD"
export VK_ICD_FILENAMES="$ICD" VK_DRIVER_FILES="$ICD"
export XDG_RUNTIME_DIR=$(mktemp -d /tmp/carla-runtime.XXXXXX)
chmod 700 "$XDG_RUNTIME_DIR"
cd "$CARLA_ROOT"
# -NoAsyncLoadingThread was used as a recovery diagnostic in the original deployment.
# Keep all rendering/physics settings otherwise unchanged.
exec ./CarlaUE4.sh -RenderOffScreen -NoAsyncLoadingThread -quality-level=Epic -carla-rpc-port=2000 -nosound -log

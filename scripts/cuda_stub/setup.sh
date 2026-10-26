#!/usr/bin/env bash
# Prepares DIR (default /tmp/tensomite-cuda) so that `-Dcuda` builds link and
# NVRTC runs on a machine with no NVIDIA driver: NVRTC from PyPI, and a stand-in
# libcuda whose every call fails with CUDA_ERROR_NO_DEVICE. Then:
#
#     LIBRARY_PATH=DIR LD_LIBRARY_PATH=DIR crystal spec -Dcuda spec/cuda_spec.cr
#
# The compiler specs run for real; the ones that need a GPU are pending.
set -euo pipefail
dir="${1:-/tmp/tensomite-cuda}"
mkdir -p "$dir"
python3 -m venv "$dir/venv"
"$dir/venv/bin/pip" install --quiet --disable-pip-version-check "nvidia-cuda-nvrtc==13.1.*"
lib="$(dirname "$(find "$dir/venv" -name 'libnvrtc.so.13' | head -1)")"
ln -sf "$lib/libnvrtc.so.13" "$dir/libnvrtc.so"
ln -sf "$lib/libnvrtc.so.13" "$dir/libnvrtc.so.13"
for builtins in "$lib"/libnvrtc-builtins.so*; do ln -sf "$builtins" "$dir/"; done
cc -shared -fPIC -o "$dir/libcuda.so" "$(dirname "$0")/libcuda.c"
echo "$dir"

#!/usr/bin/env bash
# Prepares DIR (default /tmp/tensomite-cuda) so that `-Dcuda` builds link and
# run on a machine with no NVIDIA driver: NVRTC from PyPI, a stand-in libcuda
# whose one "GPU" uses host memory (libcuda.c), and the CUDA device's kernels
# compiled as host C++ for it to run (emulate.h). Then:
#
#     export LIBRARY_PATH=DIR LD_LIBRARY_PATH=DIR TENSOMITE_CUDA_KERNELS=DIR/libkernels.so
#     crystal spec -Dcuda
#
# NVRTC compiles the kernels for real, and the CUDA device runs them on the
# host. Without TENSOMITE_CUDA_KERNELS, the device runs every op on host kernels.
set -euo pipefail
dir="${1:-/tmp/tensomite-cuda}"
mkdir -p "$dir"
python3 -m venv "$dir/venv"
"$dir/venv/bin/pip" install --quiet --disable-pip-version-check "nvidia-cuda-nvrtc==13.1.*"
lib="$(dirname "$(find "$dir/venv" -name 'libnvrtc.so.13' | head -1)")"
ln -sf "$lib/libnvrtc.so.13" "$dir/libnvrtc.so"
ln -sf "$lib/libnvrtc.so.13" "$dir/libnvrtc.so.13"
for builtins in "$lib"/libnvrtc-builtins.so*; do ln -sf "$builtins" "$dir/"; done
here="$(cd "$(dirname "$0")" && pwd)"
cc -shared -fPIC -o "$dir/libcuda.so" "$here/libcuda.c" -ldl
python3 "$here/register.py" "$here/../../src/tensomite/cuda/kernels.cu" > "$dir/registry.inc"
cat "$here/emulate.h" "$here/../../src/tensomite/cuda/kernels.cu" "$dir/registry.inc" > "$dir/kernels.cpp"
c++ -std=c++20 -O2 -ffp-contract=off -shared -fPIC -pthread -o "$dir/libkernels.so" "$dir/kernels.cpp"
echo "$dir"

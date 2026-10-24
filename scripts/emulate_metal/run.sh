#!/bin/sh
# Runs tensomite's Metal matmul kernels on the CPU and checks that every tiled
# variant gives split-K's bits, on a set of shapes. Exits non-zero on a mismatch.
#
#     scripts/emulate_metal/run.sh
#
# A reference for working on Metal kernels without a Mac, not part of CI: it
# checks indexing and the order of arithmetic, not MSL's own rules, which only
# the Metal compiler applies. Needs g++ with C++20.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "${work:?}"' EXIT

# The prelude, then the kernels from the serial matmul to the Metal 3 section.
python3 - "$root/src/tensomite/metal" "$work/kernels.inc" <<'PY'
import pathlib, sys
metal, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
prelude = (metal / "prelude.metal").read_text()
prelude = prelude.replace("#include <metal_stdlib>", "").replace("using namespace metal;", "")
source = (metal / "kernels.metal").read_text()
start = source.index("// Output element i of a batched matrix product")
end = source.index("#if __METAL_VERSION__ >= 300")
out.write_text(prelude + "\n" + source[start:end])
PY

g++ -std=c++20 -O1 -ffp-contract=off -fno-fast-math -pthread -Wno-attributes \
  -I"$here" -I"$work" "$here/matmul.cpp" -o "$work/matmul"
"$work/matmul"

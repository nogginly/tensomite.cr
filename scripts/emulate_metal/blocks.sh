#!/bin/sh
# Runs tensomite's Metal kernels for block formats on the CPU and checks that
# they give the host's bits, for Q8_0, Q4_0 and Q4_1. Exits non-zero on a
# mismatch.
#
#     scripts/emulate_metal/blocks.sh
#
# Like run.sh, a reference for working without a Mac, not part of CI: it checks
# indexing and the order of arithmetic, not MSL's own rules. On a Mac, division
# may differ from the host's in the last bit, so the GPU is checked within a
# tolerance there. Needs g++ with C++20 and _Float16, and crystal.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "${work:?}"' EXIT

python3 - "$root/src/tensomite/metal" "$work/blocks.inc" <<'PY'
import pathlib, sys
metal, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
prelude = (metal / "prelude.metal").read_text()
prelude = prelude.replace("#include <metal_stdlib>", "").replace("using namespace metal;", "")
source = (metal / "kernels.metal").read_text()
report = source[source.index("static void report("):source.index("// Output element i is column i % width")]
start = source.index("// Block formats (Q8_0, Q4_0, Q4_1)")
end = source.index("#if __METAL_VERSION__ >= 300")
out.write_text(prelude + "\n" + report + "\n" + source[start:end])
PY

crystal run "$here/blocks_data.cr" -- "$work/data.bin"
g++ -std=c++20 -O1 -ffp-contract=off -fno-fast-math -Wno-attributes \
  -I"$here" -I"$work" "$here/blocks.cpp" -o "$work/blocks"
"$work/blocks" "$work/data.bin"

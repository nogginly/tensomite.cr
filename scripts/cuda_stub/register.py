"""Prints a REGISTER line (see emulate.h) for each kernel in a CUDA source file.

A kernel is a function declared `extern "C" __global__`, or one a BINARY,
UNARY or TILED macro defines. One that uses shared memory, directly or through
the tiled matmul templates, is marked to run its threads as OS threads:
every kernel that synchronises its threads shares memory between them, and may
reach __syncthreads through a helper or a macro.
"""
import re
import sys

source = "".join(open(path).read() for path in sys.argv[1:])
for name in re.findall(r"^(?:BINARY|UNARY)\((\w+),", source, re.M):
    print(f"REGISTER({name}, false)")
for name in re.findall(r"^TILED\((\w+),", source, re.M):
    print(f"REGISTER({name}, true)")
parts = re.split(r'(?=^extern "C" __global__ void )', source, flags=re.M)
for part in parts:
    match = re.match(r'extern "C" __global__ void (\w+)\(', part)
    if match:
        print(f"REGISTER({match.group(1)}, {'true' if '__shared__' in part or '__syncthreads' in part or 'matmul_tiled' in part else 'false'})")

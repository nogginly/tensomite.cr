"""Prints a REGISTER line (see emulate.h) for each kernel in a CUDA source file.

A kernel is a function declared `extern "C" __global__`, or one a BINARY or
UNARY macro defines. One that calls __syncthreads is marked to run its threads
as OS threads.
"""
import re
import sys

source = open(sys.argv[1]).read()
for name in re.findall(r"^(?:BINARY|UNARY)\((\w+),", source, re.M):
    print(f"REGISTER({name}, false)")
parts = re.split(r'(?=^extern "C" __global__ void )', source, flags=re.M)
for part in parts:
    match = re.match(r'extern "C" __global__ void (\w+)\(', part)
    if match:
        print(f"REGISTER({match.group(1)}, {'true' if '__syncthreads' in part else 'false'})")

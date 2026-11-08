#!/bin/sh
# Downloads SmolLM2-135M-Instruct as GGUF files, at F16, Q8_0 and Q4_0, as
# llama.cpp's tools made them (bartowski's conversions on Hugging Face).
#
#     scripts/fetch_smollm2_gguf.sh [DIR]    # default: models/smollm2-135m-instruct-gguf
#     TENSOMITE_SMOLLM2_GGUF=models/smollm2-135m-instruct-gguf/SmolLM2-135M-Instruct-f16.gguf crystal spec spec/gguf_spec.cr
set -eu
dir="${1:-models/smollm2-135m-instruct-gguf}"
base="https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main"
mkdir -p "$dir"
for kind in f16 Q8_0 Q4_0; do
  file="SmolLM2-135M-Instruct-$kind.gguf"
  [ -f "$dir/$file" ] || curl -fsSL -o "$dir/$file" "$base/$file"
done
echo "$dir"

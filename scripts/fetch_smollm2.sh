#!/bin/sh
# Downloads SmolLM2-135M-Instruct's weights and tokeniser from Hugging Face.
#
#     scripts/fetch_smollm2.sh [DIR]    # default: models/smollm2-135m-instruct
#     TENSOMITE_SMOLLM2_DIR=models/smollm2-135m-instruct crystal spec
set -eu
dir="${1:-models/smollm2-135m-instruct}"
base="https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main"
mkdir -p "$dir"
for file in config.json generation_config.json tokenizer.json tokenizer_config.json model.safetensors; do
  [ -f "$dir/$file" ] || curl -fsSL -o "$dir/$file" "$base/$file"
done
echo "$dir"

#!/bin/sh
# Downloads a SmolLM2 Instruct model's weights and tokeniser from Hugging Face:
# SmolLM2-135M-Instruct by default, or the one MODEL names.
#
#     scripts/fetch_smollm2.sh [DIR] [MODEL]    # default: models/smollm2-135m-instruct SmolLM2-135M-Instruct
#     scripts/fetch_smollm2.sh models/smollm2-1.7b-instruct SmolLM2-1.7B-Instruct
#     TENSOMITE_SMOLLM2_DIR=models/smollm2-135m-instruct crystal spec
set -eu
dir="${1:-models/smollm2-135m-instruct}"
model="${2:-SmolLM2-135M-Instruct}"
base="https://huggingface.co/HuggingFaceTB/$model/resolve/main"
mkdir -p "$dir"
for file in config.json generation_config.json tokenizer.json tokenizer_config.json model.safetensors; do
  [ -f "$dir/$file" ] || curl -fsSL -o "$dir/$file" "$base/$file"
done
echo "$dir"

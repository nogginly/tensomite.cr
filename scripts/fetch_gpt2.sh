#!/bin/sh
# Downloads GPT-2 small's weights, tokeniser files and config from Hugging Face
# into DIR (default models/gpt2), skipping files already there.
#
#     scripts/fetch_gpt2.sh [DIR]
set -eu
dir="${1:-models/gpt2}"
base="https://huggingface.co/openai-community/gpt2/resolve/main"
mkdir -p "$dir"
for file in model.safetensors vocab.json merges.txt config.json; do
  if [ ! -s "$dir/$file" ]; then
    curl -fL --retry 3 -o "$dir/$file" "$base/$file"
  fi
done
echo "$dir"

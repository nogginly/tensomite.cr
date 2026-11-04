#!/bin/sh
# Downloads a GPT-2's weights, tokeniser files and config from Hugging Face into
# DIR, skipping files already there. SIZE is small (the default, 124M
# parameters), medium (355M), large (774M) or xl (1.5B); DIR defaults to
# models/gpt2 for small and models/gpt2-SIZE for the others.
#
#     scripts/fetch_gpt2.sh [SIZE] [DIR]
#     scripts/fetch_gpt2.sh DIR
set -eu
case "${1:-small}" in
  small | medium | large | xl) size="${1:-small}"; dir="${2:-}" ;;
  *) size=small; dir="$1" ;;
esac
case "$size" in
  small) repo=gpt2 ;;
  *) repo="gpt2-$size" ;;
esac
dir="${dir:-models/$repo}"
base="https://huggingface.co/openai-community/$repo/resolve/main"
mkdir -p "$dir"
for file in model.safetensors vocab.json merges.txt config.json; do
  if [ ! -s "$dir/$file" ]; then
    curl -fL --retry 3 -o "$dir/$file" "$base/$file"
  fi
done
echo "$dir"

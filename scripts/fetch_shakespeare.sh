#!/bin/sh
# Downloads Tiny Shakespeare, about 1.1 MB of plays from Karpathy's char-rnn, to
# DIR/input.txt (default data/shakespeare), skipping it when already there.
#
#     scripts/fetch_shakespeare.sh [DIR]
set -eu
dir="${1:-data/shakespeare}"
url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
mkdir -p "$dir"
if [ ! -s "$dir/input.txt" ]; then
  curl -fL --retry 3 -o "$dir/input.txt" "$url"
fi
echo "$dir/input.txt"

#!/bin/sh
# Writes a corpus of Crystal source to OUT (default data/crystal/input.txt): every
# `.cr` file under SRC (default the installed compiler's standard library), each
# after a `# File: path` line, in an order shuffled by a checksum of the path.
# The shuffle spreads every part of the library through the text, so the last
# tenth `apps/finetune` holds out for validation is a sample of whole files from
# all over it, not just the end of the alphabet.
#
#     scripts/make_code_corpus.sh [SRC] [OUT]
set -eu
src="${1:-$(crystal env CRYSTAL_PATH | tr ':' '\n' | tail -n 1)}"
out="${2:-data/crystal/input.txt}"
[ -d "$src" ] || { echo "no directory $src" >&2; exit 1; }
mkdir -p "$(dirname "$out")"
: > "$out"
(cd "$src" && find . -name '*.cr' -type f | sed 's|^\./||') |
  while IFS= read -r path; do
    printf '%s %s\n' "$(printf '%s' "$path" | cksum | cut -d ' ' -f 1)" "$path"
  done |
  LC_ALL=C sort -n | cut -d ' ' -f 2- |
  while IFS= read -r path; do
    printf '# File: %s\n\n' "$path" >> "$out"
    cat "$src/$path" >> "$out"
    printf '\n' >> "$out"
  done
echo "$out: $(grep -c '^# File: ' "$out") files, $(wc -c < "$out" | tr -d ' ') bytes"

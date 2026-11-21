#!/bin/sh
# Times builds of a Crystal program in three ways: from an empty cache, again
# with the cache warm, and after a one-line code change, for `--release` (one
# LLVM module) and `-O3` split into a module per type. Prints a Markdown table.
#
#     scripts/time_builds.sh [PROGRAM.cr] [EXTRA_FLAGS...]
#     scripts/time_builds.sh apps/finetune/finetune.cr
#     scripts/time_builds.sh bench/train_gpt2.cr -Dcuda
#
# The change is made to a copy of a small wrapper that requires PROGRAM, so no
# tracked file is touched. Each flag set has its own cache, emptied first.
set -eu
program=${1:-apps/finetune/finetune.cr}
[ "$#" -gt 0 ] && shift
root=$(cd "$(dirname "$0")/.." && pwd)
work=$(mktemp -d)
wrapper="$root/zz_time_builds_$$.cr"
trap 'rm -rf "${work:?}" "$wrapper"' EXIT

seconds() {
  start=$(date +%s.%N)
  crystal build "$@" "$wrapper" -o "$work/program" > "$work/log" 2>&1 || { cat "$work/log" >&2; exit 1; }
  echo "$(date +%s.%N) - $start" | bc | xargs printf "%.1f"
}

echo "flags|cold s|warm s|after an edit s"
echo "-----|------|------|---------------"
for flags in "--release" "-O3"; do
  export CRYSTAL_CACHE_DIR="$work/cache"
  rm -rf "$CRYSTAL_CACHE_DIR"
  printf 'require "./%s"\nputs "time_builds 1" if ARGV.includes?("--time-builds")\n' "$program" > "$wrapper"
  # shellcheck disable=SC2086
  cold=$(seconds $flags "$@")
  # shellcheck disable=SC2086
  warm=$(seconds $flags "$@")
  printf 'require "./%s"\nputs "time_builds 2" if ARGV.includes?("--time-builds")\n' "$program" > "$wrapper"
  # shellcheck disable=SC2086
  edit=$(seconds $flags "$@")
  echo "$flags $*|$cold|$warm|$edit"
done

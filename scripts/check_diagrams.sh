#!/bin/sh
# Renders every Mermaid block in the given Markdown files with mermaid-cli and
# reports each one that fails to parse. Exits non-zero if any fail.
#
#     scripts/check_diagrams.sh [FILE.md ...]   # default: every *.md at the root
#
# Set PUPPETEER_EXECUTABLE_PATH to use an installed Chromium instead of the one
# puppeteer downloads, e.g. /opt/pw-browsers/chromium in Claude Code's cloud
# containers.
set -eu
work=$(mktemp -d)
trap 'rm -rf "${work:?}"' EXIT
[ "$#" -gt 0 ] || set -- *.md

python3 - "$work" "$@" <<'PY'
import pathlib, re, sys
work, files = pathlib.Path(sys.argv[1]), sys.argv[2:]
for path in files:
    blocks = re.findall(r"```mermaid\n(.*?)```", pathlib.Path(path).read_text(), re.S)
    for n, block in enumerate(blocks, 1):
        (work / f"{pathlib.Path(path).stem}-{n:02}.mmd").write_text(block)
PY

config="$work/puppeteer.json"
if [ -n "${PUPPETEER_EXECUTABLE_PATH:-}" ]; then
  printf '{"executablePath": "%s", "args": ["--no-sandbox"]}' "$PUPPETEER_EXECUTABLE_PATH" > "$config"
else
  printf '{"args": ["--no-sandbox"]}' > "$config"
fi

failed=0
for diagram in "$work"/*.mmd; do
  [ -e "$diagram" ] || continue
  if npx -y -p @mermaid-js/mermaid-cli@11.4.2 mmdc -q -p "$config" -i "$diagram" -o "${diagram%.mmd}.svg" >"$work/err.txt" 2>&1; then
    echo "ok    $(basename "$diagram" .mmd)"
  else
    echo "FAIL  $(basename "$diagram" .mmd)"
    sed 's/^/      /' "$work/err.txt" | head -5
    failed=1
  fi
done
exit "$failed"

"""Pads Markdown tables the way the maintainer's editor does: no outer pipes, and
every column padded to its widest cell, including the last.

    python3 scripts/format_tables.py DESIGN.md DEVELOPMENT.md

Tables inside fenced code blocks are left alone. Rerunning is a no-op.
"""

import re, sys, unicodedata

def width(s):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)

def fmt(lines):
    rows = [[c.strip() for c in l.rstrip("\n").split("|")] for l in lines]
    sep = 1
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    w = [max(width(r[i]) for j, r in enumerate(rows) if j != sep) for i in range(n)]
    out = []
    for j, r in enumerate(rows):
        if j == sep:
            out.append("|".join("-" * w[i] for i in range(n)) + "\n")
        else:
            out.append("|".join(r[i] + " " * (w[i] - width(r[i])) for i in range(n)) + "\n")
    return out

for path in sys.argv[1:]:
    lines = open(path).readlines()
    res, i, fence = [], 0, False
    while i < len(lines):
        l = lines[i]
        if l.startswith("```"):
            fence = not fence
        if (not fence and "|" in l and i + 1 < len(lines)
                and re.fullmatch(r"-+(\|-+)+\s*", lines[i + 1])):
            j = i + 2
            while j < len(lines) and "|" in lines[j] and lines[j].strip():
                j += 1
            res += fmt(lines[i:j]); i = j; continue
        res.append(l); i += 1
    open(path, "w").writelines(res)

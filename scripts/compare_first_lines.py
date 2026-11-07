#!/usr/bin/env python3
"""Compares two models' first lines in `bench/code_eval.cr --verbose` output.

For each model it counts the completions that continue the `def` line before
the body, and those whose first line is the real one, as the evaluation's
"first line" column does. Then it lists the methods where only one of the two
models matched, with the real first line and each model's.

    scripts/compare_first_lines.py EVAL.txt MODEL_A MODEL_B
    scripts/compare_first_lines.py runs/crystal-large-3500/eval.txt runs/crystal/best runs/crystal-large-3500/best
"""
import re
import sys

HEADER = re.compile(r"(?m)^(real|models/[^ :\n]+|runs/[^ :\n]+)( \(no end\))?:")


def methods(text):
    """Returns `(title, {name: completion})` per method, `real` holding the real body."""
    text = text.split("\nmodel|body loss", 1)[0]
    for chunk in re.split(r"(?m)^## ", text)[1:]:
        title, rest = chunk.split("\n", 1)
        marks = list(HEADER.finditer(rest))
        parts = {}
        for i, mark in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(rest)
            parts[mark.group(1)] = rest[mark.end():end].rstrip("\n")
        yield title, parts


def first_line(completion):
    """The completion's first line as the evaluation compares it, or None when
    the completion continues the `def` line, which can never match."""
    if not completion.startswith("\n"):
        return None
    return completion[1:].split("\n", 1)[0]


def main(path, a, b):
    rows = list(methods(open(path).read()))
    for name in (a, b):
        missing = [t for t, parts in rows if name not in parts]
        if missing:
            sys.exit(f"{name} is not in {path}")
    print(f"{len(rows)} methods")
    for name in (a, b):
        continues = sum(first_line(p[name]) is None for _, p in rows)
        matches = sum(first_line(p[name]) == first_line(p["real"]) for _, p in rows)
        print(f"{name}: first line matches {matches}, continues the def line {continues}")
    only = {a: [], b: []}
    for title, parts in rows:
        real = first_line(parts["real"])
        hit = {name: first_line(parts[name]) == real for name in (a, b)}
        for name, other in ((a, b), (b, a)):
            if hit[name] and not hit[other]:
                only[name].append((title, parts))
    for name, other in ((a, b), (b, a)):
        print(f"\nonly {name} matches: {len(only[name])}")
        for title, parts in only[name]:
            print(f"## {title}")
            print(f"   real:  {first_line(parts['real'])}")
            print(f"   {name}: {first_line(parts[name])}")
            print(f"   {other}: {first_line(parts[other])}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])

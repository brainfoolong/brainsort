#!/usr/bin/env python3
"""The README's results table, derived from results/counts.csv so no number is typed by hand.

usage: python3 scripts/scorecard.py                  print the Markdown block
       python3 scripts/scorecard.py --write README.md  replace the block between the markers
       python3 scripts/scorecard.py --check README.md  exit 1 if the README block is out of date (CI)

The block counts, at n = 100,000 and over every key type and input pattern,
the cells in which brainsort has the lowest memory traffic, the fewest
comparisons and the least scratch memory: first against the other stable
sorts, then against every sort in the benchmark. The radix baselines are
brainsort's own building blocks and never count as opponents. These numbers
are deterministic, so the same table comes out on every machine; the timing
lives on the website.
"""
import csv
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COUNTS = os.path.join(ROOT, "results", "counts.csv")
CANDIDATE = "brainsort"
BASELINES = {"radix11", "radix16"}
N = 100000
METRICS = [("traffic_bytes", "memory traffic (bytes moved)"), ("compares", "comparisons"), ("aux_peak_bytes", "scratch memory")]
BEGIN, END = "<!-- scorecard:begin -->", "<!-- scorecard:end -->"


def load():
    cells = defaultdict(dict)
    with open(COUNTS, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["ok"] == "1" and int(r["n"]) == N:
                cells[(r["type"], r["dataset"])][r["algorithm"]] = r
    return cells


def wins(cells, metric, stable_only):
    won = total = 0
    for cell in cells.values():
        if CANDIDATE not in cell:
            continue
        opp = [float(r[metric]) for a, r in cell.items()
               if a != CANDIDATE and a not in BASELINES and (r["stable"] == "1" or not stable_only)]
        if not opp:
            continue
        total += 1
        won += float(cell[CANDIDATE][metric]) <= min(opp)
    return won, total


def block():
    cells = load()
    lines = [BEGIN,
             f"| deterministic, n = {N:,}, {len(cells)} cells | against the stable sorts | against every sort |",
             "|---|---:|---:|"]
    for metric, label in METRICS:
        ws, ts = wins(cells, metric, True)
        wa, ta = wins(cells, metric, False)
        lines.append(f"| {label}: brainsort lowest or tied | **{ws} / {ts}** | {wa} / {ta} |")
    lines.append(END)
    return "\n".join(lines)


def main():
    args = sys.argv[1:]
    text = block()
    if not args:
        print(text)
        return
    mode, path = args[0], args[1] if len(args) > 1 else os.path.join(ROOT, "README.md")
    with open(path, encoding="utf-8") as f:
        readme = f.read()
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    if not pattern.search(readme):
        sys.exit(f"{path}: no {BEGIN} ... {END} block")
    updated = pattern.sub(lambda _: text, readme)
    if mode == "--check":
        if updated != readme:
            sys.exit(f"{path}: the scorecard block is out of date; run: python3 scripts/scorecard.py --write {path}")
        print(f"{path}: scorecard block is current")
    elif mode == "--write":
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(updated)
        print(f"{path}: scorecard block {'updated' if updated != readme else 'unchanged'}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()

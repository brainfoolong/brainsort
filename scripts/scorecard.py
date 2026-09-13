#!/usr/bin/env python3
"""The README's results tables, derived from results/counts.csv and
results/rust-counts.csv so no number is typed by hand.

usage: python3 scripts/scorecard.py                  print the Markdown blocks
       python3 scripts/scorecard.py --write README.md  replace the blocks between the markers
       python3 scripts/scorecard.py --check README.md  exit 1 if a README block is out of date (CI)

The C++ block counts, at n = 100,000 and over every key type and input
pattern, the cells in which brainsort has the lowest memory traffic, the
fewest comparisons and the least scratch memory: first against the other
stable sorts, then against every sort in the benchmark. The radix baselines
are brainsort's own building blocks and never count as opponents. The Rust
block does the same against the Rust sorts as shipped on the two of those
metrics that can be counted without their source (comparisons and scratch
memory) plus the compare flips. These numbers are deterministic, so the same
tables come out on every machine; the timing lives on the website.
"""
import csv
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COUNTS = os.path.join(ROOT, "results", "counts.csv")
RUST_COUNTS = os.path.join(ROOT, "results", "rust-counts.csv")
CANDIDATE = "brainsort"
BASELINES = {"radix11", "radix16"}
N = 100000
METRICS = [("traffic_bytes", "memory traffic (bytes moved)"), ("compares", "comparisons"), ("aux_peak_bytes", "scratch memory")]
RUST_METRICS = [("compares", "comparisons"), ("cmp_flips", "compare flips"), ("aux_peak_bytes", "scratch memory")]
BEGIN, END = "<!-- scorecard:begin -->", "<!-- scorecard:end -->"
RUST_BEGIN, RUST_END = "<!-- scorecard-rust:begin -->", "<!-- scorecard-rust:end -->"


def load(path=COUNTS):
    cells = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
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


def block(cells, metrics, begin, end, what):
    lines = [begin,
             f"| deterministic, n = {N:,}, {len(cells)} cells | against the stable {what}s | against every {what} |",
             "|---|---:|---:|"]
    for metric, label in metrics:
        ws, ts = wins(cells, metric, True)
        wa, ta = wins(cells, metric, False)
        lines.append(f"| {label}: brainsort lowest or tied | **{ws} / {ts}** | {wa} / {ta} |")
    lines.append(end)
    return "\n".join(lines)


def blocks():
    """[(begin, end, text)]: the C++ block, and the Rust block when its file exists."""
    out = [(BEGIN, END, block(load(), METRICS, BEGIN, END, "sort"))]
    if os.path.exists(RUST_COUNTS):
        out.append((RUST_BEGIN, RUST_END, block(load(RUST_COUNTS), RUST_METRICS, RUST_BEGIN, RUST_END, "Rust sort")))
    return out


def main():
    args = sys.argv[1:]
    if not args:
        print("\n\n".join(t for _, _, t in blocks()))
        return
    mode, path = args[0], args[1] if len(args) > 1 else os.path.join(ROOT, "README.md")
    with open(path, encoding="utf-8") as f:
        readme = f.read()
    updated = readme
    for begin, end, text in blocks():
        pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.S)
        if not pattern.search(updated):
            sys.exit(f"{path}: no {begin} ... {end} block")
        updated = pattern.sub(lambda _: text, updated)
    if mode == "--check":
        if updated != readme:
            sys.exit(f"{path}: a scorecard block is out of date; run: python3 scripts/scorecard.py --write {path}")
        print(f"{path}: scorecard blocks are current")
    elif mode == "--write":
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(updated)
        print(f"{path}: scorecard blocks {'updated' if updated != readme else 'unchanged'}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()

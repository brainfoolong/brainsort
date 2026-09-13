#!/usr/bin/env python3
"""Build the website: two self-contained HTML pages from everything in results/.

usage: python3 scripts/website.py [--out site/index.html] [--results results]
                               [--repo URL] [--allow-stale]

The C++ library and the Rust crate each get a complete page with the same
sections and the same controls: index.html is the C++ page, rust.html next
to it the Rust page, and a switch at the top of each leads to the other.
The deterministic numbers are on both pages (the crate is held to the C++
golden counts by its tests); every timed number is on the page of the
language that was timed, and the one place the languages meet, the C++ and
Rust calls on the same machine, is labelled as such on both pages.

Inputs, all discovered in the results directory:

  counts*.csv      the deterministic numbers (sortbench --counts-only): one
                   row per (size, key type, dataset, algorithm), the same on
                   every machine. counts.csv is the golden file of the test
                   suite (10 to 1,000,000); counts-<n>.csv files add
                   further sizes, e.g. COUNTS_SIZES=10000000 scripts/counts.sh.
                   The C++ page's deterministic section.
  rust-counts*.csv the deterministic numbers of the Rust sorts as shipped
                   (brainsort-bench --rust-counts): comparisons, compare
                   flips, key reads and scratch memory, the metrics a Rust
                   program can count of a sort it did not write. The Rust
                   page's deterministic section.
  <id>.csv         the timing of one machine (sortbench --timing-only), with
  <id>.meta.json   its run stamp: id, host, OS, CPU, compiler, code
                   fingerprint, settings.
  <id>.api.md      the public-API benchmark of that machine
                   (brainsort_api_bench), stamped the same way.
  <id>.rust.api.md the Rust port's public-API benchmark on the same machine
                   (brainsort-bench --bench), stamped the same way with the
                   fingerprint of the Rust sources.

A timing file describes the code it was measured on. Its stamp carries a
fingerprint of the measured sources (include/, src/, third_party/,
CMakeLists.txt; the same function as sortbench/stamp.hpp); the page is
built from the current tree, so a file whose fingerprint differs is stale
and left out, with a note. --allow-stale includes such files, marked stale.
"""
import csv
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEASURED_PATHS = ["include", "src", "third_party", "CMakeLists.txt"]
VERSION_RE = re.compile(r'#define BRAINSORT_VERSION "([^"]+)"')

CANDIDATE = "brainsort"
# The pool: the sorts that can do what brainsort does, as shipped (stable,
# every key type, an arbitrary comparator; decision 0017).
UPSTREAM = ["std::stable_sort", "gfx::timsort", "boost::spinsort", "boost::flat_stable_sort"]

TYPES = [
    ("int32", 8, "int32_t key in an 8-byte element", "counters, small ids"),
    ("double", 16, "double key in a 16-byte element", "prices, measurements, timestamps"),
    ("int64", 16, "int64_t key in a 16-byte element", "database ids, nanosecond timestamps"),
    ("string", 16, "pointer + length, variable-length bytes", "names, identifiers, URLs"),
]
STANDARD = ["random", "sorted", "reverse", "nearly_sorted", "few_unique"]
EXTRA = ["all_equal", "runs", "organ_pipe", "small_range", "sawtooth", "prefixed", "sparse_bits"]
DATASETS = {
    "random": "uniformly random keys (strings: random lowercase words of 3 to 12 letters)",
    "sorted": "already sorted ascending",
    "reverse": "sorted descending",
    "nearly_sorted": "sorted, then 1% of the positions swapped at random",
    "few_unique": "random keys drawn from only 100 distinct values",
    "all_equal": "every key identical",
    "runs": "a concatenation of ascending runs of random length (16 to 2000)",
    "organ_pipe": "ascending, then descending",
    "small_range": "random keys from only 4 distinct values",
    "sawtooth": "ascending ramps of length 1000, repeated",
    "prefixed": "keys that share a long common prefix, like customer-00012345",
    "sparse_bits": "int32 only: random keys in which only bits 0-3 and 28-31 vary",
}
# name: (what it is, where the design ships, extra memory)
ALGOS = {
    "brainsort": ("the candidate: a scout pass picks a route (sorted, reversed, few runs, displaced elements, or radix)", "this repository", "about n/2 elements on random input, 0 on sorted or reversed input"),
    "std::stable_sort": ("the standard library's stable sort, the toolchain's own, as shipped: a merge sort in libstdc++ and the MSVC STL, in libc++ 20+ a radix sort for integers", "every C++ program that calls std::stable_sort", "n/2"),
    "gfx::timsort": ("cpp-TimSort 2.1.0, upstream code as shipped", "the C++ port of the TimSort in CPython and OpenJDK", "up to n/2"),
    "boost::spinsort": ("Boost.Sort spinsort 1.92.0, upstream code as shipped: an adaptive merge sort", "Boost", "n/2"),
    "boost::flat_stable_sort": ("Boost.Sort flat_stable_sort 1.92.0, upstream code as shipped: a block merge sort with a small buffer", "Boost", "n/256 + 8 KiB"),
}
ALGO_ORDER = ["brainsort", "std::stable_sort", "gfx::timsort", "boost::spinsort", "boost::flat_stable_sort"]
# The Rust page's pool: the crate against the Rust sorts as shipped, counted
# through what a Rust program can observe (rust-counts.csv, decision 0010).
RUST_ALGOS = {
    "brainsort": ("the candidate, the Rust crate: a scout pass picks a route (sorted, reversed, few runs, displaced elements, or radix)", "this repository, crates.io", "about n/2 elements on random input, 0 on sorted or reversed input"),
    "slice::sort": ("the standard library's stable sort, driftsort, as shipped in Rust 1.81+", "every Rust program that calls slice::sort", "n elements, n/2 above a size threshold"),
    "slice::sort_by_cached_key": ("the standard library's sort by a cached key: the keys and indices in a buffer, sorted with the same sort, then one permutation; the standard route for a sort by key", "every Rust program that calls slice::sort_by_cached_key", "n keys and indices, plus the sort's own"),
    "glidesort": ("the glidesort crate 0.1, upstream code as published: the stable sort driftsort grew out of", "crates.io", "n/8 elements by default"),
}
RUST_ALGO_ORDER = ["brainsort", "slice::sort", "slice::sort_by_cached_key", "glidesort"]
RUST_UPSTREAM = [a for a in RUST_ALGO_ORDER if a != "brainsort"]
API_TYPES = {
    "int32_t": "std::vector<int32_t>",
    "int64_t": "std::vector<int64_t>",
    "double": "std::vector<double>",
    "std::string": "std::vector<std::string> of random 3 to 12 letter words",
    "64-byte struct by int64": "std::vector of a 64-byte struct, sorted by its int64_t field (brainsort by projection, the others by comparator)",
    "int32_t by comparator": "std::vector<int32_t>, every sort called with a comparator; brainsort infers the key the comparator compares and verifies it",
    "64-byte struct by int64 by comparator": "std::vector of a 64-byte struct, every sort called with a comparator on its int64_t field; brainsort infers the key",
}


def group_of(name):
    return "candidate" if name == CANDIDATE else "upstream"


def num(s):
    return None if s in ("", None) else float(s)


# ---- the code fingerprint: must match sortbench/stamp.hpp -------------------
def code_fingerprint(root, measured_paths=None):
    files = []
    for p in measured_paths or MEASURED_PATHS:
        base = os.path.join(root, p)
        if os.path.isfile(base):
            files.append((p, base))
        elif os.path.isdir(base):
            for d, _, names in os.walk(base):
                for name in names:
                    full = os.path.join(d, name)
                    if os.path.isfile(full) and not os.path.islink(full):
                        files.append((os.path.relpath(full, root).replace(os.sep, "/"), full))
    if not files:
        return ""
    files.sort()
    h = 0xcbf29ce484222325
    prime = 0x100000001b3
    mask = (1 << 64) - 1
    for rel, full in files:
        data = rel.encode("utf-8") + b"\0"
        with open(full, "rb") as f:
            data += f.read().replace(b"\r", b"") + b"\0"
        for c in data:
            h = ((h ^ c) * prime) & mask
    return f"{h:016x}"


def version():
    try:
        with open(os.path.join(ROOT, "include", "brainsort", "detail", "config.hpp"), encoding="utf-8") as f:
            m = VERSION_RE.search(f.read())
            return m.group(1) if m else "unknown"
    except OSError:
        return "unknown"


# ---- deterministic numbers ---------------------------------------------------
DET_COLS = ["traffic", "compares", "aux", "reads", "writes", "l1", "l2", "flips", "table", "keybytes", "route", "passes", "depth"]


def load_counts(paths):
    """{(n, type, dataset, algorithm): [values in DET_COLS order]}, plus {algorithm: stable}."""
    det, stable = {}, {}
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["ok"] != "1":
                    continue
                key = (int(r["n"]), r["type"], r["dataset"], r["algorithm"])
                det[key] = [num(r["traffic_bytes"]), num(r["compares"]), num(r["aux_peak_bytes"]), num(r["reads"]), num(r["writes"]),
                            num(r["l1_misses"]), num(r["l2_misses"]), num(r["cmp_flips"]),
                            num(r["table_reads"]) + num(r["table_writes"]), num(r["key_bytes"]),
                            "" if r["route"] in ("-", "") else r["route"], num(r.get("radix_passes") or ""), num(r.get("max_depth") or "")]
                stable[r["algorithm"]] = r["stable"] == "1"
    return det, stable


RUST_DET_COLS = ["compares", "aux", "flips", "keyreads"]


def load_rust_counts(paths):
    """{(n, type, dataset, algorithm): [values in RUST_DET_COLS order]}, plus {algorithm: stable}."""
    det, stable = {}, {}
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["ok"] != "1":
                    continue
                key = (int(r["n"]), r["type"], r["dataset"], r["algorithm"])
                det[key] = [num(r["compares"]), num(r["aux_peak_bytes"]), num(r["cmp_flips"]), num(r["key_reads"])]
                stable[r["algorithm"]] = r["stable"] == "1"
    return det, stable


def det_block(det, stable, order, table, cols, extra_sizes):
    """The page's deterministic pool: the algorithms, sizes, types and inputs the rows have."""
    algo_names = [a for a in order if any(k[3] == a for k in det)] + sorted({k[3] for k in det} - set(order))
    algos = [{"name": a, "group": group_of(a), "stable": stable.get(a, False),
              "what": table.get(a, ("", "", ""))[0], "where": table.get(a, ("", "", ""))[1], "memory": table.get(a, ("", "", ""))[2]}
             for a in algo_names]
    present_ds = {k[2] for k in det}
    return {
        "sizes": sorted({k[0] for k in det} | set(extra_sizes)),
        "types": [{"name": t, "elemBytes": b, "desc": d, "real": r} for t, b, d, r in TYPES if any(k[1] == t for k in det)],
        "datasets": [{"name": d, "description": DATASETS.get(d, ""), "standard": d in STANDARD}
                     for d in STANDARD + EXTRA + sorted(present_ds - set(STANDARD + EXTRA)) if d in present_ds],
        "algos": algos, "detCols": cols,
        "det": {"cols": ["n", "t", "d", "a"] + cols, "rows": [[k[0], k[1], k[2], k[3]] + v for k, v in sorted(det.items())]},
    }


# ---- timing ------------------------------------------------------------------
TIMING_COLS = ["wall", "wallmin", "cpu", "instr", "cycles", "rss", "reps", "batch"]


def measured(s):
    return None if s in ("", "0", None) else float(s)


def read_meta(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def short_label(meta, pid):
    os_ = (meta.get("os") or "").replace("Linux (WSL2)", "WSL2").replace(" (32-bit)", "-32")
    cc = meta.get("compiler") or ""
    cc = re.sub(r"\s*-O2$", "", cc)
    m = re.match(r"(Apple Clang|GCC|Clang|MSVC)\s+(\d+)", cc)
    cc_short = f"{m.group(1)} {m.group(2)}" if m else cc.split(" ")[0]
    if cc_short.startswith("Apple Clang"):
        cc_short = "Apple Clang"
    parts = [p for p in (os_, meta.get("arch") or "", cc_short) if p]
    return " ".join(parts) if parts else pid


def is_shared_runner(meta):
    host = (meta.get("host") or "").lower()
    return "shared runner" in host or "github actions" in host


def load_timing(path, current_code, allow_stale):
    pid = os.path.basename(path)[:-4]
    meta = read_meta(path[:-4] + ".meta.json")
    code = meta.get("code") or ""
    stale = "unknown" if not code or not current_code else ("stale" if code != current_code else "fresh")
    rows = []
    backend = ""
    sizes = set()
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            backend = r["backend"] or backend
            ok = r["ok"] == "1"
            n = int(r["n"])
            sizes.add(n)
            rows.append([n, r["type"], r["dataset"], r["algorithm"], 1 if ok else 0, r["error"],
                         num(r["wall_ns_med"]) if ok and r["wall_ns_med"] else None,
                         num(r["wall_ns_min"]) if ok and r["wall_ns_min"] else None,
                         measured(r["cpu_ns_med"]) if ok else None,
                         measured(r["instr_med"]) if ok else None,
                         measured(r["cycles_med"]) if ok else None,
                         num(r["rss_delta_bytes"]) if ok and r["rss_delta_bytes"] else None,
                         int(r["reps"] or 0), int(r.get("batch") or 1)])
    platform = {
        "id": pid, "label": short_label(meta, pid), "os": meta.get("os", ""), "arch": meta.get("arch", ""),
        "compiler": meta.get("compiler", ""), "cpu": meta.get("cpu", ""), "host": meta.get("host", ""),
        "backend": backend, "measured": meta.get("measured", ""), "commit": meta.get("commit", ""), "code": code,
        "reps": meta.get("reps", ""), "rounds": meta.get("rounds", ""), "pinned": meta.get("pinned", ""),
        "sizes": sorted(sizes), "stale": stale, "shared": is_shared_runner(meta), "cells": len(rows),
    }
    return platform, rows


STAMP_RE = re.compile(r"<!--\s*stamp\s*(\{.*?\})\s*-->", re.S)


RUST_MEASURED_PATHS = ["rust/brainsort/src", "rust/brainsort/Cargo.toml", "rust/brainsort-bench/src"]

RUST_API_TYPES = {
    "i32": "Vec<i32>",
    "i64": "Vec<i64>",
    "f64": "Vec<f64>",
    "String": "Vec<String> of random 3 to 12 letter words",
    "64-byte struct by i64": "Vec of a 64-byte struct, sorted by its i64 field (brainsort by key, the others by comparator)",
    "i32 by comparator": "Vec<i32>, every sort called with a comparator: brainsort's sort_by, the comparison sort, against the others",
    "64-byte struct by i64 by comparator": "Vec of a 64-byte struct, every sort called with a comparator on its i64 field: brainsort's sort_by",
    "i32 by comparator, inferred": "Vec<i32>, every sort called with a comparator: brainsort's sort_by_inferred, which infers the key the comparator compares and verifies it",
    "64-byte struct by i64 by comparator, inferred": "Vec of a 64-byte struct, every sort called with a comparator on its i64 field: brainsort's sort_by_inferred",
}

# C++ API benchmark type -> Rust API benchmark type: the cells the two
# languages are compared on. Same generators (identical integer and string
# inputs; doubles from the same distribution), same sizes, same machine id.
LANG_PAIRS = {
    "int32_t": "i32", "int64_t": "i64", "double": "f64", "std::string": "String",
    "64-byte struct by int64": "64-byte struct by i64",
    "int32_t by comparator": "i32 by comparator",
    "64-byte struct by int64 by comparator": "64-byte struct by i64 by comparator",
}


def load_api_table(path, suffix):
    """A public-API benchmark file: the opponents come from the header row."""
    pid = os.path.basename(path)[:-len(suffix)]
    with open(path, encoding="utf-8") as f:
        text = f.read()
    m = STAMP_RE.search(text)
    stamp = {}
    if m:
        try:
            stamp = json.loads(m.group(1))
        except json.JSONDecodeError:
            stamp = {}
    note, opponents, rows = "", [], []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("<!--") or line.startswith('"') or line.startswith("}") or line.startswith("-->"):
            continue
        if not line.startswith("|"):
            if not note and not line.startswith("{"):
                note = line
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells and cells[0] == "type":
            opponents = cells[4:-1]
            continue
        if len(cells) < 6 or set(cells[0]) <= set("-:"):
            continue
        try:
            others = [None if c == "n/a" else float(c) for c in cells[4:-1]]
            rows.append([cells[0], int(cells[1]), cells[2], float(cells[3])] + others)
        except ValueError:
            continue
    return pid, stamp, note, opponents, rows


def main():
    args = sys.argv[1:]

    def opt(name, default):
        if name in args:
            i = args.index(name)
            v = args[i + 1]
            del args[i:i + 2]
            return v
        return default

    out = opt("--out", os.path.join("site", "index.html"))
    results = opt("--results", "results")
    repo = opt("--repo", "https://github.com/BrainFooLong/brainsort")
    allow_stale = "--allow-stale" in args
    if allow_stale:
        args.remove("--allow-stale")
    if args:
        sys.exit(f"unknown arguments: {' '.join(args)}\n{__doc__}")

    current_code = code_fingerprint(ROOT)
    counts_paths = sorted(glob.glob(os.path.join(results, "counts*.csv")))
    det, stable = load_counts(counts_paths)
    if not det:
        sys.exit(f"no counts*.csv in {results}/: run scripts/counts.sh (or counts.ps1) first")
    for p in counts_paths:
        print(f"{p}: deterministic rows")
    rust_counts_paths = sorted(glob.glob(os.path.join(results, "rust-counts*.csv")))
    rust_det, rust_stable = load_rust_counts(rust_counts_paths)
    for p in rust_counts_paths:
        print(f"{p}: deterministic rows of the Rust sorts")
    if not rust_det:
        print(f"no rust-counts*.csv in {results}/: the Rust page's deterministic section will be empty (cd rust && cargo run --release -p brainsort-bench -- --rust-counts)")

    platforms, timing = [], []
    skipped = []
    for path in sorted(glob.glob(os.path.join(results, "*.csv"))):
        name = os.path.basename(path)
        if name.startswith("counts") or name.startswith("rust-counts"):
            continue
        plat, rows = load_timing(path, current_code, allow_stale)
        note = {"fresh": "current", "stale": f"STALE: measured code {plat['code']}, tree is {current_code}", "unknown": "no code fingerprint in the stamp"}[plat["stale"]]
        print(f"{path}: {plat['label']}, {len(rows)} rows, sizes {plat['sizes']}, {note}")
        if plat["stale"] == "stale" and not allow_stale:
            skipped.append(plat["id"])
            continue
        platforms.append(plat)
        timing.extend([[plat["id"]] + r for r in rows])
    # Quiet machines (performance counters, no other tenants) before shared runners.
    platforms.sort(key=lambda p: (p["shared"], p["label"]))

    api_platforms, api_rows, api_opponents = [], [], []
    for path in sorted(glob.glob(os.path.join(results, "*.api.md"))):
        if path.endswith(".rust.api.md"):
            continue
        pid, stamp, note, opponents, rows = load_api_table(path, ".api.md")
        code = stamp.get("code") or ""
        stale = "unknown" if not code or not current_code else ("stale" if code != current_code else "fresh")
        if stale == "stale" and not allow_stale:
            print(f"{path}: STALE, left out")
            continue
        if not rows:
            print(f"{path}: no table, left out")
            continue
        plat = next((p for p in platforms if p["id"] == pid), None)
        api_platforms.append({"id": pid, "label": plat["label"] if plat else short_label(stamp, pid), "note": note,
                              "stale": stale, "reps": stamp.get("reps", ""), "cpu": stamp.get("cpu", plat["cpu"] if plat else ""),
                              "shared": plat["shared"] if plat else is_shared_runner(stamp), "measured": stamp.get("measured", "")})
        if opponents and not api_opponents:
            api_opponents = opponents
        api_rows.extend([[pid] + r[:4] + [r[4:]] for r in rows])
        print(f"{path}: {len(rows)} rows, {stale}")
    api_platforms.sort(key=lambda p: (p["shared"], p["label"]))

    # The Rust port: one file per machine, paired with the C++ file of the
    # same machine id by the comparison section. Staleness is judged against
    # the fingerprint of the Rust sources.
    rust_code = code_fingerprint(ROOT, RUST_MEASURED_PATHS)
    rust_platforms, rust_rows, rust_opponents = [], [], []
    for path in sorted(glob.glob(os.path.join(results, "*.rust.api.md"))):
        pid, stamp, note, opponents, rows = load_api_table(path, ".rust.api.md")
        code = stamp.get("code") or ""
        stale = "unknown" if not code or not rust_code else ("stale" if code != rust_code else "fresh")
        if stale == "stale" and not allow_stale:
            print(f"{path}: STALE (Rust sources changed), left out")
            skipped.append(pid + " (rust)")
            continue
        if not rows:
            print(f"{path}: no table, left out")
            continue
        if opponents and not rust_opponents:
            rust_opponents = opponents
        plat = next((p for p in platforms if p["id"] == pid), None)
        rustc = re.sub(r"\s*\(.*", "", stamp.get("compiler") or "rustc")   # "rustc 1.98.0"
        rust_platforms.append({"id": pid, "label": (plat["label"] if plat else short_label({**stamp, "compiler": ""}, pid)) + " · " + rustc,
                               "note": note, "stale": stale, "reps": stamp.get("reps", ""), "compiler": stamp.get("compiler", ""),
                               "cpu": stamp.get("cpu", plat["cpu"] if plat else ""),
                               "shared": plat["shared"] if plat else is_shared_runner(stamp), "measured": stamp.get("measured", "")})
        rust_rows.extend([[pid] + r[:4] + [r[4:]] for r in rows])
        print(f"{path}: {len(rows)} rows, {stale}")
    rust_platforms.sort(key=lambda p: (p["shared"], p["label"]))

    # The deterministic pool of each page: the C++ harness on the C++ page, the
    # Rust sorts as shipped on the Rust page.
    det_pages = {
        "cpp": det_block(det, stable, ALGO_ORDER, ALGOS, DET_COLS, {r[1] for r in timing}),
        "rust": det_block(rust_det, rust_stable, RUST_ALGO_ORDER, RUST_ALGOS, RUST_DET_COLS, {r[2] for r in rust_rows}),
    }
    data = {
        "version": version(), "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code": current_code, "repo": repo,
        "candidate": CANDIDATE,
        "platforms": platforms,
        "timing": {"cols": ["p", "n", "t", "d", "a", "ok", "error"] + TIMING_COLS, "rows": timing},
        "skipped": skipped,
        # The public call on a plain vector, one benchmark per language, the same shape: brainsort,
        # then the opponents in header order; `std` is the index of the standard library's stable sort.
        "bench": {
            "cpp": {"platforms": api_platforms, "opponents": api_opponents, "std": 0, "code": current_code,
                    "types": [{"name": t, "desc": d} for t, d in API_TYPES.items() if any(r[1] == t for r in api_rows)],
                    "sizes": sorted({r[2] for r in api_rows}), "datasets": [d for d in STANDARD if any(r[3] == d for r in api_rows)],
                    "cols": ["p", "t", "n", "d", "bs", "others"], "rows": api_rows},
            "rust": {"platforms": rust_platforms, "opponents": rust_opponents, "std": 0, "code": rust_code,
                     "types": [{"name": t, "desc": d} for t, d in RUST_API_TYPES.items() if any(r[1] == t for r in rust_rows)],
                     "sizes": sorted({r[2] for r in rust_rows}), "datasets": [d for d in STANDARD if any(r[3] == d for r in rust_rows)],
                     "cols": ["p", "t", "n", "d", "bs", "others"], "rows": rust_rows},
        },
        "pairs": [[c, r] for c, r in LANG_PAIRS.items()],
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pages = {"cpp": out, "rust": os.path.join(os.path.dirname(out), "rust.html")}
    for lang, path in pages.items():
        other = "rust" if lang == "cpp" else "cpp"
        page = dict(data)
        page.update(det_pages[lang])
        if lang == "rust":   # the harness (instrumented C++ builds, hardware counters) is C++ only
            page["platforms"] = []
            page["timing"] = {"cols": data["timing"]["cols"], "rows": []}
        js = json.dumps(page, separators=(",", ":")).replace("<", "\\u003c")
        html = re.sub(rf"<!--{other}-->.*?<!--/{other}-->", "", TEMPLATE, flags=re.S)
        html = html.replace(f"<!--{lang}-->", "").replace(f"<!--/{lang}-->", "")
        html = html.replace("__DATA__", js).replace("__VERSION__", data["version"]).replace("__LANG__", lang)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(html)
    print(f"wrote {pages['cpp']} (C++) and {pages['rust']} (Rust): {len(det)} deterministic rows at n = {', '.join(str(n) for n in sorted({k[0] for k in det}))}, "
          f"{len(rust_det)} of the Rust sorts; "
          f"{len(platforms)} machine(s), {len(timing)} timed rows; C++ API benchmark on {len(api_platforms)} machine(s), {len(api_rows)} rows; "
          f"Rust API benchmark on {len(rust_platforms)} machine(s), {len(rust_rows)} rows"
          + (f"; left out as stale: {', '.join(skipped)}" if skipped else ""))


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>brainsort __VERSION__: <!--cpp-->C++<!--/cpp--><!--rust-->Rust<!--/rust--> benchmark results</title>
<meta name="description" content="brainsort, a stable sort for integer, floating-point and string keys, measured against the stable sorts that ship in today's standard libraries and the common libraries: deterministic work counts on every input size, and timing per machine.">
<style>
:root {
  color-scheme: light;
  --bg: #faf9f6; --bg-2: #f1efea; --bg-3: #e8e5de; --card: #ffffff; --border: #e3e0d8; --grid: #ece9e2;
  --text: #17160f; --text-2: #55534b; --text-3: #7f7c72;
  --accent: #2a78d6; --accent-soft: #e6f0fc; --accent-ink: #1c5cab;
  --c-candidate: #2a78d6; --c-upstream: #eb6834; --c-port: #1baf7a; --c-baseline: #eda100;
  --win-1: #e3eefb; --win-2: #b7d3f6; --win-3: #86b6ef; --win-4: #3987e5;
  --loss-1: #fbe4e4; --loss-2: #f5bdbd; --loss-3: #ee8f8f; --loss-4: #e34948;
  --neutral: #eeece6; --best: #e4f3e4; --warn: #b35a00; --bad: #d03b3b;
  --shadow: 0 1px 2px rgba(23,22,15,.05), 0 8px 24px rgba(23,22,15,.06);
  --shadow-lg: 0 12px 40px rgba(23,22,15,.14);
  --radius: 14px;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #161614; --bg-2: #1e1e1c; --bg-3: #292926; --card: #1b1b19; --border: #33332f; --grid: #2a2a27;
    --text: #f4f3ee; --text-2: #bdbbb1; --text-3: #8a8880;
    --accent: #5b9bea; --accent-soft: #1d2b3e; --accent-ink: #9cc3f5;
    --c-candidate: #4f93e8; --c-upstream: #e0653a; --c-port: #22b07e; --c-baseline: #d59a1c;
    --win-1: #1e2c3d; --win-2: #1c3b63; --win-3: #1f5297; --win-4: #3987e5;
    --loss-1: #3a2222; --loss-2: #5a2626; --loss-3: #8a3232; --loss-4: #e66767;
    --neutral: #353531; --best: #1f2e1f; --warn: #f0a050; --bad: #ff7b7b;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.35);
    --shadow-lg: 0 12px 40px rgba(0,0,0,.6);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #161614; --bg-2: #1e1e1c; --bg-3: #292926; --card: #1b1b19; --border: #33332f; --grid: #2a2a27;
  --text: #f4f3ee; --text-2: #bdbbb1; --text-3: #8a8880;
  --accent: #5b9bea; --accent-soft: #1d2b3e; --accent-ink: #9cc3f5;
  --c-candidate: #4f93e8; --c-upstream: #e0653a; --c-port: #22b07e; --c-baseline: #d59a1c;
  --win-1: #1e2c3d; --win-2: #1c3b63; --win-3: #1f5297; --win-4: #3987e5;
  --loss-1: #3a2222; --loss-2: #5a2626; --loss-3: #8a3232; --loss-4: #e66767;
  --neutral: #353531; --best: #1f2e1f; --warn: #f0a050; --bad: #ff7b7b;
  --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.35);
  --shadow-lg: 0 12px 40px rgba(0,0,0,.6);
}
* { box-sizing: border-box; }
html { scroll-padding-top: 130px; scroll-behavior: smooth; }
body { margin: 0; padding-block: 0 80px; padding-inline: 16px; background: var(--bg); color: var(--text);
       font: 15px/1.6 system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; -webkit-font-smoothing: antialiased; }
main { max-width: 1180px; margin: 0 auto; }
a { color: var(--accent); text-decoration-color: color-mix(in srgb, var(--accent) 35%, transparent); text-underline-offset: 3px; }
a:hover { text-decoration-color: var(--accent); }
h1 { font-size: clamp(34px, 6vw, 52px); font-weight: 750; letter-spacing: -.03em; margin: 0; line-height: 1.05; }
h2 { font-size: clamp(22px, 3vw, 30px); font-weight: 700; letter-spacing: -.02em; margin: 0 0 10px; line-height: 1.2; }
h3 { font-size: 17px; font-weight: 650; margin: 34px 0 6px; letter-spacing: -.01em; }
h4 { font-size: 13px; font-weight: 600; margin: 22px 0 8px; color: var(--text-2); text-transform: uppercase; letter-spacing: .06em; }
p, li { color: var(--text-2); max-width: 78ch; }
p.lead { font-size: 19px; color: var(--text); max-width: 66ch; line-height: 1.5; margin: 18px 0 0; }
.muted { color: var(--text-3); font-size: 13px; }
.note { font-size: 14px; color: var(--text-3); margin: 6px 0 14px; max-width: 90ch; }
code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .92em; }
code { background: var(--bg-2); padding: 1px 6px; border-radius: 6px; }
pre { background: var(--bg-2); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; overflow-x: auto; font-size: 13px; line-height: 1.55; }
pre code { background: none; padding: 0; }

/* ---- header ---- */
header { padding: 52px 0 12px; }
.topline { display: flex; flex-wrap: wrap; gap: 10px 14px; align-items: center; }
.badge { display: inline-block; font-size: 12.5px; font-weight: 600; padding: 3px 10px; border-radius: 999px; border: 1px solid var(--border); color: var(--text-2); background: var(--card); }
.badge.accent { background: var(--accent-soft); color: var(--accent-ink); border-color: transparent; }
.topline .ci { display: inline-flex; line-height: 0; }
.topline .ci img { height: 20px; }
/* the language switch: two complete pages, one per language, the same sections on each */
nav.lang { display: flex; gap: 10px; margin: 22px 0 0; }
nav.lang a { flex: 1 1 200px; max-width: 320px; text-decoration: none; padding: 12px 16px; border-radius: var(--radius); border: 1px solid var(--border); background: var(--card); color: var(--text-2); }
nav.lang a b { display: block; font-size: 17px; color: var(--text); }
nav.lang a span { display: block; font-size: 13px; margin-top: 2px; }
nav.lang a:hover { border-color: var(--accent); }
nav.lang[data-lang="cpp"] a.cpp, nav.lang[data-lang="rust"] a.rust { border-color: var(--accent); background: var(--accent-soft); color: var(--accent-ink); box-shadow: inset 0 -3px 0 var(--accent); }
nav.lang[data-lang="cpp"] a.cpp b, nav.lang[data-lang="rust"] a.rust b { color: var(--accent-ink); }
nav.lang a b::after { font-size: 11px; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; margin-left: 10px; color: var(--text-3); }
nav.lang[data-lang="cpp"] a.cpp b::after, nav.lang[data-lang="rust"] a.rust b::after { content: "this page"; color: var(--accent-ink); }
.links { display: flex; flex-wrap: wrap; gap: 8px 10px; margin: 22px 0 0; }
.links a { font-size: 14px; font-weight: 500; text-decoration: none; padding: 7px 13px; border-radius: 999px; border: 1px solid var(--border); background: var(--card); color: var(--text); }
.links a:hover { border-color: var(--accent); color: var(--accent); }
.stamp { font-size: 13px; color: var(--text-3); margin: 16px 0 0; }
nav.toc { display: flex; flex-wrap: wrap; gap: 6px 22px; margin: 26px 0 0; padding: 14px 0 0; border-top: 1px solid var(--border); font-size: 14px; counter-reset: toc; }
nav.toc a { text-decoration: none; color: var(--text-2); counter-increment: toc; }
nav.toc a::before { content: counter(toc, decimal-leading-zero) " "; color: var(--accent); font-weight: 600; font-variant-numeric: tabular-nums; margin-right: 2px; }
nav.toc a:hover { color: var(--accent); }

/* ---- sections ---- */
section, details.section { margin-top: 72px; }
.eyebrow { display: block; font-size: 12.5px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--accent); margin: 0 0 8px; }
.eyebrow::before { content: ""; display: inline-block; width: 22px; height: 2px; background: var(--accent); vertical-align: middle; margin-right: 8px; border-radius: 2px; }
.intro { font-size: 16px; color: var(--text-2); max-width: 74ch; margin: 0 0 18px; }

/* ---- sticky controls ---- */
.panel { position: sticky; top: 0; z-index: 5; margin: 36px -16px 0; padding: 10px 16px 10px;
         background: color-mix(in srgb, var(--bg) 88%, transparent); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
         border-bottom: 1px solid var(--border); }
.panel .inner { max-width: 1180px; margin: 0 auto; }
.panel .row { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 12px; margin: 4px 0; }
.lbl { font-size: 11.5px; font-weight: 600; color: var(--text-3); min-width: 70px; text-transform: uppercase; letter-spacing: .07em; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
button.chip { font: inherit; font-size: 13.5px; font-weight: 500; padding: 5px 13px; border-radius: 999px; border: 1px solid var(--border);
              background: var(--card); color: var(--text-2); cursor: pointer; display: inline-flex; align-items: center; gap: 7px; line-height: 1.3;
              transition: border-color .12s, background .12s, color .12s; }
button.chip:hover { border-color: var(--accent); color: var(--text); }
button.chip[aria-pressed="true"] { background: var(--accent); border-color: var(--accent); color: #fff; box-shadow: 0 2px 8px color-mix(in srgb, var(--accent) 35%, transparent); }
button.chip i { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
button.chip.dim { opacity: .55; }
button.chip.dim[aria-pressed="true"] { opacity: 1; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 20px; margin: 12px 0 0; font-size: 13px; color: var(--text-2); }
.legend span { display: inline-flex; align-items: center; gap: 2px; }
.legend i, table.data td .sw, .sw { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 7px; vertical-align: -1px; }

/* ---- tiles ---- */
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; margin: 18px 0 10px; }
.tile { border: 1px solid var(--border); border-radius: var(--radius); padding: 18px 20px 16px; background: var(--card); box-shadow: var(--shadow); }
.tile .k { font-size: 12.5px; font-weight: 600; color: var(--text-3); text-transform: uppercase; letter-spacing: .06em; }
.tile .v { font-size: 34px; font-weight: 750; letter-spacing: -.03em; margin: 6px 0 4px; line-height: 1.1; color: var(--text); }
.tile .v small { font-size: 15px; font-weight: 500; color: var(--text-3); letter-spacing: 0; }
.tile .s { font-size: 13.5px; color: var(--text-2); line-height: 1.5; }
.tile.win .v { color: var(--accent); } .tile.loss .v { color: var(--loss-4); }
.tile.hero { background: linear-gradient(135deg, var(--accent-soft), var(--card) 70%); }

/* ---- tables ---- */
.tablewrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 12px; margin: 10px 0; background: var(--card); box-shadow: var(--shadow); }
table.data { border-collapse: collapse; width: 100%; font-size: 13.5px; font-variant-numeric: tabular-nums; }
table.data th, table.data td { padding: 9px 12px; border-bottom: 1px solid var(--border); text-align: right; white-space: nowrap; }
table.data th { color: var(--text-3); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .05em; background: var(--bg-2); position: sticky; top: 0; }
table.data th:first-child, table.data td:first-child { text-align: left; }
table.data tr:last-child td { border-bottom: 0; }
table.data td.best { background: var(--best); font-weight: 600; }
table.data td .rel { color: var(--text-3); font-weight: 400; font-size: 11px; margin-left: 4px; }
table.data.wrap td, table.data.wrap th { white-space: normal; text-align: left; }
table.heat td { text-align: center; padding: 0; }
table.heat td:first-child { text-align: left; padding: 8px 12px; font-weight: 500; }
table.heat td.cell > div { padding: 9px 8px; min-width: 70px; border-radius: 6px; margin: 2px; }
table.heat td.cell.w { font-weight: 600; }
table.heat td.l1 > div { background: var(--loss-1); } table.heat td.l2 > div { background: var(--loss-2); }
table.heat td.l3 > div { background: var(--loss-3); } table.heat td.l4 > div { background: var(--loss-4); color: #fff; }
table.heat td.w1 > div { background: var(--win-1); } table.heat td.w2 > div { background: var(--win-2); }
table.heat td.w3 > div { background: var(--win-3); } table.heat td.w4 > div { background: var(--win-4); color: #fff; }
table.heat td.n0 > div { background: var(--neutral); }
.scale { display: flex; align-items: center; gap: 10px; font-size: 12.5px; color: var(--text-3); margin: 8px 0 4px; flex-wrap: wrap; }
.scale .bar { display: flex; height: 12px; border-radius: 6px; overflow: hidden; width: 200px; }
.scale .bar i { flex: 1; }

/* ---- tags ---- */
.tag { display: inline-block; font-size: 10.5px; line-height: 1.3; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; padding: 2px 8px; border-radius: 999px;
       background: var(--bg-3); color: var(--text-2); margin-left: 6px; vertical-align: 2px; }
.tag.det { background: var(--accent-soft); color: var(--accent-ink); }
.tag.warn { background: color-mix(in srgb, var(--warn) 18%, transparent); color: var(--warn); }
.tag.bad { background: color-mix(in srgb, var(--bad) 18%, transparent); color: var(--bad); }

/* ---- cards and charts ---- */
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 340px), 1fr)); gap: 14px; }
.card { border: 1px solid var(--border); border-radius: var(--radius); padding: 14px 16px 10px; background: var(--card); min-width: 0; box-shadow: var(--shadow); }
.card h3 { margin: 0 0 2px; font-size: 15px; }
.card .desc { font-size: 12.5px; color: var(--text-3); margin: 0 0 8px; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
svg text { font-family: inherit; }
.tip { position: fixed; z-index: 10; pointer-events: none; background: var(--card); color: var(--text); border: 1px solid var(--border);
       border-radius: 10px; padding: 10px 12px; font-size: 12.5px; box-shadow: var(--shadow-lg); max-width: min(380px, 90vw); }
.tip b { display: block; margin-bottom: 4px; }
.tip table { border-collapse: collapse; }
.tip td { padding: 1px 8px 1px 0; color: var(--text-2); white-space: nowrap; }
.tip td:last-child { color: var(--text); text-align: right; font-variant-numeric: tabular-nums; }

/* ---- collapsibles ---- */
details { border: 1px solid var(--border); border-radius: 12px; margin: 14px 0; background: var(--card); box-shadow: var(--shadow); }
details > summary { cursor: pointer; padding: 13px 16px; color: var(--text); font-weight: 600; list-style: none; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; border-radius: 12px; transition: background .12s; }
details > summary:hover { background: var(--bg-2); }
details[open] > summary { border-bottom: 1px solid var(--border); border-radius: 12px 12px 0 0; }
details > summary::-webkit-details-marker { display: none; }
details > summary::before { content: ""; flex: none; width: 22px; height: 22px; border-radius: 6px; background: var(--accent-soft);
  background-image: linear-gradient(var(--accent), var(--accent)), linear-gradient(var(--accent), var(--accent));
  background-size: 10px 2px, 2px 10px; background-position: center; background-repeat: no-repeat; transition: transform .15s; }
details[open] > summary::before { background-size: 10px 2px, 0 0; }
details > summary .muted { font-weight: 400; }
details > summary .hint { margin-left: auto; font-size: 12.5px; font-weight: 500; color: var(--accent); }
details > summary .hint::after { content: "Show"; }
details[open] > summary .hint::after { content: "Hide"; }
details > .body { padding: 6px 16px 16px; }
details.section { border: 0; box-shadow: none; background: none; }
details.section > summary { display: block; padding: 22px 26px; border: 1px solid var(--border); border-radius: var(--radius); background: var(--card); box-shadow: var(--shadow); position: relative; }
details.section > summary:hover { border-color: var(--accent); background: var(--card); }
details.section[open] > summary { border-radius: var(--radius); margin-bottom: 8px; border-color: var(--border); }
details.section > summary::before { position: absolute; right: 24px; top: 26px; width: 28px; height: 28px; border-radius: 8px; background-size: 12px 2px, 2px 12px; }
details.section[open] > summary::before { background-size: 12px 2px, 0 0; }
details.section > summary h2 { margin: 6px 0 6px; padding-right: 48px; }
details.section > summary .muted { display: block; font-size: 14px; }
details.section > summary .hint { display: block; margin: 8px 0 0; font-size: 13.5px; }
details.section > summary .hint::after { content: "Open this section \2193"; }
details.section[open] > summary .hint::after { content: "Collapse this section \2191"; }
details.section > .body { padding: 8px 0 0; }

/* ---- guides and callouts ---- */
.callout { border-left: 3px solid var(--accent); background: var(--accent-soft); color: var(--text); border-radius: 0 10px 10px 0; padding: 10px 16px; margin: 14px 0; font-size: 14px; max-width: 90ch; }
.callout.warn { border-color: var(--warn); background: color-mix(in srgb, var(--warn) 12%, transparent); }
.guide { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 250px), 1fr)); gap: 14px; margin-top: 18px; }
.guide .g { border: 1px solid var(--border); border-radius: var(--radius); background: var(--card); padding: 16px 18px; box-shadow: var(--shadow); }
.guide .g b { display: block; color: var(--text); font-size: 15px; margin-bottom: 4px; }
.guide .g p { font-size: 14px; margin: 0; }
.guide .g .n { display: inline-flex; width: 26px; height: 26px; border-radius: 8px; background: var(--accent-soft); color: var(--accent-ink); font-weight: 700; font-size: 13px; align-items: center; justify-content: center; margin-bottom: 10px; }
.machine { border: 1px solid var(--border); border-left: 4px solid var(--accent); border-radius: 12px; padding: 12px 16px; background: var(--card); font-size: 13.5px; color: var(--text-2); margin: 12px 0; box-shadow: var(--shadow); }
.machine b { color: var(--text); }
.empty { color: var(--text-3); font-style: italic; }
.fail { color: var(--bad); }
.cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr)); gap: 12px 40px; }
dl { margin: 6px 0; } dt { font-weight: 600; color: var(--text); margin-top: 12px; } dd { margin: 2px 0 0; color: var(--text-2); max-width: 80ch; font-size: 14px; }
.row.inline { display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: center; margin: 10px 0; }
select { font: inherit; font-size: 13.5px; padding: 5px 10px; border-radius: 8px; border: 1px solid var(--border); background: var(--card); color: var(--text); }
footer { margin-top: 80px; padding-top: 18px; border-top: 1px solid var(--border); font-size: 13px; color: var(--text-3); }
@media (max-width: 640px) {
  .panel .row { gap: 4px 8px; } .lbl { min-width: 100%; }
  .tile .v { font-size: 28px; }
  html { scroll-padding-top: 240px; }
  details.section > summary { padding: 18px 18px; }
  details.section > summary::before { right: 16px; top: 18px; }
}
/* ---- motion: a short fade and lift as a block scrolls into view; nothing moves without JS or with reduced motion ---- */
.reveal { opacity: 0; transform: translateY(14px); transition: opacity .55s ease-out, transform .55s cubic-bezier(.2,.7,.2,1); }
.reveal.in { opacity: 1; transform: none; }
.panel { transition: box-shadow .25s; }
.panel.stuck { box-shadow: 0 6px 20px rgba(23,22,15,.08); }
:root[data-theme="dark"] .panel.stuck { box-shadow: 0 6px 20px rgba(0,0,0,.4); }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .panel.stuck { box-shadow: 0 6px 20px rgba(0,0,0,.4); } }
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
  .reveal { opacity: 1; transform: none; transition: none; }
}
@media print { .panel { position: static; } details { border: 0; } .reveal { opacity: 1; transform: none; } }
</style>
</head>
<body>
<main>
<header>
  <div class="topline"><h1>brainsort</h1><span class="badge accent">version __VERSION__</span><span class="badge">stable sort</span><!--cpp--><span class="badge">header-only C++20</span><!--/cpp--><!--rust--><span class="badge">Rust crate</span><!--/rust--><a class="ci" id="ci-badge" href="#"><img alt="CI status" src=""></a></div>
  <nav class="lang" data-lang="__LANG__" aria-label="language"><a class="cpp" href="index.html"><b>C++</b><span>the header-only library, measured against the C++ sorts</span></a><a class="rust" href="rust.html"><b>Rust</b><span>the crate, measured against the Rust sorts</span></a></nav>
  <p class="lead">A stable sorting algorithm for keys that map to an ordered integer: numbers, strings, dates, ids. <!--cpp-->This page is the C++ library: measured against the stable sorts that ship in today's C++ standard libraries and in the common libraries, on the same inputs, the same compiler flags and the same machines. The Rust crate has its own page with the same sections, built from its own measurements.<!--/cpp--><!--rust-->This page is the Rust crate: the same algorithm as the C++ library, held to the same deterministic counts by its tests, and timed on its own against the stable sorts the Rust ecosystem ships, on the same inputs and the same machines. The C++ library has its own page with the same sections.<!--/rust--></p>
  <div class="links" id="links"></div>
  <p class="stamp" id="stamp"></p>
  <nav class="toc">
    <a href="#use">Use it</a><a href="#glance">At a glance</a><a href="#reading">How to read this page</a><a href="#det">What each algorithm does</a>
    <!--cpp--><a href="#timing">Measured time per machine</a><a href="#api">The library on plain vectors</a><!--/cpp--><!--rust--><a href="#api">The crate on plain vectors</a><!--/rust--><a href="#method">Method</a><a href="#tables">All numbers</a>
  </nav>
</header>

<section id="use">
<span class="eyebrow">In your own code</span>
<h2>Use it</h2>
<!--cpp-->
<p class="intro">Copy the <a id="link-header" href="#">single header</a> into your project, or add <code>include/</code> to your include path and include <code>brainsort/brainsort.hpp</code>. Header-only C++20, MIT licensed. GCC, Clang, MSVC and Apple Clang; x86-64 with AVX2 and BMI2 chosen at run time, ARM64 and 32-bit x86 with the same code minus the vector paths.</p>
<pre><code>#include "brainsort.hpp"

std::vector&lt;int&gt; v = ...;
brainsort::sort(v);                                              // stable, by value
brainsort::sort(v.begin(), v.end());                             // any random-access range
brainsort::sort(rows, [](const Row&amp; r) { return r.id; });        // by a key
brainsort::sort(rows, [](const Row&amp; r) { return std::pair(r.group, brainsort::desc(r.score)); });
brainsort::sort(rows, [](const Row&amp; a, const Row&amp; b) { return a.name &lt; b.name; });  // a comparator: merge sort</code></pre>
<div class="guide">
<div class="g"><b>Keys</b><p>Any integer, bool, character type, float, double, enum, pointer, <code>std::string</code>, <code>std::string_view</code>, C string, <code>std::chrono</code> duration or time point, a <code>std::pair</code>, <code>std::tuple</code> or <code>std::array</code> of those, or <code>brainsort::desc(key)</code> for a reversed order. Your own key type takes a <code>brainsort::key_traits</code> specialisation.</p></div>
<div class="g"><b>Elements</b><p>Anything move-assignable, in any random-access range: <code>std::vector</code>, <code>std::deque</code>, <code>std::array</code>, <code>std::span</code>, C arrays, pointer pairs. The keys are sorted first, then the elements are permuted once. Every overload is stable.</p></div>
<div class="g"><b>Three things to know</b><p>It uses memory: about 1.5 small records per element while sorting, plus one element per element for the final permutation of plain structs, and it keeps up to 32 MiB of freed blocks for the next call (<code>brainsort::release_memory()</code> frees them). An arbitrary comparator gets a comparison sort, the library's own stable merge sort, without the radix wins. A range of 2<sup>32</sup> elements or more goes to <code>std::stable_sort</code>.</p></div>
</div>
<p class="note">The full contract, every overload and how the elements are handled: <a id="link-readme" href="#">README, The library</a>; step by step with examples: <a id="link-usage-cpp" href="#">the C++ usage guide</a>.</p>
<!--/cpp-->
<!--rust-->
<p class="intro">Add the crate <a id="link-crate" href="#">brainsort</a> to your project. It is a port of the C++ library function by function, held to the C++ golden numbers by its test suite: every element read and write, every comparison and table access, the peak memory and the exact access sequence must match, cell for cell. No dependencies, <code>no_std</code> + <code>alloc</code>, MIT licensed; AVX2 and BMI2 chosen at run time on x86-64, the same code minus the vector paths elsewhere.</p>
<pre><code>[dependencies]
brainsort = "__VERSION__"</code></pre>
<pre><code>let mut v = vec![3, 1, 2];
brainsort::sort(&amp;mut v);                                         // stable, by value
brainsort::sort_by_key(&amp;mut rows, |r| r.id);                     // by a key
brainsort::sort_by_key_ref(&amp;mut rows, |r| r.name.as_str());      // by a borrowed key
brainsort::sort_by_key(&amp;mut rows, |r| (r.id, brainsort::Desc(r.score)));
brainsort::sort_by(&amp;mut rows, |a, b| a.name.cmp(&amp;b.name));       // a comparator</code></pre>
<div class="guide">
<div class="g"><b>Keys</b><p>Every integer, bool, char, f32, f64, Duration, str, String, byte strings, tuples and arrays of those, or <code>Desc(key)</code> for a reversed order. Your own key type implements <code>Key</code>.</p></div>
<div class="g"><b>Elements</b><p>Anything, in a slice or a Vec. The keys are sorted first, then the elements are permuted once. Every function is stable.</p></div>
<div class="g"><b>Three things to know</b><p>It uses memory: about 1.5 small records per element while sorting, plus one element per element for the final permutation of plain structs, and it keeps up to 32 MiB of freed blocks for the next call (<code>brainsort::release_memory()</code> frees them). An arbitrary comparator (<code>sort_by</code>) gets a comparison sort, the crate's own stable merge sort, without the radix wins. A slice of 2<sup>32</sup> elements or more goes to the standard library's stable sort.</p></div>
</div>
<p class="note">The API docs are on <a id="link-docsrs" href="#">docs.rs</a>; step by step with examples: <a id="link-usage-rust" href="#">the Rust usage guide</a>.</p>
<!--/rust-->
</section>

<section id="glance">
<span class="eyebrow">Overview</span>
<h2>At a glance</h2>
<p class="intro" id="glance-text"></p>
<p class="note"><b>Why these sorts.</b> An opponent has to offer what brainsort offers, so that a win or a loss means something: it is stable, it sorts every key type on this page and it takes an arbitrary comparator. And it is the upstream code as it ships, never a port written here. <!--cpp-->That is std::stable_sort, cpp-TimSort and Boost.Sort's spinsort and flat_stable_sort; std::sort, pdqsort and the radix libraries are unstable or sort scalar keys only, an easier job, and are not opponents.<!--/cpp--><!--rust-->That is slice::sort, its cached-key form and glidesort; slice::sort_unstable and the radix crates are unstable or sort scalar keys only, an easier job, and are not opponents.<!--/rust--></p>
<div class="tiles" id="glance-tiles"></div>
<p class="note" id="glance-note"></p>
</section>

<section id="reading">
<span class="eyebrow">Reading guide</span>
<h2>How to read this page</h2>
<p class="intro">Six things to know, then every number below makes sense.</p>
<div class="guide">
<div class="g"><span class="n">1</span><b>A cell</b><p>One key type on one input pattern at one size. In every cell brainsort is compared with the <em>best</em> of the other sorts shown, whichever that is. <em>38 of 45</em> means brainsort is best or tied in 38 cells.</p></div>
<div class="g"><span class="n">2</span><b>Lower is better</b><p>On every number here: fewer bytes moved, fewer comparisons, less memory, less time.</p></div>
<div class="g"><span class="n">3</span><b>Two kinds of numbers</b><p><span class="tag det" style="margin-left:0">deterministic</span> numbers describe what an algorithm <em>does</em>: bytes moved, comparisons made, memory asked for. Counted, the same on every machine, held by the test suite. <span class="tag" style="margin-left:0">measured</span> numbers are clocks and hardware counters: they depend on the machine and vary between runs, so they are shown per machine.</p></div>
<div class="g"><span class="n">4</span><b>Only like for like</b><p>Every sort on this page is stable (it keeps equal keys in their original order, which sorting records by one field needs), sorts every key type shown and takes an arbitrary comparator, and every one is upstream code run as shipped. A sort that does less (an unstable sort, a radix sort for scalar keys only) has an easier job and is not an opponent; a port written here could be slower than the original and is not one either.</p></div>
<div class="g"><span class="n">5</span><b>Sizes</b><p>Every number exists at several input sizes. The <em>size</em> control changes the whole page; the <em>across sizes</em> charts show a metric per element as the input grows.</p></div>
<div class="g"><span class="n">6</span><b>Colours</b><p>Blue: brainsort ahead. Red: behind. Grey: a tie within 5%. Darker is a larger margin. In charts:</p><div class="legend"><span><i style="background:var(--c-candidate)"></i>brainsort</span><!--cpp--><span><i style="background:var(--c-upstream)"></i>upstream code, as shipped</span><!--/cpp--><!--rust--><span><i style="background:var(--c-upstream)"></i>the Rust sorts, as shipped</span><!--/rust--></div></div>
</div>
</section>

<div class="panel" id="panel"><div class="inner">
  <div class="row"><span class="lbl">size</span><div class="chips" id="c-size"></div></div>
  <div class="row"><span class="lbl">key type</span><div class="chips" id="c-types"></div></div>
  <div class="row"><span class="lbl">inputs</span><div class="chips" id="c-datasets"></div></div>
</div></div>

<section id="det">
<span class="eyebrow">Same on every machine</span>
<h2>What each algorithm does <span class="tag det">deterministic</span></h2>
<div id="det-fallback"></div>
<p class="intro">These numbers are the same on every computer. They are counted, not timed: <!--cpp-->how many bytes an algorithm moves, how many comparisons it makes, how much memory it asks for.<!--/cpp--><!--rust-->how many comparisons a sort makes, how often their outcome flips, how often the cached-key sort asks for a key, how much memory it asks for. The other sorts are the stable sorts a Rust program can call, as shipped: the standard library's driftsort, its sort by a cached key and the glidesort crate, counted through what a program can observe of them (the comparator they are given, the key function they call, the allocator). What cannot be observed of a sort one did not write, the bytes it moves, is counted for brainsort alone in the golden file and compared with the C++ pool on <a href="index.html">the C++ page</a>; the crate's own bytes moved are exactly those, held by its test suite.<!--/rust--> Choose a metric:</p>
<div class="chips" id="c-det-metric"></div>
<p class="note" id="det-metric-note"></p>
<p class="note" id="det-nocontest" hidden></p>
<div id="det-contest">
<div class="tiles" id="det-tiles"></div>
<h3>Where brainsort wins and loses</h3>
<p class="note" id="det-heat-note"></p><div class="scale"><span>brainsort behind</span><div class="bar"><i style="background:var(--loss-4)"></i><i style="background:var(--loss-3)"></i><i style="background:var(--loss-2)"></i><i style="background:var(--loss-1)"></i><i style="background:var(--neutral)"></i><i style="background:var(--win-1)"></i><i style="background:var(--win-2)"></i><i style="background:var(--win-3)"></i><i style="background:var(--win-4)"></i></div><span>brainsort ahead</span><span class="muted">grey: a tie within 5%. Each cell is the best other sort divided by brainsort.</span></div>
<div class="tablewrap"><table class="data heat" id="det-heat"></table></div>
<h3>Across sizes</h3>
<p class="note" id="det-sizes-note"></p>
<div class="row inline"><span class="lbl" style="min-width:0">input</span><select id="det-sizes-ds"></select></div>
<div class="grid" id="det-sizes"></div>
<h3>One cell in detail</h3>
<div class="row inline"><span class="lbl" style="min-width:0">key type</span><select id="det-bar-t"></select><span class="lbl" style="min-width:0">input</span><select id="det-bar-d"></select></div>
<div class="grid" id="det-bar"></div>
</div>
<details id="det-tables-wrap"><summary>Full tables for this metric<span class="muted" id="det-tables-sum"></span><span class="hint"></span></summary><div class="body" id="det-tables"></div></details>
</section>

<!--cpp-->
<details class="section" id="timing">
<summary><span class="eyebrow">Per machine</span><h2>Measured time per machine <span class="tag">measured</span></h2><span class="muted" id="timing-sum"></span><span class="hint"></span></summary>
<div class="body">
<p class="intro" id="timing-intro"></p><div class="legend"><span><i style="background:var(--c-candidate)"></i>brainsort</span><span><i style="background:var(--c-upstream)"></i>upstream code, as shipped</span></div>
<div class="row inline"><span class="lbl" style="min-width:0">machine</span><div class="chips" id="c-platform"></div></div>
<div class="machine" id="machine"></div>
<div class="chips" id="c-meas-metric"></div>
<p class="note" id="meas-metric-note"></p>
<div class="tiles" id="meas-tiles"></div>
<h3>Where brainsort wins and loses</h3>
<p class="note" id="meas-heat-note"></p><div class="scale"><span>brainsort behind</span><div class="bar"><i style="background:var(--loss-4)"></i><i style="background:var(--loss-3)"></i><i style="background:var(--loss-2)"></i><i style="background:var(--loss-1)"></i><i style="background:var(--neutral)"></i><i style="background:var(--win-1)"></i><i style="background:var(--win-2)"></i><i style="background:var(--win-3)"></i><i style="background:var(--win-4)"></i></div><span>brainsort ahead</span><span class="muted">grey: a tie within 5%. Each cell is the best other sort divided by brainsort.</span></div>
<div class="tablewrap"><table class="data heat" id="meas-heat"></table></div>
<h3>Across sizes</h3>
<p class="note" id="meas-sizes-note"></p>
<div class="row inline"><span class="lbl" style="min-width:0">input</span><select id="meas-sizes-ds"></select></div>
<div class="grid" id="meas-sizes"></div>
<h3>One cell in detail</h3>
<div class="row inline"><span class="lbl" style="min-width:0">key type</span><select id="meas-bar-t"></select><span class="lbl" style="min-width:0">input</span><select id="meas-bar-d"></select></div>
<div class="grid" id="meas-bar"></div>
<details><summary>Full tables for this metric<span class="muted" id="meas-tables-sum"></span><span class="hint"></span></summary><div class="body" id="meas-tables"></div></details>
<h3>All machines side by side</h3>
<p class="note" id="allplat-note"></p>
<div class="tablewrap"><table class="data" id="allplat"></table></div>
</div>
</details>
<!--/cpp-->

<details class="section" id="api">
<summary><span class="eyebrow"><!--cpp-->What a user gets<!--/cpp--><!--rust-->Per machine: what a user gets<!--/rust--></span><h2><!--cpp-->The library on plain vectors<!--/cpp--><!--rust-->The crate on plain vectors<!--/rust--> <span class="tag">measured</span></h2><span class="muted" id="api-sum"></span><span class="hint"></span></summary>
<div class="body">
<p class="intro" id="api-intro"></p>
<div class="row inline"><span class="lbl" style="min-width:0">machine</span><div class="chips" id="c-api-platform"></div><span class="lbl" style="min-width:0;margin-left:8px">elements</span><div class="chips" id="c-api-n"></div></div>
<div class="machine" id="api-machine"></div>
<div class="tiles" id="api-tiles"></div>
<p class="note" id="api-heat-note"></p><div class="scale"><span>brainsort behind</span><div class="bar"><i style="background:var(--loss-4)"></i><i style="background:var(--loss-3)"></i><i style="background:var(--loss-2)"></i><i style="background:var(--loss-1)"></i><i style="background:var(--neutral)"></i><i style="background:var(--win-1)"></i><i style="background:var(--win-2)"></i><i style="background:var(--win-3)"></i><i style="background:var(--win-4)"></i></div><span>brainsort ahead</span><span class="muted">grey: a tie within 5%. Each cell is the best other sort divided by brainsort.</span></div>
<div class="tablewrap"><table class="data heat" id="api-heat"></table></div>
<div id="api-tables"></div>
<h3>All machines side by side</h3>
<p class="note" id="api-allplat-note"></p>
<div class="tablewrap"><table class="data" id="api-allplat"></table></div>
<h3>C++ and Rust on the same machine</h3>
<p class="note" id="pair-note"></p>
<div class="tablewrap"><table class="data" id="pair"></table></div>
</div>
</details>

<section id="method">
<span class="eyebrow">Behind the numbers</span>
<h2>Method</h2>
<div class="cols">
<div>
<h3>How the numbers are made</h3>
<ul>
<li>Every (size, key type, input, algorithm) cell runs in a <b>fresh process</b>, pinned to one CPU where the OS allows it, so heap state and memory figures do not depend on what ran before.</li>
<li>The input comes from a fixed seed and is the same for every algorithm. Every element carries its original position, so the harness can check the result: keys in order, no element lost or duplicated, and for a stable sort the exact order of equal keys.</li>
<li><b>Deterministic numbers</b> come from one run of an instrumented build that counts every element read and write, every comparison, every table access and every allocation, and feeds the access sequence to a fixed cache model. The upstream code is counted through an element wrapper with the same definitions, so a swap is two reads and two writes for everyone, and through the global allocator (and, for Boost.Sort, its malloc calls) for scratch memory.</li>
<!--cpp--><li><b>Measured numbers</b> come from repeated timed runs of the plain build: the median of the repetitions, and the best median of two rounds over the whole matrix, because interference can only make a run slower. Below 100,000 elements the timed region sorts enough independent copies to reach 100,000 elements and reports the time per sort, so the clock's own cost stays small. Above, the repetitions shrink in proportion, never below three.</li>
<li>All algorithms are compiled with the same flags (<code>-O2</code>, no <code>-march=native</code>). brainsort chooses its AVX2 and BMI2 paths at run time.</li>
<li><b>The library on plain vectors</b> is a separate benchmark: <code>brainsort::sort</code> on a <code>std::vector</code> against <code>std::stable_sort</code>, cpp-TimSort and the two Boost.Sort stable sorts on the same generated inputs, the median of a few runs, every result checked, the same flags as above.</li><!--/cpp-->
<!--rust--><li><b>The crate on plain vectors</b> is the timed measurement of the Rust crate: <code>brainsort::sort</code> on a <code>Vec</code> against the standard library's stable sorts and the glidesort crate as published on crates.io, on the same generated inputs as the C++ benchmark, the median of a few runs, every result checked. Everything is built by cargo in release mode with no target-cpu flags; brainsort chooses its AVX2 and BMI2 paths at run time. The algorithm-level harness with instrumented ports and hardware counters is C++ code and is on the C++ page; its deterministic counts above hold for the crate.</li>
<li>Where both languages were measured on one machine, the same machine id pairs them, so a C++ and a Rust number side by side always come from the same computer.</li><!--/rust-->
<li>Every timing file carries a fingerprint of the code it measured. This page is built from the current sources; a file that measured other code is left out, so no time on this page describes older code than the numbers next to it.</li>
</ul>
<h3>Reproduce it</h3>
<!--cpp--><pre><code>sh scripts/counts.sh        # the deterministic numbers (results/counts.csv, any machine)
sh scripts/bench.sh         # the timing of this machine, then both pages of the website
python3 scripts/website.py     # the website alone, from results/</code></pre>
<p class="note">Windows: <code>scripts\counts.ps1</code>, <code>scripts\bench.ps1</code>. Details in <a id="link-setup" href="#">setup.md</a>.</p><!--/cpp-->
<!--rust--><pre><code>cd rust
cargo test --release                                             # the crate against the golden counts, among the tests
cargo run --release -p brainsort-bench -- --golden               # the counted view of every cell, checked against results/counts.csv
cargo run --release -p brainsort-bench -- --rust-counts          # the Rust sorts as shipped: results/rust-counts.csv, the deterministic numbers of this page
cargo run --release -p brainsort-bench -- --rust-golden          # that file recomputed and checked
cargo run --release -p brainsort-bench -- --bench --out ../results/&lt;id&gt;.rust.api.md   # the timing of this machine
python3 ../scripts/website.py                                    # both pages of the website, from results/</code></pre>
<p class="note"><code>scripts/bench.sh</code> and <code>scripts\bench.ps1</code> run the Rust benchmark after the C++ one when cargo is installed. Details in <a id="link-setup" href="#">setup.md</a>.</p><!--/rust-->
</div>
<div>
<h3>What each metric means</h3>
<dl id="glossary"></dl>
</div>
</div>
<h3>The algorithms</h3>
<div class="tablewrap"><table class="data wrap" id="algos"></table></div>
<!--cpp--><p class="note">"Best opponent" on this page means the best of the other sorts in a cell. Every sort here is upstream code run as shipped, counted through an element wrapper and the allocator; nothing is a port written here. The pool is the C++ sorts that can do what brainsort does: stable, every key type, an arbitrary comparator. std::sort, pdqsort and the radix libraries are not in it, because an unstable sort or a scalar-only radix sort has an easier job, so a comparison with them would not be like for like. Rust 1.81+ ships driftsort and CPython 3.11+ merges with powersort; the Rust page counts driftsort as shipped, on what a Rust program can observe.</p><!--/cpp-->
<!--rust--><p class="note">"Best opponent" on this page means the best of the other sorts in a cell. Every sort here is the published code, run unchanged: the deterministic numbers count what it asks of the program (comparisons, key reads, memory), the timed numbers what it costs. The pool is the Rust sorts that can do what brainsort does: stable, every key type, an arbitrary comparator. slice::sort_unstable and the radix crates (radsort, voracious_radix_sort, rdst) are not in it, because an unstable sort or a scalar-only radix sort has an easier job, so a comparison with them would not be like for like; the Rust ecosystem has two such sorts, the standard library's and glidesort, and the cached-key form of the standard one is the route a program takes for a sort by key. Element moves cannot be observed from outside in Rust, so bytes moved, reads, writes and the cache model are not counted for these sorts; the C++ page has them for the C++ pool.</p><!--/rust-->
<h3>The key types</h3>
<div class="tablewrap"><table class="data wrap" id="types"></table></div>
<h3>The inputs</h3>
<div class="tablewrap"><table class="data wrap" id="datasets"></table></div>
</section>

<section id="tables">
<span class="eyebrow">Raw data</span>
<h2>All numbers</h2>
<p>Every value behind this page, for the selected size and key types: the deterministic columns once, the measured columns per machine. Open a cell to render it.</p>
<div id="all-tables"></div>
</section>

<footer id="footer"></footer>
</main>
<div class="tip" id="tip" hidden></div>
<script>
'use strict';
const DATA = __DATA__;

// ---- data access ---------------------------------------------------------------
function table(t) { const idx = Object.fromEntries(t.cols.map((c, i) => [c, i])); return { idx, rows: t.rows, get: (r, c) => r[idx[c]] }; }
const LANG = '__LANG__', OTHER_LANG = LANG === 'cpp' ? 'rust' : 'cpp';   // this page's language, and the other page's
const LANG_NAME = { cpp: 'C++', rust: 'Rust' };
const DET = table(DATA.det), TIM = table(DATA.timing);
// The public call on a plain vector: this page's language (BENCH) and, for the pairing table, the other one (OTHER).
const BENCH = DATA.bench[LANG], OTHER = DATA.bench[OTHER_LANG], BT = table(BENCH), OT = table(OTHER);
const detIdx = new Map(DET.rows.map(r => [`${r[0]}|${r[1]}|${r[2]}|${r[3]}`, r]));
const detRow = (n, t, d, a) => detIdx.get(`${n}|${t}|${d}|${a}`);
const timIdx = new Map(TIM.rows.map(r => [`${r[0]}|${r[1]}|${r[2]}|${r[3]}|${r[4]}`, r]));
const timRow = (p, n, t, d, a) => timIdx.get(`${p}|${n}|${t}|${d}|${a}`);
const benchIdx = new Map(BT.rows.map(r => [`${r[0]}|${r[1]}|${r[2]}|${r[3]}`, r]));
const benchRow = (p, t, n, d) => benchIdx.get(`${p}|${t}|${n}|${d}`);
const otherIdx = new Map(OT.rows.map(r => [`${r[0]}|${r[1]}|${r[2]}|${r[3]}`, r]));
const otherRow = (p, t, n, d) => otherIdx.get(`${p}|${t}|${n}|${d}`);
const BPLAT = Object.fromEntries(BENCH.platforms.map(p => [p.id, p]));
const OPLAT = Object.fromEntries(OTHER.platforms.map(p => [p.id, p]));
const benchDefaultN = () => BENCH.sizes.includes(100000) ? 100000 : (BENCH.sizes[BENCH.sizes.length - 1] || 0);
// This page's benchmark type -> the other language's type for the same cell.
const PAIR = Object.fromEntries(DATA.pairs.map(([c, r]) => LANG === 'cpp' ? [c, r] : [r, c]));
const ALGO = Object.fromEntries(DATA.algos.map(a => [a.name, a]));
const PLAT = Object.fromEntries(DATA.platforms.map(p => [p.id, p]));
const REF_N = DATA.sizes.includes(100000) ? 100000 : DATA.sizes[DATA.sizes.length - 1];
// Not every size has every kind of number: the deterministic runs may stop below the
// largest timed size (CI adds the million-element counts). Fall back to the nearest size below.
const DET_SIZES = [...new Set(DET.rows.map(r => r[0]))].sort((a, b) => a - b);
function detN() { if (DET_SIZES.includes(state.n)) return state.n; const below = DET_SIZES.filter(n => n < state.n); return below.length ? below[below.length - 1] : DET_SIZES[0]; }
const hasTiming = (pid, n) => TIM.rows.some(r => r[0] === pid && r[1] === n);

// ---- formatting ----------------------------------------------------------------
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmtN = n => n >= 1e6 ? (n / 1e6) + ' M' : n >= 1e3 ? (n / 1e3) + ' k' : String(n);
const fmtNum = n => Math.round(n).toLocaleString('en-US');
// Milliseconds as the API benchmarks print them: three decimals, more below one ms so that a sort of ten elements keeps three significant digits.
const fmtMs = v => v >= 1 ? v.toFixed(3) : v >= 0.01 ? v.toFixed(4) : v.toFixed(6);
function compact(v) { if (v == null) return 'n/a'; if (v >= 1e9) return (v / 1e9).toFixed(2) + ' G'; if (v >= 1e6) return (v / 1e6).toFixed(2) + ' M'; if (v >= 1e3) return (v / 1e3).toFixed(1) + ' k'; return String(Math.round(v * 10) / 10); }
function bytes(v) { if (v == null) return 'n/a'; if (v === 0) return '0'; if (v < 1024) return Math.round(v) + ' B'; if (v < 1048576) return (v / 1024).toFixed(1) + ' KiB'; if (v < 1073741824) return (v / 1048576).toFixed(2) + ' MiB'; return (v / 1073741824).toFixed(2) + ' GiB'; }
function ns(v) { if (v == null) return 'n/a'; if (v < 1e3) return v.toFixed(0) + ' ns'; if (v < 1e6) return (v / 1e3).toFixed(v < 1e4 ? 2 : 1) + ' µs'; if (v < 1e9) return (v / 1e6).toFixed(v < 1e7 ? 3 : 2) + ' ms'; return (v / 1e9).toFixed(2) + ' s'; }
function perElem(v, unit) { if (v == null) return 'n/a'; if (unit === 'ns') return v < 10 ? v.toFixed(2) + ' ns' : v.toFixed(1) + ' ns'; if (unit === 'bytes') return v.toFixed(1) + ' B'; return v < 10 ? v.toFixed(2) : v.toFixed(1); }
function niceStep(max, ticks) { const raw = max / ticks, p = Math.pow(10, Math.floor(Math.log10(raw))), r = raw / p; return (r < 1.5 ? 1 : r < 3.5 ? 2 : r < 7.5 ? 5 : 10) * p; }
const timeStamp = s => s ? String(s).replace('T', ' ').replace('Z', ' UTC') : 'unknown time';

// ---- metrics -------------------------------------------------------------------
// Deterministic first, in order of what matters for real work: bytes moved,
// comparisons, memory; then the finer counts, table accesses last because
// they have no contest. Measured: time first.
const ALL_DET_METRICS = [
  { key: 'traffic', label: 'Memory traffic', unit: 'bytes', fmt: bytes, per: 'bytes', noun: 'bytes moved', less: 'less',
    plain: 'How many bytes the algorithm read and wrote in total: elements moved in the array and in scratch buffers, the count tables a radix sort keeps, and the string bytes it looked at. The closest single number to how much work the memory system does, and every algorithm pays it in the same currency.' },
  { key: 'compares', label: 'Comparisons', unit: '', fmt: compact, per: '', noun: 'comparisons', less: 'fewer',
    plain: 'How many times two keys were compared. The classic cost measure of a sort. A radix sort makes almost none, which is the whole point of it.' },
  { key: 'aux', label: 'Scratch memory', unit: 'bytes', fmt: bytes, per: 'bytes', noun: 'scratch memory', less: 'less',
    plain: 'The most extra heap memory the algorithm asked for at any moment, beyond the array itself. In-place sorts use none; merge sorts and radix sorts trade memory for speed. brainsort keeps it at about half the array on random input and at zero on sorted or reversed input.' },
  { key: 'reads', label: 'Element reads', unit: '', fmt: compact, per: '', noun: 'element reads', less: 'fewer',
    plain: 'How many elements were loaded from the array or from a scratch buffer.' },
  { key: 'writes', label: 'Element writes', unit: '', fmt: compact, per: '', noun: 'element writes', less: 'fewer',
    plain: 'How many elements were stored into the array or into a scratch buffer.' },
  { key: 'l1', label: 'Cache misses (L1 model)', unit: '', fmt: compact, per: '', noun: 'L1 misses', less: 'fewer',
    plain: 'Misses in a fixed model of a first-level CPU cache (32 KiB, 8-way, 64-byte lines) fed with the algorithm’s access pattern. A machine-independent measure of locality: scattered accesses miss, streaming accesses do not. A radix scatter is charged here where a quicksort is not; the real CPU’s larger caches absorb much of it.' },
  { key: 'l2', label: 'Cache misses (L2 model)', unit: '', fmt: compact, per: '', noun: 'L2 misses', less: 'fewer',
    plain: 'Misses in the model’s second level (1 MiB, 16-way), probed on an L1 miss.' },
  { key: 'flips', label: 'Compare flips', unit: '', fmt: compact, per: '', noun: 'compare flips', less: 'fewer',
    plain: 'How often a comparison’s outcome differed from the one before. A branch predictor learns patterns; flips are what it cannot learn, so this is a machine-independent proxy for mispredicted branches, the main cost of a quicksort on random data.' },
  { key: 'keybytes', label: 'Key bytes (strings)', unit: 'bytes', fmt: bytes, per: 'bytes', noun: 'key bytes', less: 'fewer',
    plain: 'For string keys: how many bytes of key text were loaded, by comparisons (up to the first differing byte) and by the radix chunk extraction. Zero for fixed-size keys.' },
  { key: 'keyreads', label: 'Key reads', unit: '', fmt: compact, per: '', noun: 'key reads', less: 'fewer',
    plain: 'How often slice::sort_by_cached_key asked an element for its key: once per element, before it sorts the keys with their indices. Not counted for the comparator sorts, which never ask, nor for brainsort, whose key work is counted as table accesses on the C++ page.',
    noContest: 'A number here is the one pass over the keys the cached-key sort makes, and only it pays it: the comparator sorts never ask for a key, and brainsort’s key work is counted in a different unit (table accesses, on the C++ page), so there is no contest to score. The full tables below carry the raw counts.' },
  { key: 'table', label: 'Table accesses', unit: '', fmt: compact, per: '', noun: 'table accesses', less: 'fewer',
    plain: 'Reads and writes of the count tables a radix sort keeps: one small array per pass, touched once per element to count and once to place. Comparison sorts have none; this is what a radix sort pays instead of comparisons. Memory traffic counts these bytes next to the element traffic.',
    noContest: 'A large number here is not a loss, even though it looks like one. A radix sort touches a table of a few thousand entries a few times per element, and in exchange moves each element a few times instead of log n times and makes almost no comparisons. The table is capped at 13-bit digits, 32 KiB, so it stays in a first-level cache and each access is cheap. Every comparison sort has zero, so there is no contest to score. Memory traffic adds these table bytes to brainsort’s total and is the metric that decides; the full tables below carry the raw counts.' },
];
// The metrics this page's deterministic file has (the C++ harness counts everything; the Rust file what a program can observe).
const DET_METRICS = ALL_DET_METRICS.filter(m => DATA.detCols.includes(m.key));
const MEAS_METRICS = [
  { key: 'wall', label: 'Wall time', unit: 'ns', fmt: ns, per: 'ns', noun: 'time', less: 'less', faster: true,
    plain: 'Elapsed time of one sort: the median of the repetitions, best of two rounds. What a user waits for.' },
  { key: 'cycles', label: 'CPU cycles', unit: '', fmt: compact, per: '', noun: 'cycles', less: 'fewer',
    plain: 'Cycles the CPU spent in the sort (a hardware counter on Linux, the thread cycle time on Windows). Like time, but independent of clock frequency changes.' },
  { key: 'instr', label: 'Instructions', unit: '', fmt: compact, per: '', noun: 'instructions', less: 'fewer',
    plain: 'Retired user-space instructions, available on Linux with performance counters. Nearly deterministic. A radix sort runs more, cheaper instructions than a quicksort: no unpredictable branches.' },
  { key: 'cpu', label: 'CPU time', unit: 'ns', fmt: ns, per: 'ns', noun: 'CPU time', less: 'less',
    plain: 'Thread CPU time (Linux; the Windows clock is too coarse). Equal to wall time unless the thread was descheduled.' },
  { key: 'rss', label: 'Peak memory growth', unit: 'bytes', fmt: bytes, per: 'bytes', noun: 'peak memory growth', less: 'less',
    plain: 'How much the process’s peak resident memory grew during the sort, in whole pages. The practical memory cost, allocator behaviour included.' },
];
const gcolor = g => `var(--c-${g})`;
const color = a => gcolor(ALGO[a] ? ALGO[a].group : 'upstream');
const DASH = { 'brainsort': '', 'std::stable_sort': '', 'gfx::timsort': '6 3', 'boost::spinsort': '2 3', 'boost::flat_stable_sort': '8 3 2 3',
  'slice::sort': '', 'slice::sort_by_cached_key': '6 3', 'glidesort': '2 3' };


// ---- state ---------------------------------------------------------------------
const state = {
  n: REF_N, types: DATA.types.map(t => t.name), datasets: 'all',
  det: DET_METRICS[0].key, p: DATA.platforms.length ? DATA.platforms[0].id : '', meas: 'wall',
  benchP: BENCH.platforms.length ? BENCH.platforms[0].id : '', benchN: BENCH.sizes.includes(100000) ? 100000 : (BENCH.sizes[BENCH.sizes.length - 1] || 0),
  detDs: 'random', detBarT: DATA.types[0] ? DATA.types[0].name : '', detBarD: 'random',
  measDs: 'random', measBarT: DATA.types[0] ? DATA.types[0].name : '', measBarD: 'random',
};
function readHash() {
  const q = new URLSearchParams(location.hash.slice(1));
  const n = parseInt(q.get('n') || '', 10); if (DATA.sizes.includes(n)) state.n = n;
  const ty = q.get('type'); if (ty && ty !== 'all') { const l = ty.split(',').filter(x => DATA.types.some(t => t.name === x)); if (l.length) state.types = l; }
  if (q.get('inputs') === 'standard') state.datasets = 'standard';
  const dm = q.get('det'); if (DET_METRICS.some(m => m.key === dm)) state.det = dm;
  const mm = q.get('meas'); if (MEAS_METRICS.some(m => m.key === mm)) state.meas = mm;
  const p = q.get('machine'); if (PLAT[p]) state.p = p;
  const ap = q.get('api_machine'); if (BPLAT[ap]) state.benchP = ap;
  const an = parseInt(q.get('api_n') || '', 10); if (BENCH.sizes.includes(an)) state.benchN = an;
  for (const id of (q.get('open') || '').split(',')) { const el = document.getElementById(id); if (el && el.tagName === 'DETAILS') el.open = true; }
}
function writeHash() {
  const q = new URLSearchParams();
  if (state.n !== REF_N) q.set('n', state.n);
  if (state.types.length !== DATA.types.length) q.set('type', state.types.join(','));
  if (state.datasets !== 'all') q.set('inputs', 'standard');
  if (state.det !== DET_METRICS[0].key) q.set('det', state.det);
  if (state.meas !== 'wall') q.set('meas', state.meas);
  if (DATA.platforms.length && state.p !== DATA.platforms[0].id) q.set('machine', state.p);
  if (BENCH.platforms.length && state.benchP !== BENCH.platforms[0].id) q.set('api_machine', state.benchP);
  if (BENCH.sizes.length && state.benchN !== benchDefaultN()) q.set('api_n', state.benchN);
  const open = ['timing', 'api'].filter(id => { const el = document.getElementById(id); return el && el.open; }); if (open.length) q.set('open', open.join(','));
  const h = q.toString().replace(/%3A/g, ':').replace(/%2C/g, ',');
  history.replaceState(null, '', h ? '#' + h : location.pathname + location.search);
  // The other language's page opens with the same size, key types, inputs and metric.
  for (const a of document.querySelectorAll('nav.lang a')) a.setAttribute('href', (a.classList.contains('cpp') ? 'index.html' : 'rust.html') + (h ? '#' + h : ''));
}
const selTypes = () => DATA.types.filter(t => state.types.includes(t.name));
const selDatasets = () => DATA.datasets.filter(d => state.datasets === 'all' || d.standard);
// Shown algorithms: brainsort, then the others. Every sort of the pool is an opponent.
function shownAlgos() { return DATA.algos; }
const isOpponent = a => a !== DATA.candidate;
const detMetric = () => DET_METRICS.find(m => m.key === state.det);
const measMetric = () => MEAS_METRICS.find(m => m.key === state.meas);

// ---- generic comparison: brainsort against the best shown opponent ---------------
// get(a) -> value or null. Returns null when brainsort or every opponent is missing.
function compareCell(get) {
  const b = get(DATA.candidate); if (b == null) return null;
  let best = null;
  for (const a of shownAlgos()) { if (!isOpponent(a.name)) continue; const v = get(a.name); if (v == null) continue; if (!best || v < best.v) best = { name: a.name, v }; }
  if (!best) return null;
  const win = b <= best.v;
  const ratio = b === 0 && best.v === 0 ? 1 : b === 0 ? Infinity : best.v === 0 ? 0 : best.v / b;   // > 1: brainsort ahead
  return { b, o: best.v, opp: best.name, win, ratio };
}
function heatClass(c) {
  if (!c) return '';
  if (c.ratio === Infinity) return 'w4 w'; if (c.ratio === 0) return 'l4';
  const l = Math.log2(c.ratio);
  if (Math.abs(l) < 0.07) return 'n0';
  const s = Math.abs(l) < 0.4 ? 1 : Math.abs(l) < 1 ? 2 : Math.abs(l) < 2 ? 3 : 4;
  return l > 0 ? 'w' + s + ' w' : 'l' + s;
}
function ratioText(c) { if (!c) return 'n/a'; if (c.ratio === Infinity) return '0 vs work'; if (c.ratio === 0) return 'work vs 0'; return c.ratio.toFixed(2) + 'x'; }
function winPhrase(m, r) { return m.faster ? `${r.toFixed(1)}x faster` : `${r.toFixed(1)}x ${m.less} ${m.noun}`; }
function lossPhrase(m, r) { return m.faster ? `${r.toFixed(2)}x slower` : `${r.toFixed(2)}x more ${m.noun}`; }
// The score over the selected types and datasets at size n: wins, biggest win, worst loss.
function score(getFor) {
  const cs = [];
  for (const t of selTypes()) for (const d of selDatasets()) { const c = compareCell(getFor(t.name, d.name)); if (c) cs.push({ t: t.name, d: d.name, c }); }
  const wins = cs.filter(x => x.c.win);
  const best = wins.filter(x => isFinite(x.c.ratio) && x.c.ratio > 1.05).sort((a, b) => b.c.ratio - a.c.ratio)[0];
  const inf = wins.find(x => x.c.ratio === Infinity);
  const worst = cs.filter(x => !x.c.win && x.c.ratio > 0).sort((a, b) => a.c.ratio - b.c.ratio)[0];
  const zeroLoss = cs.filter(x => !x.c.win && x.c.ratio === 0).sort((a, b) => b.c.b - a.c.b)[0];
  return { cells: cs, wins, best, inf, worst, zeroLoss };
}
const detGet = (n, t, d) => a => { const r = detRow(n, t, d, a); return r ? DET.get(r, state.det) : null; };
const detGetM = (n, t, d, m) => a => { const r = detRow(n, t, d, a); return r ? DET.get(r, m) : null; };
const timGet = (p, n, t, d, m) => a => { const r = timRow(p, n, t, d, a); return r && TIM.get(r, 'ok') ? TIM.get(r, m) : null; };

// ---- tiles, heat map, tables, charts (shared by both sections) --------------------
function tiles(host, m, sc, extra, atN) {
  const n = sc.cells.length;
  if (!n) { host.innerHTML = `<div class="tile empty">Nothing to compare: no other sort has ${esc(m.label.toLowerCase())} here.</div>`; return; }
  let bw = 'none', bs = 'brainsort is never ahead here', wc = 'none', ws = 'brainsort is best or tied in every cell';
  if (m.key === 'aux' && sc.wins.some(x => x.c.b === 0)) { bw = '0 bytes'; bs = 'no scratch memory at all on ' + [...new Set(sc.wins.filter(x => x.c.b === 0).map(x => x.d))].join(', ') + ' input'; }
  else if (sc.inf) { bw = 'all the work saved'; bs = `the others need ${m.noun}, brainsort none, on ${sc.inf.t} ${sc.inf.d}`; }
  else if (sc.best) { bw = winPhrase(m, sc.best.c.ratio); bs = `than ${sc.best.c.opp}, the best other sort on ${sc.best.t} ${sc.best.d} (${m.fmt(sc.best.c.b)} vs ${m.fmt(sc.best.c.o)})`; }
  if (m.key === 'aux' && sc.zeroLoss) { wc = m.fmt(sc.zeroLoss.c.b); ws = `where the in-place sorts use none (${sc.zeroLoss.t} ${sc.zeroLoss.d})`; }
  else if (sc.zeroLoss) { wc = m.fmt(sc.zeroLoss.c.b) + ' vs 0'; ws = `${sc.zeroLoss.c.opp} needs no ${m.noun} at all on ${sc.zeroLoss.t} ${sc.zeroLoss.d}, brainsort ${m.fmt(sc.zeroLoss.c.b)}`; }
  else if (sc.worst) { wc = lossPhrase(m, 1 / sc.worst.c.ratio); ws = `than ${sc.worst.c.opp} on ${sc.worst.t} ${sc.worst.d} (${m.fmt(sc.worst.c.b)} vs ${m.fmt(sc.worst.c.o)}${m.faster ? ', +' + ns(sc.worst.c.b - sc.worst.c.o) : ''})`; }
  host.innerHTML =
    `<div class="tile hero ${sc.wins.length * 2 >= n ? 'win' : 'loss'}"><div class="k">${esc(m.label)}${extra ? ' · ' + esc(extra) : ''}</div><div class="v">${sc.wins.length} <small>of ${n} cells</small></div><div class="s">where brainsort is ${m.faster ? 'fastest' : 'lowest'} or tied against every other sort, at n = ${fmtN(atN || state.n)}</div></div>` +
    `<div class="tile"><div class="k">Biggest win</div><div class="v">${esc(bw)}</div><div class="s">${esc(bs)}</div></div>` +
    `<div class="tile"><div class="k">Worst loss</div><div class="v">${esc(wc)}</div><div class="s">${esc(ws)}</div></div>`;
}
function heat(host, m, getFor, tipKind, n) {
  const types = selTypes();
  let h = '<thead><tr><th>input</th>' + types.map(t => `<th title="${esc(t.desc)}">${esc(t.name)}</th>`).join('') + '</tr></thead><tbody>';
  for (const d of selDatasets()) {
    h += `<tr><td title="${esc(d.description)}">${esc(d.name)}</td>`;
    for (const t of types) {
      const c = compareCell(getFor(t.name, d.name));
      h += c ? `<td class="cell ${heatClass(c)}" data-tip="${tipKind}" data-t="${esc(t.name)}" data-d="${esc(d.name)}"><div>${ratioText(c)}</div></td>` : `<td class="cell"><div class="muted">${detRow(n, t.name, d.name, DATA.candidate) || timRow(state.p, n, t.name, d.name, DATA.candidate) ? 'n/a' : '–'}</div></td>`;
    }
    h += '</tr>';
  }
  host.innerHTML = h + '</tbody>';
}
// One group: every sort of the pool is stable, so there is nothing to separate.
function stabilityGroups() { return [{ key: 'all', label: 'every sort', algos: shownAlgos() }]; }
function matrixTables(host, m, getFor, summaryEl) {
  let h = '';
  for (const t of selTypes()) {
    for (const g of stabilityGroups()) {
      let rows = '', any = false;
      for (const d of selDatasets()) {
        const get = getFor(t.name, d.name);
        const vals = g.algos.map(a => get(a.name));
        if (!vals.some(v => v != null)) continue;
        any = true;
        const best = Math.min(...vals.filter(v => v != null));
        rows += `<tr><td title="${esc(d.description)}">${esc(d.name)}</td>` + vals.map((v, i) => v == null ? '<td class="muted">n/a</td>' :
          `<td class="${v === best ? 'best' : ''}">${esc(m.fmt(v))}<span class="rel">${v === best ? 'best' : best === 0 ? '' : (v / best).toFixed(2) + 'x'}</span></td>`).join('') + '</tr>';
      }
      if (any) h += `<h4>${esc(t.name)}</h4><div class="tablewrap"><table class="data"><thead><tr><th>input</th>` +
        g.algos.map(a => `<th><span class="sw" style="background:${color(a.name)}"></span>${esc(a.name)}</th>`).join('') + `</tr></thead><tbody>${rows}</tbody></table></div>`;
    }
  }
  host.innerHTML = h || '<p class="empty">nothing measured here</p>';
  if (summaryEl) summaryEl.textContent = ` ${m.label.toLowerCase()}, n = ${fmtN(state.n)}, ${selTypes().length} key type${selTypes().length === 1 ? '' : 's'}`;
}
// Line chart: metric per element against n (log x), one line per shown algorithm.
function sizesCharts(host, m, getAt, note) {
  host.innerHTML = '';
  const sizes = DATA.sizes;
  const algos = shownAlgos();
  for (const t of selTypes()) {
    const series = algos.map(a => ({ a: a.name, pts: sizes.map(n => { const v = getAt(n, t.name, a.name); return v == null ? null : v / n; }) })).filter(s => s.pts.some(v => v != null));
    if (!series.length) continue;
    const vals = series.flatMap(s => s.pts.filter(v => v != null));
    const max = Math.max(1e-9, ...vals), step = niceStep(max, 4), yMax = Math.ceil(max / step - 1e-9) * step || max;
    const W = 460, H = 250, L = 56, R = 110, T = 12, B = 34, pw = W - L - R, ph = H - T - B;
    const lx = Math.log10(sizes[0]), hx = Math.log10(sizes[sizes.length - 1]);
    const x = n => L + (sizes.length > 1 ? (Math.log10(n) - lx) / (hx - lx) : 0.5) * pw, y = v => T + ph - (v / yMax) * ph;
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(m.label)} per element for ${esc(t.name)} against input size">`;
    for (let v = 0; v <= yMax * 1.0001; v += step) s += `<line x1="${L}" x2="${L + pw}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/><text x="${L - 6}" y="${y(v) + 3.5}" font-size="10" fill="var(--text-3)" text-anchor="end">${esc(perElem(v, m.per))}</text>`;
    for (const n of sizes) s += `<text x="${x(n)}" y="${H - 14}" font-size="10" fill="var(--text-3)" text-anchor="middle">${esc(fmtN(n))}</text>`;
    s += `<text x="${L + pw / 2}" y="${H - 2}" font-size="10" fill="var(--text-3)" text-anchor="middle">elements</text>`;
    // Direct labels at the right end, pushed apart so they do not collide.
    // Labels are placed at the line's last point, top to bottom, each pushed below the one above it.
    const ends = series.map(se => { let last = null; se.pts.forEach((v, i) => { if (v != null) last = { i, v }; }); return { se, last }; }).filter(e => e.last).sort((a, b) => b.last.v - a.last.v);
    let bottom = -Infinity;
    for (const e of ends) { let py = y(e.last.v); if (py < bottom + 11) py = bottom + 11; bottom = py; e.ly = py; }
    const overflow = bottom - (T + ph + 4); if (overflow > 0) for (const e of ends) e.ly -= overflow;
    for (const e of ends) {
      const se = e.se, d = se.pts.map((v, i) => v == null ? null : `${x(sizes[i]).toFixed(1)},${y(v).toFixed(1)}`);
      let path = '', pen = false; d.forEach(p => { if (!p) { pen = false; return; } path += (pen ? ' L' : ' M') + p; pen = true; });
      s += `<path d="${path}" fill="none" stroke="${color(se.a)}" stroke-width="${se.a === DATA.candidate ? 2.6 : 1.8}" stroke-dasharray="${DASH[se.a] || ''}" stroke-linecap="round"/>`;
      se.pts.forEach((v, i) => { if (v == null) return; s += `<circle cx="${x(sizes[i])}" cy="${y(v)}" r="${se.a === DATA.candidate ? 4 : 3}" fill="${color(se.a)}" stroke="var(--bg)" stroke-width="1.5" data-tip="pt" data-n="${sizes[i]}" data-t="${esc(t.name)}" data-a="${esc(se.a)}" data-v="${v}"/>`; });
      s += `<text x="${L + pw + 8}" y="${e.ly + 3.5}" font-size="10.5" fill="var(--text-2)" font-weight="${se.a === DATA.candidate ? 600 : 400}">${esc(se.a)}</text>`;
    }
    s += '</svg>';
    const card = document.createElement('div'); card.className = 'card';
    card.innerHTML = `<h3>${esc(t.name)}</h3><p class="desc">${esc(m.label)} per element, ${esc(note)}</p>${s}`;
    host.appendChild(card);
  }
  if (!host.innerHTML) host.innerHTML = `<p class="empty">${esc(m.label)} was not measured here.</p>`;
}
// Horizontal bars for one cell.
function barChart(host, m, get, title, desc) {
  const algos = shownAlgos().map(a => ({ a: a.name, v: get(a.name) })).filter(s => s.v != null);
  host.innerHTML = '';
  if (!algos.length) { host.innerHTML = `<p class="empty">nothing measured for this cell</p>`; return; }
  const max = Math.max(1e-9, ...algos.map(s => s.v)), best = Math.min(...algos.map(s => s.v));
  const W = 560, labelW = 170, valueW = 96, barH = 16, band = 26, top = 6, bottom = 24, pw = W - labelW - valueW, H = top + algos.length * band + bottom;
  const step = niceStep(max, 4), sMax = Math.ceil(max / step - 1e-9) * step || max;
  const x = v => labelW + (v / sMax) * pw;
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(m.label)} for ${esc(title)}">`;
  for (let v = 0; v <= sMax * 1.0001; v += step) s += `<line x1="${x(v)}" x2="${x(v)}" y1="${top}" y2="${H - bottom}" stroke="var(--grid)"/><text x="${x(v)}" y="${H - 8}" font-size="10" fill="var(--text-3)" text-anchor="middle">${esc(m.fmt(v))}</text>`;
  s += `<line x1="${labelW}" x2="${labelW}" y1="${top}" y2="${H - bottom}" stroke="var(--border)"/>`;
  algos.forEach((se, i) => {
    const yy = top + i * band + (band - barH) / 2, w = Math.max(0, x(se.v) - labelW), rr = Math.min(4, w / 2);
    s += `<text x="${labelW - 8}" y="${yy + barH / 2 + 4}" font-size="12" fill="var(--text)" text-anchor="end" font-weight="${se.a === DATA.candidate ? 600 : 400}">${esc(se.a)}</text>`;
    s += `<path d="M${labelW} ${yy} h${Math.max(0, w - rr)} a${rr} ${rr} 0 0 1 ${rr} ${rr} v${Math.max(0, barH - 2 * rr)} a${rr} ${rr} 0 0 1 -${rr} ${rr} h-${Math.max(0, w - rr)} z" fill="${color(se.a)}"/>`;
    s += `<text x="${labelW + w + 6}" y="${yy + barH / 2 + 4}" font-size="11" fill="${se.v === best ? 'var(--text)' : 'var(--text-2)'}" font-weight="${se.v === best ? 600 : 400}">${esc(m.fmt(se.v))}${se.v !== best && best > 0 ? ' (' + (se.v / best).toFixed(2) + 'x)' : ''}</text>`;
  });
  s += '</svg>';
  const card = document.createElement('div'); card.className = 'card'; card.style.gridColumn = '1 / -1';
  card.innerHTML = `<h3>${esc(title)}</h3><p class="desc">${esc(desc)}</p>${s}`;
  host.appendChild(card);
}

// ---- controls ------------------------------------------------------------------
function chips(host, items, isOn, onClick, cls) {
  if (!host) return;   // a control of a section this page does not have (the harness is C++ only)
  host.innerHTML = '';
  for (const it of items) {
    const b = document.createElement('button'); b.className = 'chip' + (cls ? ' ' + cls : '') + (it.color ? ' sw' : '') + (it.cls ? ' ' + it.cls : '');
    b.innerHTML = (it.color ? `<i style="background:${it.color}"></i>` : '') + esc(it.label);
    b.setAttribute('aria-pressed', String(!!isOn(it)));
    if (it.title) b.title = it.title;
    b.onclick = () => { onClick(it); refresh(); render(); };
    host.appendChild(b);
  }
}
function selectOptions(el, items, value, onChange) {
  if (!el) return;
  el.innerHTML = items.map(it => `<option value="${esc(it.id)}"${it.id === value ? ' selected' : ''}>${esc(it.label)}</option>`).join('');
  el.onchange = () => { onChange(el.value); render(); };
}
function refresh() {
  chips(document.getElementById('c-size'), DATA.sizes.map(n => ({ id: n, label: fmtN(n) + ' elements', cls: DET_SIZES.includes(n) ? '' : 'dim',
    title: DET_SIZES.includes(n) ? '' : 'timing only in this build of the page: no counted run at this size (scripts/counts.sh with COUNTS_SIZES adds one)' })), it => state.n === it.id, it => { state.n = it.id; });
  chips(document.getElementById('c-types'), [...DATA.types.map(t => ({ id: t.name, label: t.name, title: t.desc + ': ' + t.real })), { id: 'all', label: 'all types' }],
    it => it.id === 'all' ? state.types.length === DATA.types.length : state.types.length === 1 && state.types[0] === it.id,
    it => { state.types = it.id === 'all' ? DATA.types.map(t => t.name) : [it.id]; });
  chips(document.getElementById('c-datasets'), [{ id: 'standard', label: 'the 5 standard inputs' }, { id: 'all', label: 'all ' + DATA.datasets.length + ' inputs' }], it => state.datasets === it.id, it => { state.datasets = it.id; });
  chips(document.getElementById('c-det-metric'), DET_METRICS.map(m => ({ id: m.key, label: m.label, title: m.plain })), it => state.det === it.id, it => { state.det = it.id; });
  chips(document.getElementById('c-platform'), DATA.platforms.map(p => ({ id: p.id, label: p.label + (p.shared ? '' : ' ★'), title: (p.cpu || '') + (p.host ? ', ' + p.host : '') })), it => state.p === it.id, it => { state.p = it.id; });
  const p = PLAT[state.p];
  chips(document.getElementById('c-meas-metric'), MEAS_METRICS.filter(m => p && TIM.rows.some(r => r[0] === p.id && TIM.get(r, m.key) != null)).map(m => ({ id: m.key, label: m.label, title: m.plain })), it => state.meas === it.id, it => { state.meas = it.id; });
  chips(document.getElementById('c-api-platform'), BENCH.platforms.map(p => ({ id: p.id, label: p.label + (p.shared ? '' : ' ★'), title: p.cpu || '' })), it => state.benchP === it.id, it => { state.benchP = it.id; });
  chips(document.getElementById('c-api-n'), BENCH.sizes.map(n => ({ id: n, label: fmtN(n) })), it => state.benchN === it.id, it => { state.benchN = it.id; });
  const dsItems = selDatasets().map(d => ({ id: d.name, label: d.name }));
  if (!dsItems.some(d => d.id === state.detDs)) state.detDs = dsItems[0] ? dsItems[0].id : '';
  if (!dsItems.some(d => d.id === state.measDs)) state.measDs = dsItems[0] ? dsItems[0].id : '';
  if (!dsItems.some(d => d.id === state.detBarD)) state.detBarD = dsItems[0] ? dsItems[0].id : '';
  if (!dsItems.some(d => d.id === state.measBarD)) state.measBarD = dsItems[0] ? dsItems[0].id : '';
  const tyItems = DATA.types.map(t => ({ id: t.name, label: t.name }));
  if (!selTypes().some(t => t.name === state.detBarT)) state.detBarT = selTypes()[0] ? selTypes()[0].name : '';
  if (!selTypes().some(t => t.name === state.measBarT)) state.measBarT = selTypes()[0] ? selTypes()[0].name : '';
  selectOptions(document.getElementById('det-sizes-ds'), dsItems, state.detDs, v => { state.detDs = v; });
  selectOptions(document.getElementById('meas-sizes-ds'), dsItems, state.measDs, v => { state.measDs = v; });
  selectOptions(document.getElementById('det-bar-t'), tyItems.filter(t => state.types.includes(t.id)), state.detBarT, v => { state.detBarT = v; });
  selectOptions(document.getElementById('det-bar-d'), dsItems, state.detBarD, v => { state.detBarD = v; });
  selectOptions(document.getElementById('meas-bar-t'), tyItems.filter(t => state.types.includes(t.id)), state.measBarT, v => { state.measBarT = v; });
  selectOptions(document.getElementById('meas-bar-d'), dsItems, state.measBarD, v => { state.measBarD = v; });
}

// ---- header and glance ---------------------------------------------------------------
function renderHeader() {
  document.getElementById('links').innerHTML = LANG === 'cpp' ?
    `<a href="${esc(DATA.repo)}">Source and README</a><a href="${esc(DATA.repo)}/blob/main/single_include/brainsort.hpp">Single header</a>` +
    `<a href="${esc(DATA.repo)}/blob/main/docs/usage-cpp.md">C++ usage guide</a><a href="${esc(DATA.repo)}/blob/main/setup.md">Build, test, reproduce</a><a href="${esc(DATA.repo)}/tree/main/results">Raw data (CSV)</a>` :
    `<a href="${esc(DATA.repo)}">Source and README</a><a href="https://crates.io/crates/brainsort">crates.io</a><a href="https://docs.rs/brainsort">API docs on docs.rs</a>` +
    `<a href="${esc(DATA.repo)}/blob/main/docs/usage-rust.md">Rust usage guide</a><a href="${esc(DATA.repo)}/blob/main/setup.md">Build, test, reproduce</a><a href="${esc(DATA.repo)}/tree/main/results">Raw data</a>`;
  const ci = document.getElementById('ci-badge');
  ci.href = `${DATA.repo}/actions/workflows/ci.yml`;
  ci.firstElementChild.src = `${DATA.repo}/actions/workflows/ci.yml/badge.svg`;
  const setHref = (id, href) => { const el = document.getElementById(id); if (el) el.href = href; };   // the Use section differs per page
  setHref('link-setup', `${DATA.repo}/blob/main/setup.md`);
  setHref('link-header', `${DATA.repo}/blob/main/single_include/brainsort.hpp`);
  setHref('link-readme', `${DATA.repo}#the-library`);
  setHref('link-usage-cpp', `${DATA.repo}/blob/main/docs/usage-cpp.md`);
  setHref('link-usage-rust', `${DATA.repo}/blob/main/docs/usage-rust.md`);
  setHref('link-crate', 'https://crates.io/crates/brainsort');
  setHref('link-docsrs', 'https://docs.rs/brainsort');
  const machines = LANG === 'cpp' ? DATA.platforms.length : BENCH.platforms.length;
  document.getElementById('stamp').textContent = `${LANG_NAME[LANG]} page, generated ${timeStamp(DATA.generated)} from code ${(LANG === 'cpp' ? DATA.code : BENCH.code) || 'unknown'}. ` +
    `${DATA.algos.length} algorithms, ${DATA.types.length} key types, ${DATA.datasets.length} input patterns, ${DATA.sizes.length} sizes (${DATA.sizes.map(fmtN).join(', ')} elements), ${machines} machine${machines === 1 ? '' : 's'} timed.`;
  document.getElementById('footer').innerHTML = `brainsort ${esc(DATA.version)} · MIT license · <a href="${esc(DATA.repo)}">${esc(DATA.repo.replace(/^https?:\/\//, ''))}</a> · ` +
    `this is the ${LANG_NAME[LANG]} page; <a href="${OTHER_LANG === 'cpp' ? 'index.html' : 'rust.html'}">the ${LANG_NAME[OTHER_LANG]} page</a> has the same sections for the ${OTHER_LANG === 'cpp' ? 'C++ library' : 'Rust crate'} · ` +
    `generated by scripts/website.py from the files in results/; nothing on it is typed by hand.` +
    (DATA.skipped.length ? ` Left out as stale (measured other code): ${esc(DATA.skipped.join(', '))}.` : '');
}
function renderGlance() {
  const n = detN(), nt = state.n;
  // Deterministic, against every other sort, all inputs, all types: the core claim.
  const saveTypes = state.types, saveDs = state.datasets;
  state.types = DATA.types.map(t => t.name); state.datasets = 'all';
  const GL = LANG === 'cpp' ? ['traffic', 'compares', 'aux'] : ['compares', 'aux', 'flips'];   // the page's headline metrics, the first the hero
  const scores = {}; for (const k of GL) scores[k] = score((t, d) => detGetM(n, t, d, k));
  // Measured wall time per machine.
  const meas = [];
  for (const p of LANG === 'cpp' ? DATA.platforms : []) {
    const s1 = score((t, d) => timGet(p.id, nt, t, d, 'wall'));
    if (s1.cells.length) meas.push({ p, s: s1 });
  }
  state.types = saveTypes; state.datasets = saveDs;
  const cells = scores[GL[0]].cells.length;
  const tiles = [];
  const REF = LANG === 'cpp' ? 'std::stable_sort' : 'slice::sort';   // the standard library's stable sort of the page
  const rnd = detRow(n, 'int32', 'random', DATA.candidate), srt = detRow(n, 'int32', 'random', REF);
  const TILE = {
    traffic: ['Bytes moved', 'cells in which brainsort moves the fewest bytes of every sort', ''],
    compares: ['Comparisons', 'cells with the fewest comparisons of every sort', LANG === 'rust' ? '. brainsort’s scout pass makes one per element; the others compare to sort' : ''],
    aux: ['Scratch memory', 'cells with the least scratch memory of every sort', (rnd && srt ? `; on random int32 brainsort asks for ${bytes(DET.get(rnd, 'aux'))} and ${REF} for ${bytes(DET.get(srt, 'aux'))}` : '') + (LANG === 'cpp' ? '. boost::flat_stable_sort gets by on n/256 + 8 KiB' : '')],
    flips: ['Compare flips', 'cells with the fewest changes of comparison outcome of every sort, a machine-independent proxy for mispredicted branches', ''],
  };
  GL.forEach((k, i) => {
    const sc = scores[k], [label, what, tail] = TILE[k];
    tiles.push(`<div class="tile ${i === 0 ? 'hero ' : ''}${sc.wins.length * 2 >= cells ? 'win' : 'loss'}"><div class="k">${label} <span class="tag det">deterministic</span></div><div class="v">${sc.wins.length} <small>of ${cells}</small></div><div class="s">${what}${tail}</div></div>`);
  });
  if (meas.length) {
    const ws = meas.map(x => x.s.wins.length), tot = meas[0].s.cells.length;
    const rng = a => Math.min(...a) === Math.max(...a) ? String(a[0]) : `${Math.min(...a)} to ${Math.max(...a)}`;
    tiles.push(`<div class="tile"><div class="k">Wall time <span class="tag">measured</span></div><div class="v">${esc(rng(ws))} <small>of ${tot}</small></div><div class="s">cells in which brainsort is the fastest sort at n = ${fmtN(nt)}, across ${meas.length} machine${meas.length === 1 ? '' : 's'}. Per machine below.</div></div>`);
  }
  // The public call on a plain vector, per machine.
  const bn = BENCH.sizes.includes(nt) ? nt : benchDefaultN(), bs = BENCH.platforms.map(p => benchScore(p.id, bn)).filter(x => x.cells);
  if (bs.length) {
    const rng = a => Math.min(...a) === Math.max(...a) ? String(a[0]) : `${Math.min(...a)} to ${Math.max(...a)}`;
    tiles.push(`<div class="tile"><div class="k">${LANG === 'cpp' ? 'On plain vectors' : 'The crate on plain vectors'} <span class="tag">measured</span></div><div class="v">${esc(rng(bs.map(x => x.wins)))} <small>of ${bs[0].cells}</small></div><div class="s">cells in which <code>brainsort::sort</code> on a plain ${LANG === 'cpp' ? '<code>std::vector</code>' : '<code>Vec</code>'} is the fastest of every ${LANG_NAME[LANG]} sort measured (${esc(BENCH.opponents.join(', '))}) at n = ${fmtN(bn)}, across ${bs.length} machine${bs.length === 1 ? '' : 's'}; ${esc(rng(bs.map(x => x.winsSt)))} of ${bs[0].cells} against <code>${esc(BENCH.opponents[BENCH.std])}</code> alone, the standard library's. <a href="#api">Per machine below.</a></div></div>`);
  }
  document.getElementById('glance-tiles').innerHTML = tiles.join('');
  const r1 = detRow(n, 'int32', 'random', DATA.candidate), rp = detRow(n, 'int32', 'random', REF);
  document.getElementById('glance-text').innerHTML = LANG === 'cpp' ?
    `brainsort is a <b>stable</b> sort that avoids comparing keys wherever the key type allows it, and does less work when the input already has structure. ` +
    `On random keys it moves a fraction of the bytes a merge sort moves${r1 && rp ? ` (${bytes(DET.get(r1, 'traffic'))} against ${bytes(DET.get(rp, 'traffic'))} for std::stable_sort on ${fmtN(n)} random int32 keys)` : ''} and makes almost no comparisons. ` +
    `It pays for that with scratch memory of about half the array, and it does not win everywhere: on input that is already sorted or has only a handful of distinct values, the adaptive comparison sorts finish in one pass and brainsort’s scout pass costs a little extra. Every such case is on this page. Every other sort here is stable, upstream code as shipped, and takes every key type and any comparator: the sorts that can do what brainsort does.` :
    `The crate is a <b>stable</b> sort that avoids comparing keys wherever the key type allows it, and does less work when the input already has structure. ` +
    `On random keys it makes almost no comparisons${r1 && rp ? ` (${compact(DET.get(r1, 'compares'))} against ${compact(DET.get(rp, 'compares'))} for slice::sort on ${fmtN(n)} random int32 keys)` : ''} and wins on input with structure. ` +
    `It pays with scratch memory of about half the array. Bytes moved, its strongest number, cannot be counted of the Rust sorts as shipped, so that comparison is on <a href="index.html">the C++ page</a>, where the crate’s own counts are exactly the C++ library’s. Every case where brainsort loses is on this page. Every other sort here is stable, published code as shipped, and takes every key type and any comparator: the sorts that can do what brainsort does.`;
  document.getElementById('glance-note').textContent = `The tiles count every key type and every input pattern: ${cells} cells at n = ${fmtN(n)}${n !== nt ? ` for the deterministic numbers (no counted run at ${fmtN(nt)} in this build of the page)` : ''}. Change the size in the controls below to see another. A cell counts as a win when brainsort is best or tied.`;
}

// ---- the deterministic section --------------------------------------------------------
function renderDet() {
  const m = detMetric(), n = detN();
  document.getElementById('det-fallback').innerHTML = n === state.n ? '' :
    `<div class="callout warn">No counted run at ${fmtN(state.n)} elements is in this build of the page (scripts/counts.sh with COUNTS_SIZES=${state.n} adds one; at ten million that is over an hour of counting). The deterministic numbers below are at ${fmtN(n)}; the measured section has the ${fmtN(state.n)} timing.</div>`;
  document.getElementById('det-metric-note').textContent = m.plain + ' Lower is better.';
  // A cost only one side pays: no win/loss score, no heat map, no charts; the full tables still list it.
  document.getElementById('det-contest').hidden = !!m.noContest;
  const nc = document.getElementById('det-nocontest'); nc.hidden = !m.noContest; nc.textContent = m.noContest || '';
  const wrap = document.getElementById('det-tables-wrap');
  if (m.noContest) {
    if (wrap.open) matrixTables(document.getElementById('det-tables'), m, (t, d) => detGet(n, t, d), document.getElementById('det-tables-sum'));
    else document.getElementById('det-tables-sum').textContent = ` ${m.label.toLowerCase()}, n = ${fmtN(n)}`;
    return;
  }
  const sc = score((t, d) => detGet(n, t, d));
  tiles(document.getElementById('det-tiles'), m, sc, '', n);
  document.getElementById('det-heat-note').textContent = `${m.label} of the best other sort divided by brainsort’s, at n = ${fmtN(n)}. Hover a cell for the numbers.`;
  heat(document.getElementById('det-heat'), m, (t, d) => detGet(n, t, d), 'det', n);
  document.getElementById('det-sizes-note').textContent = `${m.label} divided by the number of elements, as the input grows. A flat line is linear work; a rising line is the n log n of a comparison sort or a cache effect. Every size has its own instrumented run.`;
  sizesCharts(document.getElementById('det-sizes'), m, (nn, t, a) => { const r = detRow(nn, t, state.detDs, a); return r ? DET.get(r, state.det) : null; }, `${state.detDs} input`);
  const dd = DATA.datasets.find(d => d.name === state.detBarD);
  barChart(document.getElementById('det-bar'), m, detGet(n, state.detBarT, state.detBarD), `${state.detBarT} · ${state.detBarD} · n = ${fmtN(n)}`, (dd ? dd.description + '. ' : '') + m.label + ' of every sort shown.');
  if (wrap.open) matrixTables(document.getElementById('det-tables'), m, (t, d) => detGet(n, t, d), document.getElementById('det-tables-sum'));
  else document.getElementById('det-tables-sum').textContent = ` ${m.label.toLowerCase()}, n = ${fmtN(n)}`;
}

// ---- the measured section ----------------------------------------------------------------
function renderMeas() {
  const sec = document.getElementById('timing');
  const quiet = DATA.platforms.filter(p => !p.shared).length, shared = DATA.platforms.length - quiet;
  document.getElementById('timing-sum').textContent = DATA.platforms.length ? `${DATA.platforms.length} machine${DATA.platforms.length === 1 ? '' : 's'}: ${DATA.platforms.map(p => p.label).join(', ')}` : 'no timing files';
  if (!DATA.platforms.length) { document.getElementById('timing-intro').textContent = 'No timing file is present. Run scripts/bench.sh or scripts/bench.ps1.'; return; }
  document.getElementById('timing-intro').innerHTML =
    `Time is what a user feels, and it depends on the machine. Each machine below was measured on its own; pick one. ` +
    (quiet ? `Machines marked ★ are quiet developer workstations with hardware performance counters. ` : '') +
    (shared ? `The others are shared GitHub Actions runners: other tenants, no fixed CPU, no performance counters, so their times are noisier and only the ranking and the rough ratios should be read from them. ` : '') +
    `The deterministic numbers above do not change from machine to machine; these do.`;
  const p = PLAT[state.p]; if (!p) return;
  const m = measMetric() || MEAS_METRICS[0]; if (!MEAS_METRICS.some(mm => mm.key === state.meas && TIM.rows.some(r => r[0] === p.id && TIM.get(r, mm.key) != null))) state.meas = 'wall';
  const mm = measMetric();
  document.getElementById('machine').innerHTML =
    `<b>${esc(p.label)}</b>${p.shared ? '' : ' ★'} · ${esc(p.cpu || 'unknown CPU')} · ${esc(p.os)} ${esc(p.arch)} · ${esc(p.compiler)}${p.host ? ' · ' + esc(p.host) : ''}<br>` +
    `<span class="muted">counters: ${esc(p.backend)} · ${esc(String(p.reps))} repetitions at 100 k (scaled elsewhere), best of ${esc(String(p.rounds))} round${p.rounds == 1 ? '' : 's'}${p.pinned !== '' && p.pinned != null && p.pinned >= 0 ? ', pinned to CPU ' + esc(String(p.pinned)) : ', not pinned'} · sizes ${p.sizes.map(fmtN).join(', ')} · measured ${esc(timeStamp(p.measured))}${p.commit ? ' at commit ' + esc(p.commit) : ''}${p.stale === 'stale' ? ' <span class="tag bad">stale: measured other code</span>' : p.stale === 'unknown' ? ' <span class="tag warn">code not verifiable</span>' : ''}</span>`;
  document.getElementById('meas-metric-note').textContent = mm.plain + ' Lower is better.';
  const n = state.n;
  const sc = score((t, d) => timGet(p.id, n, t, d, mm.key));
  tiles(document.getElementById('meas-tiles'), mm, sc, p.label, n);
  document.getElementById('meas-heat-note').textContent = `${mm.label} of the best other sort divided by brainsort’s on ${p.label}, n = ${fmtN(n)}. Hover a cell for the numbers.`;
  heat(document.getElementById('meas-heat'), mm, (t, d) => timGet(p.id, n, t, d, mm.key), 'meas', n);
  document.getElementById('meas-sizes-note').textContent = `${mm.label} per element as the input grows, on ${p.label}. Small inputs sit in the cache and cost few nanoseconds per element; large ones pay for memory bandwidth.`;
  sizesCharts(document.getElementById('meas-sizes'), mm, (nn, t, a) => timGet(p.id, nn, t, state.measDs, mm.key)(a), `${state.measDs} input, ${p.label}`);
  const dd = DATA.datasets.find(d => d.name === state.measBarD);
  barChart(document.getElementById('meas-bar'), mm, timGet(p.id, n, state.measBarT, state.measBarD, mm.key), `${state.measBarT} · ${state.measBarD} · n = ${fmtN(n)} · ${p.label}`, (dd ? dd.description + '. ' : '') + mm.label + ' of every sort shown.');
  const wrap = document.getElementById('meas-tables').parentElement;
  if (wrap.open) matrixTables(document.getElementById('meas-tables'), mm, (t, d) => timGet(p.id, n, t, d, mm.key), document.getElementById('meas-tables-sum'));
  else document.getElementById('meas-tables-sum').textContent = ` ${mm.label.toLowerCase()}, n = ${fmtN(n)}, ${p.label}`;
  // Every machine: wins of wall time, and the random-input speed-up per key type.
  document.getElementById('allplat-note').textContent = `Wall time at n = ${fmtN(n)} on every machine: how many cells brainsort wins against every other sort, and by how much it beats (or trails) the best other sort on random input, per key type. Quiet workstations are marked ★.`;
  let h = '<thead><tr><th>machine</th><th>CPU</th><th>fastest or tied</th>' + selTypes().map(t => `<th>random ${esc(t.name)}</th>`).join('') + '</tr></thead><tbody>';
  for (const q of DATA.platforms) {
    const s = score((t, d) => timGet(q.id, n, t, d, 'wall'));
    if (!s.cells.length) continue;
    h += `<tr><td>${esc(q.label)}${q.shared ? '' : ' ★'}${q.stale === 'stale' ? ' <span class="tag bad">stale</span>' : ''}</td><td style="text-align:left">${esc(q.cpu || 'unknown')}</td><td>${s.wins.length} of ${s.cells.length}</td>` +
      selTypes().map(t => { const c = compareCell(timGet(q.id, n, t.name, 'random', 'wall')); return c ? `<td class="${c.win ? 'best' : ''}">${c.win ? c.ratio.toFixed(2) + 'x faster' : (1 / c.ratio).toFixed(2) + 'x slower'}<span class="rel">${esc(c.opp)}</span></td>` : '<td class="muted">n/a</td>'; }).join('') + '</tr>';
  }
  document.getElementById('allplat').innerHTML = h + '</tbody>';
}

// ---- the public call on a plain vector: this page's language, per machine --------------------
// One cell: brainsort against the best other sort measured; `st` is the standard library's.
function benchCmp(r) {
  if (!r) return null;
  const b = BT.get(r, 'bs'), others = BT.get(r, 'others');
  const os = others.map((v, i) => [BENCH.opponents[i], v]).filter(x => x[1] != null).sort((x, y) => x[1] - y[1]);
  if (!os.length) return null;
  return { b, o: os[0][1], opp: os[0][0], win: b <= os[0][1], ratio: os[0][1] / b, st: others[BENCH.std] };
}
function benchScore(pid, n) {
  let cells = 0, wins = 0, winsSt = 0;
  for (const t of BENCH.types) for (const d of BENCH.datasets) { const c = benchCmp(benchRow(pid, t.name, n, d)); if (!c) continue; cells++; if (c.win) wins++; if (c.st != null && c.b <= c.st) winsSt++; }
  return { cells, wins, winsSt };
}
const bestOther = (T, B, r) => B.opponents.map((o, i) => [o, T.get(r, 'others')[i]]).filter(x => x[1] != null).sort((x, y) => x[1] - y[1])[0];
function renderBench() {
  const B = BENCH, vec = LANG === 'cpp' ? 'std::vector' : 'Vec';
  document.getElementById('api-sum').textContent = B.platforms.length ? `brainsort::sort on a ${vec} against ${B.opponents.join(', ')}; ${B.platforms.length} machine${B.platforms.length === 1 ? '' : 's'}` : `no ${LANG_NAME[LANG]} benchmark files`;
  if (!B.platforms.length) { document.getElementById('api-intro').textContent = `No ${LANG_NAME[LANG]} benchmark file is present.`; return; }
  const p = BPLAT[state.benchP]; if (!p) return;
  const n = state.benchN;
  document.getElementById('api-intro').innerHTML = LANG === 'cpp' ?
    `What a user of the library gets: <code>brainsort::sort(v)</code> on a plain <code>std::vector&lt;T&gt;</code> against ${B.opponents.map(o => '<code>' + esc(o) + '</code>').join(', ')} on the same vector, wall time, every result checked. This includes everything the API adds on top of the algorithm: building the key records, permuting the elements afterwards, and allocating and freeing memory on every call (the harness above keeps its memory warm across repetitions; a single call pays page faults). Sizes: ${B.sizes.map(fmtN).join(', ')} elements. Every opponent is stable upstream code called with the same comparator; brainsort's comparator overload is the library's own comparison sort, so the comparator rows show what the library does when it cannot use keys.` :
    `What a user of the crate gets: <code>brainsort::sort(&amp;mut v)</code> on a plain <code>Vec&lt;T&gt;</code> against the stable Rust sorts as shipped: <code>slice::sort</code> (the standard library's stable sort, driftsort), <code>slice::sort_by_cached_key</code> (the same sort on a buffer of keys and indices, the standard route for a sort by key; not on strings, whose key would be the string itself) and the <code>glidesort</code> crate. Wall time, every result checked, the same generated inputs and sizes as the C++ benchmark. This includes everything the API adds on top of the algorithm: building the key records, permuting the elements afterwards, and allocating and freeing memory on every call. Sizes: ${B.sizes.map(fmtN).join(', ')} elements. brainsort's <code>sort_by</code> is the crate's own comparison sort, so the comparator rows show what the crate does when it cannot use keys. This is the one timed measurement of the crate: the algorithm-level harness with hardware counters is C++ code and is on <a href="index.html">the C++ page</a>; its deterministic counts above hold for the crate.`;
  document.getElementById('api-machine').innerHTML = `<b>${esc(p.label)}</b>${p.shared ? '' : ' ★'} · ${esc(p.cpu || '')}${p.compiler ? ' · ' + esc(p.compiler) : ''} · median of ${esc(String(p.reps || '?'))} runs · measured ${esc(timeStamp(p.measured))}${p.stale === 'stale' ? ' <span class="tag bad">stale</span>' : ''}`;
  const cs = []; for (const t of B.types) for (const d of B.datasets) { const c = benchCmp(benchRow(p.id, t.name, n, d)); if (c) cs.push({ t: t.name, d, c }); }
  const wins = cs.filter(x => x.c.win), winsSt = cs.filter(x => x.c.st != null && x.c.b <= x.c.st);
  const best = wins.filter(x => x.c.ratio > 1.05).sort((a, b) => b.c.ratio - a.c.ratio)[0], worst = cs.filter(x => !x.c.win).sort((a, b) => a.c.ratio - b.c.ratio)[0];
  document.getElementById('api-tiles').innerHTML = cs.length ?
    `<div class="tile ${wins.length * 2 >= cs.length ? 'win' : 'loss'}"><div class="k">Fastest or tied · n = ${fmtN(n)}</div><div class="v">${wins.length} <small>of ${cs.length}</small></div><div class="s">cells where brainsort::sort beats or ties every other ${LANG_NAME[LANG]} sort measured; ${winsSt.length} of ${cs.length} against ${esc(B.opponents[B.std])} alone, the standard library's</div></div>` +
    `<div class="tile"><div class="k">Biggest win</div><div class="v">${best ? best.c.ratio.toFixed(1) + 'x faster' : 'none'}</div><div class="s">${best ? `than ${esc(best.c.opp)} on ${esc(best.t)} ${esc(best.d)} (${fmtMs(best.c.b)} vs ${fmtMs(best.c.o)} ms)` : 'brainsort::sort is never ahead here'}</div></div>` +
    `<div class="tile"><div class="k">Worst loss</div><div class="v">${worst ? (1 / worst.c.ratio).toFixed(2) + 'x slower' : 'none'}</div><div class="s">${worst ? `than ${esc(worst.c.opp)} on ${esc(worst.t)} ${esc(worst.d)} (${fmtMs(worst.c.b)} vs ${fmtMs(worst.c.o)} ms)` : 'fastest or tied in every cell'}</div></div>` :
    '<div class="tile empty">nothing measured at this size on this machine</div>';
  document.getElementById('api-heat-note').textContent = `Wall time at n = ${fmtN(n)} on ${p.label}: the fastest other ${LANG_NAME[LANG]} sort divided by brainsort::sort. Hover a cell for the numbers.`;
  let h = '<thead><tr><th>input</th>' + B.types.map(t => `<th title="${esc(t.desc)}">${esc(t.name)}</th>`).join('') + '</tr></thead><tbody>';
  for (const d of B.datasets) {
    h += `<tr><td>${esc(d)}</td>` + B.types.map(t => { const c = benchCmp(benchRow(p.id, t.name, n, d)); return c ? `<td class="cell ${heatClass(c)}" data-tip="api" data-t="${esc(t.name)}" data-d="${esc(d)}"><div>${c.ratio.toFixed(2)}x</div></td>` : '<td class="cell"><div class="muted">–</div></td>'; }).join('') + '</tr>';
  }
  document.getElementById('api-heat').innerHTML = h + '</tbody>';
  let t2 = '';
  for (const t of B.types) {
    let rows = '';
    for (const d of B.datasets) {
      const r = benchRow(p.id, t.name, n, d); if (!r) continue;
      const vals = [BT.get(r, 'bs'), ...BT.get(r, 'others')], best = Math.min(...vals.filter(v => v != null));
      rows += `<tr><td>${esc(d)}</td>` + vals.map(v => v == null ? '<td class="muted">n/a</td>' : `<td class="${v === best ? 'best' : ''}">${fmtMs(v)}<span class="rel">${v === best ? 'best' : (v / best).toFixed(2) + 'x'}</span></td>`).join('') + '</tr>';
    }
    if (rows) t2 += `<h4>${esc(t.name)} · ${esc(t.desc)} · ms</h4><div class="tablewrap"><table class="data"><thead><tr><th>input</th><th><span class="sw" style="background:${gcolor('candidate')}"></span>brainsort::sort</th>${B.opponents.map(o => `<th><span class="sw" style="background:${gcolor('upstream')}"></span>${esc(o)}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table></div>`;
  }
  document.getElementById('api-tables').innerHTML = t2;
  // Every machine: wins, and the random-input speed-up per type.
  document.getElementById('api-allplat-note').textContent = `Wall time at n = ${fmtN(n)} on every machine: how many cells brainsort::sort wins against every other ${LANG_NAME[LANG]} sort measured, and by how much it beats (or trails) the best other sort on random input, per type. Quiet workstations are marked ★.`;
  let ap = '<thead><tr><th>machine</th><th>CPU</th><th>fastest or tied</th>' + B.types.map(t => `<th>random ${esc(t.name)}</th>`).join('') + '</tr></thead><tbody>';
  for (const q of B.platforms) {
    const sc = benchScore(q.id, n); if (!sc.cells) continue;
    ap += `<tr><td>${esc(q.label)}${q.shared ? '' : ' ★'}${q.stale === 'stale' ? ' <span class="tag bad">stale</span>' : ''}</td><td style="text-align:left">${esc(q.cpu || 'unknown')}</td><td>${sc.wins} of ${sc.cells}</td>` +
      B.types.map(t => { const c = benchCmp(benchRow(q.id, t.name, n, 'random')); return c ? `<td class="${c.win ? 'best' : ''}">${c.win ? c.ratio.toFixed(2) + 'x faster' : (1 / c.ratio).toFixed(2) + 'x slower'}<span class="rel">${esc(c.opp)}</span></td>` : '<td class="muted">n/a</td>'; }).join('') + '</tr>';
  }
  document.getElementById('api-allplat').innerHTML = ap + '</tbody>';
  // C++ and Rust, same machine id, same size, same type and input. The same columns on both pages.
  const op = OPLAT[p.id];
  let pr = '', cells = 0;
  if (op) {
    for (const t of B.types) {
      const ot = PAIR[t.name]; if (!ot) continue;
      let rows = '';
      for (const d of B.datasets) {
        const br = benchRow(p.id, t.name, n, d), or = otherRow(p.id, ot, n, d); if (!br || !or) continue;
        const [cr, rr, cT, rT, cB, rB] = LANG === 'cpp' ? [br, or, BT, OT, BENCH, OTHER] : [or, br, OT, BT, OTHER, BENCH];   // C++ first
        const cb = cT.get(cr, 'bs'), rb = rT.get(rr, 'bs'), ratio = rb / cb;
        const co = bestOther(cT, cB, cr), ro = bestOther(rT, rB, rr);
        const cls = Math.abs(Math.log2(ratio)) < 0.07 ? 'n0' : ratio < 1 ? (ratio < 0.5 ? 'w3' : ratio < 0.8 ? 'w2' : 'w1') : (ratio > 2 ? 'l3' : ratio > 1.25 ? 'l2' : 'l1');
        rows += `<tr><td>${esc(d)}</td><td>${fmtMs(cb)}</td><td>${fmtMs(rb)}</td><td class="cell ${cls}"><div>${ratio.toFixed(2)}x</div></td><td>${co ? fmtMs(co[1]) + '<span class="rel">' + esc(co[0]) + '</span>' : 'n/a'}</td><td>${ro ? fmtMs(ro[1]) + '<span class="rel">' + esc(ro[0]) + '</span>' : 'n/a'}</td></tr>`;
        cells++;
      }
      if (rows) pr += `<tbody><tr><th colspan="6" style="text-align:left">${esc(LANG === 'cpp' ? t.name : ot)} · ${esc(LANG === 'cpp' ? ot : t.name)}</th></tr>${rows}</tbody>`;
    }
  }
  document.getElementById('pair-note').textContent = cells ?
    `${cells} cells at n = ${fmtN(n)} where both languages were measured on ${p.label}: the same generators (integer and string inputs identical, doubles from the same distribution), the same size, wall time in ms. The algorithm is the same step for step; the difference is the compiler, the standard library and the allocator. Blue: the Rust build is faster; red: the C++ build. The last two columns show what each language's ecosystem offers in that cell, which is a different pool on each side.` :
    (op ? `Nothing measured in both languages at n = ${fmtN(n)} on this machine.` : `No ${LANG_NAME[OTHER_LANG]} benchmark of the machine ${p.label} is on this page, so there is nothing to pair these numbers with.`);
  document.getElementById('pair').innerHTML = cells ? `<thead><tr><th>input</th><th>C++ brainsort</th><th>Rust brainsort</th><th>Rust / C++</th><th>best other C++</th><th>best other Rust</th></tr></thead>${pr}` : '';
}

// ---- method tables and glossary ------------------------------------------------------------
function renderMethod() {
  document.getElementById('glossary').innerHTML =
    DET_METRICS.map(m => `<dt>${esc(m.label)} <span class="tag det">deterministic</span></dt><dd>${esc(m.plain)}</dd>`).join('') +
    MEAS_METRICS.map(m => `<dt>${esc(m.label)} <span class="tag">measured</span></dt><dd>${esc(m.plain)}</dd>`).join('');
  document.getElementById('algos').innerHTML = '<thead><tr><th>algorithm</th><th>what it is</th><th>where the design ships</th><th>stable</th><th>extra memory</th></tr></thead><tbody>' +
    DATA.algos.map(a => `<tr><td><span class="sw" style="background:${color(a.name)}"></span><b>${esc(a.name)}</b></td><td>${esc(a.what)}</td><td>${esc(a.where)}</td><td>${a.stable ? 'yes' : 'no'}</td><td>${esc(a.memory)}</td></tr>`).join('') + '</tbody>';
  document.getElementById('types').innerHTML = '<thead><tr><th>key type</th><th>element</th><th>real-world case</th></tr></thead><tbody>' +
    DATA.types.map(t => `<tr><td><b>${esc(t.name)}</b></td><td>${esc(t.desc)}</td><td>${esc(t.real)}</td></tr>`).join('') + '</tbody>';
  document.getElementById('datasets').innerHTML = '<thead><tr><th>input</th><th>what it looks like</th><th>set</th></tr></thead><tbody>' +
    DATA.datasets.map(d => `<tr><td><b>${esc(d.name)}</b></td><td>${esc(d.description)}</td><td>${d.standard ? 'standard' : 'additional'}</td></tr>`).join('') + '</tbody>';
}

// ---- all numbers ------------------------------------------------------------------------------
function allTable(t, d) {
  const n = state.n, algos = shownAlgos(), plats = DATA.platforms.filter(p => TIM.rows.some(r => r[0] === p.id && r[1] === n));   // the harness: empty on the Rust page
  const measCols = MEAS_METRICS.filter(m => plats.some(p => algos.some(a => { const r = timRow(p.id, n, t, d, a.name); return r && TIM.get(r, m.key) != null; })));
  let h = '';
  for (const g of stabilityGroups()) {
    let rows = '';
    const detBest = Object.fromEntries(DET_METRICS.map(m => [m.key, Math.min(...g.algos.map(a => { const r = detRow(n, t, d, a.name); return r ? DET.get(r, m.key) : Infinity; }))]));
    for (const a of g.algos) {
      const r = detRow(n, t, d, a.name); if (!r) continue;
      rows += `<tr><td><span class="sw" style="background:${color(a.name)}"></span>${esc(a.name)}</td><td class="muted">deterministic</td>` +
        DET_METRICS.map(m => { const v = DET.get(r, m.key); return `<td class="${v === detBest[m.key] ? 'best' : ''}">${esc(m.fmt(v))}</td>`; }).join('') + measCols.map(() => '<td></td>').join('') +
        `<td style="text-align:left" class="muted">${esc(DET.get(r, 'route') || '')}${DET.get(r, 'passes') ? ', ' + DET.get(r, 'passes') + ' radix passes' : ''}</td></tr>`;
    }
    for (const p of plats) {
      const best = Object.fromEntries(measCols.map(m => [m.key, Math.min(...g.algos.map(a => { const r = timRow(p.id, n, t, d, a.name); const v = r && TIM.get(r, 'ok') ? TIM.get(r, m.key) : null; return v == null ? Infinity : v; }))]));
      for (const a of g.algos) {
        const r = timRow(p.id, n, t, d, a.name); if (!r) continue;
        rows += `<tr><td><span class="sw" style="background:${color(a.name)}"></span>${esc(a.name)}</td><td class="muted">${esc(p.label)}</td>` + DET_METRICS.map(() => '<td></td>').join('') +
          (TIM.get(r, 'ok') ? measCols.map(m => { const v = TIM.get(r, m.key); return `<td class="${v != null && v === best[m.key] ? 'best' : ''}">${v == null ? '<span class="muted">n/a</span>' : esc(m.fmt(v))}</td>`; }).join('') + `<td style="text-align:left" class="muted">${TIM.get(r, 'reps')} reps${TIM.get(r, 'batch') > 1 ? ', ' + TIM.get(r, 'batch') + ' copies per timing' : ''}, min ${esc(ns(TIM.get(r, 'wallmin')))}</td>`
            : `<td colspan="${measCols.length + 1}" class="fail">failed: ${esc(TIM.get(r, 'error'))}</td>`) + '</tr>';
      }
    }
    if (rows) h += `<h4>${esc(g.label)}</h4><div class="tablewrap"><table class="data"><thead><tr><th>algorithm</th><th>source</th>` +
      DET_METRICS.map(m => `<th title="${esc(m.plain)}">${esc(m.label)}</th>`).join('') + measCols.map(m => `<th title="${esc(m.plain)}">${esc(m.label)}</th>`).join('') + '<th>notes</th></tr></thead><tbody>' + rows + '</tbody></table></div>';
  }
  h += apiTable(t, d);
  return h || '<p class="empty">nothing here</p>';
}
// The public call on a plain vector for one harness cell, this page's language, per machine, wall ms.
// The harness key types map to the vector element types.
const API_TYPE = LANG === 'cpp' ? { int32: 'int32_t', int64: 'int64_t', double: 'double', string: 'std::string' } : { int32: 'i32', int64: 'i64', double: 'f64', string: 'String' };
function apiTable(t, d) {
  const n = state.n, bt = API_TYPE[t]; if (!bt) return '';
  let rows = '';
  for (const p of BENCH.platforms) {
    const r = benchRow(p.id, bt, n, d); if (!r) continue;
    const vals = [BT.get(r, 'bs'), ...BT.get(r, 'others')], best = Math.min(...vals.filter(v => v != null));
    rows += `<tr><td class="muted">${esc(p.label)}</td>` + vals.map(v => `<td class="${v === best ? 'best' : ''}">${v == null ? '<span class="muted">n/a</span>' : fmtMs(v)}</td>`).join('') + '</tr>';
  }
  if (!rows) return '';
  return `<h4>${esc(LANG === 'cpp' ? 'the library' : 'the crate')} on plain vectors · ${esc(bt)} · wall ms</h4><div class="tablewrap"><table class="data"><thead><tr><th>machine</th><th>brainsort::sort</th>${BENCH.opponents.map(o => `<th>${esc(o)}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table></div>`;
}

function renderAll() {
  const host = document.getElementById('all-tables');
  const open = new Set([...host.querySelectorAll('details[open]')].map(x => x.dataset.key));
  host.innerHTML = '';
  for (const t of selTypes()) for (const d of selDatasets()) {
    if (!detRow(state.n, t.name, d.name, DATA.candidate) && !DATA.platforms.some(p => timRow(p.id, state.n, t.name, d.name, DATA.candidate))) continue;
    const key = t.name + '/' + d.name;
    const det = document.createElement('details'); det.dataset.key = key;
    det.innerHTML = `<summary>${esc(t.name)} · ${esc(d.name)} · n = ${fmtN(state.n)}<span class="muted">${esc(d.description)}</span><span class="hint"></span></summary><div class="body"></div>`;
    det.addEventListener('toggle', () => { det.querySelector('.body').innerHTML = det.open ? allTable(t.name, d.name) : ''; });
    host.appendChild(det);
    if (open.has(key)) det.open = true;
  }
}

// ---- tooltip -------------------------------------------------------------------------------
const tip = document.getElementById('tip');
function showTip(html) { tip.innerHTML = html; tip.hidden = false; }
function oppRows(get, m) {
  return shownAlgos().map(a => ({ a: a.name, v: get(a.name) })).filter(x => x.v != null).sort((x, y) => x.v - y.v)
    .map(x => `<tr><td><span class="sw" style="background:${color(x.a)}"></span>${esc(x.a)}</td><td>${esc(m.fmt(x.v))}</td></tr>`).join('');
}
document.addEventListener('mouseover', e => {
  const g = e.target.closest && e.target.closest('[data-tip]'); if (!g) return;
  const k = g.dataset.tip;
  if (k === 'det') { const m = detMetric(), get = detGet(state.n, g.dataset.t, g.dataset.d), c = compareCell(get); showTip(`<b>${esc(g.dataset.t)} · ${esc(g.dataset.d)} · n = ${fmtN(state.n)} · ${esc(m.label.toLowerCase())}</b><table>${oppRows(get, m)}${c ? `<tr><td>brainsort ${c.win ? 'ahead' : 'behind'} of the best other sort by</td><td>${c.ratio === Infinity || c.ratio === 0 ? 'all' : (c.win ? c.ratio : 1 / c.ratio).toFixed(2) + 'x'}</td></tr>` : ''}</table>`); }
  else if (k === 'meas') { const m = measMetric(), get = timGet(state.p, state.n, g.dataset.t, g.dataset.d, m.key), c = compareCell(get); showTip(`<b>${esc(g.dataset.t)} · ${esc(g.dataset.d)} · n = ${fmtN(state.n)} · ${esc(m.label.toLowerCase())} on ${esc(PLAT[state.p].label)}</b><table>${oppRows(get, m)}${c ? `<tr><td>brainsort ${c.win ? 'ahead' : 'behind'} by</td><td>${(c.win ? c.ratio : 1 / c.ratio).toFixed(2) + 'x'}</td></tr>` : ''}</table>`); }
  else if (k === 'api') { const r = benchRow(state.benchP, g.dataset.t, state.benchN, g.dataset.d); if (r) showTip(`<b>${esc(g.dataset.t)} · ${esc(g.dataset.d)} · n = ${fmtN(state.benchN)}</b><table><tr><td>brainsort::sort</td><td>${fmtMs(BT.get(r, 'bs'))} ms</td></tr>${BT.get(r, 'others').map((v, i) => `<tr><td>${esc(BENCH.opponents[i])}</td><td>${v == null ? 'n/a' : fmtMs(v) + ' ms'}</td></tr>`).join('')}</table>`); }
  else if (k === 'pt') { const m = g.closest('#det-sizes') ? detMetric() : measMetric(); showTip(`<b>${esc(g.dataset.a)} · ${esc(g.dataset.t)} · n = ${fmtN(+g.dataset.n)}</b><table><tr><td>${esc(m.label)} per element</td><td>${esc(perElem(+g.dataset.v, m.per))}</td></tr><tr><td>total</td><td>${esc(m.fmt(+g.dataset.v * +g.dataset.n))}</td></tr></table>`); }
});
document.addEventListener('mousemove', e => { if (tip.hidden) return; const pad = 14; let x = e.clientX + pad, y = e.clientY + pad; if (x + tip.offsetWidth > innerWidth - 8) x = e.clientX - tip.offsetWidth - pad; if (y + tip.offsetHeight > innerHeight - 8) y = e.clientY - tip.offsetHeight - pad; tip.style.left = x + 'px'; tip.style.top = y + 'px'; });
document.addEventListener('mouseout', e => { if (e.target.closest && e.target.closest('[data-tip]') && !(e.relatedTarget && e.relatedTarget.closest && e.relatedTarget.closest('[data-tip]'))) tip.hidden = true; });

// ---- go ----------------------------------------------------------------------------------------
function render() { writeHash(); renderGlance(); renderDet(); if (LANG === 'cpp') renderMeas(); renderBench(); renderAll(); }
document.getElementById('det-tables-wrap').addEventListener('toggle', renderDet);
if (LANG === 'cpp') document.getElementById('meas-tables').parentElement.addEventListener('toggle', renderMeas);
for (const id of ['timing', 'api']) { const el = document.getElementById(id); if (el) el.addEventListener('toggle', writeHash); }
readHash(); renderHeader(); renderMethod(); refresh(); render();
window.addEventListener('hashchange', () => { readHash(); refresh(); render(); });

// ---- motion: reveal blocks as they scroll into view, once each; shadow the control panel while it is stuck ----
// The blocks are the static children of every section (heading, intro, tile grid, chart grid, ...), never the
// re-rendered content inside them, so a chip click does not replay anything. The class is added here, not in the
// markup, so the page is complete without JS.
if ('IntersectionObserver' in window && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
  const blocks = document.querySelectorAll('main > section > *, main > details.section > summary, main > details.section > .body > *, footer');
  const io = new IntersectionObserver(entries => {
    for (const e of entries) if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
  }, { rootMargin: '0px 0px -10% 0px', threshold: 0 });
  const vh = innerHeight;
  for (const b of blocks) {
    if (b.getBoundingClientRect().top < vh * .9) continue;   // already on screen at load: no fade-in
    b.classList.add('reveal'); io.observe(b);
  }
  const panel = document.getElementById('panel'), sentinel = document.createElement('div');
  panel.before(sentinel);
  new IntersectionObserver(([e]) => panel.classList.toggle('stuck', e.boundingClientRect.top < 0 && !e.isIntersecting), { threshold: 0 }).observe(sentinel);
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()

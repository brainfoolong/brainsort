# Setup: build, test, measure, publish

From a fresh clone to a verified build, a passing test suite, your own
benchmark results and the website built from them. For what the project is,
see the [README](README.md); for the results, the
[website](https://brainfoolong.github.io/brainsort/).

## 1. Prerequisites

Brainsort and the benchmark are plain C++20, header-only. The only third-party
code is the two upstream sorting implementations vendored under
[third_party/](third_party/README.md), which the benchmark runs as opponents.

| | Linux / WSL2 | Windows | macOS |
|---|---|---|---|
| compiler | GCC 13+ or Clang 16+ | MinGW-w64 GCC 13+ (MSYS2), or MSVC 2022 | Apple Clang 15+ |
| build | CMake 3.20+ and Ninja (optional: the script falls back to plain `g++`) | CMake 3.20+ and Ninja, or Visual Studio | CMake 3.20+ and Ninja |
| website | Python 3 | Python 3 | Python 3 |

**To use the library** you need none of this: one header
([single_include/brainsort.hpp](single_include/brainsort.hpp)) and any C++20
compiler.

### Ubuntu / Debian / WSL2

```sh
sudo apt update
sudo apt install g++-13 cmake ninja-build python3 git
```

If `g++ --version` reports something older than 13, set `export CXX=g++-13`.

### Windows (MSYS2)

1. Install [MSYS2](https://www.msys2.org/).
2. In the **MSYS2 UCRT64** shell: `pacman -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja`
3. Add `C:\msys64\ucrt64\bin` to `PATH` so PowerShell finds `g++`, `cmake` and `ninja`.

## 2. Build

```sh
sh scripts/build-linux.sh        # Linux, WSL, macOS, MSYS2 -> build-linux/
```

```powershell
.\scripts\build-windows.ps1      # Windows PowerShell      -> build-win\
```

Both honour `BUILD_DIR`. By hand:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

With MSVC: `cmake -S . -B build -A x64` (any installed Visual Studio 2022 or newer), then
`cmake --build build --config Release` and `ctest --test-dir build -C Release`.

Every build produces:

| program | what it is |
|---|---|
| `brainsort_tests` | the library test suite (assertions on) |
| `brainsort_tests_scalar` | the same suite with the vector paths compiled out (`BRAINSORT_NO_SIMD`), the code every non-x86 target runs |
| `brainsort_single_header` | compiles against `single_include/brainsort.hpp` |
| `brainsort_fuzz_smoke` | the fuzz target driven by random inputs |
| `brainsort_api_bench` | the public API against `std::sort`, `std::stable_sort` and pdqsort on plain vectors |
| `sortbench` | the benchmark driver |
| `sortbench_tests` | the benchmark's correctness suite and golden-file check (assertions on) |

| CMake option | default | effect |
|---|---|---|
| `BRAINSORT_BUILD_TESTS` | `ON` | the library tests, single-header check, fuzz smoke test and API benchmark |
| `BRAINSORT_BUILD_BENCHMARK` | `ON` | `sortbench` and `sortbench_tests` |
| `BRAINSORT_BUILD_FUZZERS` | `OFF` | `brainsort_fuzz`, a libFuzzer target (Clang only) |
| `BRAINSORT_SANITIZE` | empty | e.g. `address,undefined` or `thread` (GCC/Clang) |
| `SORTBENCH_NATIVE` | `OFF` | adds `-march=native`; keep it off for numbers comparable with the published ones |

After a change to `include/brainsort/`, regenerate the single header with
`python3 scripts/amalgamate.py` (`--check` is what CTest and CI run).

## 3. Test

```sh
ctest --test-dir build-linux --output-on-failure
```

This runs the library suite (about 54,000 checks: every key type on ten
input patterns and 25 sizes, containers and element kinds, the floating-point
total order, allocation failure at every allocation point, throwing
projections, concurrent sorts, random shapes, 10 million elements), the same
suite on the scalar code, the single-header build and its currency check,
3,000 fuzz inputs, and the benchmark suite. Each program also runs by hand,
e.g. `./build-linux/sortbench_tests`.

`sortbench_tests` covers every algorithm on every key type across the input
patterns, 42 sizes from 0 to 10,000 (including every internal threshold) and
3 seeds, both the counted and the uncounted code path, with the array view's
bounds assertions on. Stable algorithms must match `std::stable_sort`
exactly. The suite also checks that the verifier rejects wrong output.

**The golden file.** The last test recomputes every row of
[results/counts.csv](results/counts.csv): the deterministic numbers of every
(size, key type, input, algorithm) cell at 1,000, 10,000 and 100,000 elements
(reads, writes, comparisons, table and key traffic, cache-model misses,
scratch memory, telemetry, two fingerprints), and fails on any difference.
A change to brainsort's planner or to a port is caught even when the output
is still correct. When the change was intended, regenerate the file and
commit it with the change:

```sh
sh scripts/counts.sh                        # rewrites results/counts.csv, about two minutes
./build-linux/sortbench_tests --golden      # only the golden comparison
```

```powershell
.\scripts\counts.ps1
```

The file is the same on every machine and compiler. Only the libstdc++
`std::sort` and `std::stable_sort` rows may differ with another libstdc++
version; a difference there is a warning, not a failure.

## 4. Measure

### The timing of your machine

```sh
sh scripts/bench.sh              # Linux, WSL, macOS, MSYS2
```

```powershell
.\scripts\bench.ps1              # Windows
```

The script builds, then runs every algorithm on every key type and input at
1,000, 10,000, 100,000, 1,000,000 and 10,000,000 elements, then the public-API
benchmark, then builds the website. Expect about an hour on a fast
machine. It writes:

| file | content |
|---|---|
| `results/<id>.csv` | the timing, one row per cell: wall time, CPU time, instructions, cycles, peak memory growth, repetitions |
| `results/<id>.meta.json` | the run stamp: id, host, OS, CPU, compiler, when, the code fingerprint, sizes, repetitions, rounds, seed, pinned CPU |
| `results/<id>.api.md` | the public-API benchmark (`brainsort::sort` on plain vectors against `std::sort`, `std::stable_sort`, pdqsort at 100k, 1M, 10M elements), stamped the same way |
| `site/index.html` | the website, from everything in `results/` |

`<id>` defaults to `<os>-<arch>-<compiler>`, e.g. `linux-x86-64-gcc13`.
Environment variables: `BENCH_ID`, `BENCH_HOST` (a description of the
machine that appears on the page), `BENCH_SIZES`, `BENCH_API_MAX_N`,
`BUILD_DIR`, `BENCH_NO_SITE`. Extra arguments go to `sortbench`, e.g.
`sh scripts/bench.sh --reps 11`.

How a cell is timed: one fresh process per cell, pinned to one CPU where the
OS allows it; the meter's own overhead is measured and subtracted; 21 timed
repetitions at 100,000 elements, medians reported, the lower median of two
rounds over the whole matrix kept. Below 100,000 elements the timed region
sorts enough independent copies to reach 100,000 elements and reports the
time per sort, so the clock's cost stays small; above, the repetitions
shrink in proportion, never below three. Every run is verified. The
deterministic columns are not recomputed by a timing run (`--timing-only`);
the website takes them from the counts files.

### The deterministic numbers at one million elements

`results/counts.csv` stops at 100,000 elements because the test suite
recomputes it. The website also shows one million:

```sh
COUNTS_SIZES=1000000 sh scripts/counts.sh      # -> results/counts-1000000.csv, about ten minutes
```

CI does this in its own job; the file is not committed.

### Smaller runs

```sh
./build-linux/sortbench --help
./build-linux/sortbench --algo brainsort --algo pdqsort --type int32             # one type, five standard inputs
./build-linux/sortbench --type string --all-algos --all-datasets --n 1000000     # one size
./build-linux/sortbench --type int64 --dataset nearly_sorted --all-algos --n 1000 --n 100000 --csv mine.csv
```

| option | default | meaning |
|---|---|---|
| `--n N` | 100000 | number of elements (repeatable) |
| `--reps R` | 21 | timed repetitions per cell at 100,000 elements |
| `--rounds K` | 2 | run the whole matrix K times, keep the lowest median |
| `--seed S` | 20260912 | RNG seed for the inputs |
| `--type NAME` | int32 | `int32`, `double`, `int64`, `string` (repeatable); `--all-types` |
| `--algo NAME` | | restrict to an algorithm (repeatable); `--all-algos` adds brainsort, the baselines and the `std::` references |
| `--dataset NAME` | the 5 standard | restrict to an input (repeatable); `--all-datasets` |
| `--pin CPU` | auto | pin the measuring thread; `-1` disables |
| `--counts-only` | | the deterministic columns only, no timing |
| `--timing-only` | | the timed columns only, no counted run |
| `--id`, `--host` | | the stamp's id and machine description |
| `--csv`, `--md FILE` | | output; `--csv` also writes `<name>.meta.json` |
| `--print-id`, `--print-code` | | print the default id, or the code fingerprint of the tree, and exit |
| `--no-fork` | | run in-process instead of one child per cell |

### The website

```sh
python3 scripts/website.py                          # results/ -> site/index.html
python3 scripts/website.py --out /tmp/x.html --results /tmp/results
python3 scripts/website.py --allow-stale            # include timing that measured other code, marked stale
```

The page is one self-contained HTML file with no external resources; open
it locally or serve `site/`. It reads every `counts*.csv` and every
`<id>.csv` / `.meta.json` / `.api.md` in the results directory.

**Staleness.** Every timing file carries a fingerprint of the measured
sources (`include/`, `src/`, `third_party/`, `CMakeLists.txt`), computed by
the binary that measured it; `website.py` computes the same fingerprint over
the current tree (`sortbench --print-code` prints it) and leaves out any
timing file whose fingerprint differs, with a note. So after a change under
those paths the committed workstation results drop off the page until they
are re-measured with `bench.sh`. The deterministic files need no stamp: the
test suite holds every commit to them.

### Continuous integration and publishing

[.github/workflows/ci.yml](.github/workflows/ci.yml) runs on every push to
`main`, every pull request and on demand.

| job | runner | what it does |
|---|---|---|
| Linux GCC 13 and 14 | `ubuntu-24.04` | build, full CTest, README scorecard check; GCC 13 then benchmarks and uploads its results |
| Linux Clang | `ubuntu-24.04` | build, full CTest, benchmark |
| Linux ARM64 | `ubuntu-24.04-arm` | build, full CTest, benchmark |
| macOS ARM64 | `macos-latest`, Apple Clang | build, full CTest, benchmark |
| Windows MSVC | `windows-latest` | build, full CTest, benchmark |
| Windows MinGW | `windows-latest`, MSYS2 UCRT64 | build, full CTest, benchmark |
| Deterministic counts at 1,000,000 | `ubuntu-24.04` | `scripts/counts.sh` for one million elements |
| Sanitizers, libFuzzer, Linux -m32 | `ubuntu-24.04`, Clang / GCC | tests only |
| Website | `ubuntu-24.04` | downloads every result, builds `site/`, uploads it (and, on `main`, publishes it on GitHub Pages) |

The benchmark jobs use `--reps 11` because shared runners are noisy; the
page labels them as shared runners. To publish, enable GitHub Pages for the
repository with **GitHub Actions** as the source (Settings, Pages). Nothing
is measured twice: the website is assembled from the platform jobs' uploads.

## 5. Trustworthy numbers on your own machine

**Performance counters (Linux).** The stamp's backend `linux-perf` means
instructions and cycles are measured; `linux-clock-only` means
`perf_event_open` was refused, usually by `kernel.perf_event_paranoid`.
Allow user-space counting for the current boot with
`sudo sysctl kernel.perf_event_paranoid=2`. WSL2 exposes the counters on
recent Windows builds; cloud VMs may not have them at all. Windows has no
user-mode instruction counter and no usable thread CPU clock; it reports
wall time and cycles.

**A quiet machine.** Close browsers, indexers and builds, use a performance
power plan. Pinning and the best of two rounds absorb short bursts, not a
busy machine. If numbers look noisy, use `--rounds 3`.

**Like with like.** Absolute times depend on the CPU; brainsort's radix
route likes a large L2 cache. What should reproduce anywhere is the ranking
and the rough ratios, and everything in the counts files reproduces exactly.

## 6. Adding an algorithm or an input

- **Algorithm**: write it as a template over the array view `A`, using only
  `a.get(i)`, `a.set(i, v)`, `a.less(x, y)` and, for radix keys,
  `A::key(x, chunk)`. Add one line to the table in
  [registry.hpp](include/sortbench/registry.hpp), a description in
  `scripts/website.py`, and run the tests. If the algorithm keeps tables in
  raw scratch, report their accesses on the counted path with
  `g_trace.on_table_rw(...)` as `radix.hpp` does, and wrap recursive
  functions in a `DepthScope<A::counted>`. Then regenerate
  `results/counts.csv` (section 3).
- **Input**: add a generator to
  [datasets.hpp](include/sortbench/datasets.hpp) and a description in
  `scripts/website.py`; the benchmark and the tests pick it up.

## Troubleshooting

| symptom | fix |
|---|---|
| C++20 errors, `'std::...' is not a member` | the compiler is too old: GCC 13+ (`export CXX=g++-13`, fresh build dir) |
| `cmake` or `ninja` not found on Windows | `C:\msys64\ucrt64\bin` is not on `PATH`, or the packages went into the wrong MSYS2 environment (use UCRT64) |
| backend `linux-clock-only` | section 5 (`perf_event_paranoid`) |
| CMake complains about a cached generator or compiler | delete the build directory and configure again |
| a machine is missing from the website | its timing file measured other code (the build log says `STALE`); rerun `bench.sh` / `bench.ps1` there, or pass `--allow-stale` |
| times vary a lot between runs | the machine is busy; section 5 |

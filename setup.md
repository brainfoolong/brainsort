# Setup: build, test, measure, publish

From a fresh clone to a verified build, a passing test suite, your own
benchmark results and the website built from them. For what the project is,
see the [README](README.md); for the results, the
[website](https://brainfoolong.github.io/brainsort/).

## 1. Prerequisites

Brainsort and the benchmark are plain C++20, header-only. The only third-party
code is the upstream sorting implementations vendored under
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
| `brainsort_api_bench` | the public API against `std::stable_sort`, cpp-TimSort and the Boost.Sort stable sorts on plain vectors |
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

### The Rust port

The crate lives in [rust/](rust/) as a Cargo workspace: `brainsort` (the
published crate) and `brainsort-bench` (the counted harness and the API
benchmark, not published). A stable Rust 1.86 or newer is all it needs.

```sh
cd rust
cargo test --workspace --all-features          # the library tests, the harness's unit tests
cargo run --release -p brainsort-bench -- --golden --max-n 1000000   # the golden equivalence with results/counts.csv
cargo run --release -p brainsort-bench -- --rust-golden              # results/rust-counts.csv: the Rust sorts as shipped, up to 100,000 elements
cargo run --release -p brainsort-bench -- --bench --out ../results/<id>.rust.api.md   # the API benchmark
```

The tests take `BRAINSORT_TEST_QUICK=1` to cap the sizes (what Miri and
the sanitizer runs use) and `--ignored large_inputs` for ten million
elements. The scalar code is tested with `RUSTFLAGS="--cfg brainsort_no_simd"`,
the compile-time vector build with `RUSTFLAGS="-C target-feature=+avx2,+bmi2"`.
The other checks of CI, by hand:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo +nightly doc -p brainsort --no-deps --all-features               # RUSTDOCFLAGS="-D warnings --cfg docsrs"
cargo +1.86.0 check -p brainsort --all-features --all-targets           # the minimum Rust version
cargo check -p brainsort --no-default-features --target thumbv7em-none-eabi   # no_std
cargo +nightly miri test -p brainsort --all-features --lib --test fuzz_smoke --test alloc_failure --test panic_safety   # RUSTFLAGS="--cfg brainsort_no_simd" BRAINSORT_TEST_QUICK=1
cargo deny check                                                        # licenses, advisories, bans
cargo publish -p brainsort --dry-run
cd brainsort && cargo +nightly fuzz run sort -- -max_total_time=180     # needs cargo-fuzz
```

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

**The golden file.** The last test recomputes
[results/counts.csv](results/counts.csv): the deterministic numbers of every
(size, key type, input, algorithm) cell at 1,000, 10,000, 100,000 and
1,000,000 elements (reads, writes, comparisons, table and key traffic,
cache-model misses, scratch memory, telemetry, two fingerprints), and fails
on any difference. The rows up to 100,000 elements are recomputed by
default, about a minute; `--golden-max-n 1000000` adds the million-element
rows, about ten minutes more. The Rust port is held to the same file:
`cargo run --release -p brainsort-bench -- --golden` in `rust/` recomputes
every brainsort row through the Rust code and fails on any column that
differs. A second file, [results/rust-counts.csv](results/rust-counts.csv),
holds what can be counted of the Rust sorts as shipped on the same cells
(comparisons, compare flips, key reads, scratch memory:
[docs/decisions/0010.md](docs/decisions/0010.md)); `cargo test` in `rust/`
recomputes it up to 10,000 elements (100,000 in release) and
`cargo run --release -p brainsort-bench -- --rust-golden` up to 100,000,
holding its brainsort rows to `counts.csv` as well.
A change to brainsort's planner or to a port is caught even when the output
is still correct. When the change was intended, regenerate the file and
commit it with the change:

```sh
sh scripts/counts.sh                        # rewrites results/counts.csv, about twelve minutes, then results/rust-counts.csv when cargo is installed
./build-linux/sortbench_tests --golden      # only the golden comparison
cd rust && cargo run --release -p brainsort-bench -- --rust-counts   # only the Rust sorts' file, a few minutes
```

```powershell
.\scripts\counts.ps1
```

The file is the same on every machine and compiler. Only the rows of the
sorts that run on the toolchain's standard library (`std::stable_sort`,
and the vendored cpp-TimSort and Boost.Sort sorts, which use its
containers and algorithms) may differ with another standard library or
version; a difference there is a warning, not a failure.

## 4. Measure

### The timing of your machine

```sh
sh scripts/bench.sh              # Linux, WSL, macOS, MSYS2
```

```powershell
.\scripts\bench.ps1              # Windows
```

The script builds, then runs every algorithm on every key type and input on
one set of 100,000 elements with a few repetitions (timing is indicative
and machine-bound; the deterministic counts are the measurement and cover
every size), then the public-API benchmark in C++ and, when `cargo` is
found, in Rust, then builds the website. A few minutes on a fast machine.
It writes:

| file | content |
|---|---|
| `results/<id>.csv` | the timing, one row per cell: wall time, CPU time, instructions, cycles, peak memory growth, repetitions |
| `results/<id>.meta.json` | the run stamp: id, host, OS, CPU, compiler, when, the code fingerprint, sizes, repetitions, rounds, seed, pinned CPU |
| `results/<id>.api.md` | the public-API benchmark (`brainsort::sort` on plain vectors against `std::stable_sort`, cpp-TimSort, Boost.Sort spinsort and flat_stable_sort), stamped the same way |
| `results/<id>.rust.api.md` | the same benchmark of the Rust crate against `slice::sort`, `slice::sort_by_cached_key` and glidesort, stamped with the fingerprint of the Rust sources |
| `site/index.html`, `site/rust.html` | the website, from everything in `results/`: one complete page for the C++ library and one for the Rust crate, with a switch between them |

`<id>` defaults to `<os>-<arch>-<compiler>`, e.g. `linux-x86-64-gcc13`.
Environment variables: `BENCH_ID`, `BENCH_HOST` (a description of the
machine that appears on the page), `BENCH_SIZES` (default `100000`),
`BENCH_REPS` (default 5), `BENCH_API_MAX_N` (unset: the quick API run at
100,000 elements; `1000000` or `10000000` add the larger sizes), `BUILD_DIR`,
`BENCH_NO_SITE`, `BENCH_NO_RUST`. Extra arguments go to `sortbench`, e.g.
`sh scripts/bench.sh --pin 3`.

How a cell is timed: one fresh process per cell, pinned to one CPU where the
OS allows it; the meter's own overhead is measured and subtracted; 5 timed
repetitions at 100,000 elements by default, medians reported. With
`BENCH_SIZES` set, below 100,000 elements the timed region sorts enough
independent copies to reach 100,000 elements and reports the time per sort,
so the clock's cost stays small; above, the repetitions shrink in
proportion, never below three. Every run is verified. The deterministic
columns are not recomputed by a timing run (`--timing-only`); the website
takes them from the counts files.

### More sizes

The timing scripts measure 100,000 elements by default. Other sizes, up to
ten million, are an option; the website shows every size it finds in
`results/`:

```sh
COUNTS_SIZES=10000000 sh scripts/counts.sh     # -> results/counts-10000000.csv, over an hour
BENCH_SIZES="1000 10000 100000 1000000 10000000" BENCH_API_MAX_N=10000000 sh scripts/bench.sh
```

```powershell
$env:COUNTS_SIZES = "10000000"; .\scripts\counts.ps1
$env:BENCH_SIZES = "1000 10000 100000 1000000 10000000"; $env:BENCH_API_MAX_N = "10000000"; .\scriptsench.ps1
```

The counted file at ten million is not committed; the timing files carry
whatever sizes they were measured at.

### Smaller runs

```sh
./build-linux/sortbench --help
./build-linux/sortbench --algo brainsort --algo gfx::timsort --type int32        # one type, five standard inputs
./build-linux/sortbench --type string --all-datasets --n 1000000                 # one size
./build-linux/sortbench --type int64 --dataset nearly_sorted --n 1000 --n 100000 --csv mine.csv
```

| option | default | meaning |
|---|---|---|
| `--n N` | 100000 | number of elements (repeatable) |
| `--reps R` | 21 | timed repetitions per cell at 100,000 elements |
| `--rounds K` | 2 | run the whole matrix K times, keep the lowest median |
| `--seed S` | 20260912 | RNG seed for the inputs |
| `--type NAME` | int32 | `int32`, `double`, `int64`, `string` (repeatable); `--all-types` |
| `--algo NAME` | | restrict to an algorithm (repeatable; default: every one) |
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
python3 scripts/website.py                          # results/ -> site/index.html (C++) and site/rust.html (Rust)
python3 scripts/website.py --out /tmp/x.html --results /tmp/results
python3 scripts/website.py --allow-stale            # include timing that measured other code, marked stale
```

Each page is one self-contained HTML file with no external resources; open
them locally or serve `site/`. The two pages have the same sections and
controls; the harness timing is on the C++ page only, the deterministic
counts are on both, and every timed number is on the page of the language
that was timed ([docs/decisions/0009.md](docs/decisions/0009.md)). It reads every `counts*.csv` and every
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
| Sanitizers, libFuzzer, Linux -m32 | `ubuntu-24.04`, Clang / GCC | tests only |
| Rust | see [rust.yml](.github/workflows/rust.yml) | format, clippy, docs, MSRV, `no_std`, tests on six platforms (the scalar, compile-time AVX2 and no-default-features variants and ten million elements on the Linux runners, where the compile-bound test build is fastest), 32-bit, Miri, ASan, fuzzing, cargo-deny, package dry run, the golden equivalence, the API benchmark on four runners |
| Website | `ubuntu-24.04` | downloads every result, builds `site/`, uploads it (and, on `main`, publishes it on GitHub Pages) |

The benchmark jobs use 5 repetitions; the page labels them as shared
runners. To publish, enable GitHub Pages for the repository with **GitHub
Actions** as the source (Settings, Pages). Nothing is measured twice: the
website is assembled from the platform jobs' uploads.

### Releasing

Both libraries carry one version, in nine places (the C++ header's string
and numeric macros, the crate, the two crates that depend on it, the lock
file, the README, this file, the single header) plus the changelog.
`scripts/version.py` sets them all at once and `--check`, a CTest and the
first step of the release workflow, fails when any of them disagrees. To
release:

1. Keep the changes under `## Unreleased` in `rust/brainsort/CHANGELOG.md`
   as they land. Then `python3 scripts/version.py 0.4.0`: every location is
   set, the unreleased section becomes the version's, the single header
   and the lock file are regenerated. Write the version's summary line
   under the new heading if you want one, and commit.
2. Wait for CI to be green on that commit, then tag it and push the tag:
   `git tag v0.5.0 && git push origin v0.4.0`.
3. [release.yml](.github/workflows/release.yml) checks that the tag and
   every location agree, runs the C++ and Rust test suites and the golden
   equivalence, publishes the crate on crates.io and creates the GitHub
   release with `brainsort-<version>.hpp`, a source archive and checksums.
   `workflow_dispatch` with `dry_run` runs everything but the two
   publishing steps.

crates.io credentials: the workflow uses Trusted Publishing (OIDC) once it
is configured for the crate on crates.io (crate settings, Trusted
Publishing: owner `brainfoolong`, repository `brainsort`, workflow
`release.yml`). crates.io requires the first version of a crate to be
published with an API token, so for the first release add a repository
secret `CARGO_REGISTRY_TOKEN` (an API token with the `publish-new` scope
from [crates.io/settings/tokens](https://crates.io/settings/tokens)); the
workflow uses it when the OIDC exchange is not available. After the first
release, configure Trusted Publishing, delete the secret, and set the
repository variable `BRAINSORT_CRATE_PUBLISHED` to `true` so CI runs
cargo-semver-checks against the published baseline on pull requests.

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

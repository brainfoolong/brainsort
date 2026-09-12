# Brainsort Rust port: the plan

Written for: a coding agent starting from a fresh session in this repository,
with no memory of the conversation that produced this file. Everything you
need to decide is decided here; everything you need to look up is pointed at.
Read this file top to bottom before touching anything.

Status: plan only, nothing of the Rust port exists yet. Written 2026-09-13
against commit `5014eb1` plus uncommitted changes in the working tree (README,
dashboard, results and bench scripts). Do not touch, revert or commit those
changes; they belong to other work. Put every file of the port under `rust/`
and every CI change in a new workflow file, so the two lines of work never
collide.

## 1. Goal

A native Rust port of brainsort, published on crates.io as the crate
`brainsort`, with:

- the same algorithm, verified by the same golden numbers the C++ is held to,
- the same public contract (stable, exact, defined float order, completes on
  allocation failure, 2^32 limit with a fallback),
- an API shaped like the standard library's `slice::sort*` family,
- runtime AVX2/BMI2 dispatch on x86-64 and the scalar code everywhere else,
- `no_std` + `alloc` compatible core,
- full CI: every platform the C++ is tested on, plus the Rust-specific checks
  (fmt, clippy, docs, MSRV, Miri, sanitizer, fuzz, semver, publish dry run),
- an honest benchmark against the Rust ecosystem, with results committed,
- a release workflow that publishes from a tag.

The crate name `brainsort` was checked on 2026-09-13: both
`https://crates.io/api/v1/crates/brainsort` and `.../brain-sort` return 404,
so the name is free. Reserving it is step one (section 9, phase 0).

## 2. What exists: the C++ library you are porting

Header-only C++20 under [include/brainsort/](../include/brainsort/), 3,734
lines. Read the files in this order; the line counts tell you how much each
one is.

| file | lines | what it is |
|---|---:|---|
| [brainsort.hpp](../include/brainsort/brainsort.hpp) | 356 | public API: `sort`, `sort_by_key`, `sort_with`, `stable_sort`, `desc`; the record build, the monotone pre-check, the small sort, the permutation, the fallbacks |
| [detail/config.hpp](../include/brainsort/detail/config.hpp) | 144 | compiler/platform adaptation: SIMD target attributes, `cpuid`/`xgetbv` feature detection, bswap/popcount/ctz helpers, `BRAINSORT_NO_SIMD` |
| [detail/traits.hpp](../include/brainsort/detail/traits.hpp) | 302 | the array view `A` the algorithm is written against, `elem_traits<T>`, the instrumentation hooks (`NoHooks`), `DefaultAlloc`, scratch buffer types |
| [detail/keys.hpp](../include/brainsort/detail/keys.hpp) | 363 | `key_traits<K>` adapters: fixed (integer-like), bytes (strings), composite (pair/tuple/array), `descending<K>`; the compile-time shape of a key (`PartList`, up to 32 leaves); `compare_keys` |
| [detail/records.hpp](../include/brainsort/detail/records.hpp) | 318 | the four record types `Rec32` (8 B), `Rec64` (16 B), `StrRec` (16 B), `CompRec<Kinds...>`; `elem_traits` for each; `record_for<K>`; `RecordBuilder` |
| [detail/algorithm.hpp](../include/brainsort/detail/algorithm.hpp) | 1,843 | the algorithm: scout pass (scalar + AVX2), routes 1-5, split, radix plan, dictionary radix, partition sort, `Scratch`, `sort_range`, `brainsort_impl` |
| [detail/radix.hpp](../include/brainsort/detail/radix.hpp) | 340 | radix core: `DigitPlan`, digit width choice, histogram prefix sums (scalar + AVX2), PEXT key extraction, `live_passes` |
| [detail/mergesort.hpp](../include/brainsort/detail/mergesort.hpp) | 68 | top-down merge sort: the fallback beyond 2^32 elements |

The README sections to read: "The library" (the contract you must keep, with
the tables of key types and the memory numbers), "How brainsort works" (every
route and every mechanism, with the reasons), "Not repeating the classic
sorting bugs", and "Ideas that were tested and rejected" (do not re-try
those). [CHANGELOG.md](../CHANGELOG.md) lists the 1.0.0 contract in one
place.

### 2.1 The contract to keep, verbatim

From the header comment of brainsort.hpp and the README:

- Every sort is stable and exact.
- Keys: every integer, bool, chars, f32/f64, enums (Rust: user impl),
  pointers, strings and byte strings, durations/time points, pairs, tuples,
  arrays of keys, `desc(key)`. A user key type implements the key trait.
- Floats: `-0.0 == +0.0`; NaN with the sign bit clear sorts after `+inf`,
  with it set before `-inf`, ordered by payload. This is NOT `f64::total_cmp`
  (which orders `-0.0 < +0.0`). Document the difference in the Rust docs.
- Elements: anything movable. The elements are permuted once, after the keys
  were sorted. Keys are computed exactly once per element.
- Memory: about 1.5 records per element while sorting (8 B records for keys
  up to 32 bits, 16 B up to 64 bits or a string, more for composites) plus
  one element per element for the final permutation of `Copy` elements.
- Allocation failure: the sort completes anyway through the comparison
  fallback with the same order (Rust: `slice::sort_by` on the key compare).
  The range is untouched when records or scratch cannot be allocated; if only
  the permutation buffer fails, permute in place through cycles.
- A key function that panics leaves the slice unchanged (strong guarantee):
  all keys are computed before anything moves.
- Limits: at most 2^32 - 1 elements per call, strings shorter than 2^32
  bytes; beyond either, the comparison fallback with the same order.
- No shared mutable state; thread safe (Rust: `Send + Sync` free functions,
  no statics except the CPU feature cache).
- Below 32 elements (`kSmallSort`): insertion sort on the elements by key, no
  allocation.

### 2.2 How the algorithm is instrumented, and why it matters for the port

The algorithm is generic over a view type `A` (traits.hpp lines 1-30). The
view carries `hooks` and a `static constexpr bool counted`. Every element
read/write, table access, key chunk load and comparison in the algorithm goes
through the view, and every hook call sits under `if constexpr (A::counted)`,
so the library build has zero overhead while the benchmark build counts
everything. The benchmark's `TraceHooks` (include/sortbench/core.hpp ~line
452) records: reads, writes, compares, cmp_flips, table_reads, table_writes,
table_bytes, key_bytes, aux_peak_bytes, aux_allocs, max_depth, the route
taken and the route counters (giveups, splits, split_retries, part_sorts,
dict_tries, dict_hits, plan_retries, radix_passes, passes_skipped), an FNV-1a
`trace_hash` over the access sequence (region-relative addresses, op code)
and a fixed cache model (l1/l2 misses).

[results/counts.csv](../results/counts.csv) is the golden file: 564 rows,
one per (type, dataset, algorithm) at n = 100,000, seed 20260912. The 48
brainsort rows (4 types x 12 datasets) are the port's target. The test
[src/test.cpp](../src/test.cpp) `test_golden()` recomputes every row and
fails on any mismatch. The Rust port must reproduce the same hooks at the
same call sites, so the same counts come out. This is the single strongest
proof that the port is the same algorithm, and it is the reason the port
keeps the view/hooks structure instead of "just writing a radix sort".

Datasets: [include/sortbench/datasets.hpp](../include/sortbench/datasets.hpp)
generates them with `std::mt19937_64` and `std::uniform_int_distribution` /
`uniform_real_distribution`. The distributions are implementation-defined
(libstdc++ and MSVC map differently), so do NOT reimplement them in Rust.
Instead add a `--dump-datasets DIR` mode to the C++ driver (section 5.2) and
let the Rust golden test read the dump.

The 12 datasets: random, sorted, reverse, nearly_sorted, few_unique,
all_equal, runs, organ_pipe, small_range, sawtooth, prefixed, sparse_bits
(int32 only, but generated for every type). The 4 types: int32 (8-byte
element: key + id), double (16 B), int64 (16 B), string (16 B: ptr + len,
plus the character pool).

## 3. Crate design

### 3.1 Layout

```
rust/
  Cargo.toml                  workspace: members = ["brainsort", "brainsort-bench"], resolver = "3"
  rust-toolchain.toml         channel = "stable" (CI overrides for nightly jobs)
  brainsort/                  THE PUBLISHED CRATE
    Cargo.toml
    README.md                 crate README (docs.rs front page); symlinks do not work on crates.io, keep a real file
    CHANGELOG.md
    LICENSE                   copy of the root MIT license
    build.rs                  declares the brainsort_no_simd cfg (check-cfg)
    src/
      lib.rs                  public API, crate docs, feature gates, re-exports
      key.rs                  Key trait, Desc<K>, built-in impls, key compare
      shape.rs                the compile-time shape of a key (leaves: fixed bits / bytes / desc)
      record.rs               Rec32, Rec64, StrRec, CompRec; ElemTraits impls; record building; index_of/set_index/key_of
      view.rs                 the array view, the Hooks trait + NoHooks, the Alloc trait + Global alloc, buffers (AuxBuffer, AuxRaw, DispBuf)
      sort.rs                 sort_by_key_impl: small sort, sort_if_monotone, sort_records, permute, fallbacks (= brainsort.hpp minus the API surface)
      algorithm/
        mod.rs                sort_range, brainsort_impl, Scratch
        scout.rs              ScoutResult, scout_scalar, scout_tail, scout (dispatch), compute_mask
        monotone.rs           reverse_all, stable_reverse (routes 1, 2)
        runs.rs               merge_with_buffer, sort_few_runs (route 3)
        displaced.rs          DispBuf, restore_displaced, sort_displaced (route 4)
        radix_route.rs        Pivot, pick_pivot, dict_pick, split_*, RadixPlan, choose_plan, HistStore, radix_run, dict_sort, radix_part, sort_split_parts, radix_route (route 5)
        partition.rs          split2, partition2, partition_sort_few (the 2-4 distinct key partition sort)
      radix.rs                DigitPlan, make_plan, digit_bits_for, radix_prefix, prefix_sums, live_passes, pext key
      mergesort.rs            the 2^32 fallback, and the merge sort port for the bench
      simd/
        mod.rs                dispatch: have_avx2(), have_bmi2(); cfg(brainsort_no_simd) forces scalar
        x86.rs                every AVX2/BMI2 kernel: scout_avx2_32/64, reverse_avx2, lt_mask, split_forward_avx2, split2_avx2, partition2_avx2, prefix_avx2 (u16/u32), radix_exec_pext
      fallback.rs             the comparison fallback (slice::sort_by with the key compare) and the > 2^32 path
    tests/
      api.rs                  the port of tests/api_tests.cpp (section 5.1)
      float_order.rs          the documented float order, NaN, signed zeros, extremes
      alloc_failure.rs        fail every allocation in turn (a counting Alloc that fails at call k)
      panic_safety.rs         a key fn that panics on element i leaves the slice unchanged, for every i in a small range
      golden.rs               the golden equivalence test (section 5.2); #[ignore] unless BRAINSORT_DATASETS is set
      shapes.rs               proptest random shapes: every key kind, every size 0..=300, every pattern
      fuzz_smoke.rs           the fuzz body on 3,000 seeded inputs, on stable
    benches/
      api.rs                  criterion: sort/sort_by_key vs slice::sort, slice::sort_unstable, rdst, voracious_radix_sort (section 5.5)
    fuzz/                     cargo-fuzz crate (publish = false, excluded from the package)
      Cargo.toml
      fuzz_targets/sort.rs    the port of tests/fuzz_sort.cpp
  brainsort-bench/            PRIVATE (publish = false): the sortbench counterpart
    Cargo.toml
    src/
      datasets.rs             reads the C++ dataset dump; also its own generators for the timing bench
      hooks.rs                the counting Hooks + counting Alloc (the port of TraceHooks / Trace, without the cache model at first)
      counted_run.rs          runs one (type, dataset) through the counted view, verifies, produces the counts.csv columns
      main.rs                 CLI: --counts (write counts-rust.csv), --golden (compare with ../results/counts.csv), --bench
```

Naming: the crate is `brainsort`, the library target is `brainsort`, the
public module path is `brainsort::`. The bench crate is never published.

### 3.2 Public API

Mirror `slice::sort*`. Free functions on `&mut [T]`, nothing on iterators.

```rust
/// Stable sort by the elements themselves.
pub fn sort<T: Key>(v: &mut [T]);

/// Stable sort by an owned key computed once per element (like sort_by_cached_key).
pub fn sort_by_key<T, K: Key, F: FnMut(&T) -> K>(v: &mut [T], f: F);

/// Stable sort by a key borrowed from the element (e.g. `|r| r.name.as_str()`).
pub fn sort_by_key_ref<T, K: Key + ?Sized, F: for<'a> FnMut(&'a T) -> &'a K>(v: &mut [T], f: F);

/// Stable sort with an arbitrary comparator: the slow path, delegates to slice::sort_by.
pub fn sort_by<T, F: FnMut(&T, &T) -> Ordering>(v: &mut [T], f: F);

/// Reverses the order of any key: `sort_by_key(&mut v, |r| (r.group, Desc(r.score)))`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Desc<K>(pub K);
pub fn desc<K: Key>(k: K) -> Desc<K>;

/// Extension trait so `v.brainsort()` / `v.brainsort_by_key(f)` read naturally. Optional, keep it small.
pub trait SliceExt<T> { fn brainsort(&mut self) where T: Key; fn brainsort_by_key<K: Key>(&mut self, f: impl FnMut(&T) -> K); }
```

Why two key functions: Rust cannot express "a closure that returns either an
owned key or a borrow of its argument" in one signature (the same limitation
makes `slice::sort_by_key` unable to return `&str`). C++ handles both through
`KeyHolder` and `materialise`; Rust needs `sort_by_key` (owned; keys are
stored for the duration of the sort, exactly like the C++ `materialise`
path) and `sort_by_key_ref` (borrowed; records point into the elements,
like the C++ by-reference projection).

`sort_by` exists so one crate covers every case, as `sort_with` does in C++.
Document plainly that it is `slice::sort_by`.

### 3.3 The `Key` trait

Port `key_traits` and the compile-time shape. Rust has no specialisation and
no variadic templates, so the design is:

```rust
pub trait Key {
    /// The leaves of this key, in order. Computed at compile time; at most 32.
    const SHAPE: Shape;                 // Shape { parts: [Part; 32], n: usize }; Part::Fixed { bits: u8, exact: bool, desc: bool } | Part::Bytes { desc: bool }
    /// Writes each leaf: fixed leaves as an order-preserving u64 radix value, byte leaves as a slice.
    fn write_parts(&self, w: &mut impl PartSink);   // PartSink { fn fixed(&mut self, radix: u64); fn bytes(&mut self, b: &[u8]); }
    /// The natural order, independent of the radix form: used by the small sort, the monotone check and the fallbacks.
    fn cmp_key(&self, other: &Self) -> Ordering;
}
```

Helper traits with blanket impls keep user code short:

- `FixedKey { type Radix: u32 | u64; const BITS: u32; const EXACT: bool; fn to_radix(&self) -> Self::Radix; fn from_radix(r) -> Self where EXACT }`
  for integers (sign bit flip), bool, char, f32/f64 (the documented
  transform: positive `bits | 2^(w-1)`, negative `2^(w-1) - magnitude`),
  `core::time::Duration` (secs and nanos need 94 bits: make it a composite
  of two fixed parts, not one u64), `*const T`/`*mut T` (address),
  `NonZero*`, `Wrapping<T>`, `core::cmp::Reverse<K>` (same as `Desc`).
- `BytesKey { fn bytes(&self) -> &[u8] }` for `str`, `String`, `[u8]`,
  `Vec<u8>`, `&str`, `&[u8]`, `Box<str>`, `Cow<str>`, `Rc/Arc<str>`,
  `CStr`, `OsStr` (std feature, via `as_encoded_bytes`), `Path` (std).
- Composites: `(A,)` through 12-tuples, `[K; N]`. No `Option<K>`: the C++
  has no optional adapter and says so; leave it out and document it.
- `Desc<K>` and `core::cmp::Reverse<K>`: the reversed shape.
- No derive macro in v1 (a proc-macro crate is a second package to
  publish); a declarative macro for newtype keys is enough:
  `brainsort::fixed_key!(UserId => u64, |k| k.0);`.

The record type is chosen from `K::SHAPE` at compile time by a plain `if`
chain on constants inside `sort_by_key_impl`; the optimiser removes the dead
arms. Order, as in `record_for<K>` (records.hpp ~line 139): all fixed and
total bits <= 32 -> `Rec32`; all fixed and <= 64 -> `Rec64`; exactly one
bytes leaf -> `StrRec<DESC>`; otherwise `CompRec`.

`CompRec<Kinds...>` is variadic in C++. In Rust, without generic const
expressions, make the composite record a fixed-size struct sized by a small
set of const generics chosen at the same `if` chain: `CompRec<const NF: usize,
const NB: usize>` (NF fixed slots as `[u64; NF]`, NB byte slots as
`[BytesSlot; NB]`), instantiated for a handful of (NF, NB) combinations that
cover the shapes (say NF in {1, 2, 3, 4, 8, 16, 32} and NB in {0, 1, 2, 4,
8}); a shape that does not fit takes the next larger record, with the unused
slots zero. Per-part descending flags live in `K::SHAPE`, read as a constant
inside `ElemTraits for CompRec`, so the chunk compare is still branch-free
on the flag after monomorphisation. Verify with `cargo asm` or a benchmark
that this is not slower than the C++ on composite keys; if it is, add more
instantiations, never runtime-sized records.

### 3.4 The view, hooks and allocation

Port traits.hpp faithfully. It is the seam the golden test depends on.

```rust
pub(crate) trait Hooks {
    const COUNTED: bool;                         // false for NoHooks: every call is `if H::COUNTED { ... }`
    type DepthScope: Default;                    // increments max_depth on the counted build
    fn stats() -> ...;                           // giveups, splits, ..., note_route(route); thread-local on the counted build
    fn on_read(p: *const u8, bytes: usize); fn on_write(..); fn on_table_read(..); fn on_table_write(..); fn on_table_rw(..);
    fn on_table_sweep(p, entries, entry_bytes, read: bool, write: bool);
    fn on_key_chunk<T>(t: &T, chunk: i32);
}
pub(crate) trait Alloc { fn alloc_array<U>(n: usize) -> Result<NonNull<U>, AllocError>; unsafe fn free_array<U>(p, n); }
```

Stats for NoHooks must compile to nothing: an empty struct with inherent
no-op methods. On the counted build the bench crate supplies the impl. The
counted impl needs global state; use a `thread_local!` in the bench crate,
not a static in the library.

Expose the generic entry points to the bench crate behind a cargo feature
`__internals` (double underscore, `#[doc(hidden)]`, documented as "not part
of the public API, no semver guarantee"). This is how serde exposes
`__private`. The published crate's public surface stays the four functions,
`Key`, `Desc`, the helper traits and the macro.

Allocation must never abort. Use `alloc::alloc::alloc` with a checked
`Layout` and return `Err` on null (do not use `Vec::with_capacity`, which
aborts). The permutation buffer for `Copy` elements: try first, fall back to
the cycle walk.

### 3.5 SIMD and dispatch

- Kernels in `simd/x86.rs`, each `#[target_feature(enable = "avx2")]` (or
  `"bmi2"`) and `unsafe fn`, called only after the runtime check, exactly as
  the C++ `BRAINSORT_TARGET_AVX2` functions are.
- Dispatch in `simd/mod.rs`: with the `std` feature,
  `is_x86_feature_detected!("avx2")` cached in a relaxed `AtomicU8`
  tri-state (what memchr does). Without `std`, the vector path is compiled
  in only when `cfg(target_feature = "avx2")` is set at compile time;
  otherwise scalar. Document this.
- The C++ checks `xgetbv` for OS YMM support; `is_x86_feature_detected!`
  already does. Do not roll your own cpuid.
- A `--cfg brainsort_no_simd` (RUSTFLAGS, not a cargo feature: features must
  be additive and this one removes code) compiles the scalar code on x86-64,
  the twin of `BRAINSORT_NO_SIMD`. CI runs the whole test suite once each way.
  Declare it in `build.rs` with `println!("cargo::rustc-check-cfg=cfg(brainsort_no_simd)")`
  so `check-cfg` does not warn.
- `_pext_u64` / `_pext_u32` are in `core::arch::x86_64`; keep the C++ rule
  that PEXT is chosen only when it saves a pass and the CPU reports BMI2.
  Keep parity with the C++ plan logic, do not "improve" it.
- Every AVX2 kernel has a scalar twin in the C++ (`*_scalar`) that also
  handles the tail. Port the scalar one first, make the golden test pass on
  it (the counts do not depend on the vector path: `brainsort_tests_scalar`
  and the golden file agree), then add the kernel and check the golden test
  still passes and the output is bit-identical.

### 3.6 Unsafe policy

- The scatter, split and permutation loops use raw pointers or
  `get_unchecked`; bounds checks there cost more than the branch-free design
  saves. Every `unsafe` block gets a `// SAFETY:` comment naming the
  invariant (index < n established where).
- `#![deny(unsafe_op_in_unsafe_fn)]`, `#![warn(missing_docs, clippy::undocumented_unsafe_blocks)]`.
- No `unsafe` in `key.rs`, `shape.rs` or the public API surface.
- Records are `Copy` plain-old-data; the element permutation moves with
  `ptr::read`/`ptr::write` through cycles and never runs user code, so a
  panic cannot leave a hole. Keys are computed before anything moves.
- `cargo miri test` must pass on the scalar build (Miri has partial AVX2
  support; do not depend on it). `-Zsanitizer=address` must pass on the
  vector build.

### 3.7 Features and targets

| feature | default | what it adds |
|---|---|---|
| `std` | yes | runtime CPU detection, `OsStr`/`Path` keys |
| no default features | | the whole algorithm on `alloc`; compile-time SIMD only |
| `__internals` | no | the generic entry points and the Hooks/Alloc traits, for the bench crate and the alloc-failure test |

- `#![no_std]` at the crate root with `extern crate alloc;` and
  `#[cfg(feature = "std")] extern crate std;`.
- Edition 2024, `rust-version = "1.86"` (safe `#[target_feature]` functions
  and edition 2024 need it). Verify the exact MSRV with `cargo msrv` in
  phase 1 and pin what it finds; do not claim lower than tested.
- Targets tested: x86_64 Linux/Windows(msvc, gnu)/macOS, aarch64 Linux/macOS,
  i686 Linux (scalar), and a `cargo check` for `thumbv7em-none-eabi`
  (`--no-default-features`) to prove `no_std`.

## 4. Porting map and order

Port in this order; each step has a test that passes before the next starts.

1. `radix.rs` from radix.hpp: pure functions, unit-testable on their own
   (`make_plan`, `digit_bits_for`, `radix_prefix` against a naive prefix sum,
   `live_passes`). Scalar only.
2. `view.rs` from traits.hpp: the view, `NoHooks`, `Alloc`, buffers. Then
   `record.rs` from records.hpp and `key.rs`/`shape.rs` from keys.hpp with
   unit tests on the radix transforms (every integer type round-trips through
   `to_radix`/`from_radix`; float transform agrees with the documented order
   on a table of edge values).
3. `algorithm/scout.rs` (scalar), `monotone.rs`, `runs.rs`: routes 1-3.
   Test: sorted, reversed (with ties), organ pipe, <= 4 runs at many sizes.
4. `algorithm/displaced.rs`: route 4 with its give-up path. Test:
   nearly_sorted at 1%, and inputs that force the give-up (the C++ test suite
   has these; port them).
5. `algorithm/radix_route.rs` + `partition.rs`: route 5, the split, plans,
   dictionary radix, partition sort. Test: random, few_unique, small_range,
   prefixed, sparse_bits, plus the plan_retry and split_retry paths.
6. `mergesort.rs`, `fallback.rs`, `sort.rs`, `lib.rs`: the API. Test:
   `tests/api.rs` (section 5.1).
7. `brainsort-bench` counting hooks + the C++ dataset dump + `tests/golden.rs`
   (section 5.2). This is the milestone: 48 rows match.
8. `simd/x86.rs`: one kernel at a time, golden test after each, output
   bit-identical to the scalar build on every dataset and 1,000 random shapes.
9. Fuzz target, Miri, sanitizer, benches, docs, CI, publish (sections 5-8).

Function-level map for the big file, so nothing is missed (line numbers in
algorithm.hpp as of `5014eb1`):

| C++ | line | Rust |
|---|---:|---|
| `ScoutResult`, `commit_now`, `track_pair`, `finish_runs`, `scout_scalar`, `compute_mask`, `scout_block`, `scout_tail`, `scout` | 75-364 | `algorithm/scout.rs` |
| `scout_avx2_32`, `scout_avx2_64`, `gt64`, `scout_avx2`, `reverse_avx2` | 227-349 | `simd/x86.rs` |
| `reverse_all`, `stable_reverse` | 365-422 | `algorithm/monotone.rs` |
| `merge_with_buffer`, `sort_few_runs` | 423-483 | `algorithm/runs.rs` |
| `DispBuf`, `restore_displaced`, `sort_displaced` | 484-697 | `algorithm/displaced.rs` |
| `Pivot`, `DictMuls`, `dict_pick`, `pick_pivot` | 698-833 | `algorithm/radix_route.rs` |
| `split_forward_scalar`, `split_backward_scalar`, `split_forward` | 834-1071 | `algorithm/radix_route.rs` |
| `CompressLut`, `lt_mask`, `split_forward_avx2` | 928-1053 | `simd/x86.rs` |
| `digit_width`, `RadixPlan`, `choose_plan`, `HistStore`, `radix_run`, `dict_sort`, `Scratch`, `radix_part` | 1072-1403 | `algorithm/radix_route.rs`, `Scratch` in `algorithm/mod.rs` |
| `radix_exec_pext` | 1260 | `simd/x86.rs` |
| `split2_scalar`, `partition2_scalar`, `split2`, `partition2`, `sort_split_parts`, `partition_sort_few` | 1404-1694 | `algorithm/partition.rs` |
| `split2_avx2`, `partition2_avx2` | 1496-1604 | `simd/x86.rs` |
| `radix_route`, `sort_range`, `brainsort_impl`, `brainsort_view` | 1695-1843 | `algorithm/radix_route.rs`, `algorithm/mod.rs` |

Rules while porting:

- Same constants, same thresholds, same sample size (512), same digit caps
  (16 bits whole-array, 13 bits split path), same `kSmallSort = 32`, same
  give-up ratios. The README explains each; do not tune.
- Same hook call sites. If a C++ loop calls `on_read` once per element, the
  Rust loop does too, on the same element, in the same order.
- Same allocation order and sizes, so `aux_peak_bytes`, `aux_allocs` and the
  trace regions match.
- Where Rust forces a structural change (no goto, no variadics, borrow
  checker on the two-buffer alternation), keep the behaviour and write a
  comment naming the C++ function it mirrors.
- Do not port the benchmark's opponents (introsort, timsort, pdqsort ports).
  The Rust opponents are the ecosystem's real crates (section 5.5).

## 5. Verification

### 5.1 The API test suite (port of tests/api_tests.cpp)

Read the C++ file first; it is the spec. Port every section:

- the natural order of every key type written independently of the library
  (a plain `Ord`-based reference, floats by the documented rule),
- key generators and the ten input patterns,
- the main matrix: every key type x 10 patterns x 25 sizes from 0 to 100,000,
  through `sort` (self-keyed), `sort_by_key` and `sort_by_key_ref`; check
  order by the reference, stability (carry the original index), and that the
  output is a permutation,
- the documented float order (NaN payloads, both signs, infinities, signed
  zeros, subnormals),
- element kinds: `Copy` structs, `String`s, `Box<T>`, `Rc`, zero-sized, large
  (256-byte) structs, `Vec<&str>`, arrays, slices of a `VecDeque::make_contiguous`,
- allocation failure at every allocation point (a counting `Alloc` behind
  `__internals` that fails call k, for every k, and checks the result is
  still sorted and stable),
- a key function that panics at element i, for every i, then the slice is
  unchanged (`catch_unwind`, std feature only),
- eight threads sorting concurrently,
- 400 random shapes,
- 10 million elements (`#[ignore]`d by default; CI runs it once on Linux).

Target: the same order of magnitude of checks as the C++ (about 54,000).

### 5.2 The golden equivalence test

This is the proof of "same algorithm". Three parts.

**C++ side: dump the datasets.** Add `--dump-datasets DIR` to
[src/bench.cpp](../src/bench.cpp): for each type in {int32, double, int64,
string} and each of the 12 datasets, write `DIR/<type>-<dataset>.bin` for
n = 100,000, seed 20260912, exactly what `generate_dataset<T>` produces.
Format: little-endian; int32 as 4 bytes per key, int64 as 8, double as 8
IEEE bits, string as u32 length + bytes per key, in input order. The id is
the index, not stored. Keep this change minimal and separate from the
uncommitted dashboard work; it is one new flag and one function. Add a CI
step that dumps and checks the file sizes.

**Rust side: the counting harness.** `brainsort-bench` implements `Hooks`
and `Alloc` with counters matching `include/sortbench/core.hpp` (`Trace`,
`TraceHooks`, `AlgoStats`) and reproduces the columns of counts.csv for the
brainsort rows: reads, writes, compares, table_reads, table_writes,
table_bytes, key_bytes, cmp_flips, aux_peak_bytes, aux_allocs, max_depth,
route, r_sorted, r_reverse, r_runs, r_displaced, r_radix, giveups, splits,
split_retries, part_sorts, dict_tries, dict_hits, plan_retries,
radix_passes, passes_skipped, order_hash. The semantics of each counter are
in core.hpp (for example a three-way compare counts as one comparison;
`on_table_sweep` adds `entries` to reads and/or writes). `order_hash` is the
FNV-1a over the output ids (see counted_run.hpp line ~112); it proves the
stable output is right, which for a stable sort is the unique answer, so it
is a correctness check, not an algorithm check. The algorithm check is every
other column.

`trace_hash`, `l1_misses`, `l2_misses` and `traffic_bytes` need the
region-relative address model and the cache model. Tier 2: port them after
the tier-1 columns match, because they also pin the allocation order and the
buffer layout. The plan considers the port equivalent when tier 1 matches on
all 48 rows and tier 2 matches on at least the int32 and int64 rows.

**The test.** `rust/brainsort/tests/golden.rs` reads
`$BRAINSORT_DATASETS/<type>-<dataset>.bin`, builds the same element records
the C++ benchmark sorts (int32: 8-byte `{key, id}`; double/int64: 16-byte;
string: `{ptr, len, id}` into a pool), runs them through the counted view,
and compares every tier-1 column with the brainsort rows of
`../../results/counts.csv`. It is `#[ignore]` when the env var is unset and
prints how to produce the dump. CI builds the C++ driver, dumps, then runs
`cargo test --test golden -- --ignored` (section 6, job `golden`).

Also write `results/counts-rust.csv` from the bench crate with the same
schema, so `scripts/scorecard.py` and `scripts/dashboard.py` can read it
later. Do not change those scripts in this work.

### 5.3 Fuzzing

`rust/brainsort/fuzz/fuzz_targets/sort.rs` with `cargo-fuzz` +
`arbitrary`: the input bytes choose a key kind (i32, i64, u64, f64, bytes,
(u32, Desc<i16>), composite with two strings) and the elements; sort; verify
order, stability and permutation against `slice::sort_by` on the key compare.
CI runs it for three minutes on nightly with ASan, like the C++ job.
`tests/fuzz_smoke.rs` runs the same body on 3,000 seeded random inputs on
stable, so every `cargo test` covers it.

### 5.4 Miri and sanitizers

- `cargo +nightly miri test --lib --tests` with `RUSTFLAGS="--cfg brainsort_no_simd"`
  and `MIRIFLAGS="-Zmiri-strict-provenance"` on the whole suite minus the
  large and golden tests. Expect it to be slow; give it a quick knob (an env
  var `BRAINSORT_TEST_QUICK=1` that caps sizes at 4,000) rather than skipping
  tests.
- `RUSTFLAGS="-Zsanitizer=address" cargo +nightly test --target x86_64-unknown-linux-gnu`
  with the vector paths on.

### 5.5 Benchmarks

`benches/api.rs` with criterion 0.5: brainsort `sort`/`sort_by_key` against
`slice::sort` (driftsort, stable), `slice::sort_unstable` (ipnsort),
`slice::sort_by_cached_key`, `rdst` (`RadixSort`), `voracious_radix_sort`
(`voracious_stable_sort`), `radsort`; types i32, i64, f64, `String`, `&str`,
64-byte struct by an i64 field; datasets random, sorted, reverse,
nearly_sorted, few_unique; n = 100k, 1M, 10M. Same generators as the C++
[bench/api_bench.cpp](../bench/api_bench.cpp) (its `make` function), seed
fixed. Write the tables to `results/api-bench-rust-linux.md` and
`...-windows.md` with a `--markdown` flag on the bench crate (criterion's
own reports go to `target/`, do not commit them).

Honesty rules from the C++ README apply: compare against the opponents' own
code as shipped, report where brainsort loses, never `-C target-cpu=native`
for the committed numbers, state the machine, compiler and commit in the
file. Rust 1.81+ `slice::sort` handles runs and nearly sorted input well;
expect smaller margins than the C++ tables show and say so.

## 6. CI

New file `.github/workflows/rust.yml`, triggered on push to main, pull
requests and `workflow_dispatch`. Use `dtolnay/rust-toolchain`,
`Swatinem/rust-cache`, `taiki-e/install-action` for tools. All jobs
`working-directory: rust`. Leave `ci.yml` alone unless the user agrees to a
`paths-ignore` (section 10).

| job | runner | toolchain | what |
|---|---|---|---|
| `fmt` | ubuntu-24.04 | stable | `cargo fmt --all --check` |
| `clippy` | ubuntu-24.04 | stable | `cargo clippy --workspace --all-targets --all-features -- -D warnings` |
| `docs` | ubuntu-24.04 | nightly | `RUSTDOCFLAGS="-D warnings --cfg docsrs" cargo doc --no-deps --all-features` |
| `test` matrix | ubuntu-24.04, ubuntu-24.04-arm, macos-latest (ARM64), windows-latest (msvc), windows-latest (`x86_64-pc-windows-gnu`) | stable, beta | `cargo test --workspace --all-features`; on ubuntu x86 also the `--no-default-features` variant |
| `test-scalar` | ubuntu-24.04 | stable | `RUSTFLAGS="--cfg brainsort_no_simd" cargo test --workspace --all-features` |
| `test-avx2-static` | ubuntu-24.04 | stable | `RUSTFLAGS="-C target-feature=+avx2,+bmi2" cargo test` (compile-time dispatch path) |
| `msrv` | ubuntu-24.04 | 1.86 (the `rust-version`) | `cargo check --workspace --all-features`; `cargo test -p brainsort` |
| `i686` | ubuntu-24.04 | stable, target `i686-unknown-linux-gnu` | `sudo apt install gcc-multilib`; `cargo test -p brainsort --target i686-unknown-linux-gnu` |
| `no_std` | ubuntu-24.04 | stable, target `thumbv7em-none-eabi` | `cargo check -p brainsort --no-default-features --target thumbv7em-none-eabi` |
| `miri` | ubuntu-24.04 | nightly + miri | section 5.4, with the quick knob |
| `asan` | ubuntu-24.04 | nightly | section 5.4 |
| `fuzz` | ubuntu-24.04 | nightly + cargo-fuzz | `cargo fuzz run sort -- -max_total_time=180 -max_len=8192` |
| `golden` | ubuntu-24.04 | stable + gcc-13, cmake, ninja | build `sortbench` (`-DBRAINSORT_BUILD_TESTS=OFF`), `./build/sortbench --dump-datasets $RUNNER_TEMP/ds`, then `BRAINSORT_DATASETS=$RUNNER_TEMP/ds cargo test -p brainsort --features __internals --test golden -- --ignored --nocapture` |
| `large` | ubuntu-24.04 | stable | `cargo test -p brainsort --release -- --ignored large` (the 10M test) |
| `bench-smoke` | ubuntu-24.04 | stable | `cargo bench -p brainsort --bench api -- --test` so the bench cannot rot |
| `semver` | ubuntu-24.04, PRs only | stable + cargo-semver-checks | `cargo semver-checks check-release -p brainsort` (enable once 0.1.0 is on crates.io) |
| `deny` | ubuntu-24.04 | stable + cargo-deny | `cargo deny check` (licenses, advisories, bans); commit `deny.toml` |
| `package` | ubuntu-24.04 | stable | `cargo publish -p brainsort --dry-run` and `cargo package --list` diffed against a committed expected list, so a forgotten file fails CI |

`permissions: contents: read`, `concurrency` group like ci.yml, `fail-fast:
false`. Cache with `Swatinem/rust-cache` keyed on the job.

Release: `.github/workflows/rust-publish.yml` on tags `rust-v*`:
checks the tag matches `rust/brainsort/Cargo.toml` version, runs `cargo test`
once on Linux, then `cargo publish -p brainsort`. Authenticate with crates.io
Trusted Publishing (`rust-lang/crates-io-auth-action`, OIDC, `permissions:
id-token: write`); this needs the user to configure the trusted publisher on
crates.io for this repository after the first manual publish. Fallback:
`CARGO_REGISTRY_TOKEN` secret. Never store a token in the repo.

## 7. Publishing

### 7.1 Manifest (`rust/brainsort/Cargo.toml`)

```toml
[package]
name = "brainsort"
version = "0.0.1"                      # phase 0 placeholder; 0.1.0 at the first real release
edition = "2024"
rust-version = "1.86"
license = "MIT"
description = "Stable, exact sort for keys that map to an ordered integer, with runtime AVX2/BMI2 dispatch"
repository = "<FILL IN: the GitHub URL; the repo has no remote configured>"
documentation = "https://docs.rs/brainsort"
readme = "README.md"
keywords = ["sort", "sorting", "radix", "stable", "simd"]      # max 5
categories = ["algorithms", "no-std"]
include = ["src/**", "Cargo.toml", "README.md", "CHANGELOG.md", "LICENSE", "build.rs"]   # keep benches/fuzz/test data out; check with cargo package --list
authors = ["BrainFooLong"]             # email only if the user wants it public

[package.metadata.docs.rs]
all-features = true
rustdoc-args = ["--cfg", "docsrs"]

[features]
default = ["std"]
std = []
__internals = []

[dependencies]                          # none in the library. Keep it that way.

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
proptest = "1"
rdst = "0.20"
voracious_radix_sort = "1"
radsort = "0.1"

[[bench]]
name = "api"
harness = false
```

Copy `LICENSE` into `rust/brainsort/` (the package must carry the file).
Zero runtime dependencies is a selling point for a sort crate; do not add
any.

### 7.2 Versioning

- `0.0.1`: placeholder to reserve the name (phase 0). Publish a crate whose
  only public item is a doc comment and a `pub const VERSION`, so no one can
  depend on a half API. The description says "under development, see the
  repository".
- `0.1.0`: the first real release: the whole plan, golden test green, all CI
  green, benchmarks committed.
- Semver: 0.x means breaking changes bump the minor. `cargo semver-checks`
  in CI from 0.1.0 on.
- Yanking, never deleting. `cargo yank --version x.y.z` only for a broken
  release, with a `CHANGELOG` entry.
- The C++ library is 1.0.0. Do not sync the numbers; they are two packages.

### 7.3 Before `cargo publish`

- `cargo publish --dry-run -p brainsort`
- `cargo package --list -p brainsort` and read it: no test data, no bench
  results, no `target/`.
- `cargo doc --open` and read every public item's docs. The crate README is
  the docs.rs front page: put the five-line "Use it" example, the key table,
  the memory note and the honest results summary there, and link to the
  repository README for the rest.
- `cargo login` needs the user's crates.io token (section 10). The agent
  does not have it; stop and ask when reaching that step.

## 8. Documentation to update

- `rust/brainsort/README.md` (new): the crate front page, see 7.3.
- Root [README.md](../README.md): a "Rust" subsection under "Use it" with
  `cargo add brainsort` and the same five lines in Rust; a line in "Layout"
  for `rust/`; a "Results" pointer to the Rust benchmark tables. Keep the
  edits to a few lines; the README is under change in the working tree, so
  make these edits last and only in those spots.
- [setup.md](../setup.md): a section "Rust port: build, test, bench" with the
  commands from sections 5 and 6, and how to run the golden test locally
  (build sortbench, dump, set the env var).
- `rust/brainsort/CHANGELOG.md`: keep-a-changelog format; `0.0.1` and
  `0.1.0` entries.
- Root [CHANGELOG.md](../CHANGELOG.md): one line under a new "Rust" heading
  pointing at the crate changelog.
- `.gitignore`: add `rust/target/`, `rust/**/fuzz/corpus/`,
  `rust/**/fuzz/artifacts/`.

## 9. Phases and acceptance criteria

Each phase ends with a commit (ask the user before committing if in doubt; the
working tree has unrelated uncommitted changes, so always `git add` by path,
never `git add -A`).

**Phase 0: reserve the name.**
Workspace skeleton, `rust/brainsort` with docs-only `lib.rs`, manifest as in
7.1, `cargo publish --dry-run` clean. Publishing itself needs the user's
token: prepare everything and hand over the exact command. Done when
`https://crates.io/crates/brainsort` exists at 0.0.1.

**Phase 1: scalar core (the bulk of the work).**
Steps 1-6 of section 4. Done when `tests/api.rs`, `float_order.rs`,
`alloc_failure.rs`, `panic_safety.rs`, `shapes.rs` pass on x86-64 Linux and
Windows, with `--cfg brainsort_no_simd` and without (they are the same code
at this point), and `cargo msrv` confirms `rust-version`.

**Phase 2: golden equivalence.**
Section 5.2. Done when the 48 brainsort rows match on tier 1 columns, the
test runs in CI (`golden` job) and `results/counts-rust.csv` is written by
the bench crate. Record any column that cannot match and why in this file
under a new "Deviations" heading; the expectation is zero.

**Phase 3: SIMD.**
Step 8 of section 4. Done when the golden test passes on the vector build,
every dataset's output is bit-identical between the two builds, the
`test-scalar`, `test-avx2-static` and `asan` jobs are green, and a quick
benchmark shows the vector kernels are actually taken (a counter behind
`__internals`, or `perf`).

**Phase 4: hardening.**
Fuzz target + smoke test, Miri green, i686 and ARM64 green, `no_std` check
green, `deny.toml`, `cargo doc` clean under `-D warnings`, every public item
documented with an example that compiles as a doctest.

**Phase 5: benchmarks and docs.**
Section 5.5, committed result files, README/setup/CHANGELOG edits from
section 8. The numbers in the crate README come from these files and name
the machine and commit.

**Phase 6: release 0.1.0.**
Version bump, changelog, tag `rust-v0.1.0`, the publish workflow, trusted
publishing configured by the user. Done when docs.rs has built the crate and
`cargo add brainsort` in a fresh project sorts a `Vec<i32>`.

Definition of done for the whole plan: every job in section 6 exists and is
green on main; 0.1.0 is on crates.io; the golden test is part of CI; the
benchmark files are committed; the root README points at the crate.

## 10. Open items that need the user

1. The GitHub repository URL for `repository` in Cargo.toml (no git remote is
   configured in this checkout).
2. A crates.io token for `cargo login` at phase 0 and phase 6, or the user
   runs `cargo publish` themselves from the prepared tree. Trusted publishing
   setup on crates.io for the release workflow.
3. Whether the author email should appear in the manifest (`authors` is
   optional and public).
4. Whether `.github/workflows/ci.yml` may get `paths-ignore: [rust/**]`.

Do not block phases 1-5 on any of these; only phase 0's actual publish and
phase 6 need them.

## 11. Risks and traps

- **`is_x86_feature_detected!` is std-only.** Without `std` only compile-time
  features apply. Documented in 3.5; do not try to emulate cpuid in `no_std`.
- **Miri and AVX2.** Run Miri on the scalar cfg. Miri failures on the vector
  build are not actionable.
- **Rust's `f64` is not `Ord`.** The `Key` impl for floats defines the
  documented order; `cmp_key` must use the radix transform, never
  `partial_cmp`. Test NaN with several payloads and both signs.
- **Composite records without variadics.** Section 3.3. Measure before
  accepting the const-generic bucket approach; do not fall back to
  heap-allocated per-record parts.
- **Two-buffer alternation and the borrow checker.** The C++ passes two views
  `src`/`dst` over distinct buffers and swaps roles. In Rust pass two `&mut
  [T]` (or two raw views) that are provably disjoint (the scratch is a
  separate allocation; the split path's in-place compaction writes below the
  read cursor, which needs raw pointers with a SAFETY comment).
- **Allocation failure semantics.** `Vec` aborts on OOM. Every buffer goes
  through the crate's `Alloc` with `Result`. The alloc-failure test exists to
  catch a stray `Vec::with_capacity`.
- **Golden datasets.** Never regenerate them in Rust; consume the C++ dump.
  If the C++ golden file is regenerated (the repo convention: `sortbench
  --counts-only` after an intended change), the Rust test follows
  automatically because it reads the same CSV.
- **Benchmark honesty.** The C++ README's rule is that every claim is
  checked against upstream code as shipped, and its own ports are known to
  be 5-50% slower than upstream. In Rust the "upstream" is
  `slice::sort`/`sort_unstable` and the radix crates as published. Report
  losses. Do not compare against the C++ numbers across languages as if they
  were the same machine run.
- **Hidden features.** `__internals` is not semver-stable; say so in its doc
  comment and never mention it in the README.
- **The uncommitted working tree.** README.md, results/, scripts/ and
  src/bench.cpp are modified and uncommitted by other work. Your only C++
  change is the `--dump-datasets` flag in src/bench.cpp; make it small, and
  tell the user it touches a file with pending changes.

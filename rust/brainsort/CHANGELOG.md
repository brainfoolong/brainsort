# Changelog

The crate follows the version of the C++ library: one number, one
algorithm. Notable changes per version; the format is
[Keep a Changelog](https://keepachangelog.com/).

## Unreleased

- The split kernels (the median split and the partition sort, AVX2) read
  the reference key of their varying-bits mask before the vector loop
  compacts over the first element; the scalar tail had read it after,
  which could set or miss bits of the mask that decides which radix
  passes run (not reachable through the sort, whose split starts at a
  thousand elements). The kernels and the timed scalar splits overflow
  only on a buffered element, as the counted splits do, instead of on a
  buffer that is exactly full with kept elements left: a spurious retry
  less. Found by new unit tests that run every vector kernel against its
  scalar twin on exactly sized buffers, which the AddressSanitizer job
  now covers where Miri cannot.
- Documented what a comparator that is not a total order gets from
  `sort_by` and `sort_by_inferred`: an unspecified order, or the panic the
  standard library's sort documents, with every element in the slice
  exactly once. The fuzz body now feeds such comparators (random per call,
  a hash of the pair) and orders no byte window of the element expresses,
  on 16- and 32-byte elements, and repeats an input up to 64 times so the
  routes that start at thousands of elements are fuzzed.
- `sort` on a plain slice of keys of up to 32 bits (`i32`, `u32`, the 8-
  and 16-bit integers, `bool`, `char`, `f32`, any `Key` whose radix form
  inverts) sorts the keys themselves, 4 bytes per element instead of an
  8-byte record with an index, and writes them back; the memory of such a
  sort halves. `-0.0` is kept, as for `f64`. At 100,000 elements
  (7800X3D, Windows): `i32` random 0.40 to 0.37 ms, from 0.96x of the
  fastest radix crate to 1.07x; few distinct values 0.17 to 0.14 ms.
- `sort_by_inferred` verifies its result with one comparator call per
  pair of neighbours that share the window key, and stops testing
  narrower windows once a wider one agrees. Same result on every input.
- The crate compiles on Rust 1.86, its stated minimum, again (a `let`
  chain had crept into `sort_by_inferred`).
- `sort_by_inferred` and the `PlainBytes` trait: a comparator sort that
  guesses the key the comparator compares (an aligned window of the
  element read as an integer or a float, ascending or descending) from a
  sample of its answers, sorts by that key and verifies the result with
  one comparator pass, falling back to `sort_by` on the untouched slice
  when no window agrees; the same result as `sort_by` on every input, and
  `sort_by` itself below 4,096 elements. Primitive numbers and arrays of
  them are `PlainBytes`; a struct without padding opts in with `unsafe
  impl`. At 100,000 elements (7800X3D, Windows) against `sort_by`: `i32`
  random 1.23 to 0.61 ms, from 0.72x of `slice::sort_unstable_by` to 1.5x
  ahead; 64-byte rows by comparator random 2.4 to 1.5 ms; on WSL2 1.22 to
  0.59 ms and 2.4 to 1.4 ms.
- `sort` on a plain slice of 64-bit keys (`i64`, `u64`, `isize`, `usize`,
  `f64`, raw pointers, any `Key` whose radix form inverts) sorts the keys
  themselves, 8 bytes per element instead of a 16-byte record with an
  index, and writes them back; the memory of such a sort halves. `-0.0`
  is kept: the negative zeros are put back by their rank among the zeros.
  At 100,000 elements (7800X3D, Windows): `i64` random 0.99 to about
  0.75 ms, `f64` random 0.98 to about 0.8 ms, few distinct values 0.22 to
  0.17 and 0.30 to 0.23 ms; the random rows went from 0.90x of the fastest
  radix crate to level or ahead; on WSL2 `i64` random 0.92 to 0.80 ms and
  `f64` random 0.95 to 0.77 ms.
- `sort_by` sorts elements over 16 bytes through an array of indices and
  permutes them once, as the C++ comparator overload does: the comparison
  sort moves 4 bytes per element instead of the element, and a comparator
  that panics there leaves the slice unchanged. 64-byte rows by comparator
  at 100,000 elements (7800X3D): random 5.7 to 2.6 ms on Windows and 4.2
  to 2.5 ms on WSL2, few distinct values 3.2 to 1.2 ms and 2.1 to 1.4 ms,
  from behind `slice::sort_unstable_by` to level with it or ahead.
- `sort_by` takes the displaced-element route for nearly sorted elements
  of up to 64 bytes, as the C++ comparator overload does; it had stopped
  at 16, the key path's limit, and sent such input to the full sort.
- The prescans walk the non-descending prefix and the non-ascending run
  after the first descent with one compare and one branch per pair, and
  count only after that; sorted and reversed input costs the same as the
  standard library's run detection. Every decision is unchanged.

- The AVX2 prescan of plain slices takes the key kind as a compile-time
  constant, as the C++ template does; the run-time match inside its loop
  had cost about 1.7 times the C++ pass, which is what the sorted and
  reversed cells of the API benchmark paid.

## 0.3.0

The first release of the Rust port.

- `sort`, `sort_by_key`, `sort_by_key_ref` and `sort_by`, all stable.
- The `Key` trait for every integer, `bool`, `char`, `f32`, `f64`,
  `Duration`, raw pointers, `str`, `String`, `[u8]`, `Vec<u8>`, `CStr`,
  `OsStr`, `Path` and their owned forms, tuples of up to twelve keys,
  arrays of keys, `Desc` and `Reverse`; the `fixed_key!` and `bytes_key!`
  macros for user types.
- The same algorithm as C++ brainsort 0.3.0, verified against its golden
  work counts; AVX2 and BMI2 kernels chosen at run time on x86-64.
- `no_std` + `alloc` with the `std` feature off.

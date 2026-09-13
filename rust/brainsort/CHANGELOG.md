# Changelog

The crate follows the version of the C++ library: one number, one
algorithm. Notable changes per version; the format is
[Keep a Changelog](https://keepachangelog.com/).

## Unreleased

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

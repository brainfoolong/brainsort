# Changelog

The crate follows the version of the C++ library: one number, one
algorithm. Notable changes per version; the format is
[Keep a Changelog](https://keepachangelog.com/).

## Unreleased

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

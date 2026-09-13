# brainsort

A stable sort for keys that map to an ordered integer: numbers, strings,
dates, ids. It avoids comparing keys wherever the key type allows it and
does less work when the input already has structure. No dependencies,
`no_std` + `alloc`, AVX2 and BMI2 chosen at run time on x86-64.

This is the Rust port of the C++ library of the same name. It runs the
same algorithm step for step: the test suite recomputes the C++ golden
file of work counts (every element read and write, every comparison and
table access, the peak memory and the exact access sequence) through the
Rust code and requires every number to match. Results, both languages,
every machine: [brainfoolong.github.io/brainsort](https://brainfoolong.github.io/brainsort/).

```toml
[dependencies]
brainsort = "0.3"
```

```rust
let mut v = vec![3, 1, 2];
brainsort::sort(&mut v);                                         // stable, by value

struct Row { id: u64, name: String, score: f64 }
let mut rows: Vec<Row> = vec![];
brainsort::sort_by_key(&mut rows, |r| r.id);                     // by a key
brainsort::sort_by_key_ref(&mut rows, |r| r.name.as_str());      // by a borrowed key
brainsort::sort_by_key(&mut rows, |r| (r.id, brainsort::Desc(r.score)));
brainsort::sort_by(&mut rows, |a, b| a.name.cmp(&b.name));       // a comparator
let mut ids = vec![3u64, 1, 2];
brainsort::sort_by_inferred(&mut ids, |a, b| b.cmp(a));          // a comparator whose key is inferred
```

Every sort is stable. Keys can be any integer, `bool`, `char`, `f32`,
`f64`, `Duration`, raw pointer, `str`, `String`, `[u8]`, `Vec<u8>`,
`CStr`, `OsStr`, `Path` and their owned forms, a tuple or array of those,
or `Desc(key)` for a reversed order; your own key type implements `Key`
(the `fixed_key!` and `bytes_key!` macros do it in one line). Elements can
be anything: they are permuted once, after the keys were sorted.

## What to know

- **Memory.** About 1.5 small records per element while sorting (8 bytes
  for keys up to 32 bits, 16 bytes up to 64 bits or a string, more for
  composite keys), plus one element per element for the final permutation.
  Sorted and reversed input, and nearly sorted small elements, need no
  memory beyond the displaced elements. Freed blocks of 64 KiB and more are
  kept, up to 32 MiB, for the next sort; `release_memory()` frees them,
  `set_memory_cache_limit(0)` disables the cache.
- **Floating point.** `-0.0 == +0.0`; a NaN with the sign bit clear sorts
  after `+inf`, with it set before `-inf`. A total order, so no input is
  unsafe; it is not `f64::total_cmp`, which orders `-0.0` before `+0.0`.
- **A comparator** gets a comparison sort, with sorted, reversed and
  nearly sorted input handled on the elements first: long natural runs
  are merged as they are, anything else is the standard library's stable
  sort; elements over 16 bytes are sorted through indices and moved once.
  `sort_by_inferred` first guesses the window of the element the
  comparator compares, from a sample of its answers, sorts by it as a key
  and verifies; the elements must be `PlainBytes` (no padding).
- **Allocation failure** completes the sort through the standard library's
  stable sort with the same order. A key function that panics during the
  first pass leaves the slice unchanged; later, every element is still in
  the slice, in an unspecified order.
- **Limits.** At most 2^32 - 1 elements per call and strings shorter than
  2^32 bytes; beyond either, the standard library's stable sort.

## Results in short

On plain vectors against the stable Rust sorts as shipped (`slice::sort`,
`slice::sort_by_cached_key`, `glidesort`), on the website's machines:
brainsort wins on random, nearly sorted, few-unique, string and struct
inputs, most by two to four times, and loses on input that is already
sorted, where its first pass over the keys costs more than theirs. Every
cell is on the website with the machine and the commit it was measured on.

## Features

- `std` (default): run-time CPU feature detection, a mutex for the memory
  cache, `OsStr` and `Path` keys. Without it the crate is `no_std` +
  `alloc` and the vector paths are used only when enabled at compile time
  (`-C target-feature=+avx2,+bmi2`).

Rust 1.86 or newer. MIT license.

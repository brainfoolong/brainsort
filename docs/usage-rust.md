# Using brainsort from Rust

Written for: a Rust developer adding the crate to a project. The C++ guide
is [usage-cpp.md](usage-cpp.md); the API reference is on
[docs.rs/brainsort](https://docs.rs/brainsort).

## Get it

```sh
cargo add brainsort
```

No dependencies. The crate is `no_std` + `alloc` with the `std` feature
off (`default-features = false`); with `std` (the default) it detects AVX2
and BMI2 at run time on x86-64 and supports `OsStr` and `Path` keys.
Rust 1.86 or newer.

## Sort

Every function is stable: elements with equal keys keep their input order.
They take a `&mut [T]`, so a `Vec<T>`, an array or any slice works.

```rust
let mut v = vec![3, 1, 2];
brainsort::sort(&mut v);                       // by the values
```

By a key computed from each element, called once per element:

```rust
struct Row { id: u64, name: String, score: f64 }
let mut rows: Vec<Row> = vec![];
brainsort::sort_by_key(&mut rows, |r| r.id);
brainsort::sort_by_key(&mut rows, |r| (r.id, brainsort::Desc(r.score)));   // then by score, descending
```

By a key borrowed from the element, such as a string field, so nothing is
copied:

```rust
brainsort::sort_by_key_ref(&mut rows, |r| r.name.as_str());
brainsort::sort_by_key_ref(&mut rows, |r| &r.id);
```

`sort_by_key` and `sort_by_key_ref` exist because Rust cannot express "a
closure that returns either an owned key or a borrow of its argument" in
one signature (the same limit makes `slice::sort_by_key` unable to return
`&str`). Owned keys are computed once and kept for the duration of the
sort; borrowed keys are read in place.

With a comparator, when the order is not a key:

```rust
brainsort::sort_by(&mut rows, |a, b| a.name.cmp(&b.name));
```

A comparator gets a comparison sort: sorted, reversed and nearly sorted
input is handled on the elements as the key sorts do, everything else is
the standard library's `slice::sort_by`, on the elements themselves up to
16 bytes and on an array of indices beyond that, so the passes move 4
bytes per element and each element moves once, at the end. Prefer a key
whenever the order is one; that is where the radix wins are.

When the comparator is the order of a field but the call site only has
the comparator, `sort_by_inferred` finds the field, from 4,096 elements
on: it asks the comparator about a sample of adjacent pairs, looks for an aligned window
of the element (an integer or a float of 8 to 64 bits, ascending or
descending) whose order agrees with every answer, sorts by that window
as a key and checks the result with one more comparator pass, putting
runs the comparator calls equal back into input order. If no window
agrees or the check fails, the slice is untouched and `sort_by` takes
over, so the result is always that of `sort_by`. The elements must be
`PlainBytes`, every byte initialised: the primitive numbers and arrays
of them are, a struct opts in with `unsafe impl` once it has no padding.
`sort_by` cannot do this on its own because a slice of an arbitrary type
may hold padding bytes it must not read.

```rust
#[derive(Clone, Copy)]
#[repr(C)]
struct Point { x: f64, y: f64, id: u64 }
// SAFETY: three 8-byte fields, no padding.
unsafe impl brainsort::PlainBytes for Point {}
brainsort::sort_by_inferred(&mut points, |a, b| a.y.total_cmp(&b.y));
```

## Keys

| key | order |
|---|---|
| every integer (`i8` to `i128`, `u8` to `u128`, `isize`, `usize`), `bool`, `char` | numeric |
| `f32`, `f64` | numeric; `-0.0 == +0.0`; NaN with the sign bit clear after `+inf`, with it set before `-inf` |
| `core::time::Duration` | by duration |
| `*const T`, `*mut T` | by address |
| `str`, `String`, `[u8]`, `Vec<u8>`, `Box<str>`, `Rc<str>`, `Arc<str>`, `Cow<str>`, `CStr`, `CString`, `OsStr`, `OsString`, `Path`, `PathBuf` | lexicographic, unsigned bytes, shorter prefix first |
| `Wrapping<T>`, `NonZero*` | as the integer |
| tuples of up to twelve keys, `[K; N]` | lexicographic, member by member |
| `&K`, `&mut K`, `Box<K>`, `Rc<K>`, `Arc<K>` | as the key |
| `Desc<K>`, `core::cmp::Reverse<K>` | the reverse of any of the above |

The floating-point order is a total order, so no input is unsafe; it is
not `f64::total_cmp`, which orders `-0.0` before `+0.0`.

Your own key type implements `Key`. The two macros cover the common
shapes:

```rust
#[derive(Clone, Copy)]
struct UserId(u32);
brainsort::fixed_key!(UserId, 32, |k| k.0 as u64);          // 32 bits, order-preserving u64

struct Tag { name: String }
brainsort::bytes_key!(Tag, |k| k.name.as_bytes());          // a byte string
```

A composite of several keys is a tuple or array of them, up to 32 leaves
in total; `Option<K>` is not a key (decide where `None` goes and map it).
Implementing `Key` by hand (a `Shape`, a `Slots` type, `write_parts` and
`cmp_key`) is documented on docs.rs.

## Elements

Anything. The keys are sorted first, in an array of small records, then
the elements are permuted once with bitwise moves; no element is cloned or
dropped on the way. Plain slices of integers, `f32` and `f64` are sorted
as bare keys, without an index, and written back.

## What to know

- **Memory.** About 1.5 small records per element while sorting (4 bytes
  for a plain slice of numbers of up to 32 bits, 8 bytes for other keys up
  to 32 bits and for a plain slice of 64-bit numbers, 16 bytes for other
  keys up to 64 bits or a string, more for composites), plus one
  element per element for the permutation; a
  comparator on elements over 16 bytes uses 4 bytes per element for the
  indices instead of the records. Sorted and reversed input, and nearly
  sorted elements of up to 16 bytes (64 with a comparator), need no
  memory beyond the displaced elements.
- **Memory is kept.** Freed blocks of 64 KiB and more are kept, up to
  32 MiB, for the next sort of the process. `set_memory_cache_limit(bytes)`
  changes the limit (0 disables the cache); `release_memory()` frees the
  blocks at any time. Without `std` the cache is behind a spin lock.
- **Failure.** If an allocation fails the sort completes through the
  standard library's stable sort; allocation never aborts inside the
  library. A key function that panics during the first pass over the keys
  leaves the slice unchanged; one that panics later (the nearly sorted
  route compares elements again), or a comparator that panics, leaves
  every element in the slice, exactly once, in an unspecified order; the
  comparison sort of elements over 16 bytes runs on indices and moves
  nothing before its last comparison. A comparator that is not a total
  order gets an unspecified order, or the panic the standard library's
  sort documents for it; every element stays in the slice exactly once.
- **Limits.** At most 2^32 - 1 elements per call and strings shorter than
  2^32 bytes; beyond either the call goes to the standard library's
  stable sort.
- **Threads.** Concurrent sorts are fine; the block cache is the only
  shared state, under its own lock.

## Build configuration

- `--cfg brainsort_no_simd` (in `RUSTFLAGS`): compile the scalar code on
  x86-64 too, what every other architecture runs.
- `-C target-feature=+avx2,+bmi2`: use the vector kernels without run-time
  detection; the only way to get them without the `std` feature.

The same algorithm as the C++ library: the test suite recomputes the C++
golden work counts through the Rust code and requires every number to
match. How it works is described in the [README](../README.md#how-it-works)
and in [docs/decisions/](decisions/).

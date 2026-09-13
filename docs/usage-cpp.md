# Using brainsort from C++

Written for: a C++ developer adding the library to a project. The Rust
guide is [usage-rust.md](usage-rust.md); the contract in full is in the
[README](../README.md#the-library).

## Get it

One header, C++20, no dependencies. Either copy
[single_include/brainsort.hpp](../single_include/brainsort.hpp) next to
your sources:

```cpp
#include "brainsort.hpp"
```

or add `include/` to the include path and include the modular header:

```cpp
#include "brainsort/brainsort.hpp"
```

With CMake, `add_subdirectory(brainsort)` and link `brainsort::brainsort`;
the target is an interface library that only sets the include path. GCC,
Clang, MSVC and Apple Clang are supported; on x86-64 the AVX2 and BMI2
paths are chosen at run time, every other architecture runs the same
algorithm without the vector kernels.

## Sort

Every call is stable: elements with equal keys keep their input order.

```cpp
std::vector<int> v = {3, 1, 2};
brainsort::sort(v);                          // by the values
brainsort::sort(v.begin(), v.end());         // any random-access range: vector, deque, array, span, pointers
```

By a key computed from each element:

```cpp
struct Row { long id; std::string name; double score; };
std::vector<Row> rows = ...;
brainsort::sort(rows, [](const Row& r) { return r.id; });
brainsort::sort(rows, [](const Row& r) -> const std::string& { return r.name; });   // by reference: no copy
brainsort::sort(rows, [](const Row& r) { return std::pair(r.id, brainsort::desc(r.score)); });
```

A projection that returns a reference is read in place; one that returns
a value is called once per element and the values are kept for the
duration of the sort. `sort_by_key` is the same as the projection form of
`sort`, for code that wants to say so. `brainsort::desc(key)` reverses the
order of any key, also inside a pair or tuple.

With a comparator, when the order cannot be expressed as a key:

```cpp
brainsort::sort(rows, [](const Row& a, const Row& b) { return a.name < b.name; });
brainsort::sort_with(rows, [](const Row& a, const Row& b) { return a.id > b.id; });
```

A comparator on trivially copyable elements, from 4,096 of them on, first
has its key inferred: the library asks it about a sample of adjacent pairs, looks for an
aligned window of the element (an integer or a floating-point number of
8 to 64 bits, ascending or descending) whose order agrees with every
answer, sorts by that window as a key and checks the result with one
more comparator pass, putting runs the comparator calls equal back into
input order. A comparator that is the order of a field thus gets the
radix sort; one that is not (the check fails, or no window agrees) costs
a sample and at worst one verification pass, and the range, still
untouched, goes to the library's comparison sort: a stable partition sort
that merges long runs as they are and strips repeated values in one
pass, with elements over 16 bytes sorted through an index array and
moved once. Elements that are not trivially copyable go to
`std::stable_sort`. Every path is stable and handles sorted, reversed and
nearly sorted input in place; prefer a key whenever you have one, it
skips the guessing.

`brainsort::stable_sort` is an alias of `brainsort::sort` for code that
replaces `std::stable_sort` by name.

## Keys

| key | order |
|---|---|
| `bool`, every integer, `char`, `wchar_t`, `char8_t`, `char16_t`, `char32_t` | numeric |
| `float`, `double` | numeric; `-0.0 == +0.0`; NaN with the sign bit clear after `+inf`, with it set before `-inf` |
| enums | by the underlying integer |
| pointers (except `char*`) | by address |
| `std::string`, `std::string_view` (`char` and `char8_t`), `const char*`, `char[N]` | lexicographic, unsigned bytes, shorter prefix first |
| `std::chrono::duration`, `std::chrono::time_point` | by tick count |
| `std::pair`, `std::tuple`, `std::array` of keys | lexicographic, member by member |
| `brainsort::desc(key)` | the reverse of any of the above |

Your own key type takes a `brainsort::key_traits` specialisation, in one of
two shapes:

```cpp
struct UserId { uint32_t value; };
struct Tag { std::string name; };

namespace brainsort {
template <> struct key_traits<UserId> {              // a fixed-width key
    static constexpr key_kind kind       = key_kind::fixed;
    static constexpr int      radix_bits = 32;         // 1..64
    using radix_type = uint32_t;                       // uint32_t up to 32 bits, else uint64_t
    static constexpr bool     exact      = false;      // true if from_radix exists and inverts to_radix
    static uint32_t to_radix(const UserId& k) noexcept { return k.value; }   // order-preserving
};
template <> struct key_traits<Tag> {                 // a byte-string key
    static constexpr key_kind kind   = key_kind::bytes;
    static constexpr bool     owning = true;           // the key object owns its bytes
    static std::string_view bytes(const Tag& t) noexcept { return t.name; }
};
}
```

A composite of several keys is a `std::pair`, `std::tuple` or `std::array`
of them; up to 32 leaves in total.

## Elements

Anything move-assignable. The keys are sorted first, in an array of small
records, then the elements are permuted once; element size does not
matter for the sort itself. Plain arrays of integers and floating-point
numbers are sorted as bare keys, without an index, and written back.

## What to know

- **Memory.** About 1.5 small records per element while sorting (4 bytes
  for a plain array of numbers of up to 32 bits, 8 bytes for other keys up
  to 32 bits and for a plain array of 64-bit numbers, 16 bytes for other
  keys up to 64 bits or a string, more for composites), plus one
  element per element for the permutation of trivially copyable elements. Sorted and reversed input, and nearly
  sorted small elements, need no memory beyond the displaced elements.
- **Memory is kept.** Freed blocks of 64 KiB and more are kept, up to
  32 MiB, for the next sort of the process. `BRAINSORT_MEMORY_CACHE`
  (defined before the include) sets the limit in bytes, 0 disables it;
  `brainsort::release_memory()` frees the blocks at any time.
- **Failure.** If an allocation fails the sort completes through
  `std::stable_sort`. A projection that throws during the first pass over
  the keys leaves the range unchanged; one that throws later, or a
  comparator that throws, leaves every element in the range in an
  unspecified order. Nothing else throws.
- **Limits.** At most 2^32 - 1 elements per call and strings shorter than
  2^32 bytes; beyond either the call goes to `std::stable_sort`.
- **Threads.** Concurrent sorts are fine; the block cache is the only
  shared state, under its own mutex.

## Build flags

- `BRAINSORT_NO_SIMD`: compile the scalar code on x86-64 too (what every
  other architecture runs; the test suite builds once each way).
- `BRAINSORT_MEMORY_CACHE`: the byte limit of the block cache.

The algorithm and its decisions are described in the
[README](../README.md#how-it-works) and in [docs/decisions/](decisions/).

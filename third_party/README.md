# Upstream reference implementations

These files are the sorting implementations that real software ships, vendored
**unmodified** so the benchmark can run them as opponents under the same
compiler, flags, inputs and process isolation as everything else. Do not edit
them; if a newer upstream version is wanted, replace the whole file and update
this table.

| file | project | version / commit | license |
|---|---|---|---|
| `pdqsort/pdqsort.h` | [orlp/pdqsort](https://github.com/orlp/pdqsort) | `b1ef26a55cdb60d236a5cb199c4234c704f46726` (2021-03-14), sha256 `1916aff2...237fc0` | zlib |
| `gfx/timsort.hpp` | [timsort/cpp-TimSort](https://github.com/timsort/cpp-TimSort) | 2.1.0, `978afc12e37bf919651472f8e308303be470884e` (2024-11-28), sha256 `a5cf0f0f...5e9c9` | MIT |

Both licenses are reproduced in the file headers. The benchmark calls them
exactly as an application would: `pdqsort(first, last, comp)`,
`pdqsort_branchless(first, last, comp)` and `gfx::timsort(first, last, comp)`
with the same key comparator that `std::sort` and our own ports use.

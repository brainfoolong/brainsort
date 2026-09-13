# Upstream reference implementations

These files are the sorting implementations that real software ships,
vendored **unmodified** so the benchmark can run them as opponents under the
same compiler, flags, inputs and process isolation as everything else. Do not
edit them; if a newer upstream version is wanted, replace the whole file (or
the whole directory) and update this table.

| files | project | version / commit | license |
|---|---|---|---|
| `gfx/timsort.hpp` | [timsort/cpp-TimSort](https://github.com/timsort/cpp-TimSort) | 2.1.0, `978afc12e37bf919651472f8e308303be470884e` (2024-11-28), sha256 `a5cf0f0f...5e9c9` | MIT |
| `boost/sort/` (13 headers: `spinsort/`, `flat_stable_sort/`, `insert_sort/`, `common/`) | [boostorg/sort](https://github.com/boostorg/sort) | Boost 1.92.0, tag `boost-1.92.0`, `48c424bee18ee2d886f83571cc592f0319b501e3` | BSL-1.0 (`boost/LICENSE_1_0.txt`) |

The Boost.Sort set is exactly the include closure of `spinsort.hpp` and
`flat_stable_sort.hpp`; it depends on no other Boost library. The licenses
are reproduced in the file headers. The benchmark calls the code exactly as
an application would: `gfx::timsort(first, last, comp)`,
`boost::sort::spinsort(first, last, comp)` and
`boost::sort::flat_stable_sort(first, last, comp)` with the same key
comparator that `std::stable_sort` gets. Boost.Sort takes its scratch from
`std::malloc`; the benchmark includes its headers with that name and
`std::free` redirected to counted twins (see `include/sortbench/registry.hpp`),
so the code stays as shipped and its scratch is counted like everyone else's.

The pool is the C++ sorts that can do what brainsort does: stable, every key
type, an arbitrary comparator ([docs/decisions/0017.md](../docs/decisions/0017.md)).

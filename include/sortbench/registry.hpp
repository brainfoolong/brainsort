// Table of algorithms under test, per element type. Each entry provides an
// uncounted entry point (used for timing / hardware counters) and a counted
// one (used for access counts and the correctness check of the instrumented
// path). The opponents are upstream code, vendored or taken from the
// toolchain as shipped, and counted through an element wrapper whose copy
// operations and comparator operands report to g_trace, so the same
// definitions of "read", "write" and "compare" hold for every algorithm.
//
// The pool is what brainsort can be compared with like for like: every
// opponent is stable, sorts every key type of the harness and takes an
// arbitrary comparator (docs/decisions/0017.md). Nothing here is our own
// implementation of somebody else's algorithm.
#pragma once
#include "sortbench/algorithms/brainsort.hpp"
#include "sortbench/core.hpp"

// Upstream reference implementations, vendored unmodified (see third_party/README.md).
#include "gfx/timsort.hpp"

// Boost.Sort takes its scratch memory from std::malloc, which the global
// operator new replacement of trace_alloc.hpp does not see. Its headers are
// included with the two names redirected to counted twins, so the Boost code
// stays exactly as shipped and its scratch is counted like everyone else's:
// the same peak, the same allocation count, a region for the cache model.
// Every standard header Boost.Sort includes is included first, so the macros
// meet only Boost's own calls.
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>
namespace sb::registry_detail {
inline void* counted_malloc(std::size_t n) {
    void* p = std::malloc(n ? n : 1);
    if (p) g_trace.on_alloc(p, n);
    return p;
}
inline void counted_free(void* p) {
    if (!p) return;
    g_trace.on_free(p);
    std::free(p);
}
}  // namespace sb::registry_detail
namespace std {
using sb::registry_detail::counted_free;
using sb::registry_detail::counted_malloc;
}  // namespace std
#define malloc counted_malloc
#define free counted_free
#include "boost/sort/flat_stable_sort/flat_stable_sort.hpp"
#include "boost/sort/spinsort/spinsort.hpp"
#undef malloc
#undef free

#include <cstring>
#include <string>

namespace sb {

template <class T>
struct AlgoInfo {
    const char* name;
    const char* used_by;         // where this algorithm ships in modern software
    bool        stable;
    bool        comparison_sort; // false for radix-based sorts (may do zero comparisons)
    bool        counts_accesses; // reads/writes/scratch are counted (true for every entry now; kept for the reports)
    void (*run_raw)(T*, size_t);     // uncounted
    void (*run_counted)(T*, size_t); // counted (reads/writes/compares via g_counters)
};

namespace registry_detail {

template <class T, void (*F)(Array<T, false>), void (*G)(Array<T, true>)>
struct Wrap {
    static void raw(T* p, size_t n)     { F(Array<T, false>(p, n)); }
    static void counted(T* p, size_t n) { G(Array<T, true>(p, n)); }
};

// Function objects (not function pointers) so the comparison is inlined into
// the upstream algorithms exactly as a user's comparator would be.
template <class T> struct KeyLess {
    bool operator()(const T& a, const T& b) const { return KeyTraits<T>::less(a, b); }
};
// The counted path of the upstream code runs on Traced<T>: the same bytes,
// but every copy or move of an element reports its source and destination
// to g_trace, which counts a read when the source is a buffer (the array or
// heap scratch) and a write when the destination is one; a temporary on the
// stack is neither. The comparator reports its operands the same way. That
// makes the upstream counts mean exactly what the Array view's do: a swap is
// two reads and two writes, a compare of an array element against a local
// pivot is one read.
template <class T> struct Traced {
    T v;
    Traced() = default;
    Traced(const Traced& o) : v(o.v)             { g_trace.on_move(&o, this, sizeof(T)); }
    Traced(Traced&& o) noexcept : v(o.v)         { g_trace.on_move(&o, this, sizeof(T)); }
    Traced& operator=(const Traced& o)           { v = o.v; g_trace.on_move(&o, this, sizeof(T)); return *this; }
    Traced& operator=(Traced&& o) noexcept       { v = o.v; g_trace.on_move(&o, this, sizeof(T)); return *this; }
};
static_assert(sizeof(Traced<Item>) == sizeof(Item) && alignof(Traced<Item>) == alignof(Item));
static_assert(sizeof(Traced<DblItem>) == sizeof(DblItem) && sizeof(Traced<I64Item>) == sizeof(I64Item) && sizeof(Traced<StrItem>) == sizeof(StrItem));
template <class T> struct TracedLess {
    bool operator()(const Traced<T>& a, const Traced<T>& b) const {
        g_trace.on_operand(&a, sizeof(T));
        g_trace.on_operand(&b, sizeof(T));
        return g_trace.counted_less(a.v, b.v);
    }
};
// Traced<T> is a standard-layout wrapper with T as its only member, so the
// array is sorted in place through the wrapper type; nothing is copied.
template <class T> Traced<T>* traced(T* p) { return reinterpret_cast<Traced<T>*>(p); }

// The toolchain's std::stable_sort: libstdc++, libc++ or the MSVC STL,
// whichever the build uses.
template <class T> void std_stable_sort_raw(T* p, size_t n)     { std::stable_sort(p, p + n, KeyLess<T>{}); }
template <class T> void std_stable_sort_counted(T* p, size_t n) { std::stable_sort(traced(p), traced(p) + n, TracedLess<T>{}); }
// gfx/timsort.hpp: the widely used C++ port of CPython's listobject.c / OpenJDK's TimSort.java.
template <class T> void timsort_ref_raw(T* p, size_t n)         { gfx::timsort(p, p + n, KeyLess<T>{}); }
template <class T> void timsort_ref_counted(T* p, size_t n)     { gfx::timsort(traced(p), traced(p) + n, TracedLess<T>{}); }
// Boost.Sort's two single-threaded stable sorts.
template <class T> void spinsort_raw(T* p, size_t n)            { boost::sort::spinsort(p, p + n, KeyLess<T>{}); }
template <class T> void spinsort_counted(T* p, size_t n)        { boost::sort::spinsort(traced(p), traced(p) + n, TracedLess<T>{}); }
template <class T> void flat_stable_raw(T* p, size_t n)         { boost::sort::flat_stable_sort(p, p + n, KeyLess<T>{}); }
template <class T> void flat_stable_counted(T* p, size_t n)     { boost::sort::flat_stable_sort(traced(p), traced(p) + n, TracedLess<T>{}); }

#define SB_ENTRY(fn) \
    registry_detail::Wrap<T, &fn<Array<T, false>>, &fn<Array<T, true>>>::raw, \
    registry_detail::Wrap<T, &fn<Array<T, false>>, &fn<Array<T, true>>>::counted

template <class T>
std::vector<AlgoInfo<T>> make_algorithms() {
    return {
        // The candidate, written against the counted Array view so reads,
        // writes and scratch memory are measured from the inside.
        {"brainsort",               "candidate: scout pass -> sorted/reverse/runs/displaced-merge/compressed radix", true, false, true, SB_ENTRY(brainsort)},
        // Upstream code, compiled as shipped; counted through the Traced<T>
        // wrapper. These are the opponents every claim is measured against.
        {"std::stable_sort",        "the standard library's stable sort, as shipped (libstdc++, libc++ or MSVC STL)", true, true, true, registry_detail::std_stable_sort_raw<T>, registry_detail::std_stable_sort_counted<T>},
        {"gfx::timsort",            "upstream cpp-TimSort 2.1.0, C++ port of CPython/OpenJDK TimSort",                true, true, true, registry_detail::timsort_ref_raw<T>,     registry_detail::timsort_ref_counted<T>},
        {"boost::spinsort",         "upstream Boost.Sort spinsort, n/2 buffer",                                        true, true, true, registry_detail::spinsort_raw<T>,        registry_detail::spinsort_counted<T>},
        {"boost::flat_stable_sort", "upstream Boost.Sort flat_stable_sort, n/256 + 8 KiB buffer",                     true, true, true, registry_detail::flat_stable_raw<T>,     registry_detail::flat_stable_counted<T>},
    };
}
#undef SB_ENTRY

}  // namespace registry_detail

template <class T>
inline const std::vector<AlgoInfo<T>>& algorithms() {
    static const std::vector<AlgoInfo<T>> v = registry_detail::make_algorithms<T>();
    return v;
}

template <class T>
inline const AlgoInfo<T>* find_algorithm(const std::string& name) {
    for (const auto& a : algorithms<T>()) if (name == a.name) return &a;
    return nullptr;
}

// Calls f(TypeTag<T>{}) for the element type named `type`. Returns false if
// the name is unknown.
template <class T> struct TypeTag { using type = T; };

template <class F>
inline bool with_type(const std::string& type, F&& f) {
    if (type == "int32")  { f(TypeTag<Item>{});    return true; }
    if (type == "double") { f(TypeTag<DblItem>{}); return true; }
    if (type == "int64")  { f(TypeTag<I64Item>{}); return true; }
    if (type == "string") { f(TypeTag<StrItem>{}); return true; }
    return false;
}

}  // namespace sb

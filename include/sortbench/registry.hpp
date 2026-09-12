// Table of algorithms under test, per element type. Each entry provides an
// uncounted entry point (used for timing / hardware counters) and a counted
// one (used for access counts and the correctness check of the instrumented
// path). The upstream code is counted through an element wrapper whose copy
// operations and comparator operands report to g_trace, so the same
// definitions of "read", "write" and "compare" hold for every algorithm.
#pragma once
#include "sortbench/algorithms/brainsort.hpp"
#include "sortbench/algorithms/heapsort.hpp"
#include "sortbench/algorithms/introsort.hpp"
#include "sortbench/algorithms/mergesort.hpp"
#include "sortbench/algorithms/pdqsort.hpp"
#include "sortbench/algorithms/radix.hpp"
#include "sortbench/algorithms/timsort.hpp"
#include "sortbench/core.hpp"

// Upstream reference implementations, vendored unmodified (see third_party/README.md).
#include "gfx/timsort.hpp"
#include "pdqsort/pdqsort.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

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
// the std:: algorithms exactly as it is in our own implementations.
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

template <class T> void std_sort_raw(T* p, size_t n)            { std::sort(p, p + n, KeyLess<T>{}); }
template <class T> void std_sort_counted(T* p, size_t n)        { std::sort(traced(p), traced(p) + n, TracedLess<T>{}); }
template <class T> void std_stable_sort_raw(T* p, size_t n)     { std::stable_sort(p, p + n, KeyLess<T>{}); }
template <class T> void std_stable_sort_counted(T* p, size_t n) { std::stable_sort(traced(p), traced(p) + n, TracedLess<T>{}); }
// orlp/pdqsort.h. With a user comparator the upstream header selects its
// "non-branchless" partition; the branchless block partition is what it
// selects on its own for arithmetic keys with std::less, and what Rust's
// sort_unstable used, so both are run.
template <class T> void pdq_ref_raw(T* p, size_t n)                { ::pdqsort(p, p + n, KeyLess<T>{}); }
template <class T> void pdq_ref_counted(T* p, size_t n)            { ::pdqsort(traced(p), traced(p) + n, TracedLess<T>{}); }
template <class T> void pdq_branchless_ref_raw(T* p, size_t n)     { ::pdqsort_branchless(p, p + n, KeyLess<T>{}); }
template <class T> void pdq_branchless_ref_counted(T* p, size_t n) { ::pdqsort_branchless(traced(p), traced(p) + n, TracedLess<T>{}); }
// gfx/timsort.hpp: the widely used C++ port of CPython's listobject.c / OpenJDK's TimSort.java.
template <class T> void timsort_ref_raw(T* p, size_t n)            { gfx::timsort(p, p + n, KeyLess<T>{}); }
template <class T> void timsort_ref_counted(T* p, size_t n)        { gfx::timsort(traced(p), traced(p) + n, TracedLess<T>{}); }

#define SB_ENTRY(fn) \
    registry_detail::Wrap<T, &fn<Array<T, false>>, &fn<Array<T, true>>>::raw, \
    registry_detail::Wrap<T, &fn<Array<T, false>>, &fn<Array<T, true>>>::counted

template <class T>
std::vector<AlgoInfo<T>> make_algorithms() {
    // Our own ports, written against the counted Array view so reads, writes
    // and scratch memory can be measured. They are algorithmically faithful,
    // but the view costs them 5-50% of wall time compared with the upstream
    // code below, so speed claims are made against the upstream code.
    std::vector<AlgoInfo<T>> v = {
        {"introsort", "port of libstdc++ std::sort (also .NET Array.Sort)",                              false, true, true, SB_ENTRY(introsort)},
        {"timsort",   "port of OpenJDK TimSort (Java Arrays.sort(Object[]), Android, V8; CPython before 3.11)", true, true, true, SB_ENTRY(timsort)},
        {"pdqsort",   "port of orlp pdqsort, non-branchless path (Go sort.Slice/sort.Ints since 1.19, Boost)", false, true, true, SB_ENTRY(pdqsort)},
        {"mergesort", "classic top-down merge sort with an n-element buffer",                              true, true, true, SB_ENTRY(merge_sort)},
        {"heapsort",  "port of libstdc++ make_heap/sort_heap (bottom-up, like the Linux kernel's sort())", false, true, true, SB_ENTRY(heapsort)},
        // Candidate developed here.
        {"brainsort", "candidate: scout pass -> sorted/reverse/runs/displaced-merge/compressed radix", true, false, true, SB_ENTRY(brainsort)},
    };
    if constexpr (!KeyTraits<T>::chunked) {
        v.push_back({"radix11", "baseline: LSD radix, 11-bit digits",                     true, false, true, SB_ENTRY(radix_sort11)});
        v.push_back({"radix16", "experiment: LSD radix, 16-bit digits, 512 KiB tables",  true, false, true, SB_ENTRY(radix_sort16)});
    }
    // Upstream code, compiled as shipped; counted through the Traced<T>
    // wrapper. These are the opponents every speed claim is measured against.
    v.push_back({"std::sort",               "libstdc++ introsort, as shipped",                                              false, true, true, registry_detail::std_sort_raw<T>,             registry_detail::std_sort_counted<T>});
    v.push_back({"std::stable_sort",        "libstdc++ merge sort with an n/2 buffer, as shipped",                          true,  true, true, registry_detail::std_stable_sort_raw<T>,      registry_detail::std_stable_sort_counted<T>});
    v.push_back({"orlp::pdqsort",           "upstream pdqsort.h, non-branchless partition (what a custom comparator gets)", false, true, true, registry_detail::pdq_ref_raw<T>,              registry_detail::pdq_ref_counted<T>});
    v.push_back({"orlp::pdqsort_branchless","upstream pdqsort.h, branchless block partition (what plain int/double keys get; Rust sort_unstable < 1.81)", false, true, true, registry_detail::pdq_branchless_ref_raw<T>, registry_detail::pdq_branchless_ref_counted<T>});
    v.push_back({"gfx::timsort",            "upstream cpp-TimSort 2.1.0, C++ port of CPython/OpenJDK TimSort",             true,  true, true, registry_detail::timsort_ref_raw<T>,          registry_detail::timsort_ref_counted<T>});
    return v;
}
#undef SB_ENTRY

}  // namespace registry_detail

constexpr size_t kPrimaryAlgorithmCount = 5;

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

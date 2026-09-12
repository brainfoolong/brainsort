// brainsort: classic top-down merge sort with a full-size scratch buffer.
// The fallback for ranges of 2^32 elements or more (positions inside the
// main algorithm are 32-bit), and the merge sort port of the benchmark.
// The input is copied to the scratch buffer once and the recursion then
// alternates between the two buffers so no per-level copy-back is needed.
// Leaves of <= 16 elements are insertion sorted. O(n log n) always, stable,
// n extra elements of memory.
#pragma once
#include "brainsort/detail/traits.hpp"

namespace brainsort {
namespace detail {
namespace merge_detail {

constexpr size_t kLeaf = 16;

// Merge sorted src[lo,mid) and src[mid,hi) into dst[lo,hi). Stable: on ties
// the left element wins.
template <class A>
inline void merge(A src, A dst, size_t lo, size_t mid, size_t hi) {
    size_t i = lo, j = mid, k = lo;
    if (i < mid && j < hi) {
        typename A::value_type x = src.get(i), y = src.get(j);
        for (;;) {
            if (src.less(y, x)) {
                dst.set(k++, y);
                if (++j == hi) break;
                y = src.get(j);
            } else {
                dst.set(k++, x);
                if (++i == mid) break;
                x = src.get(i);
            }
        }
    }
    for (; i < mid; ++i) dst.set(k++, src.get(i));
    for (; j < hi;  ++j) dst.set(k++, src.get(j));
}

// Precondition: src[lo,hi) and dst[lo,hi) hold identical contents.
// Postcondition: dst[lo,hi) is sorted.
template <class A>
void split_merge(A src, A dst, size_t lo, size_t hi) {
    [[maybe_unused]] const typename A::hooks::DepthScope depth{};
    if (hi - lo <= kLeaf) {
        insertion_sort(dst, lo, hi);
        return;
    }
    const size_t mid = lo + (hi - lo) / 2;
    split_merge(dst, src, lo, mid);   // sorted halves end up in src
    split_merge(dst, src, mid, hi);
    merge(src, dst, lo, mid, hi);
}

}  // namespace merge_detail

template <class A>
inline void merge_sort(A a) {
    const size_t n = a.size();
    if (n < 2) return;
    AuxBuffer<typename A::value_type, A> buf(n);
    A b = buf.arr();
    copy_forward(a, 0, b, 0, n);
    merge_detail::split_merge(b, a, 0, n);
}

}  // namespace detail
}  // namespace brainsort

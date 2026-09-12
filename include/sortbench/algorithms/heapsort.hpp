// Heapsort, the bottom-up variant that actually ships: libstdc++'s
// std::make_heap/std::sort_heap (__adjust_heap + __push_heap) and, in the same
// spirit, the Linux kernel's lib/sort.c. Instead of comparing the sifted value
// against both children at every level (2 compares per level), the hole is
// moved down to a leaf choosing the larger child (1 compare per level) and the
// value is then pushed back up (usually 1 or 2 more compares in total). That
// is about n*log2(n) comparisons instead of 2*n*log2(n).
//
// The sequence of comparisons and moves is exactly that of libstdc++, so the
// introsort port that uses this as its depth-limit fallback matches std::sort
// comparison for comparison. In-place, O(n log n) worst case, not stable.
#pragma once
#include "sortbench/core.hpp"

namespace sb {

namespace heap_detail {

// libstdc++ __push_heap: move the hole at `hole` towards `top` while the
// parent is smaller than `value`, then store `value`.
template <class A>
inline void push_heap(A a, size_t first, size_t hole, size_t top, typename A::value_type value) {
    size_t parent = (hole - 1) / 2;
    while (hole > top) {
        typename A::value_type p = a.get(first + parent);
        if (!a.less(p, value)) break;
        a.set(first + hole, p);
        hole   = parent;
        parent = (hole - 1) / 2;
    }
    a.set(first + hole, value);
}

// libstdc++ __adjust_heap: the hole at `hole` (within a heap of `len`
// elements starting at a[first]) is moved down to a leaf along the larger
// children, then `value` is pushed up from there.
template <class A>
inline void adjust_heap(A a, size_t first, size_t hole, size_t len, typename A::value_type value) {
    const size_t top = hole;
    size_t second_child = hole;
    while (second_child < (len - 1) / 2) {
        second_child = 2 * (second_child + 1);
        typename A::value_type r = a.get(first + second_child);
        typename A::value_type l = a.get(first + second_child - 1);
        if (a.less(r, l)) { --second_child; r = l; }
        a.set(first + hole, r);
        hole = second_child;
    }
    if ((len & 1) == 0 && second_child == (len - 2) / 2) {
        second_child = 2 * (second_child + 1);
        a.set(first + hole, a.get(first + second_child - 1));
        hole = second_child - 1;
    }
    push_heap(a, first, hole, top, value);
}

}  // namespace heap_detail

template <class A>
inline void heapsort_range(A a, size_t first, size_t last) {
    const size_t len = last - first;
    if (len < 2) return;
    // std::make_heap
    for (size_t parent = (len - 2) / 2;; --parent) {
        heap_detail::adjust_heap(a, first, parent, len, a.get(first + parent));
        if (parent == 0) break;
    }
    // std::sort_heap: repeated __pop_heap
    for (size_t end = len; end > 1; --end) {
        typename A::value_type value = a.get(first + end - 1);
        a.set(first + end - 1, a.get(first));
        heap_detail::adjust_heap(a, first, 0, end - 1, value);
    }
}

template <class A>
inline void heapsort(A a) { heapsort_range(a, 0, a.size()); }

}  // namespace sb

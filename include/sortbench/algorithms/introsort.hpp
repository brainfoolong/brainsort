// Introsort as implemented by libstdc++ std::sort (Musser 1997):
//   * median-of-3 quicksort with Hoare-style unguarded partitioning,
//   * recursion depth limited to 2*log2(n), after which heapsort takes over,
//   * partitions of <= 16 elements are left unsorted and finished by one final
//     insertion-sort pass (the first block guarded, the rest unguarded).
// In-place, O(n log n) worst case, not stable.
#pragma once
#include "sortbench/algorithms/heapsort.hpp"
#include "sortbench/core.hpp"

namespace sb {

namespace intro_detail {

constexpr size_t kThreshold = 16;

// Move the median of a[x], a[y], a[z] to a[result] (libstdc++ __move_median_to_first).
template <class A>
inline void move_median_to_first(A a, size_t result, size_t x, size_t y, size_t z) {
    typename A::value_type vx = a.get(x), vy = a.get(y), vz = a.get(z);
    if (a.less(vx, vy)) {
        if (a.less(vy, vz))      a.swap(result, y);
        else if (a.less(vx, vz)) a.swap(result, z);
        else                     a.swap(result, x);
    } else if (a.less(vx, vz))   a.swap(result, x);
    else if (a.less(vy, vz))     a.swap(result, z);
    else                         a.swap(result, y);
}

// Hoare partition of [first,last) around a[pivot_idx]; sentinels guaranteed by
// the median-of-3 choice, so the inner scans need no bounds checks.
template <class A>
inline size_t unguarded_partition(A a, size_t first, size_t last, size_t pivot_idx) {
    const typename A::value_type pivot = a.get(pivot_idx);
    for (;;) {
        while (a.less(a.get(first), pivot)) ++first;
        --last;
        while (a.less(pivot, a.get(last))) --last;
        if (!(first < last)) return first;
        a.swap(first, last);
        ++first;
    }
}

template <class A>
inline size_t unguarded_partition_pivot(A a, size_t first, size_t last) {
    const size_t mid = first + (last - first) / 2;
    move_median_to_first(a, first, first + 1, mid, last - 1);
    return unguarded_partition(a, first + 1, last, first);
}

template <class A>
inline void introsort_loop(A a, size_t first, size_t last, size_t depth_limit) {
    [[maybe_unused]] const DepthScope<A::counted> depth;
    while (last - first > kThreshold) {
        if (depth_limit == 0) {
            if constexpr (A::counted) ++g_stats.fallbacks;
            heapsort_range(a, first, last);
            return;
        }
        --depth_limit;
        const size_t cut = unguarded_partition_pivot(a, first, last);
        introsort_loop(a, cut, last, depth_limit);
        last = cut;
    }
}

template <class A>
inline void final_insertion_sort(A a, size_t first, size_t last) {
    if (last - first > kThreshold) {
        insertion_sort(a, first, first + kThreshold);
        unguarded_insertion_sort(a, first + kThreshold, last);
    } else {
        insertion_sort(a, first, last);
    }
}

}  // namespace intro_detail

template <class A>
inline void introsort(A a) {
    const size_t n = a.size();
    if (n < 2) return;
    intro_detail::introsort_loop(a, 0, n, 2 * floor_log2(n));
    intro_detail::final_insertion_sort(a, 0, n);
}

}  // namespace sb

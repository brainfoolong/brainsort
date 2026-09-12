// pdqsort (pattern-defeating quicksort, Orson Peters 2015/2021). This is the
// algorithm behind Rust's slice::sort_unstable and Go's sort.Slice/sort.Ints
// (since Go 1.19). Faithful port of the reference "non-branchless" path:
//   * insertion sort below 24 elements,
//   * median-of-3 pivot, Tukey's ninther above 128 elements,
//   * detection of already-partitioned input, finished with a bounded
//     partial insertion sort (adaptive to presorted data),
//   * partition_left to skip runs of elements equal to the pivot,
//   * on a highly unbalanced partition, deliberately shuffle a few elements to
//     defeat adversarial patterns; after log2(n) bad partitions fall back to
//     heapsort.
// In-place, O(n log n) worst case, O(n) on many patterns, not stable.
#pragma once
#include "sortbench/algorithms/heapsort.hpp"
#include "sortbench/core.hpp"

#include <utility>

namespace sb {

namespace pdq_detail {

constexpr size_t kInsertionSortThreshold    = 24;
constexpr size_t kNintherThreshold          = 128;
constexpr size_t kPartialInsertionSortLimit = 8;

template <class A>
inline void sort2(A a, size_t x, size_t y) {
    // Function arguments are unsequenced: read in one fixed order so the
    // trace hash is the same on every compiler.
    const typename A::value_type vy = a.get(y), vx = a.get(x);
    if (a.less(vy, vx)) a.swap(x, y);
}
template <class A>
inline void sort3(A a, size_t x, size_t y, size_t z) {
    sort2(a, x, y);
    sort2(a, y, z);
    sort2(a, x, y);
}

// Insertion sort that gives up (returns false) once it has moved more than
// kPartialInsertionSortLimit elements in total.
template <class A>
inline bool partial_insertion_sort(A a, size_t begin, size_t end) {
    if (begin == end) return true;
    size_t limit = 0;
    for (size_t cur = begin + 1; cur != end; ++cur) {
        size_t sift   = cur;
        size_t sift_1 = cur - 1;
        typename A::value_type   vs     = a.get(sift);
        typename A::value_type   vs1    = a.get(sift_1);
        if (a.less(vs, vs1)) {
            typename A::value_type tmp = vs;
            do {
                a.set(sift--, vs1);
                if (sift == begin) break;
                vs1 = a.get(--sift_1);
            } while (a.less(tmp, vs1));
            a.set(sift, tmp);
            limit += cur - sift;
        }
        if (limit > kPartialInsertionSortLimit) return false;
    }
    return true;
}

// Partition [begin,end) around pivot a[begin]: elements < pivot to the left,
// >= pivot to the right. Returns pivot position and whether the range was
// already partitioned.
template <class A>
inline std::pair<size_t, bool> partition_right(A a, size_t begin, size_t end) {
    const typename A::value_type pivot = a.get(begin);
    size_t first = begin;
    size_t last  = end;

    while (a.less(a.get(++first), pivot)) {}
    if (first - 1 == begin) {
        while (first < last && !a.less(a.get(--last), pivot)) {}
    } else {
        while (!a.less(a.get(--last), pivot)) {}
    }
    const bool already_partitioned = first >= last;

    while (first < last) {
        a.swap(first, last);
        while (a.less(a.get(++first), pivot)) {}
        while (!a.less(a.get(--last), pivot)) {}
    }
    const size_t pivot_pos = first - 1;
    a.set(begin, a.get(pivot_pos));
    a.set(pivot_pos, pivot);
    return {pivot_pos, already_partitioned};
}

// Mirror image: elements <= pivot to the left, > pivot to the right. Used when
// the pivot equals the element just before the range, so all equal elements
// are swept into the left partition and skipped.
template <class A>
inline size_t partition_left(A a, size_t begin, size_t end) {
    const typename A::value_type pivot = a.get(begin);
    size_t first = begin;
    size_t last  = end;

    while (a.less(pivot, a.get(--last))) {}
    if (last + 1 == end) {
        while (first < last && !a.less(pivot, a.get(++first))) {}
    } else {
        while (!a.less(pivot, a.get(++first))) {}
    }
    while (first < last) {
        a.swap(first, last);
        while (a.less(pivot, a.get(--last))) {}
        while (!a.less(pivot, a.get(++first))) {}
    }
    const size_t pivot_pos = last;
    a.set(begin, a.get(pivot_pos));
    a.set(pivot_pos, pivot);
    return pivot_pos;
}

template <class A>
void pdqsort_loop(A a, size_t begin, size_t end, size_t bad_allowed, bool leftmost) {
    [[maybe_unused]] const DepthScope<A::counted> depth;
    for (;;) {
        const size_t size = end - begin;

        if (size < kInsertionSortThreshold) {
            if (leftmost) insertion_sort(a, begin, end);
            else          unguarded_insertion_sort(a, begin, end);
            return;
        }

        const size_t s2 = size / 2;
        if (size > kNintherThreshold) {
            sort3(a, begin, begin + s2, end - 1);
            sort3(a, begin + 1, begin + (s2 - 1), end - 2);
            sort3(a, begin + 2, begin + (s2 + 1), end - 3);
            sort3(a, begin + (s2 - 1), begin + s2, begin + (s2 + 1));
            a.swap(begin, begin + s2);
        } else {
            sort3(a, begin + s2, begin, end - 1);
        }

        // If a[begin-1] == pivot, everything equal to the pivot is swept left
        // and skipped, which makes many-duplicates inputs O(n log k).
        if (!leftmost) {
            const typename A::value_type prev = a.get(begin - 1), first = a.get(begin);
            if (!a.less(prev, first)) {
                begin = partition_left(a, begin, end) + 1;
                continue;
            }
        }

        auto [pivot_pos, already_partitioned] = partition_right(a, begin, end);

        const size_t l_size = pivot_pos - begin;
        const size_t r_size = end - (pivot_pos + 1);
        const bool highly_unbalanced = l_size < size / 8 || r_size < size / 8;

        if (highly_unbalanced) {
            if constexpr (A::counted) ++g_stats.bad_parts;
            if (--bad_allowed == 0) {
                if constexpr (A::counted) ++g_stats.fallbacks;
                heapsort_range(a, begin, end);
                return;
            }
            if (l_size >= kInsertionSortThreshold) {
                a.swap(begin, begin + l_size / 4);
                a.swap(pivot_pos - 1, pivot_pos - l_size / 4);
                if (l_size > kNintherThreshold) {
                    a.swap(begin + 1, begin + (l_size / 4 + 1));
                    a.swap(begin + 2, begin + (l_size / 4 + 2));
                    a.swap(pivot_pos - 2, pivot_pos - (l_size / 4 + 1));
                    a.swap(pivot_pos - 3, pivot_pos - (l_size / 4 + 2));
                }
            }
            if (r_size >= kInsertionSortThreshold) {
                a.swap(pivot_pos + 1, pivot_pos + (1 + r_size / 4));
                a.swap(end - 1, end - r_size / 4);
                if (r_size > kNintherThreshold) {
                    a.swap(pivot_pos + 2, pivot_pos + (2 + r_size / 4));
                    a.swap(pivot_pos + 3, pivot_pos + (3 + r_size / 4));
                    a.swap(end - 2, end - (1 + r_size / 4));
                    a.swap(end - 3, end - (2 + r_size / 4));
                }
            }
        } else {
            if (already_partitioned &&
                partial_insertion_sort(a, begin, pivot_pos) &&
                partial_insertion_sort(a, pivot_pos + 1, end)) {
                return;
            }
        }

        // Recurse on the left, iterate on the right.
        pdqsort_loop(a, begin, pivot_pos, bad_allowed, leftmost);
        begin    = pivot_pos + 1;
        leftmost = false;
    }
}

}  // namespace pdq_detail

template <class A>
inline void pdqsort(A a) {
    const size_t n = a.size();
    if (n < 2) return;
    pdq_detail::pdqsort_loop(a, 0, n, floor_log2(n), true);
}

}  // namespace sb

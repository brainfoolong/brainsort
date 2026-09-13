// brainsort: the comparison sort behind the comparator overloads.
//
// A comparator gives the radix engine nothing to work with, so sort_with
// runs a stable merge sort: runs of kRun elements are insertion sorted in
// place, then merged bottom-up between the array and a buffer of the same
// size, so no level copies back. The merge takes the next element without
// a branch on the comparison (the side to take from is selected, not
// jumped to), which is what makes it faster than the library's merge sort
// on keys whose comparator is cheap; a pair of runs that is already in
// order is copied instead of merged. Trivially copyable elements only: the
// buffer holds plain copies, so if the comparator throws, every element is
// still in the range (in the buffer's order or the array's), which is the
// guarantee std::stable_sort gives.
#pragma once
#include "brainsort/detail/config.hpp"

#include <cstddef>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>

namespace brainsort {
namespace detail {
namespace comp_detail {

constexpr size_t kRun = 16;   // initial runs, insertion sorted

// Insertion sort of a[0,n) that leaves every element in place if the
// comparator throws: the element being inserted goes back into its hole.
template <class T, class Comp>
inline void insertion_run(T* a, size_t n, Comp& comp) {
    for (size_t i = 1; i < n; ++i) {
        T      v = a[i];
        size_t j = i;
        try {
            while (j > 0 && comp(v, a[j - 1])) { a[j] = a[j - 1]; --j; }
        } catch (...) {
            a[j] = v;
            throw;
        }
        a[j] = v;
    }
}

// A merge in progress: the two runs and the output cursor.
template <class T>
struct Merge {
    const T* l;
    const T* lend;
    const T* r;
    const T* rend;
    T*       out;
    bool live() const { return l != lend && r != rend; }
    void finish(void) {
        if (l != lend) std::memcpy(static_cast<void*>(out), l, static_cast<size_t>(lend - l) * sizeof(T));
        if (r != rend) std::memcpy(static_cast<void*>(out), r, static_cast<size_t>(rend - r) * sizeof(T));
    }
};

// A single merge, in counted stretches: as many steps as every run can
// give without a bounds check, on local cursors, then again.
template <class T, class Comp>
inline void merge_one(Merge<T>& m, Comp& comp) {
    const T* l = m.l;
    const T* r = m.r;
    T*       o = m.out;
    for (;;) {
        const size_t nl = static_cast<size_t>(m.lend - l), nr = static_cast<size_t>(m.rend - r);
        const size_t k  = nl < nr ? nl : nr;
        if (k == 0) break;
        for (size_t i = 0; i < k; ++i) {
            const bool t   = comp(*r, *l);
            const T*   src = t ? r : l;
            o[i] = *src;
            r += t;
            l += !t;
        }
        o += k;
    }
    m.l = l; m.r = r; m.out = o;
    m.finish();
}

// Two merges advanced in lock step, in counted stretches. Each step of one
// merge waits for its own comparison and then the load it selects, so one
// merge alone runs at that latency; two independent ones overlap in the
// pipeline.
template <class T, class Comp>
inline void merge_two(Merge<T>& a, Merge<T>& b, Comp& comp) {
    const T* al = a.l;
    const T* ar = a.r;
    const T* bl = b.l;
    const T* br = b.r;
    T*       ao = a.out;
    T*       bo = b.out;
    for (;;) {
        size_t k = static_cast<size_t>(a.lend - al);
        const size_t k2 = static_cast<size_t>(a.rend - ar), k3 = static_cast<size_t>(b.lend - bl), k4 = static_cast<size_t>(b.rend - br);
        k = k2 < k ? k2 : k;
        k = k3 < k ? k3 : k;
        k = k4 < k ? k4 : k;
        if (k == 0) break;
        for (size_t i = 0; i < k; ++i) {
            const bool ta = comp(*ar, *al);
            const bool tb = comp(*br, *bl);
            const T*   sa = ta ? ar : al;
            const T*   sb = tb ? br : bl;
            ao[i] = *sa;
            bo[i] = *sb;
            ar += ta; al += !ta;
            br += tb; bl += !tb;
        }
        ao += k;
        bo += k;
    }
    a.l = al; a.r = ar; a.out = ao;
    b.l = bl; b.r = br; b.out = bo;
    merge_one(a, comp);
    merge_one(b, comp);
}

// Split the merge of L[0,p) and R[0,q) at output position k: the smallest i
// (j = k - i) such that R[j-1] sorts before L[i], so that L[0,i) and R[0,j)
// are exactly the first k elements of the stable merge. A binary search of
// log(n) comparisons.
template <class T, class Comp>
inline size_t merge_split(const T* L, size_t p, const T* R, size_t q, size_t k, Comp& comp) {
    size_t lo = k > q ? k - q : 0, hi = k < p ? k : p;
    while (lo < hi) {
        const size_t i = lo + (hi - lo) / 2;
        if (!comp(R[k - i - 1], L[i])) lo = i + 1;
        else hi = i;
    }
    return lo;
}

// One level: merge the runs of `width` elements of src into dst, two
// independent merges at a time. While the level has many pairs, two pairs
// advance together; once pairs are few and long, each pair is split at its
// middle output position into two halves that advance together.
constexpr size_t kSplitWidth = 2048;   // from this run length on, pairs are split rather than paired

template <class T, class Comp>
inline void merge_level(const T* src, T* dst, size_t n, size_t width, Comp& comp) {
    Merge<T> pending{};
    bool     have = false;
    for (size_t lo = 0; lo < n; lo += 2 * width) {
        const size_t mid = lo + width < n ? lo + width : n;
        const size_t hi  = lo + 2 * width < n ? lo + 2 * width : n;
        if (mid == hi || !comp(src[mid], src[mid - 1])) {   // a lone run, or two runs already in order
            std::memcpy(static_cast<void*>(dst + lo), src + lo, (hi - lo) * sizeof(T));
            continue;
        }
        const T* L = src + lo;
        const T* R = src + mid;
        const size_t p = mid - lo, q = hi - mid;
        if (width < kSplitWidth) {
            Merge<T> m{L, L + p, R, R + q, dst + lo};
            if (!have) { pending = m; have = true; continue; }
            merge_two(pending, m, comp);
            have = false;
            continue;
        }
        const size_t k = (hi - lo) / 2;
        const size_t i = merge_split(L, p, R, q, k, comp), j = k - i;
        Merge<T> a{L, L + i, R, R + j, dst + lo};
        Merge<T> b{L + i, L + p, R + j, R + q, dst + lo + k};
        merge_two(a, b, comp);
    }
    if (have) merge_one(pending, comp);
}

}  // namespace comp_detail

// Stable merge sort of a[0,n) by comp, through a buffer of n elements from
// Alloc. Throws std::bad_alloc, with the range untouched, if the buffer
// cannot be allocated.
template <class Alloc, class T, class Comp>
inline void stable_merge_sort(T* a, size_t n, Comp& comp) {
    static_assert(std::is_trivially_copyable_v<T>);
    using namespace comp_detail;
    if (n < 2) return;
    struct Buffer {
        T*     p;
        size_t bytes;
        explicit Buffer(size_t b) : p(static_cast<T*>(Alloc::allocate(b))), bytes(b) {}
        ~Buffer() { Alloc::deallocate(p, bytes); }
    } buf(n * sizeof(T));
    for (size_t lo = 0; lo < n; lo += kRun) insertion_run(a + lo, lo + kRun < n ? kRun : n - lo, comp);
    T* src = a;
    T* dst = buf.p;
    try {
        for (size_t width = kRun; width < n; width *= 2) {
            merge_level(src, dst, n, width, comp);
            std::swap(src, dst);
        }
    } catch (...) {
        // src holds every element; put them back if the array is not it.
        if (src != a) std::memcpy(static_cast<void*>(a), src, n * sizeof(T));
        throw;
    }
    if (src != a) std::memcpy(static_cast<void*>(a), src, n * sizeof(T));
}

}  // namespace detail
}  // namespace brainsort

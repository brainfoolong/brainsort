// brainsort: the comparison sort behind the comparator overloads.
//
// A comparator gives the radix engine nothing to work with, so sort_with
// runs a stable comparison sort through a buffer of n elements:
//
//   1. Natural runs. One pass finds the maximal non-descending runs
//      (strictly descending ones are reversed in place, which keeps equal
//      elements in order) and gives up as soon as the runs are too short
//      to pay. Long runs are merged bottom-up between the array and the
//      buffer, so an input made of a few sorted pieces costs log(pieces)
//      passes.
//   2. Otherwise a stable quicksort: each level partitions its range
//      through the buffer with one compare and two unconditional stores
//      per element (no branch, no dependency between elements), the lefts
//      compacted into the buffer and the rights into the array, then both
//      slid into place. A range whose pivot equals the pivot of the range
//      it was split from consists of elements that are all at least that
//      pivot, so the elements equal to it are stripped off in one pass:
//      an input with few distinct values costs about log(distinct) levels.
//      The smaller side recurses, the larger loops; past a depth budget a
//      range takes the merge sort instead, so the worst case stays n log n.
//   3. Ranges of at most kLeaf elements are merge sorted: runs of kRun
//      elements insertion sorted in place, then merged bottom-up between
//      the array and the buffer. The merge takes the next element without
//      a branch on the comparison (the side to take from is selected, not
//      jumped to), which is what makes it faster than the library's merge
//      sort on keys whose comparator is cheap; a pair of runs that is
//      already in order is copied instead of merged.
//
// Trivially copyable elements only: the buffer holds plain copies, so if
// the comparator throws, every element is still in the range (in the
// buffer's order or the array's), which is the guarantee std::stable_sort
// gives. Elements larger than kElementRouteMax bytes are sorted through an
// index array (brainsort.hpp), so the passes move 4-byte indices and each
// element moves once, at the end.
#pragma once
#include "brainsort/detail/config.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>

namespace brainsort {
namespace detail {
namespace comp_detail {

constexpr size_t kRun  = 16;    // initial runs of the merge sort: sixteen elements, sorted without a branch
constexpr size_t kLeaf = 512;   // ranges up to this size are merge sorted, larger ones partitioned

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

// Stable sort of a[0,4) with five comparisons and no branch: the two pairs
// are ordered, the smallest and largest of the four selected, and the two
// left over ordered. Every selection prefers the earlier element on a tie.
template <class T, class Comp>
inline void sort4(T* a, Comp& comp) {
    const T v0 = a[0], v1 = a[1], v2 = a[2], v3 = a[3];
    const bool c1 = comp(v1, v0), c2 = comp(v3, v2);
    const T lo0 = c1 ? v1 : v0, hi0 = c1 ? v0 : v1;   // the first pair in order
    const T lo1 = c2 ? v3 : v2, hi1 = c2 ? v2 : v3;   // the second
    const bool c3 = comp(lo1, lo0), c4 = comp(hi1, hi0);
    const T mn = c3 ? lo1 : lo0;
    const T mx = c4 ? hi0 : hi1;
    const T ul = c3 ? lo0 : (c4 ? lo1 : hi0);   // the two left over, ul from the earlier pair
    const T ur = c4 ? hi1 : (c3 ? hi0 : lo1);
    const bool c5 = comp(ur, ul);
    a[0] = mn;
    a[1] = c5 ? ur : ul;
    a[2] = c5 ? ul : ur;
    a[3] = mx;
}

// Merge the runs L[0,m) and R[0,m) into out[0,2m), from both ends, without
// a bounds check: the front takes m steps and the back takes m, and no run
// can go dry before a side's last step, because each has exactly m
// elements. The two chains overlap in the pipeline.
template <size_t m, class T, class Comp>
inline void merge_exact(const T* L, const T* R, T* out, Comp& comp) {
    const T* l  = L;
    const T* r  = R;
    const T* le = L + m;
    const T* re = R + m;
    T*       o  = out;
    T*       oe = out + 2 * m;
    for (size_t i = 0; i < m; ++i) {
        const bool t = comp(*r, *l);
        const bool u = comp(re[-1], le[-1]);
        o[i] = *(t ? r : l);
        oe[-1 - static_cast<std::ptrdiff_t>(i)] = *(u ? le - 1 : re - 1);
        r += t;  l += !t;
        le -= u; re -= !u;
    }
}

// Stable sort of a[0,16) without a branch: four sorts of four, two merges
// of four and four into a buffer, one merge of eight and eight back. If
// the comparator throws, a[0,16) holds every element.
template <class T, class Comp>
inline void sort16(T* a, Comp& comp) {
    sort4(a, comp); sort4(a + 4, comp); sort4(a + 8, comp); sort4(a + 12, comp);
    T tmp[16];
    merge_exact<4>(a, a + 4, tmp, comp);
    merge_exact<4>(a + 8, a + 12, tmp + 8, comp);
    try {
        merge_exact<8>(tmp, tmp + 8, a, comp);
    } catch (...) {
        std::memcpy(static_cast<void*>(a), tmp, 16 * sizeof(T));
        throw;
    }
}

// The initial runs of a[0,n): every full group of kRun elements by sort16,
// a shorter tail by insertion. If the comparator throws, a[0,n) is a
// permutation of its input.
template <class T, class Comp>
inline void initial_runs(T* a, size_t n, Comp& comp) {
    size_t lo = 0;
    for (; lo + kRun <= n; lo += kRun) sort16(a + lo, comp);
    if (lo < n) insertion_run(a + lo, n - lo, comp);
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

// A single merge from both ends, in counted stretches: the front takes
// the smaller next element (the left one on a tie), the back the larger
// last element (the right one on a tie), and the two chains of
// compare-select-load overlap in the pipeline. A stretch is as many steps
// as both ends can take without meeting or running a run dry; the tail is
// merged from the front alone.
template <class T, class Comp>
inline void merge_one(Merge<T>& m, Comp& comp) {
    const T* l  = m.l;
    const T* r  = m.r;
    const T* le = m.lend;
    const T* re = m.rend;
    T*       o  = m.out;
    T*       oe = o + (le - l) + (re - r);
    for (;;) {
        const size_t nl = static_cast<size_t>(le - l), nr = static_cast<size_t>(re - r);
        const size_t k  = (nl < nr ? nl : nr) / 2;
        if (k == 0) break;
        for (size_t i = 0; i < k; ++i) {
            const bool t  = comp(*r, *l);
            const bool u  = comp(re[-1], le[-1]);
            const T*   sf = t ? r : l;
            const T*   sb = u ? le - 1 : re - 1;
            o[i]       = *sf;
            oe[-1 - static_cast<std::ptrdiff_t>(i)] = *sb;
            r += t;  l += !t;
            le -= u; re -= !u;
        }
        o += k;
        oe -= k;
    }
    for (;;) {
        const size_t nl = static_cast<size_t>(le - l), nr = static_cast<size_t>(re - r);
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
    m.l = l; m.r = r; m.lend = le; m.rend = re; m.out = o;
    m.finish();
}

// Two merges advanced in lock step from the front, in counted stretches:
// two independent chains in the pipeline, with fewer live cursors than
// two merges from both ends would need (which spilled to the stack). The
// tails go to merge_one.
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

// One level of merging, two independent merges at a time. While the pairs
// are short, two pairs advance together; once a pair is long, it is split
// at its middle output position into two halves that advance together.
constexpr size_t kSplitWidth = 2048;   // from this run length on, pairs are split rather than paired

template <class T, class Comp>
struct Level {
    Merge<T> pending{};
    bool     have = false;
    // Merge src[lo,mid) and src[mid,hi), both sorted, into dst[lo,hi).
    void add(const T* src, T* dst, size_t lo, size_t mid, size_t hi, Comp& comp) {
        if (mid == hi || !comp(src[mid], src[mid - 1])) {   // a lone run, or two runs already in order
            std::memcpy(static_cast<void*>(dst + lo), src + lo, (hi - lo) * sizeof(T));
            return;
        }
        const T* L = src + lo;
        const T* R = src + mid;
        const size_t p = mid - lo, q = hi - mid;
        if (p < kSplitWidth && q < kSplitWidth) {
            Merge<T> m{L, L + p, R, R + q, dst + lo};
            if (!have) { pending = m; have = true; return; }
            merge_two(pending, m, comp);
            have = false;
            return;
        }
        const size_t k = (hi - lo) / 2;
        const size_t i = merge_split(L, p, R, q, k, comp), j = k - i;
        Merge<T> a{L, L + i, R, R + j, dst + lo};
        Merge<T> b{L + i, L + p, R + j, R + q, dst + lo + k};
        merge_two(a, b, comp);
    }
    void finish(Comp& comp) {
        if (have) merge_one(pending, comp);
        have = false;
    }
};

// Stable merge sort of a[0,n) through buf[0,n); the result is in a. If the
// comparator throws, a[0,n) holds every element.
template <class T, class Comp>
inline void merge_sort(T* a, T* buf, size_t n, Comp& comp) {
    initial_runs(a, n, comp);
    T* src = a;
    T* dst = buf;
    try {
        for (size_t width = kRun; width < n; width *= 2) {
            Level<T, Comp> level;
            for (size_t lo = 0; lo < n; lo += 2 * width) {
                const size_t mid = lo + width < n ? lo + width : n;
                const size_t hi  = lo + 2 * width < n ? lo + 2 * width : n;
                level.add(src, dst, lo, mid, hi, comp);
            }
            level.finish(comp);
            std::swap(src, dst);
        }
    } catch (...) {
        // src holds every element; put them back if the array is not it.
        if (src != a) std::memcpy(static_cast<void*>(a), src, n * sizeof(T));
        throw;
    }
    if (src != a) std::memcpy(static_cast<void*>(a), src, n * sizeof(T));
}

// ---- natural runs ---------------------------------------------------------

// The boundaries of the maximal non-descending runs of a[0,n) into b[0..r]
// (b[0] = 0, b[r] = n); a strictly descending run is reversed in place,
// which keeps equal elements in their order. Gives up, with false, once the
// runs seen are too short to be worth merging: more than (i >> 5) + 32 of
// them after i elements, which on unordered input happens within the first
// few hundred. b must hold max_runs(n) + 1 entries.
constexpr size_t max_runs(size_t n) { return (n >> 5) + 33; }

template <class T, class Comp>
inline bool find_runs(T* a, size_t n, uint32_t* b, size_t& r, Comp& comp) {
    r    = 0;
    b[0] = 0;
    size_t i = 0;
    while (i < n) {
        size_t j = i + 1;
        if (j < n && comp(a[j], a[j - 1])) {          // strictly descending: each element below the one before
            for (++j; j < n && comp(a[j], a[j - 1]); ++j) {}
            std::reverse(a + i, a + j);
        } else {
            for (; j < n && !comp(a[j], a[j - 1]); ++j) {}
        }
        b[++r] = static_cast<uint32_t>(j);
        i = j;
        if (r > (j >> 5) + 32) return false;
    }
    return true;
}

// Merge the runs b[0..r] of a[0,n) bottom-up through buf; the result is in
// a. b is overwritten level by level.
template <class T, class Comp>
inline void merge_runs(T* a, T* buf, size_t n, uint32_t* b, size_t r, Comp& comp) {
    T* src = a;
    T* dst = buf;
    try {
        while (r > 1) {
            Level<T, Comp> level;
            size_t w = 0;
            for (size_t k = 0; k + 1 < r; k += 2) {
                level.add(src, dst, b[k], b[k + 1], b[k + 2], comp);
                b[w++] = b[k];
            }
            if (r & 1) {   // the last run has no partner
                std::memcpy(static_cast<void*>(dst + b[r - 1]), src + b[r - 1], (n - b[r - 1]) * sizeof(T));
                b[w++] = b[r - 1];
            }
            level.finish(comp);
            b[w] = static_cast<uint32_t>(n);
            r    = w;
            std::swap(src, dst);
        }
    } catch (...) {
        if (src != a) std::memcpy(static_cast<void*>(a), src, n * sizeof(T));
        throw;
    }
    if (src != a) std::memcpy(static_cast<void*>(a), src, n * sizeof(T));
}

// ---- the stable quicksort -------------------------------------------------

// Stable partition of a[0,n) by pred, through buf: the elements pred holds
// for first, in their order, then the others in theirs. Returns how many
// pred held for. Every element is stored twice, into the next left slot of
// buf and the next right slot of a (the slot it came from or an earlier
// one), and the cursor of its side advances: no branch, no dependency from
// one element to the next. If pred throws, a[0,n) holds every element.
template <class T, class Pred>
inline size_t partition(T* a, T* buf, size_t n, Pred pred) {
    size_t nl = 0, nr = 0, i = 0;
    try {
        for (; i < n; ++i) {
            const T    x    = a[i];
            const bool left = pred(x);
            buf[nl] = x;
            a[nr]   = x;
            nl += left;
            nr += !left;
        }
    } catch (...) {   // a[0,nr) are rights, buf[0,nl) lefts, a[i,n) unread: the lefts fill a[nr,i)
        std::memcpy(static_cast<void*>(a + nr), buf, nl * sizeof(T));
        throw;
    }
    std::memmove(static_cast<void*>(a + nl), a, nr * sizeof(T));
    std::memcpy(static_cast<void*>(a), buf, nl * sizeof(T));
    return nl;
}

template <class T, class Comp>
inline const T& median3(const T& x, const T& y, const T& z, Comp& comp) {
    if (comp(y, x)) {
        if (comp(z, y)) return y;
        return comp(z, x) ? z : x;
    }
    if (comp(z, y)) return comp(z, x) ? x : z;
    return y;
}

// The pivot of a[0,n): the median of three elements, or of three such
// medians once the range is long.
template <class T, class Comp>
inline T choose_pivot(const T* a, size_t n, Comp& comp) {
    if (n < 1024) return median3(a[n / 4], a[n / 2], a[n - n / 4], comp);
    const size_t s = n / 9, h = s / 2;
    const T& m0 = median3(a[h], a[h + s], a[h + 2 * s], comp);
    const T& m1 = median3(a[h + 3 * s], a[h + 4 * s], a[h + 5 * s], comp);
    const T& m2 = median3(a[h + 6 * s], a[h + 7 * s], a[h + 8 * s], comp);
    return median3(m0, m1, m2, comp);
}

// Sort a[0,n) through buf[0,n). `ancestor` is the pivot the range was split
// off to the right of, if any: every element is at least it. `budget` is
// the number of partition levels left before the range takes the merge sort.
template <class T, class Comp>
inline void quicksort(T* a, T* buf, size_t n, const T* ancestor, unsigned budget, Comp& comp) {
    T anc{};   // the ancestor of the range the loop continues with
    for (;;) {
        if (n <= kLeaf || budget == 0) { merge_sort(a, buf, n, comp); return; }
        --budget;
        const T pivot = choose_pivot(a, n, comp);
        if (ancestor != nullptr && !comp(*ancestor, pivot)) {
            // The pivot equals the ancestor and nothing is below it: the
            // elements equal to it are the front of the sorted range.
            const size_t nl = partition(a, buf, n, [pivot, &comp](const T& x) { return !comp(pivot, x); });
            a += nl; buf += nl; n -= nl;
            continue;
        }
        const size_t nl = partition(a, buf, n, [pivot, &comp](const T& x) { return comp(x, pivot); });
        if (nl < n - nl) {   // the smaller side recurses, the larger loops
            quicksort(a, buf, nl, ancestor, budget, comp);
            anc      = pivot;
            ancestor = &anc;
            a += nl; buf += nl; n -= nl;
        } else {
            quicksort(a + nl, buf + nl, n - nl, &pivot, budget, comp);
            n = nl;
        }
    }
}

inline unsigned log2_floor(size_t n) {
    unsigned k = 0;
    while (n >>= 1) ++k;
    return k;
}

}  // namespace comp_detail

// Stable comparison sort of a[0,n) by comp, through a buffer of n elements
// from Alloc. Throws std::bad_alloc, with the range untouched, if the buffer
// cannot be allocated.
template <class Alloc, class T, class Comp>
inline void stable_comparison_sort(T* a, size_t n, Comp& comp) {
    static_assert(std::is_trivially_copyable_v<T>);
    using namespace comp_detail;
    if (n < 2) return;
    struct Buffer {
        void*  p;
        size_t bytes;
        explicit Buffer(size_t b) : p(Alloc::allocate(b)), bytes(b) {}
        ~Buffer() { Alloc::deallocate(p, bytes); }
    } buf(n * sizeof(T));
    T* const b = static_cast<T*>(buf.p);
    if (n <= kLeaf) { merge_sort(a, b, n, comp); return; }
    {   // long natural runs are merged as they are
        struct Runs {
            uint32_t* p     = nullptr;
            size_t    bytes = 0;
            ~Runs() { if (p) Alloc::deallocate(p, bytes); }
        } runs;
        try {
            runs.bytes = (max_runs(n) + 1) * sizeof(uint32_t);
            runs.p     = static_cast<uint32_t*>(Alloc::allocate(runs.bytes));
        } catch (const std::bad_alloc&) {
            runs.p = nullptr;   // no run detection then
        }
        if (runs.p != nullptr) {
            size_t r = 0;
            if (find_runs(a, n, runs.p, r, comp)) { merge_runs(a, b, n, runs.p, r, comp); return; }
        }
    }
    quicksort(a, b, n, static_cast<const T*>(nullptr), 2 * log2_floor(n) + 4, comp);
}

}  // namespace detail
}  // namespace brainsort

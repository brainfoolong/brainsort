// brainsort: the algorithm.
//
// One linear "scout" pass over the keys computes, at the same time,
//   * the OR of (key XOR key[0]): a mask of the bits that vary at all,
//   * the number of descents (a[i] < a[i-1]) and ascents (a[i] > a[i-1]),
//   * the monotone runs (timsort's definition), with their exact bounds,
//     while there are at most four of them.
// With AVX2 the scout pass is vectorised for 32-bit integer, double and 64-bit
// integer keys (the elements are compared in place, vector compares +
// popcount, vector OR) and costs a few percent of a radix pass. The pass
// decides the route:
//   1. no descents            -> already sorted, nothing to do
//   2. no ascents             -> non-increasing: reversal (O(n)), the tie
//        groups put back in input order from the recorded run bounds
//  2b. <= 4 monotone runs     -> reverse the descending runs at the recorded
//        bounds and merge pairwise from the shorter side (buffer <= n/2); no
//        second detection pass
//   3. few descents (<= n/16) -> "displaced-element" route: the elements that
//        break the ascending order are pulled out (an outlier-popping rule
//        keeps a long sorted subsequence), the kept elements are compacted in
//        place, only the displaced ones are sorted, and one backward merge
//        puts them back. O(n + k log k) for k displaced elements, memory O(k).
//   4. otherwise              -> the radix route. A strided sample gives the
//        median (split pivot), the sampled varying-bit mask and key range,
//        and whether keys repeat. When the plan needs three passes or more
//        and the data is unordered, the range is split by the pivot into two
//        parts through a buffer of about half the array, and each part is
//        radix sorted. Each part gets the cheapest plan: the raw key, the
//        contiguous varying-bit range, only the varying bits (BMI2 PEXT), or
//        key minus a base (a narrow range that straddles a power of two,
//        like a sign change). A plan made from the sample rather than the
//        exact mask is verified inside the histogram pass and replaced by
//        the exact plan if a key contradicts it, so no separate mask pass is
//        needed after an early commit. When the sample shows few distinct
//        keys, a dictionary radix sorts the part in one hashed-bucket pass
//        (verified the same way). Digits are capped at 13 bits on the split
//        path and 16 on the whole-array path; the histogram of the next pass
//        is built during the current scatter, so two tables are live at most.
// The scout pass commits to route 4 as soon as the counts prove that routes
// 1-3 are impossible (a few thousand elements on unstructured input); the
// bits it saw until then seed the plan, and the split pass computes the rest.
// Keys are whatever elem_traits<T>::radix_key exposes: integers directly,
// doubles through an order-preserving transform, strings as 7-byte chunks.
// For chunked keys (strings) groups of elements that tie on a chunk are sorted
// on the next chunk, with the same route selection per group; the largest
// group is handled by iteration, so the recursion depth is O(log n).
// Stable, exact. Scratch: about n/2 elements plus two small digit tables on
// route 4, n/2 elements on route 2b, O(k) on route 3, nothing on routes 1-2.
// Requires n < 2^32 (falls back to merge sort beyond that).
#pragma once
#include "brainsort/detail/config.hpp"
#include "brainsort/detail/mergesort.hpp"
#include "brainsort/detail/radix.hpp"
#include "brainsort/detail/traits.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <new>

namespace brainsort {
namespace detail {
namespace brain_detail {

using radix_detail::key_bits;

constexpr size_t kInsertionMax = 32;      // below this: insertion sort
constexpr size_t kSplitMin     = 1024;    // route 4 splits into two parts from this size on
constexpr size_t kMaxRuns      = 4;       // route 2b: <= 2 merge levels
constexpr size_t kMaxPop       = 4;       // route 3: how far back an outlier is looked for
constexpr size_t kRing         = 256;     // route 3: positions of the top kept elements
constexpr size_t kCommitBlock  = 1024;    // scout: how often the early-commit test runs
constexpr size_t kSample       = 512;     // route 4: pivot sample size (at most)

struct ScoutResult {
    uint64_t mask      = 0;
    bool     has_mask  = false;  // chunked keys compute the mask lazily (only the radix route needs it)
    bool     committed = false;  // stopped early: routes 1-3 are provably out, go straight to route 4
    size_t   descents  = 0;
    size_t   ascents   = 0;
    // Monotone runs by timsort's definition (non-decreasing, or strictly
    // decreasing), tracked exactly while there are at most kMaxRuns of them:
    // run j is a[bound[j], bound[j+1]) with direction dir[j] (+1
    // non-decreasing, -1 strictly decreasing). Once a (kMaxRuns+1)-th run
    // starts, tracking stops and `runs` stays at kMaxRuns + 1.
    size_t   runs      = 1;
    size_t   bound[kMaxRuns + 1] = {};
    int8_t   dir[kMaxRuns + 1]   = {};
    bool     tracking  = true;
    int8_t   cur_dir   = 0;      // direction of the run in progress; 0 while it has one element
};

// Early commit to route 4 after `scanned` elements. Rigorous, not a guess:
// routes 1 and 2 need one of the counts to be zero; route 2b needs at most
// kMaxRuns runs; route 3 gives up once the displaced elements exceed
// scanned/8 + 64, and every descent displaces at least one element.
inline bool commit_now(const ScoutResult& r, size_t scanned) {
    return r.ascents > 0 && !r.tracking && r.descents > (scanned >> 3) + 64;
}

// Feed the compare of element i against element i-1 (c < 0: descent, c > 0:
// ascent, 0: equal) into the run tracking.
inline void track_pair(ScoutResult& r, int c, size_t i) {
    if (r.cur_dir == 0) { r.cur_dir = c < 0 ? -1 : 1; return; }
    if ((c < 0) != (r.cur_dir < 0)) {   // the run ends: element i starts the next one
        if (r.runs >= kMaxRuns) { r.tracking = false; r.runs = kMaxRuns + 1; return; }
        r.dir[r.runs - 1] = r.cur_dir;
        r.bound[r.runs++] = i;
        r.cur_dir         = 0;
    }
}
inline void finish_runs(ScoutResult& r, size_t n) {
    if (!r.tracking) return;
    r.dir[r.runs - 1] = r.cur_dir < 0 ? -1 : 1;
    r.bound[r.runs]   = n;
}

// Scalar scout pass (also the counted path: every read and compare tallied).
// One three-way compare per element, starting at the shared-prefix offset.
template <class A>
inline ScoutResult scout_scalar(A a, size_t n, int chunk) {
    using T  = typename A::value_type;
    using KT = typename A::traits;
    ScoutResult r;
    T prev = a.get(0);
    const uint64_t k0 = A::key(prev, chunk);
    // The counts live in locals (the tracker takes r by reference, which
    // would otherwise keep every increment in memory) and are written back
    // at each commit check and at the end.
    size_t   desc = 0, asc = 0;
    uint64_t mask = 0;
    for (size_t i = 1; i < n; ++i) {
        const T cur = a.get(i);
        if constexpr (!KT::chunked) mask |= A::key(cur, chunk) ^ k0;   // cheap for fixed keys
        const int c = a.compare_from(cur, prev, chunk);
        // Only a pair that could end the current run (or start one) goes to
        // the tracker: on monotone data this test is all the tracking costs.
        // Equal pairs take one branch and nothing else, as they always did.
        if (c != 0) {
            desc += c < 0;
            asc  += c > 0;
            if (r.tracking && (r.cur_dir == 0 || (c < 0) != (r.cur_dir < 0))) track_pair(r, c, i);
        } else if (r.tracking && r.cur_dir <= 0) {
            track_pair(r, 0, i);
        }
        prev = cur;
        if ((i & (kCommitBlock - 1)) == 0) {
            r.descents = desc;
            r.ascents  = asc;
            if (commit_now(r, i)) { r.mask = mask; r.committed = true; return r; }
        }
    }
    r.descents = desc;
    r.ascents  = asc;
    r.mask     = mask;
    r.has_mask = !KT::chunked;
    finish_runs(r, n);
    return r;
}

// The varying-bit mask alone, for chunked keys that reach the radix route.
template <class A>
inline uint64_t compute_mask(A a, size_t n, int chunk) {
    const uint64_t k0 = A::key(a.get(0), chunk);
    uint64_t mask = 0;
    for (size_t i = 1; i < n; ++i) mask |= A::key(a.get(i), chunk) ^ k0;
    return mask;
}

// The signed key the vector paths compare: for SimdKind::i32 elements the
// signed 32-bit value in bytes 0-3 equals radix_key ^ 2^31, for i64 the
// signed 64-bit value in bytes 0-7 equals radix_key ^ 2^63.
template <class T> inline int32_t skey32(const T& e) {
    return static_cast<int32_t>(static_cast<uint32_t>(elem_traits<T>::radix_key(e, 0)) ^ 0x80000000u);
}
template <class T> inline int64_t skey64(const T& e) {
    return static_cast<int64_t>(static_cast<uint64_t>(elem_traits<T>::radix_key(e, 0)) ^ 0x8000000000000000ull);
}

#ifdef BRAINSORT_X86_64
BRAINSORT_TARGET_AVX2 inline __m256i ld256(const void* q) { return _mm256_loadu_si256(static_cast<const __m256i*>(q)); }

// Fold one block of V compare bits into the counts and the run tracking.
// dm/am have one bit per element of the block; bit bit_of[e] belongs to
// element i + e (the vector scouts leave the lanes in a permuted order
// rather than spend shuffles on straightening them). The run tracking only
// has to look inside a block when the block could end the current run: a
// descent in a non-decreasing run, an ascent or a tie in a strictly
// decreasing one. On monotone data that never happens; on unordered data
// tracking switches off after a few blocks. Either way the walk is rare.
template <int V>
inline void scout_block(ScoutResult& r, unsigned dm, unsigned am, size_t i, const uint8_t* bit_of) {
    r.descents += static_cast<size_t>(popcount(dm));
    r.ascents  += static_cast<size_t>(popcount(am));
    if (!r.tracking) return;
    const unsigned full = (1u << V) - 1;
    const unsigned eq   = ~(dm | am) & full;
    const bool walk = r.cur_dir > 0 ? dm != 0 : (r.cur_dir < 0 ? (am | eq) != 0 : true);
    if (!walk) return;
    for (int e = 0; e < V && r.tracking; ++e) {
        const unsigned b = bit_of[e];
        track_pair(r, (dm >> b) & 1 ? -1 : (((am >> b) & 1) ? 1 : 0), i + static_cast<size_t>(e));
    }
}

// Scalar tail shared by the vector scouts.
template <class T>
inline void scout_tail(ScoutResult& r, const T* p, size_t i, size_t n) {
    using KT = elem_traits<T>;
    const uint64_t k0 = KT::radix_key(p[0], 0);
    for (; i < n; ++i) {
        r.mask |= KT::radix_key(p[i], 0) ^ k0;
        const bool desc = KT::less(p[i], p[i - 1]);
        const bool asc  = KT::less(p[i - 1], p[i]);
        r.descents += desc;
        r.ascents  += asc;
        if (r.tracking && (r.cur_dir == 0 || desc != (r.cur_dir < 0))) track_pair(r, desc ? -1 : (asc ? 1 : 0), i);
    }
}

// Vectorised scout for int32 keys (8-byte elements), 8 elements per block.
// The elements are compared in place (keys sit in the even 32-bit lanes),
// and the two compare results are interleaved with a shift and a blend
// instead of gathering the keys: bit b of the block mask belongs to element
// (b >> 1) + 4 * (b & 1).
template <class T>
BRAINSORT_TARGET_AVX2 inline ScoutResult scout_avx2_32(const T* p, size_t n) {
    static_assert(sizeof(T) == 8, "SimdKind::i32 promises 8-byte elements");
    static constexpr uint8_t bit_of[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    ScoutResult r;
    const __m256i vk0   = _mm256_set1_epi32(skey32(p[0]));
    __m256i       vmask = _mm256_setzero_si256();
    size_t i = 1, next_check = kCommitBlock;
    for (; i + 8 <= n; i += 8) {
        const __m256i c0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i));
        const __m256i c1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i + 4));
        const __m256i q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i - 1));
        const __m256i q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i + 3));
        vmask = _mm256_or_si256(vmask, _mm256_or_si256(_mm256_xor_si256(c0, vk0), _mm256_xor_si256(c1, vk0)));
        const __m256i d = _mm256_blend_epi32(_mm256_cmpgt_epi32(q0, c0), _mm256_slli_epi64(_mm256_cmpgt_epi32(q1, c1), 32), 0xAA);
        const __m256i u = _mm256_blend_epi32(_mm256_cmpgt_epi32(c0, q0), _mm256_slli_epi64(_mm256_cmpgt_epi32(c1, q1), 32), 0xAA);
        scout_block<8>(r, static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(d))),
                       static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(u))), i, bit_of);
        if (i >= next_check) {
            next_check += kCommitBlock;
            if (commit_now(r, i + 8)) { r.committed = true; break; }
        }
    }
    alignas(32) uint64_t lanes[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), vmask);
    r.mask = static_cast<uint32_t>(lanes[0] | lanes[1] | lanes[2] | lanes[3]);   // the keys are the low halves
    if (r.committed) return r;   // the bits seen so far: a head start for the radix plan
    scout_tail(r, p, i, n);
    finish_runs(r, n);
    return r;
}

// Order-preserving transform of 4 doubles' bit patterns (matches the
// radix key of a double): negative -> 2^63 - magnitude, otherwise
// bits | 2^63. Both zeros map to 2^63.
BRAINSORT_TARGET_AVX2 inline __m256i f64_order_key(__m256i x) {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i sign = _mm256_set1_epi64x(static_cast<long long>(0x8000000000000000ull));
    const __m256i neg  = _mm256_cmpgt_epi64(zero, x);                       // all-ones where sign bit set
    const __m256i nk   = _mm256_sub_epi64(sign, _mm256_andnot_si256(sign, x));
    return _mm256_blendv_epi8(_mm256_or_si256(x, sign), nk, neg);
}

// Vectorised scout for 16-byte elements with a 64-bit key (double, int64),
// 8 elements per block: four vectors of two elements each, compared in place
// (keys in the even 64-bit lanes), each pair of results interleaved with one
// unpack, so bit b of the block mask belongs to element {0,2,1,3,4,6,5,7}[b].
// The keys are gathered with the same unpack for the mask.
BRAINSORT_TARGET_AVX2 inline __m256i gt_f64(__m256i x, __m256i y) {
    return _mm256_castpd_si256(_mm256_cmp_pd(_mm256_castsi256_pd(x), _mm256_castsi256_pd(y), _CMP_GT_OQ));
}
BRAINSORT_TARGET_AVX2 inline __m256i gt_i64(__m256i x, __m256i y) { return _mm256_cmpgt_epi64(x, y); }
template <bool F64> BRAINSORT_TARGET_AVX2 inline __m256i gt64(__m256i x, __m256i y) {
    if constexpr (F64) return gt_f64(x, y);
    else return gt_i64(x, y);
}
// F64: compare as doubles and transform the keys; otherwise signed 64-bit
// compare and raw keys. A template parameter, not a function argument, so
// the compare is always inlined into the block.
template <class T, bool F64>
BRAINSORT_TARGET_AVX2 inline ScoutResult scout_avx2_64(const T* p, size_t n) {
    static_assert(sizeof(T) == 16, "SimdKind::i64/f64 promise 16-byte elements");
    constexpr bool transform = F64;
    static constexpr uint8_t bit_of[8] = {0, 2, 1, 3, 4, 6, 5, 7};
    ScoutResult r;
    // Reference key in the domain the mask is built in: transformed for
    // doubles, raw for int64 (the sign flip cancels in the XOR).
    const uint64_t k0    = transform ? elem_traits<T>::radix_key(p[0], 0) : (elem_traits<T>::radix_key(p[0], 0) ^ 0x8000000000000000ull);
    const __m256i  vk0   = _mm256_set1_epi64x(static_cast<long long>(k0));
    __m256i        vmask = _mm256_setzero_si256();
    size_t i = 1, next_check = kCommitBlock;
    for (; i + 8 <= n; i += 8) {
        const __m256i c0 = ld256(p + i), c1 = ld256(p + i + 2), c2 = ld256(p + i + 4), c3 = ld256(p + i + 6);
        const __m256i q0 = ld256(p + i - 1), q1 = ld256(p + i + 1), q2 = ld256(p + i + 3), q3 = ld256(p + i + 5);
        __m256i keys0 = _mm256_unpacklo_epi64(c0, c1), keys1 = _mm256_unpacklo_epi64(c2, c3);
        if (transform) { keys0 = f64_order_key(keys0); keys1 = f64_order_key(keys1); }
        vmask = _mm256_or_si256(vmask, _mm256_or_si256(_mm256_xor_si256(keys0, vk0), _mm256_xor_si256(keys1, vk0)));
        const unsigned d0 = static_cast<unsigned>(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_unpacklo_epi64(gt64<F64>(q0, c0), gt64<F64>(q1, c1)))));
        const unsigned d1 = static_cast<unsigned>(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_unpacklo_epi64(gt64<F64>(q2, c2), gt64<F64>(q3, c3)))));
        const unsigned u0 = static_cast<unsigned>(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_unpacklo_epi64(gt64<F64>(c0, q0), gt64<F64>(c1, q1)))));
        const unsigned u1 = static_cast<unsigned>(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_unpacklo_epi64(gt64<F64>(c2, q2), gt64<F64>(c3, q3)))));
        scout_block<8>(r, d0 | (d1 << 4), u0 | (u1 << 4), i, bit_of);
        if (i >= next_check) {
            next_check += kCommitBlock;
            if (commit_now(r, i + 8)) { r.committed = true; break; }
        }
    }
    alignas(32) uint64_t lanes[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), vmask);
    r.mask = lanes[0] | lanes[1] | lanes[2] | lanes[3];
    if (r.committed) return r;
    scout_tail(r, p, i, n);
    finish_runs(r, n);
    return r;
}
template <class T>
BRAINSORT_TARGET_AVX2 inline ScoutResult scout_avx2(const T* p, size_t n) {
    constexpr SimdKind kind = elem_traits<T>::simd;
    if constexpr (kind == SimdKind::i32) return scout_avx2_32(p, n);
    else if constexpr (kind == SimdKind::f64) return scout_avx2_64<T, true>(p, n);
    else return scout_avx2_64<T, false>(p, n);
}

// In-place reversal of 8- or 16-byte elements, four or two per vector from
// each end. Type-agnostic: it moves bytes, so strings (pointer + length)
// reverse the same way.
template <class T>
BRAINSORT_TARGET_AVX2 inline void reverse_avx2(T* p, size_t n) {
    constexpr size_t V = 32 / sizeof(T);
    constexpr int    perm = sizeof(T) == 8 ? _MM_SHUFFLE(0, 1, 2, 3) : _MM_SHUFFLE(1, 0, 3, 2);
    size_t lo = 0, hi = n;
    while (hi - lo >= 2 * V) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + lo));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + hi - V));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p + lo), _mm256_permute4x64_epi64(b, perm));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p + hi - V), _mm256_permute4x64_epi64(a, perm));
        lo += V;
        hi -= V;
    }
    while (lo + 1 < hi) { --hi; std::swap(p[lo], p[hi]); ++lo; }
}
#endif

template <class A>
inline ScoutResult scout(A a, size_t n, int chunk) {
#ifdef BRAINSORT_X86_64
    if constexpr (!A::counted && A::traits::simd != SimdKind::none) {
        if (chunk == 0 && have_avx2()) {
            ScoutResult r = scout_avx2(a.data(), n);
            r.has_mask = !r.committed;
            return r;
        }
    }
#endif
    return scout_scalar(a, n, chunk);
}

// Reverse a[0,n) in place.
template <class A>
inline void reverse_all(A a, size_t n) {
#ifdef BRAINSORT_X86_64
    if constexpr (!A::counted && (sizeof(typename A::value_type) == 8 || sizeof(typename A::value_type) == 16)) {
        if (have_avx2()) { reverse_avx2(a.data(), n); return; }
    }
#endif
    reverse_range(a, 0, n);
}

// Route 2: reverse a non-increasing range while keeping equal keys in their
// original order. With the scout's run bounds the tie groups are known
// without another pass: in a non-increasing sequence a strictly decreasing
// run can only end at a tie, and a non-decreasing run consists of equal keys.
// Each group is reversed back after the whole range was reversed. Without
// the bounds (more than kMaxRuns runs) the groups are found by a scan.
template <class A>
inline void stable_reverse(A a, size_t n, const ScoutResult& s) {
    using T = typename A::value_type;
    reverse_all(a, n);
    if (s.descents == n - 1) return;   // strictly decreasing: no ties
    if (s.tracking) {
        for (size_t j = 0; j < s.runs; ++j) {
            size_t lo, hi;
            if (s.dir[j] > 0) {   // equal keys, plus the tie that ended a decreasing run before them
                lo = j > 0 && s.dir[j - 1] < 0 ? s.bound[j] - 1 : s.bound[j];
                hi = s.bound[j + 1];
            } else if (j + 1 < s.runs && s.dir[j + 1] < 0) {   // two decreasing runs meet at a tie pair
                lo = s.bound[j + 1] - 1;
                hi = lo + 2;
            } else {
                continue;
            }
            if (hi - lo > 1) reverse_range(a, n - hi, n - lo);   // the group's image after the reversal
        }
        return;
    }
    size_t i = 0;
    while (i < n) {
        size_t j = i + 1;
        const T first = a.get(i);
        while (j < n) {
            const T cur = a.get(j);
            if (a.less(first, cur) || a.less(cur, first)) break;
            ++j;
        }
        if (j - i > 1) reverse_range(a, i, j);
        i = j;
    }
}

// Stable in-place merge of a[lo,mid) and a[mid,hi) with a buffer of at least
// min(mid-lo, hi-mid) elements: the shorter run is copied out and the merge
// runs towards the free space, so the output never overtakes unread input.
// The merge branches on the compare on purpose: the runs this route sees are
// structured (an organ pipe, a few concatenated runs), so the branch predicts
// well and the loop runs ahead of the compare latency; a branch-free select
// measured 2.5x slower here, waiting on every compare.
template <class A>
inline void merge_with_buffer(A a, A buf, size_t lo, size_t mid, size_t hi) {
    using T = typename A::value_type;
    const size_t len1 = mid - lo, len2 = hi - mid;
    if (len1 == 0 || len2 == 0) return;
    if (len1 <= len2) {   // left run out, merge forward
        copy_forward(a, lo, buf, 0, len1);
        size_t i = 0, j = mid, o = lo;
        T x = buf.get(0), y = a.get(j);
        for (;;) {
            if (a.less(y, x)) {
                a.set(o++, y);
                if (++j == hi) break;
                y = a.get(j);
            } else {
                a.set(o++, x);
                if (++i == len1) break;
                x = buf.get(i);
            }
        }
        for (; i < len1; ++i) a.set(o++, buf.get(i));
    } else {              // right run out, merge backward (ties: right stays right)
        copy_forward(a, mid, buf, 0, len2);
        size_t i = mid, j = len2, o = hi;
        T x = a.get(i - 1), y = buf.get(j - 1);
        for (;;) {
            if (a.less(y, x)) {
                a.set(--o, x);
                if (--i == lo) break;
                x = a.get(i - 1);
            } else {
                a.set(--o, y);
                if (--j == 0) break;
                y = buf.get(j - 1);
            }
        }
        while (j > 0) a.set(--o, buf.get(--j));
    }
}

// Route 2b: a handful of monotone runs, with the bounds the scout recorded
// (timsort's run definition: ascending = non-decreasing, descending =
// strictly decreasing, so reversing a descending run in place keeps
// stability). No second detection pass. The buffer (n/2, the most any merge
// needs) is allocated first. Merges go pairwise, from the shorter side.
template <class A, class S>
inline void sort_few_runs(A a, S& scratch, size_t n, const ScoutResult& s) {
    A buf = scratch.ensure(n / 2 + 1);
    const size_t* bounds = s.bound;
    const size_t  runs   = s.runs;
    for (size_t j = 0; j < runs; ++j)
        if (s.dir[j] < 0) reverse_range(a, bounds[j], bounds[j + 1]);
    // Merge plan: (0,1), (2,3), then the two halves.
    if (runs >= 2) merge_with_buffer(a, buf, bounds[0], bounds[1], bounds[2]);
    if (runs == 3) merge_with_buffer(a, buf, bounds[0], bounds[2], bounds[3]);
    if (runs == 4) {
        merge_with_buffer(a, buf, bounds[2], bounds[3], bounds[4]);
        merge_with_buffer(a, buf, bounds[0], bounds[2], bounds[4]);
    }
}


// ---- route 3 ---------------------------------------------------------------

// Growable arrays for the displaced elements: the element, its original
// position, and the height of the kept stack when it was pulled out. Growth
// failure is reported, not thrown, so the route can restore the array and
// give up cleanly.
template <class A>
class DispBuf {
public:
    using T = typename A::value_type;
    static constexpr bool C = A::counted;
    explicit DispBuf(size_t cap) { allocate(cap); }
    ~DispBuf() { release(); }
    DispBuf(const DispBuf&) = delete;
    DispBuf& operator=(const DispBuf&) = delete;
    size_t capacity() const { return cap_; }
    bool   grow() {
        const size_t ncap = cap_ * 2;
        T* ne; uint32_t* np; uint32_t* nr;
        try { ne = A::template alloc_array<T>(ncap); } catch (const std::bad_alloc&) { return false; }
        try { np = A::template alloc_array<uint32_t>(ncap); } catch (const std::bad_alloc&) { A::template free_array<T>(ne, ncap); return false; }
        try { nr = A::template alloc_array<uint32_t>(ncap); } catch (const std::bad_alloc&) { A::template free_array<T>(ne, ncap); A::template free_array<uint32_t>(np, ncap); return false; }
        std::memcpy(ne, elem_, cap_ * sizeof(T));
        std::memcpy(np, pos_, cap_ * sizeof(uint32_t));
        std::memcpy(nr, rank_, cap_ * sizeof(uint32_t));
        release();
        elem_ = ne; pos_ = np; rank_ = nr; cap_ = ncap;
        return true;
    }
    T        elem(size_t i) const { if constexpr (C) A::hooks::on_read(elem_ + i, sizeof(T)); return elem_[i]; }
    void     set(size_t i, T e, uint32_t p, uint32_t r) {
        if constexpr (C) { A::hooks::on_write(elem_ + i, sizeof(T)); A::hooks::on_table_write(pos_ + i, 4); A::hooks::on_table_write(rank_ + i, 4); }
        elem_[i] = e; pos_[i] = p; rank_[i] = r;
    }
    uint32_t pos(size_t i) const { if constexpr (C) A::hooks::on_table_read(pos_ + i, 4); return pos_[i]; }
    uint32_t rank(size_t i) const { if constexpr (C) A::hooks::on_table_read(rank_ + i, 4); return rank_[i]; }
    uint32_t* ranks() { return rank_; }
    const uint32_t* positions() const { return pos_; }
private:
    void allocate(size_t cap) {
        elem_ = A::template alloc_array<T>(cap);
        try { pos_ = A::template alloc_array<uint32_t>(cap); } catch (...) { A::template free_array<T>(elem_, cap); throw; }
        try { rank_ = A::template alloc_array<uint32_t>(cap); } catch (...) { A::template free_array<T>(elem_, cap); A::template free_array<uint32_t>(pos_, cap); throw; }
        cap_ = cap;
    }
    void release() {
        if (elem_) A::template free_array<T>(elem_, cap_);
        if (pos_) A::template free_array<uint32_t>(pos_, cap_);
        if (rank_) A::template free_array<uint32_t>(rank_, cap_);
        elem_ = nullptr; pos_ = nullptr; rank_ = nullptr; cap_ = 0;
    }
    T*        elem_ = nullptr;
    uint32_t* pos_  = nullptr;
    uint32_t* rank_ = nullptr;
    size_t    cap_  = 0;
};

// Undo the compaction of route 3: a[0,nk) holds the kept elements, `disp`
// the nd displaced ones with their positions, a[scanned,n) is untouched.
// Puts every element of [0,scanned) back where it came from.
template <class A, class D>
inline void restore_displaced(A a, D& disp, size_t nd, size_t nk, size_t scanned) {
    uint32_t* idx = disp.ranks();   // ranks are not needed any more: reuse as an index array
    for (size_t t = 0; t < nd; ++t) idx[t] = static_cast<uint32_t>(t);
    const uint32_t* pos = disp.positions();
    std::sort(idx, idx + nd, [pos](uint32_t x, uint32_t y) { return pos[x] < pos[y]; });
    size_t t = nd, k = nk;
    for (size_t p = scanned; p-- > 0;) {
        if (t > 0 && pos[idx[t - 1]] == p) a.set(p, disp.elem(idx[--t]));
        else                                a.set(p, a.get(--k));   // k-1 <= p: not yet overwritten
    }
}

// Route 3. The kept (in-order) elements are compacted to a[0,nk) as the scan
// goes; displaced elements are moved to a side buffer with their original
// position. Returns false, with the array restored to its input order, if
// too many elements turn out to be displaced.
//
// Stability without storing every kept position: for a displaced element d
// let rank(d) be the number of kept elements that precede it in the input.
// Recording the kept-stack height at the moment d was pulled out and taking
// the suffix minimum over later pull-outs gives exactly that number, because
// the stack only ever loses elements from the top.
template <class A>
inline bool sort_displaced(A a, size_t n) {
    using T = typename A::value_type;
    constexpr bool C = A::counted;
    const size_t limit = n / 8;
    DispBuf<A> disp(std::min<size_t>(256, limit + 1));
    size_t   nk = 0, nd = 0;
    size_t   nk_max = 0;              // ring entries are valid for stack indices >= nk_max - kRing
    T        k1{}, k2{};              // the last two kept elements
    uint32_t ring[kRing];             // original positions of the top kept elements

    // When an element is smaller than the top of the kept subsequence, either
    // it is the outlier or the top few kept elements are (a swapped-in big
    // value, or several in a row). Pop up to kMaxPop kept elements if the
    // element fits right below them; otherwise displace the element itself.
    // The route gives up when displaced elements exceed 1/8 of what has been
    // scanned so far (plus slack), so unsuitable input costs a few thousand
    // elements instead of n/8.
    T* const p = a.data();   // the hot loop works on the raw pointer (counted: through the view)
    auto rd = [&](size_t i) -> T { if constexpr (C) return a.get(i); else return p[i]; };
    auto wr = [&](size_t i, T v) { if constexpr (C) a.set(i, v); else p[i] = v; };
    for (size_t i = 0; i < n; ++i) {
        const T it = rd(i);
        if (nk == 0 || !a.less(it, k1)) {           // extends the sorted subsequence
            ring[nk & (kRing - 1)] = static_cast<uint32_t>(i);
            wr(nk++, it);
            k2 = k1;
            k1 = it;
            continue;
        }
        size_t pops = 0;
        if (nk == 1 || !a.less(it, k2)) {
            pops = 1;
        } else {
            for (size_t j = 2; j <= kMaxPop; ++j) {  // rare path: look deeper
                if (nk <= j) { pops = j; break; }
                if (!a.less(it, rd(nk - 1 - j))) { pops = j; break; }
            }
        }
        const size_t budget = std::min(limit, (i >> 3) + 64);
        if (nd + std::max<size_t>(pops, 1) > budget + 1) { restore_displaced(a, disp, nd, nk, i); return false; }
        while (nd + std::max<size_t>(pops, 1) > disp.capacity())
            if (!disp.grow()) { restore_displaced(a, disp, nd, nk, i); return false; }
        if (pops == 0) {                            // this element is the outlier
            disp.set(nd++, it, static_cast<uint32_t>(i), static_cast<uint32_t>(nk));
            continue;
        }
        if (nk > nk_max) nk_max = nk;
        if (nk_max > kRing && nk - pops < nk_max - kRing) { restore_displaced(a, disp, nd, nk, i); return false; }
        for (size_t j = 1; j <= pops; ++j)
            disp.set(nd++, rd(nk - j), ring[(nk - j) & (kRing - 1)], static_cast<uint32_t>(nk - j));
        nk -= pops;
        ring[nk & (kRing - 1)] = static_cast<uint32_t>(i);
        wr(nk++, it);
        k1 = it;
        k2 = nk >= 2 ? rd(nk - 2) : T{};
    }
    if (nd == 0) return true;   // cannot happen when descents > 0, but harmless

    // rank(d) = suffix minimum of the recorded stack heights.
    {
        uint32_t* r  = disp.ranks();
        uint32_t  mn = r[nd - 1];
        for (size_t t = nd; t-- > 0;) { if (r[t] < mn) mn = r[t]; r[t] = mn; }
    }

    // Sort the displaced elements by (key, original position): a stable
    // bottom-up merge sort of an index array (positions break ties, so the
    // input order of the index array does not matter).
    AuxVec<uint32_t, A> idx_buf(2 * nd);
    auto idx = idx_buf.view();
    for (size_t t = 0; t < nd; ++t) idx.set(t, static_cast<uint32_t>(t));
    {
        size_t src = 0, dst = nd;   // halves of idx_buf
        for (size_t width = 1; width < nd; width *= 2) {
            for (size_t lo = 0; lo < nd; lo += 2 * width) {
                const size_t mid = std::min(lo + width, nd), hi = std::min(lo + 2 * width, nd);
                size_t i = lo, j = mid, o = lo;
                while (i < mid && j < hi) {
                    const uint32_t pi = idx.get(src + i), pj = idx.get(src + j);
                    const int c = a.compare(disp.elem(pj), disp.elem(pi));
                    if (c < 0 || (c == 0 && disp.pos(pj) < disp.pos(pi))) { idx.set(dst + o++, pj); ++j; }
                    else                                                    { idx.set(dst + o++, pi); ++i; }
                }
                while (i < mid) idx.set(dst + o++, idx.get(src + i++));
                while (j < hi)  idx.set(dst + o++, idx.get(src + j++));
            }
            std::swap(src, dst);
        }
        if (src != 0)
            for (size_t t = 0; t < nd; ++t) idx.set(t, idx.get(nd + t));
    }

    // Backward merge of the kept elements a[0,nk) and the sorted displaced
    // elements into a[0,n). The write index o = ik + id never overtakes the
    // unread kept elements. On equal keys the kept element goes after the
    // displaced one iff it is not among the rank(d) kept elements that
    // precede d in the input. Two plain less() branches, both predictable
    // on nearly sorted input (the kept element wins most of the time).
    size_t ik = nk, id = nd, o = n;
    uint32_t t  = idx.get(id - 1);
    T        ed = disp.elem(t);
    if (ik > 0) {
        T ek = a.get(ik - 1);
        for (;;) {
            bool take_kept;
            if (a.less(ed, ek))      take_kept = true;
            else if (a.less(ek, ed)) take_kept = false;
            else                     take_kept = ik - 1 >= disp.rank(t);
            if (take_kept) {
                a.set(--o, ek);
                if (--ik == 0) break;
                ek = a.get(ik - 1);
            } else {
                a.set(--o, ed);
                if (--id == 0) break;
                t  = idx.get(id - 1);
                ed = disp.elem(t);
            }
        }
    }
    while (id > 0) {   // kept elements exhausted: the rest of the displaced go in front
        a.set(--o, ed);
        if (--id == 0) break;
        t  = idx.get(id - 1);
        ed = disp.elem(t);
    }
    return true;   // the remaining kept elements are already in place
}


// ---- route 4 ---------------------------------------------------------------

// What a strided sample of the range says: the median (the split pivot), the
// sampled varying-bit mask and key range, and whether the sampled keys repeat
// enough for the dictionary radix to be worth a try.
struct Pivot {
    size_t   n_sample    = 0;
    size_t   n_ge        = 0;     // sample keys >= the pivot key
    uint64_t sample_mask = 0;     // OR of (sample key XOR key0): bits that vary in the sample
    uint64_t smin        = 0;     // smallest and largest sampled key
    uint64_t smax        = 0;
    bool     all_equal   = false; // every sampled key is the same
    bool     dict        = false; // few distinct sampled keys and a collision-free bucket hash for them
    uint64_t dict_mul    = 0;     // the multiplier of that hash
    size_t   n_distinct  = 0;     // distinct sampled keys, when counted (0: not counted); few[] holds up to four, ascending
    uint64_t few[4]      = {};
    size_t   few_cnt[4]  = {};    // how often each of them was sampled
};

// Dictionary radix: for a range with few distinct keys, one hashed bucket per
// key, the buckets laid out in key order. The count and key tables together
// are sized to the count tables a plain radix on the same part would use,
// so the route costs no extra memory: 2048 buckets for 32-bit keys (16 KiB),
// 4096 for 64-bit keys (48 KiB, inside the 64 KiB arena of 13-bit digits).
// Chunked (string) keys stay at 2048 buckets too: their split path uses
// 12-bit digit tables (32 KiB), and the dictionary must not need more.
template <class T> constexpr int kDictBits = (!elem_traits<T>::chunked && sizeof(typename elem_traits<T>::key_type) == 8) ? 12 : 11;
constexpr size_t kDictMaxDistinct = 128;   // dictionary only up to this many distinct sampled keys
template <class T> constexpr size_t dict_buckets() { return size_t(1) << kDictBits<T>; }
template <class T> constexpr size_t dict_entries() { return dict_buckets<T>() * (1 + sizeof(typename elem_traits<T>::key_type) / 4); }   // uint32 entries: cnt + rep

template <class T> inline uint32_t dict_bucket(typename elem_traits<T>::key_type k, uint64_t mul) {
    return static_cast<uint32_t>((static_cast<uint64_t>(k) * mul) >> (64 - kDictBits<T>));
}

// Deterministic odd multipliers to try (splitmix64 outputs).
struct DictMuls {
    uint64_t m[32];
    constexpr DictMuls() : m() {
        uint64_t x = 0x9E3779B97F4A7C15ull;
        for (uint64_t& v : m) {
            x += 0x9E3779B97F4A7C15ull;
            uint64_t z = x;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            v = (z ^ (z >> 31)) | 1;
        }
    }
};
inline constexpr DictMuls kDictMuls{};

// A multiplier whose bucket hash is injective on the d distinct sampled keys.
template <class T>
inline bool dict_pick(const uint64_t* keys, size_t d, uint64_t& mul) {
    using K = typename elem_traits<T>::key_type;
    for (uint64_t m : kDictMuls.m) {
        uint64_t seen[dict_buckets<T>() / 64] = {};
        bool     ok = true;
        for (size_t i = 0; i < d; ++i) {
            const uint32_t h = dict_bucket<T>(static_cast<K>(keys[i]), m);
            uint64_t&      w = seen[h >> 6];
            const uint64_t b = uint64_t(1) << (h & 63);
            if (w & b) { ok = false; break; }
            w |= b;
        }
        if (ok) { mul = m; return true; }
    }
    return false;
}

// Pivot for the split: the median radix key of a strided sample.
// Deterministic (no RNG), O(sample). For chunked keys the sample is taken on
// the current chunk. The sample's varying-bit mask and key range estimate how
// many radix passes the range will need and so whether a split pays. If the
// median key repeats in the sample, the distinct sampled keys are counted and,
// if there are few, a dictionary hash is prepared for them.
template <class A>
inline typename A::value_type pick_pivot(A a, size_t n, int chunk, Pivot& info) {
    using T  = typename A::value_type;
    size_t k = n >> 5;
    if (k > kSample) k = kSample;
    T        smp[kSample] = {};
    uint64_t keys[kSample];
    const size_t   stride = n / k;
    const uint64_t k0     = A::key(a.get(stride / 2), chunk);
    uint64_t m = 0, lo = k0, hi = k0;
    for (size_t i = 0; i < k; ++i) {
        smp[i] = a.get(i * stride + stride / 2);
        const uint64_t ki = A::key(smp[i], chunk);
        keys[i] = ki;
        m |= ki ^ k0;
        lo = ki < lo ? ki : lo;
        hi = ki > hi ? ki : hi;
    }
    info.sample_mask = m;
    info.all_equal   = m == 0;
    info.smin        = lo;
    info.smax        = hi;
    auto by_key = [chunk](T x, T y) { return A::key(x, chunk) < A::key(y, chunk); };
    std::nth_element(smp, smp + k / 2, smp + k, by_key);
    const T        pivot = smp[k / 2];
    const uint64_t pk    = A::key(pivot, chunk);
    info.n_sample = k;
    info.n_ge     = 0;
    size_t n_eq   = 0;
    for (size_t i = 0; i < k; ++i) { info.n_ge += keys[i] >= pk; n_eq += keys[i] == pk; }
    info.dict = false;
    if (n_eq >= 2 && m != 0) {   // the median repeats: few distinct keys are likely
        std::sort(keys, keys + k);
        size_t d = 0;
        for (size_t i = 0; i < k;) {
            size_t j = i + 1;
            while (j < k && keys[j] == keys[i]) ++j;
            if (d < 4) { info.few[d] = keys[i]; info.few_cnt[d] = j - i; }
            keys[d++] = keys[i];
            i = j;
        }
        info.n_distinct = d;
        if (d <= kDictMaxDistinct) info.dict = dict_pick<T>(keys, d, info.dict_mul);
    }
    return pivot;
}

// Stable two-way split of a[0,n) by key >= pivot key. One side is compacted
// in place, the other goes to buf[0,cap) (forward: the >= side, from the
// front; backward: the < side, from the back). Both sides are written every
// iteration and only the matching cursor advances, which keeps the loop free
// of unpredictable branches; the junk write always lands on the element's
// own slot or on dead space. If the buffer overflows (the sample misjudged
// which side is smaller), the buffered elements are copied into the gap they
// came from - a stable partial partition - and false is returned; the caller
// then runs the opposite direction, which is guaranteed to fit.
// On success the buffered side stays in buf and n_ge is set. When `mask` is
// given, the OR of (key XOR key0) over all elements is accumulated into it
// (the scout stopped early and did not finish it).
template <class A>
inline bool split_forward_scalar(A a, A buf, size_t n, size_t cap, uint64_t pk, int chunk, size_t& n_ge, uint64_t* mask) {
    using T  = typename A::value_type;
    using KT = typename A::traits;
    const uint64_t k0 = A::key(a.get(0), chunk);
    uint64_t m = 0;
    size_t w = 0, b = 0, i = 0;
    if constexpr (A::counted) {
        for (; i < n; ++i) {
            const T        e = a.get(i);
            const uint64_t k = A::key(e, chunk);
            m |= k ^ k0;
            if (k >= pk) { if (b == cap) break; buf.set(b++, e); }
            else a.set(w++, e);
        }
    } else {
        T*       pa  = a.data();
        T*       pb  = buf.data();
        const T* src = a.data();
        while (i < n) {
            // A stretch with enough buffer room for every element of it.
            const size_t end = std::min(n, i + (cap - b));
            if (end == i) break;
            for (; i < end; ++i) {
                const T        e = src[i];
                const uint64_t k = KT::radix_key(e, chunk);
                const size_t   g = k >= pk;
                m |= k ^ k0;
                pa[w] = e;
                pb[b] = e;
                w += 1 - g;
                b += g;
            }
        }
    }
    if (mask) *mask |= m;
    if (i < n) {   // overflow: fold the buffer back into the gap a[w, i)
        copy_forward(buf, 0, a, w, b);
        return false;
    }
    n_ge = b;      // the >= side lives in buf[0, n_ge)
    return true;
}
template <class A>
inline bool split_backward_scalar(A a, A buf, size_t n, size_t cap, uint64_t pk, int chunk, size_t& n_ge, uint64_t* mask) {
    using T  = typename A::value_type;
    using KT = typename A::traits;
    const uint64_t k0 = A::key(a.get(0), chunk);
    uint64_t m = 0;
    size_t w = n, b = cap, i = n;
    if constexpr (A::counted) {
        while (i > 0) {
            const T        e = a.get(i - 1);
            const uint64_t k = A::key(e, chunk);
            m |= k ^ k0;
            if (k >= pk) a.set(--w, e);
            else { if (b == 0) break; buf.set(--b, e); }
            --i;
        }
    } else {
        T*       pa  = a.data();
        T*       pb  = buf.data();
        const T* src = a.data();
        while (i > 0) {
            const size_t begin = i > b ? i - b : 0;
            if (begin == i) break;
            while (i > begin) {
                const T        e = src[--i];
                const uint64_t k = KT::radix_key(e, chunk);
                const size_t   g = k >= pk;
                m |= k ^ k0;
                pa[w - 1] = e;   // w-1 >= i: the element's own slot or dead space
                pb[b - 1] = e;   // b >= 1 throughout the stretch
                w -= g;
                b -= 1 - g;
            }
        }
    }
    if (mask) *mask |= m;
    if (i > 0) {   // overflow: fold the buffer back into the gap a[i, w)
        copy_forward(buf, b, a, i, cap - b);
        return false;
    }
    n_ge = n - w;  // the < side lives in buf[cap - n_lt, cap)
    return true;
}

#ifdef BRAINSORT_X86_64
// AVX2 forward split for the fixed-key element types, on the vector width
// (4 or 2 elements). Elements are compressed to the front of a vector with a
// permutation looked up by the compare mask, and the whole vector is stored:
// the lanes past the selected elements are junk that lands on dead space
// (the array side never overtakes the read position, the buffer side has
// slack). The scalar loop finishes the tail and any stretch where the
// buffer is nearly full.
struct CompressLut {
    int32_t by4[16][8];   // 4 elements of 2 lanes: mask -> lane permutation, selected first
    int32_t by2[4][8];    // 2 elements of 4 lanes
    constexpr CompressLut() : by4(), by2() {
        for (int m = 0; m < 16; ++m) {
            int o = 0;
            for (int e = 0; e < 4; ++e) if (m >> e & 1) { by4[m][o++] = 2 * e; by4[m][o++] = 2 * e + 1; }
            for (; o < 8; ++o) by4[m][o] = 0;
        }
        for (int m = 0; m < 4; ++m) {
            int o = 0;
            for (int e = 0; e < 2; ++e) if (m >> e & 1) for (int l = 0; l < 4; ++l) by2[m][o++] = 4 * e + l;
            for (; o < 8; ++o) by2[m][o] = 0;
        }
    }
};
inline constexpr CompressLut kLut{};

// Per-kind compare: returns, for a loaded vector, the "< pivot" element mask
// (bit e for element e) and ORs the element keys XOR key0 into vmask.
template <class T>
BRAINSORT_TARGET_AVX2 inline unsigned lt_mask(__m256i v, __m256i vpivot, __m256i vk0, __m256i& vmask) {
    constexpr SimdKind kind = elem_traits<T>::simd;
    if constexpr (kind == SimdKind::i32) {
        vmask = _mm256_or_si256(vmask, _mm256_xor_si256(v, vk0));
        __m256i lt = _mm256_cmpgt_epi32(vpivot, v);                                  // key lanes 0,2,4,6
        lt = _mm256_shuffle_epi32(lt, _MM_SHUFFLE(2, 2, 0, 0));                    // copy each key lane over its id lane
        return static_cast<unsigned>(_mm256_movemask_pd(_mm256_castsi256_pd(lt)));   // one bit per element
    } else if constexpr (kind == SimdKind::i64) {
        vmask = _mm256_or_si256(vmask, _mm256_xor_si256(v, vk0));
        const __m256i lt = _mm256_cmpgt_epi64(vpivot, v);                            // key lanes 0,2
        const unsigned m = static_cast<unsigned>(_mm256_movemask_pd(_mm256_castsi256_pd(lt)));
        return (m & 1) | ((m >> 1) & 2);
    } else {
        // Compare on the total-order keys (signed compare of sign-flipped keys ==
        // unsigned compare), so the vector loop agrees with the scalar tail and
        // the count pass even for NaN.
        const __m256i sign = _mm256_set1_epi64x(static_cast<long long>(0x8000000000000000ull));
        const __m256i k    = f64_order_key(v);
        vmask = _mm256_or_si256(vmask, _mm256_xor_si256(k, vk0));
        const __m256i lt = _mm256_cmpgt_epi64(vpivot, _mm256_xor_si256(k, sign));
        const unsigned m = static_cast<unsigned>(_mm256_movemask_pd(_mm256_castsi256_pd(lt)));
        return (m & 1) | ((m >> 1) & 2);
    }
}
// Broadcast a raw radix key the way lt_mask expects a pivot: int32/int64 as
// the signed key, doubles as the transformed key with the sign flipped.
template <class T>
BRAINSORT_TARGET_AVX2 inline __m256i pivot_vec_key(typename elem_traits<T>::key_type pk) {
    if constexpr (elem_traits<T>::simd == SimdKind::i32) return _mm256_set1_epi32(static_cast<int32_t>(static_cast<uint32_t>(pk) ^ 0x80000000u));
    else return _mm256_set1_epi64x(static_cast<long long>(static_cast<uint64_t>(pk) ^ 0x8000000000000000ull));
}
template <class T> BRAINSORT_TARGET_AVX2 inline __m256i pivot_vec(T pivot) { return pivot_vec_key<T>(elem_traits<T>::radix_key(pivot, 0)); }
// The reference key for the mask, in the domain the mask is folded in.
template <class T> BRAINSORT_TARGET_AVX2 inline __m256i key0_vec(T e) {
    constexpr SimdKind kind = elem_traits<T>::simd;
    if constexpr (kind == SimdKind::i32) return _mm256_set1_epi32(skey32(e));
    else if constexpr (kind == SimdKind::i64) return _mm256_set1_epi64x(skey64(e));
    else return _mm256_set1_epi64x(static_cast<long long>(elem_traits<T>::radix_key(e, 0)));
}
template <class T> BRAINSORT_TARGET_AVX2 inline uint64_t fold_mask(__m256i vmask) {
    alignas(32) uint64_t l[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(l), vmask);
    if constexpr (elem_traits<T>::simd == SimdKind::i32) {   // int32 keys in the low half of every 64-bit lane
        return static_cast<uint32_t>(l[0] | l[1] | l[2] | l[3]);
    } else {                                                  // 64-bit keys in lanes 0 and 2
        return l[0] | l[2];
    }
}

// Forward split of a[0,n) with AVX2, then the scalar loop for the rest.
template <class A>
BRAINSORT_TARGET_AVX2 inline bool split_forward_avx2(A a, A buf, size_t n, size_t cap,
                                                     typename A::value_type pivot, size_t& n_ge, uint64_t* mask) {
    using T  = typename A::value_type;
    using KT = elem_traits<T>;
    constexpr size_t V = 32 / sizeof(T);   // elements per vector
    T*       pa  = a.data();
    T*       pb  = buf.data();
    const T* src = a.data();
    const __m256i vp  = pivot_vec<T>(pivot);
    const __m256i vk0 = key0_vec<T>(src[0]);
    __m256i vmask = _mm256_setzero_si256();
    size_t w = 0, b = 0, i = 0;
    for (; i + V <= n && b + V <= cap; i += V) {
        const __m256i v  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        const unsigned lt = lt_mask<T>(v, vp, vk0, vmask);
        const unsigned ge = ~lt & ((1u << V) - 1);
        const int32_t* il = V == 4 ? kLut.by4[lt] : kLut.by2[lt];
        const int32_t* ig = V == 4 ? kLut.by4[ge] : kLut.by2[ge];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(pa + w),
                            _mm256_permutevar8x32_epi32(v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(il))));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(pb + b),
                            _mm256_permutevar8x32_epi32(v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ig))));
        w += static_cast<size_t>(popcount(lt));
        b += static_cast<size_t>(popcount(ge));
    }
    uint64_t m = fold_mask<T>(vmask);
    // Scalar rest: same predicate on the keys, same stretch logic.
    const uint64_t pk = KT::radix_key(pivot, 0);
    const uint64_t k0 = KT::radix_key(src[0], 0);
    while (i < n) {
        const size_t end = std::min(n, i + (cap - b));
        if (end == i) break;
        for (; i < end; ++i) {
            const T        e = src[i];
            const uint64_t k = KT::radix_key(e, 0);
            const size_t   g = k >= pk;
            m |= k ^ k0;
            pa[w] = e;
            pb[b] = e;
            w += 1 - g;
            b += g;
        }
    }
    if (mask) *mask |= m;
    if (i < n) {
        copy_forward(buf, 0, a, w, b);
        return false;
    }
    n_ge = b;
    return true;
}
#endif

template <class A>
inline bool split_forward(A a, A buf, size_t n, size_t cap, typename A::value_type pivot, int chunk, size_t& n_ge, uint64_t* mask) {
#ifdef BRAINSORT_X86_64
    if constexpr (!A::counted && A::traits::simd != SimdKind::none) {
        if (chunk == 0 && have_avx2()) return split_forward_avx2(a, buf, n, cap, pivot, n_ge, mask);
    }
#endif
    return split_forward_scalar(a, buf, n, cap, A::key(pivot, chunk), chunk, n_ge, mask);
}

// Digit width for `bits` key bits over m elements. The width is capped at 13
// bits for a decisive reason beyond cache locality: two 13-bit count tables
// are 64 KiB, which stays under the allocator's 128 KiB mmap threshold, so
// the histogram arena is served from the heap and reused across sorts instead
// of being mmap'd and munmap'd every call. 16-bit tables (512 KiB) are a few
// percent fewer cycles on full-entropy data but lose far more than that to
// the page faults of per-sort mmap, which a benchmark's warm repetitions hide
// but a real program's one-shot sorts pay in full. Within the cap, the width
// is chosen to balance the passes (32 bits -> 3x11, not 13+13+6).
inline int digit_width(int bits, size_t m, int max_digit) {
    const int cap = std::clamp(static_cast<int>(floor_log2(m)), 8, max_digit);
    if (bits <= cap) return bits;
    const int passes = (bits + cap - 1) / cap;
    return (bits + passes - 1) / passes;
}

// Digit-width caps. The whole-array (no-split) radix uses up to 16-bit digits,
// like the original design: its one count table is allocated once per sort,
// a pattern the allocator's dynamic mmap threshold keeps cheap. The split
// path caps at 13 bits (64 KiB tables) because its two radix calls of
// changing sizes defeat that heuristic, so a 512 KiB table would be mmap'd
// and munmap'd every sort; 13-bit tables stay under the 128 KiB mmap
// threshold and are served from the heap, and on the high-entropy data that
// takes the split the extra pass this sometimes costs is invisible next to
// the halved working set.
constexpr int kRadixMaxDigit = 16;
constexpr int kSplitMaxDigit = 13;

// The split halves the radix buffer (its reason for existing) but costs one
// extra streaming pass. That pass pays for itself only when the radix runs
// several passes, where halving the working set helps the cache and the
// memory saving is largest; on a key that sorts in one or two passes the
// extra pass is pure overhead, so the split is taken only from this many
// passes up.
constexpr int kSplitMinPasses = 3;

// A radix plan: the key function, the number of bits it yields, the digit
// width, and whether the plan is exact (derived from the true varying-bit
// mask) or a guess from the sample that the histogram pass must verify
// against every key. The key functions:
//   full   the raw radix key;
//   shift  the contiguous varying-bit range, shifted down;
//   pext   only the varying bits, compressed with BMI2 PEXT;
//   sub    key minus a base below the sampled minimum, for keys whose range
//          is narrow but straddles a power of two (a sign change, say), where
//          the XOR mask sees every bit varying.
struct RadixPlan {
    enum Kind : int { full = 0, shift = 1, pext = 2, sub = 3 };
    int      kind    = full;
    int      bits    = 0;               // key bits in the key function's domain; 0: nothing varies
    int      width   = 1;               // digit width
    int      low     = 0;               // shift: bits below `low` are constant
    uint64_t pmask   = 0;               // pext: the varying bits
    uint64_t base    = 0;               // sub: key - base
    uint64_t assumed = ~uint64_t(0);    // a valid plan needs the true XOR mask within this
    bool     exact   = true;            // no verification needed
    int    passes() const { return bits > 0 ? (bits + width - 1) / width : 0; }
    size_t table_entries() const { return size_t(2) << width; }   // two live count tables
};

// The cheapest plan for a range of m elements whose varying bits are `est`
// (exact if mask_exact, else the best estimate so far), optionally with a
// sampled key range. Fewest passes first, then markedly smaller tables, then
// exact over speculative. The sampled range is widened by
// itself on both sides, so an unsampled key outside it is rare; if one turns
// up anyway the histogram pass catches it and the exact plan takes over.
template <class K>
inline RadixPlan choose_plan(uint64_t est64, bool mask_exact, bool have_range, uint64_t rmin, uint64_t rmax,
                             size_t m, int max_digit, int force_width) {
    constexpr int kb  = key_bits<K>();
    const K       est = static_cast<K>(est64);
    RadixPlan best;
    if (mask_exact && est == 0) return best;   // nothing varies
    bool have = false;
    auto consider = [&](RadixPlan p) {
        p.width = force_width > 0 ? force_width : digit_width(p.bits, m, max_digit);
        if (!have) { best = p; have = true; return; }
        if (p.passes() != best.passes()) { if (p.passes() < best.passes()) best = p; return; }
        // Smaller tables win a tie only when they are at least 8x smaller
        // (a 512 KiB table against 8 KiB); between two cache-resident sizes
        // the extra shift or pext per element costs more than they save.
        if (p.width + 3 <= best.width) { best = p; return; }
        if (best.width + 3 <= p.width) return;
        if (p.exact != best.exact)       { if (p.exact) best = p; return; }
    };
    {   // full: always exact, whatever the mask
        RadixPlan p;
        p.kind = RadixPlan::full; p.bits = kb; p.exact = true;
        consider(p);
    }
    if (est != 0) {
        const int low = ctz(est), high = highest_bit(est);
        RadixPlan p;
        p.kind = RadixPlan::shift; p.bits = high - low + 1; p.low = low; p.exact = mask_exact;
        p.assumed = p.bits >= kb ? ~uint64_t(0) : (((uint64_t(1) << p.bits) - 1) << low);
        consider(p);
#ifdef BRAINSORT_X86_64
        if (have_bmi2()) {
            RadixPlan q;
            q.kind = RadixPlan::pext; q.bits = popcount(est); q.pmask = est; q.assumed = est; q.exact = mask_exact;
            consider(q);
        }
#endif
    }
    if (have_range && rmax > rmin) {
        const K lo = static_cast<K>(rmin), hi = static_cast<K>(rmax), r = hi - lo;
        const K base = lo < r ? K(0) : lo - r;
        const K top  = hi > static_cast<K>(~K(0)) - r ? ~K(0) : hi + r;
        RadixPlan p;
        p.kind = RadixPlan::sub; p.bits = highest_bit(top - base) + 1; p.base = base; p.exact = false;
        consider(p);
    }
    return best;
}

// Key functor for the sub plan (radix_detail has full, shift and pext).
template <class A> struct SubKey {
    using T = typename A::value_type;
    using K = typename A::key_type;
    int chunk;
    K   base;
    K operator()(T x) const { return A::key(x, chunk) - base; }
    K raw(K k) const { return k - base; }
};

// Histogram storage handed to the radix: {pointer, entry count}. Reused
// across every radix call of one sort so the count tables are allocated once.
struct HistStore {
    uint32_t* p = nullptr;
    size_t    n = 0;
};

// Radix of src[0,n) with dst[0,n) as the other buffer under plan P, with
// the result in src or dst as requested. Builds the histogram of the first
// live pass here, verifying a speculative plan on the way: the exact XOR
// mask of the keys and any key bits above the plan's width are accumulated,
// and if either shows the sample misjudged the keys the function returns
// false without moving anything (xm_out then holds the exact mask for the
// retry). Always inlined so a BMI2-targeted key functor is inlined into a
// BMI2 caller.
template <class A, class KeyFn>
BRAINSORT_ALWAYS_INLINE bool radix_exec(A src, A dst, size_t n, int chunk, const RadixPlan& P, KeyFn key,
                                        uint64_t domain_mask, bool result_in_src, HistStore hs, uint64_t& xm_out) {
    using namespace radix_detail;
    using T  = typename A::value_type;
    using K  = typename A::key_type;
    constexpr int kb = key_bits<K>();
    const DigitPlan plan = make_plan(P.bits, P.width);
    int       order[kMaxPasses];
    const int live = live_passes(plan, domain_mask, order);
    if (live == 0) {
        if (!result_in_src) copy_forward(src, 0, dst, 0, n);
        return true;
    }
    const size_t        width = plan.width();
    const size_t        need  = live > 1 ? 2 * width : width;
    AuxRaw<uint32_t, A> local(hs.p && hs.n >= need ? 0 : need);
    uint32_t*           base   = (hs.p && hs.n >= need) ? hs.p : local.data();
    uint32_t*           tab[2] = {base, live > 1 ? base + width : base};
    std::memset(tab[0], 0, width * sizeof(uint32_t));
    if constexpr (A::counted) {
        A::hooks::stats().passes_skipped += static_cast<uint64_t>(plan.passes - live);
        A::hooks::on_table_sweep(tab[0], width, sizeof(uint32_t), false, true);
    }
    const int      shift = plan.shift[order[0]];
    const uint32_t dmask = (1u << plan.bits[order[0]]) - 1;
    if (P.exact) {
        for (size_t i = 0; i < n; ++i) {
            const uint32_t dg = static_cast<uint32_t>(key(src.get(i)) >> shift) & dmask;
            if constexpr (A::counted) A::hooks::on_table_rw(tab[0] + dg, sizeof(uint32_t));
            ++tab[0][dg];
        }
    } else {
        // The XOR mask check covers shift and pext (bits outside the plan
        // must be constant); sub needs every key within [base, base + 2^bits).
        const K k0     = A::key(src.get(0), chunk);
        const K ovmask = P.kind == RadixPlan::sub && P.bits < kb ? static_cast<K>(~K(0) << P.bits) : K(0);
        K xm = 0, ov = 0;
        for (size_t i = 0; i < n; ++i) {
            const T e  = src.get(i);
            const K kr = A::key(e, chunk);
            const K u  = key.raw(kr);
            xm |= kr ^ k0;
            ov |= u & ovmask;
            const uint32_t dg = static_cast<uint32_t>(u >> shift) & dmask;
            if constexpr (A::counted) A::hooks::on_table_rw(tab[0] + dg, sizeof(uint32_t));
            ++tab[0][dg];
        }
        xm_out = xm;
        if ((xm & ~static_cast<K>(P.assumed)) != 0 || ov != 0) return false;
    }
    radix_passes<uint32_t>(src, dst, n, plan, key, order, live, 0, tab, result_in_src);
    return true;
}

#ifdef BRAINSORT_X86_64
template <class A>
BRAINSORT_TARGET_BMI2 inline bool radix_exec_pext(A src, A dst, size_t n, int chunk, const RadixPlan& P,
                                                  uint64_t domain_mask, bool result_in_src, HistStore hs, uint64_t& xm_out) {
    using K = typename A::key_type;
    return radix_exec(src, dst, n, chunk, P, radix_detail::PextKey<A>{chunk, static_cast<K>(P.pmask)}, domain_mask,
                      result_in_src, hs, xm_out);
}
#endif

// Dispatch on the plan's key function. `est` is the varying-bit mask the plan
// was made from; for an exact plan it tells which passes are trivial.
template <class A>
inline bool radix_run(A src, A dst, size_t n, int chunk, const RadixPlan& P, uint64_t est, bool result_in_src, HistStore hs, uint64_t& xm_out) {
    using namespace radix_detail;
    using K = typename A::key_type;
    [[maybe_unused]] constexpr int kb = key_bits<K>();
    const uint64_t all = ~uint64_t(0);
    switch (P.kind) {
        case RadixPlan::shift:
            return radix_exec(src, dst, n, chunk, P, ShiftKey<A>{chunk, P.low}, P.exact ? (est >> P.low) : all, result_in_src, hs, xm_out);
#ifdef BRAINSORT_X86_64
        case RadixPlan::pext:
            return radix_exec_pext(src, dst, n, chunk, P, P.exact && P.bits < kb ? (uint64_t(1) << P.bits) - 1 : all, result_in_src, hs, xm_out);
#endif
        case RadixPlan::sub:
            return radix_exec(src, dst, n, chunk, P, SubKey<A>{chunk, static_cast<K>(P.base)}, all, result_in_src, hs, xm_out);
        default:
            return radix_exec(src, dst, n, chunk, P, FullKey<A>{chunk}, P.exact ? est : all, result_in_src, hs, xm_out);
    }
}

// Dictionary radix of src[0,n): one hashed bucket per distinct key, verified
// as it is counted (every element of a bucket must carry the bucket's key),
// then the occupied buckets are laid out in key order and one scatter pass
// sorts. Returns false, with nothing moved, if two distinct keys shared a
// bucket; xm_out then holds the exact XOR mask for the fallback.
template <class A>
inline bool dict_sort(A src, A dst, size_t n, int chunk, uint64_t mul, bool result_in_src, HistStore hs, uint64_t& xm_out) {
    using T  = typename A::value_type;
    using K  = typename A::key_type;
    constexpr size_t B = dict_buckets<T>();
    uint32_t* cnt = hs.p;
    K*        rep = reinterpret_cast<K*>(hs.p + B);
    std::memset(cnt, 0, B * sizeof(uint32_t));
    if constexpr (A::counted) A::hooks::on_table_sweep(cnt, B, sizeof(uint32_t), false, true);
    const K k0 = A::key(src.get(0), chunk);
    K xm = 0, bad = 0;
    for (size_t i = 0; i < n; ++i) {
        const T        e = src.get(i);
        const K        k = A::key(e, chunk);
        const uint32_t h = dict_bucket<T>(k, mul);
        if constexpr (A::counted) { A::hooks::on_table_rw(cnt + h, sizeof(uint32_t)); A::hooks::on_table_rw(rep + h, sizeof(K)); }
        const uint32_t c = cnt[h];
        xm |= k ^ k0;
        if (c == 0) rep[h] = k;   // first sight of this bucket: rare, predictable
        else bad |= rep[h] ^ k;
        cnt[h] = c + 1;
    }
    xm_out = xm;
    if (bad) return false;
    uint32_t occ[B];
    size_t   nocc = 0;
    if constexpr (A::counted) A::hooks::on_table_sweep(cnt, B, sizeof(uint32_t), true, false);
    for (uint32_t h = 0; h < B; ++h) if (cnt[h]) occ[nocc++] = h;
    std::sort(occ, occ + nocc, [rep](uint32_t x, uint32_t y) { return rep[x] < rep[y]; });
    uint32_t sum = 0;
    for (size_t j = 0; j < nocc; ++j) {
        if constexpr (A::counted) A::hooks::on_table_rw(cnt + occ[j], sizeof(uint32_t));
        const uint32_t c = cnt[occ[j]]; cnt[occ[j]] = sum; sum += c;
    }
    if constexpr (A::counted) ++A::hooks::stats().radix_passes;
    for (size_t i = 0; i < n; ++i) {
        const T        e = src.get(i);
        const uint32_t h = dict_bucket<T>(A::key(e, chunk), mul);
        if constexpr (A::counted) A::hooks::on_table_rw(cnt + h, sizeof(uint32_t));
        dst.set(cnt[h]++, e);
    }
    if (result_in_src) copy_forward(dst, 0, src, 0, n);
    return true;
}

// The element scratch buffer, allocated on first use and sized by the route
// that needs it, so the routes that need none (sorted, reversed) cost no
// memory and the others get exactly what they ask for.
template <class A>
class Scratch {
public:
    using T = typename A::value_type;
    Scratch() = default;
    ~Scratch() { delete buf_; delete hist_; }
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;
    // A view of k elements; grows (discarding contents) if needed.
    A ensure(size_t k) {
        if (!buf_) buf_ = new AuxBuffer<T, A>(k);
        else if (buf_->size() < k) buf_->resize_discard(k);
        return buf_->arr().sub(0, k);
    }
    size_t capacity() const { return buf_ ? buf_->size() : 0; }
    // The reusable histogram arena (uint32 counters), grown to the largest
    // table a radix pass has asked for and kept for the rest of the sort.
    HistStore hist(size_t entries) {
        if (!hist_) hist_ = new AuxRaw<uint32_t, A>(entries);
        else if (hist_->size() < entries) { delete hist_; hist_ = nullptr; hist_ = new AuxRaw<uint32_t, A>(entries); }
        return HistStore{hist_->data(), hist_->size()};
    }
private:
    AuxBuffer<T, A>*     buf_  = nullptr;
    AuxRaw<uint32_t, A>* hist_ = nullptr;
};

// Radix sort of one part, src[0,n) with dst[0,n) as the other buffer, the
// result in src or dst as requested. `mask` is the part's varying-bit mask,
// exact or estimated; [rmin, rmax] the sampled key range if have_range. The
// dictionary is tried first when the sample called for it and the plan would
// otherwise need two passes or more; a speculative plan that fails its
// verification is replaced by the exact plan (the failed pass measured the
// exact mask), so no pass is ever wasted on a wrong assumption twice.
template <int DigitBits, class A, class S>
inline void radix_part(A src, A dst, size_t n, int chunk, S& scratch, uint64_t mask, bool mask_exact, const Pivot& info,
                       bool have_range, uint64_t rmin, uint64_t rmax, int max_digit, bool result_in_src) {
    using T = typename A::value_type;
    using K = typename A::key_type;
    auto done = [&] { if (!result_in_src) copy_forward(src, 0, dst, 0, n); };
    if (n < 2) { done(); return; }
    RadixPlan P = choose_plan<K>(mask, mask_exact, have_range, rmin, rmax, n, max_digit, DigitBits);
    if (P.bits == 0) { done(); return; }
    uint64_t xm = 0;
    if (info.dict && P.passes() >= 2) {
        if constexpr (A::counted) ++A::hooks::stats().dict_tries;
        if (dict_sort(src, dst, n, chunk, info.dict_mul, result_in_src, scratch.hist(dict_entries<T>()), xm)) {
            if constexpr (A::counted) ++A::hooks::stats().dict_hits;
            return;
        }
        mask = xm; mask_exact = true;
        P = choose_plan<K>(mask, true, false, 0, 0, n, max_digit, DigitBits);
        if (P.bits == 0) { done(); return; }
    }
    if (radix_run(src, dst, n, chunk, P, mask, result_in_src, scratch.hist(P.table_entries()), xm)) return;
    if constexpr (A::counted) ++A::hooks::stats().plan_retries;
    P = choose_plan<K>(xm, true, false, 0, 0, n, max_digit, DigitBits);   // exact: cannot fail
    if (P.bits == 0) { done(); return; }
    radix_run(src, dst, n, chunk, P, xm, result_in_src, scratch.hist(P.table_entries()), xm);
}

// ---- route 4 for two to four distinct keys: partition sort ------------------
// Four distinct keys (flags, enum codes, quartiles) are common, and a counting
// radix is a poor fit for them: with four buckets the counter increments in
// the scatter form store-to-load dependency chains, and a histogram pass plus
// a copy back are spent for two bits of information. A stable partition needs
// no counters. With the distinct sampled keys v0 < v1 < ... the range is
// split at a middle key exactly like the median split (one side compacted in
// place, the other in the half-size buffer), counting each side's lower key
// with popcounts on the way; then each side is partitioned at its own key
// straight into its final place, the buffered side into the array, the array
// side through the buffer. Two passes and a half instead of three, and n/2
// scratch instead of n. Two distinct keys need the first pass only.
//
// The first pass also checks that every key is one of the sampled ones; the
// partition by a key threshold is a correct stable split whatever the sample
// said, so if an unsampled key turns up the two sides continue on the general
// radix path with nothing lost.

// First pass, scalar (also the counted path): stable split of a[0,n) by
// key >= t2 into a (compacted) and buf[0,cap), with the exact XOR mask, the
// counts of keys < t1 (the lower key of the < side) and of keys in [t2, t3)
// (the lower key of the >= side), and `known`, false if some key is none of
// v[0..3]. Returns false on buffer overflow with the buffered elements folded
// back into their gap (a stable partial partition), like split_forward_scalar.
template <class A, class K>
inline bool split2_scalar(A a, A buf, size_t n, size_t cap, int chunk, const K* v, K t1, K t2, K t3,
                          uint64_t& xm, size_t& c0_lt, size_t& c0_ge, size_t& n_ge, bool& known) {
    using T  = typename A::value_type;
    const K k0 = A::key(a.get(0), chunk);
    K      m = 0;
    bool   unknown = false;
    size_t w = 0, b = 0, i = 0, c1 = 0, c3 = 0;
    for (; i < n; ++i) {
        const T e = a.get(i);
        const K k = A::key(e, chunk);
        m |= k ^ k0;
        unknown |= (k != v[0]) & (k != v[1]) & (k != v[2]) & (k != v[3]);
        c1 += k < t1;
        c3 += k < t3;
        if (k >= t2) { if (b == cap) break; buf.set(b++, e); }
        else a.set(w++, e);
    }
    xm    = m;
    known = !unknown;
    if (i < n) { copy_forward(buf, 0, a, w, b); return false; }
    n_ge  = b;
    c0_lt = c1;        // keys < t1 (all of them are < t2)
    c0_ge = c3 - w;    // keys in [t2, t3): (keys < t3) minus (keys < t2), and keys < t2 is w
    return true;
}

// Second pass, scalar: stable partition of src[0,m) by key < t into
// dst[o0, o1) (o1 = o0 + count of the lower key) and dst[o1, end1). Both
// slots are written every step and only the matching cursor advances; once a
// class is full the rest is copied plainly, so no junk write lands on a
// finished element or outside the range.
template <class A, class K>
inline void partition2_scalar(A src, size_t m, A dst, size_t o0, size_t o1, size_t end1, int chunk, K t) {
    using T  = typename A::value_type;
    size_t i = 0, c0 = o0, c1 = o1;
    for (; i < m && c0 < o1 && c1 < end1; ++i) {
        const T    e = src.get(i);
        const bool g = A::key(e, chunk) >= t;
        dst.set(c0, e);
        dst.set(c1, e);
        c0 += !g;
        c1 += g;
    }
    for (; i < m; ++i) {
        const T e = src.get(i);
        if (A::key(e, chunk) < t) dst.set(c0++, e);
        else dst.set(c1++, e);
    }
}

#ifdef BRAINSORT_X86_64
// The loaded vector in the domain the pivots of pivot_vec_key compare in.
template <class T> BRAINSORT_TARGET_AVX2 inline __m256i cmp_domain(__m256i v) {
    if constexpr (elem_traits<T>::simd == SimdKind::f64)
        return _mm256_xor_si256(f64_order_key(v), _mm256_set1_epi64x(static_cast<long long>(0x8000000000000000ull)));
    else
        return v;
}
template <class T> BRAINSORT_TARGET_AVX2 inline __m256i eq_keys(__m256i x, __m256i y) {
    if constexpr (elem_traits<T>::simd == SimdKind::i32) return _mm256_cmpeq_epi32(x, y);
    else return _mm256_cmpeq_epi64(x, y);
}

// First pass with AVX2, for 8-byte elements (four per vector): the forward
// split with the class counts and the membership check folded in. Same
// overflow contract as split2_scalar.
template <class A>
BRAINSORT_TARGET_AVX2 inline bool split2_avx2(A a, A buf, size_t n, size_t cap,
                                              const typename A::key_type* vk, typename A::key_type t1,
                                              typename A::key_type t2, typename A::key_type t3,
                                              uint64_t& xm, size_t& c0_lt, size_t& c0_ge, size_t& n_ge, bool& known) {
    using T  = typename A::value_type;
    using KT = elem_traits<T>;
    using K  = typename A::key_type;
    constexpr size_t V = 32 / sizeof(T);
    T*       pa  = a.data();
    T*       pb  = buf.data();
    const T* src = a.data();
    const __m256i vp1 = pivot_vec_key<T>(t1), vp2 = pivot_vec_key<T>(t2), vp3 = pivot_vec_key<T>(t3);
    const __m256i e0 = pivot_vec_key<T>(vk[0]), e1 = pivot_vec_key<T>(vk[1]), e2 = pivot_vec_key<T>(vk[2]), e3 = pivot_vec_key<T>(vk[3]);
    const __m256i vk0 = key0_vec<T>(src[0]);
    __m256i vmask = _mm256_setzero_si256(), junk = _mm256_setzero_si256(), vbad = _mm256_setzero_si256();
    size_t w = 0, b = 0, i = 0, c1 = 0, c3 = 0;
    for (; i + V <= n && b + V <= cap; i += V) {
        const __m256i  v  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        const unsigned lt = lt_mask<T>(v, vp2, vk0, vmask);
        const unsigned ge = ~lt & ((1u << V) - 1);
        c1 += static_cast<size_t>(popcount(lt_mask<T>(v, vp1, vk0, junk)));
        c3 += static_cast<size_t>(popcount(lt_mask<T>(v, vp3, vk0, junk)));
        const __m256i d  = cmp_domain<T>(v);
        const __m256i eq = _mm256_or_si256(_mm256_or_si256(eq_keys<T>(d, e0), eq_keys<T>(d, e1)),
                                           _mm256_or_si256(eq_keys<T>(d, e2), eq_keys<T>(d, e3)));
        vbad = _mm256_or_si256(vbad, _mm256_andnot_si256(eq, _mm256_set1_epi32(-1)));
        const int32_t* il = V == 4 ? kLut.by4[lt] : kLut.by2[lt];
        const int32_t* ig = V == 4 ? kLut.by4[ge] : kLut.by2[ge];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(pa + w),
                            _mm256_permutevar8x32_epi32(v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(il))));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(pb + b),
                            _mm256_permutevar8x32_epi32(v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ig))));
        w += static_cast<size_t>(popcount(lt));
        b += static_cast<size_t>(popcount(ge));
    }
    K    m       = static_cast<K>(fold_mask<T>(vmask));
    bool unknown = fold_mask<T>(vbad) != 0;
    const K k0 = KT::radix_key(src[0], 0);
    while (i < n) {
        const size_t end = std::min(n, i + (cap - b));
        if (end == i) break;
        for (; i < end; ++i) {
            const T      e = src[i];
            const K      k = KT::radix_key(e, 0);
            const size_t g = k >= t2;
            m |= k ^ k0;
            unknown |= (k != vk[0]) & (k != vk[1]) & (k != vk[2]) & (k != vk[3]);
            c1 += k < t1;
            c3 += k < t3;
            pa[w] = e;
            pb[b] = e;
            w += 1 - g;
            b += g;
        }
    }
    xm    = m;
    known = !unknown;
    if (i < n) { copy_forward(buf, 0, a, w, b); return false; }
    n_ge  = b;
    c0_lt = c1;
    c0_ge = c3 - w;
    return true;
}

// Second pass with AVX2 for 8-byte elements: compress-store to two
// destinations inside one array. A vector store past the class's own region
// would land on the other class's finished elements (or outside the range),
// so the vector loop runs only while both cursors have a full vector of room;
// the scalar loops finish.
template <class T>
BRAINSORT_TARGET_AVX2 inline void partition2_avx2(const T* src, size_t m, T* dst, size_t o0, size_t o1, size_t end1,
                                                  typename elem_traits<T>::key_type t) {
    using KT = elem_traits<T>;
    using K  = typename KT::key_type;
    constexpr size_t V = 32 / sizeof(T);
    const __m256i vp  = pivot_vec_key<T>(t);
    const __m256i vk0 = key0_vec<T>(src[0]);
    __m256i junk = _mm256_setzero_si256();
    size_t i = 0, c0 = o0, c1 = o1;
    for (; i + V <= m && c0 + V <= o1 && c1 + V <= end1; i += V) {
        const __m256i  v  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        const unsigned lt = lt_mask<T>(v, vp, vk0, junk);
        const unsigned ge = ~lt & ((1u << V) - 1);
        const int32_t* il = V == 4 ? kLut.by4[lt] : kLut.by2[lt];
        const int32_t* ig = V == 4 ? kLut.by4[ge] : kLut.by2[ge];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + c0),
                            _mm256_permutevar8x32_epi32(v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(il))));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + c1),
                            _mm256_permutevar8x32_epi32(v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ig))));
        c0 += static_cast<size_t>(popcount(lt));
        c1 += static_cast<size_t>(popcount(ge));
    }
    for (; i < m && c0 < o1 && c1 < end1; ++i) {
        const T      e = src[i];
        const size_t g = KT::radix_key(e, 0) >= t;
        dst[c0] = e;
        dst[c1] = e;
        c0 += 1 - g;
        c1 += g;
    }
    for (; i < m; ++i) {
        const T e = src[i];
        if (static_cast<K>(KT::radix_key(e, 0)) < t) dst[c0++] = e;
        else dst[c1++] = e;
    }
}
#endif

template <class A, class K>
inline bool split2(A a, A buf, size_t n, size_t cap, int chunk, const K* vk, K t1, K t2, K t3,
                   uint64_t& xm, size_t& c0_lt, size_t& c0_ge, size_t& n_ge, bool& known) {
#ifdef BRAINSORT_X86_64
    if constexpr (!A::counted && A::traits::simd == SimdKind::i32) {
        if (chunk == 0 && have_avx2()) return split2_avx2(a, buf, n, cap, vk, t1, t2, t3, xm, c0_lt, c0_ge, n_ge, known);
    }
#endif
    return split2_scalar(a, buf, n, cap, chunk, vk, t1, t2, t3, xm, c0_lt, c0_ge, n_ge, known);
}
template <class A, class K>
inline void partition2(A src, size_t m, A dst, size_t o0, size_t o1, size_t end1, int chunk, K t) {
#ifdef BRAINSORT_X86_64
    if constexpr (!A::counted && A::traits::simd == SimdKind::i32) {
        if (chunk == 0 && have_avx2()) { partition2_avx2(src.data(), m, dst.data(), o0, o1, end1, t); return; }
    }
#endif
    partition2_scalar(src, m, dst, o0, o1, end1, chunk, t);
}

// Sort the two sides a split left behind: the buffered side straight out of
// the buffer, the array side through the buffer. Shared by the median split
// and the partition sort's fallback.
template <int DigitBits, class A, class S>
inline void sort_split_parts(A a, A buf, size_t cap, size_t n, size_t n_ge, bool buf_holds_ge, int chunk, S& scratch, uint64_t mask,
                             const Pivot& info, bool have_range, uint64_t lo, uint64_t pk, uint64_t hi) {
    const size_t   n_lt   = n - n_ge;
    const uint64_t lt_max = pk > 0 ? pk - 1 : 0;   // the < side lies in [lo, pk), the >= side in [pk, hi]
    if (buf_holds_ge) {
        radix_part<DigitBits>(buf.sub(0, n_ge), a.sub(n_lt, n_ge), n_ge, chunk, scratch, mask, true, info, have_range, pk, hi, kSplitMaxDigit, false);
        A tmp = scratch.ensure(n_lt);   // grows only if the sample misjudged the sizes
        radix_part<DigitBits>(a.sub(0, n_lt), tmp.sub(0, n_lt), n_lt, chunk, scratch, mask, true, info, have_range, lo, lt_max, kSplitMaxDigit, true);
    } else {
        radix_part<DigitBits>(buf.sub(cap - n_lt, n_lt), a.sub(0, n_lt), n_lt, chunk, scratch, mask, true, info, have_range, lo, lt_max, kSplitMaxDigit, false);
        A tmp = scratch.ensure(n_ge);
        radix_part<DigitBits>(a.sub(n_lt, n_ge), tmp.sub(0, n_ge), n_ge, chunk, scratch, mask, true, info, have_range, pk, hi, kSplitMaxDigit, true);
    }
}

// The partition sort for two to four distinct sampled keys. Returns true
// when a[0,n) is sorted. Returns false only if the first pass overflowed the
// buffer (a skewed key distribution); the array is then a stable partial
// partition and nothing else has changed, so the caller carries on. An
// unsampled key does not return false: the two sides are finished on the
// general radix path here.
template <int DigitBits, class A, class S>
inline bool partition_sort_few(A a, S& scratch, size_t n, int chunk, const Pivot& info, bool have_range) {
    using K = typename A::key_type;
    const size_t d = info.n_distinct;
    K vk[4];
    for (size_t j = 0; j < 4; ++j) vk[j] = static_cast<K>(info.few[std::min(j, d - 1)]);
    // Split at v1 for two or three keys (the lower side is then one key and
    // needs no second pass), at v2 for four. The sample must not put clearly
    // more than half above the split, or the buffer would overflow.
    const size_t mid   = d == 4 ? 2 : 1;
    size_t       above = 0;
    for (size_t j = mid; j < d; ++j) above += info.few_cnt[j];
    if (above * 16 > info.n_sample * 9) return false;
    const K t2 = vk[mid];
    const K t1 = mid == 2 ? vk[1] : vk[0];                 // lower key of the < side (mid == 1: one key, nothing below v0)
    const K t3 = mid + 1 < d ? vk[mid + 1] : vk[d - 1];    // lower key of the >= side, or its only key
    const size_t cap = n / 2 + n / 32 + 8;
    A        buf = scratch.ensure(cap);
    uint64_t xm  = 0;
    size_t   c0_lt = 0, c0_ge = 0, n_ge = 0;
    bool     known = false;
    if (!split2(a, buf, n, cap, chunk, vk, t1, t2, t3, xm, c0_lt, c0_ge, n_ge, known)) return false;
    const size_t n_lt = n - n_ge;
    if (!known) {   // an unsampled key: general path per side, the split stands
        sort_split_parts<DigitBits>(a, buf, cap, n, n_ge, true, chunk, scratch, xm, info, have_range, info.smin, t2, info.smax);
        return true;
    }
    if (mid + 1 < d) partition2(buf.sub(0, n_ge), n_ge, a, n_lt, n_lt + c0_ge, n, chunk, t3);   // >= side: buffer -> final place
    else copy_forward(buf, 0, a, n_lt, n_ge);                                                  // one key above: just move it
    if (mid == 2) {                                                                             // < side has two keys: through the buffer
        A tmp = scratch.ensure(n_lt);
        partition2(a.sub(0, n_lt), n_lt, tmp, 0, c0_lt, n_lt, chunk, t1);
        copy_forward(tmp, 0, a, 0, n_lt);
    }
    return true;
}

// Route 4 on a[0,n). If the scratch buffer is not yet large enough for the
// whole range (the top level), the range is split by a sampled pivot with a
// buffer of about half the size; each part is then radix sorted, the part in
// the buffer straight out of it (no copy back, and for an odd number of
// passes its last scatter lands in the array). `mask` is the varying-bit
// mask if `mask_known`, otherwise the bits the scout saw before it stopped
// early; the split pass then computes the exact mask, and the no-split path
// plans from the estimate and verifies while counting. For chunked keys,
// chunks shared by every element are skipped first (chunk advances).
template <int DigitBits, class A, class S>
inline void radix_route(A a, S& scratch, size_t n, int& chunk, uint64_t mask, bool mask_known, bool unordered) {
    using T  = typename A::value_type;
    using KT = typename A::traits;
    using K  = typename A::key_type;
    Pivot info;
    T     pivot{};
    bool  have_pivot = false;
    if (n >= kSplitMin) { pivot = pick_pivot(a, n, chunk, info); have_pivot = true; }
    if constexpr (KT::chunked) {
        // A chunk shared by every element carries no information: move on to
        // the next chunk without scouting again (same elements, same order).
        // The sample tells cheaply whether that is likely; the full pass confirms.
        if (!mask_known && (!have_pivot || info.all_equal)) { mask = compute_mask(a, n, chunk); mask_known = true; }
        if (mask_known) {
            const int chunk0 = chunk;
            while (mask == 0) {
                const T e0 = a.get(0);
                [[maybe_unused]] const K k0 = A::key(e0, chunk);   // counted path: the chunk bytes this test loads
                if (KT::chunk_ends(e0, chunk)) break;
                ++chunk;
                mask = compute_mask(a, n, chunk);
            }
            if (mask == 0) return;
            if (have_pivot && chunk != chunk0) pivot = pick_pivot(a, n, chunk, info);   // re-sample on the new chunk
        }
    }
    const uint64_t est    = mask_known ? mask : (mask | info.sample_mask);
    const bool have_range = have_pivot && !info.all_equal;
    // Two to four distinct sampled keys on 8-byte elements: the partition
    // sort, no counters at all. (For 16-byte elements, two per vector, both
    // its vector and its scalar passes measured slower than the radix.)
    if constexpr (!KT::chunked && sizeof(T) == 8) {
        if (have_pivot && info.n_distinct >= 2 && info.n_distinct <= 4 &&
            partition_sort_few<DigitBits>(a, scratch, n, chunk, info, have_range)) {
            if constexpr (A::counted) ++A::hooks::stats().part_sorts;
            return;
        }
    }
    // Split only when it pays. Three conditions: the buffer is not already
    // large enough; the radix will run enough passes (planned from the
    // estimate) to amortise the split's extra streaming pass; and the data
    // is substantially unordered. The last matters because a split interleaves
    // the two halves and so destroys any key locality the input had - on
    // largely-ascending data (concatenated runs, sawtooth) the plain radix
    // scatters nearby keys to nearby buckets and the split would throw that
    // cache advantage away for no memory that matters.
    const bool want_split = have_pivot && scratch.capacity() < n && unordered &&
                            choose_plan<K>(est, mask_known, have_range, info.smin, info.smax, n, kSplitMaxDigit, DigitBits).passes() >= kSplitMinPasses;
    if (want_split) {
        const size_t   cap = n / 2 + n / 32 + 8;   // sample error margin plus vector slack; the retry is rare
        A              buf = scratch.ensure(cap);
        uint64_t*      mp  = mask_known ? nullptr : &mask;
        const uint64_t pk  = A::key(pivot, chunk);
        size_t n_ge = 0;
        bool   buf_holds_ge;   // which side ended up in the buffer
        if constexpr (A::counted) ++A::hooks::stats().splits;
        if (info.n_ge * 16 <= info.n_sample * 9) {   // >= side not clearly larger: forward (vectorised)
            buf_holds_ge = split_forward(a, buf, n, cap, pivot, chunk, n_ge, mp);
            if (!buf_holds_ge) { if constexpr (A::counted) ++A::hooks::stats().split_retries; split_backward_scalar(a, buf, n, cap, pk, chunk, n_ge, mp); }
        } else {
            buf_holds_ge = !split_backward_scalar(a, buf, n, cap, pk, chunk, n_ge, mp);
            if (buf_holds_ge) { if constexpr (A::counted) ++A::hooks::stats().split_retries; split_forward(a, buf, n, cap, pivot, chunk, n_ge, mp); }
        }
        // mask is now exact (the split accumulated it when mp != null)
        sort_split_parts<DigitBits>(a, buf, cap, n, n_ge, buf_holds_ge, chunk, scratch, mask, info, have_range, info.smin, pk, info.smax);
        return;
    }
    A tmp = scratch.ensure(n);
    radix_part<DigitBits>(a, tmp, n, chunk, scratch, est, mask_known, info, have_range, info.smin, info.smax, kRadixMaxDigit, true);
}

// Sort a[0,n) given that all its elements share the key chunks before
// `chunk`. Groups that tie on a chunk are sorted on the next chunk; all but
// the largest group recurse, the largest one loops, so the recursion depth
// is at most log2(n).
template <int DigitBits, class A, class S>
void sort_range(A a, S& scratch, size_t n, int chunk) {
    using T  = typename A::value_type;
    using KT = typename A::traits;
    [[maybe_unused]] const typename A::hooks::DepthScope depth{};
    for (;;) {
        if (n < 2) return;
        if (n <= kInsertionMax) { insertion_sort(a, 0, n); return; }

        const ScoutResult s = scout(a, n, chunk);
        if (!s.committed) {
            if (s.descents == 0) { if constexpr (A::counted) A::hooks::stats().note_route(1); return; }                            // route 1: sorted
            if (s.ascents == 0) { if constexpr (A::counted) A::hooks::stats().note_route(2); stable_reverse(a, n, s); return; }   // route 2: non-increasing
            if (s.tracking) { if constexpr (A::counted) A::hooks::stats().note_route(3); sort_few_runs(a, scratch, n, s); return; } // route 2b: at most kMaxRuns runs
            if (s.descents <= n / 16) {                                                                                            // route 3
                if constexpr (A::counted) A::hooks::stats().note_route(4);
                if (sort_displaced(a, n)) return;
                if constexpr (A::counted) ++A::hooks::stats().giveups;
            }
        }
        if constexpr (A::counted) A::hooks::stats().note_route(5);
        // "Unordered" gates the route-4 split: committed means the scout bailed
        // on high disorder; otherwise a quarter of the adjacent pairs descending
        // is disorder enough that a split's cache cost is worth paying.
        const bool unordered = s.committed || s.descents * 4 >= n;
        radix_route<DigitBits>(a, scratch, n, chunk, s.mask, s.has_mask, unordered);   // route 4

        if constexpr (!KT::chunked) return;
        else {
            // Elements tying on this chunk form contiguous groups; each group that
            // has not reached the end of its keys is sorted on the next chunk.
            size_t g = 0, big_g = 0, big_len = 0;
            while (g < n) {
                const T    eg = a.get(g);
                const auto k  = A::key(eg, chunk);
                size_t e = g + 1;
                while (e < n && A::key(a.get(e), chunk) == k) ++e;
                if (e - g > 1 && !KT::chunk_ends(eg, chunk)) {
                    if (e - g > big_len) {
                        if (big_len > 1) sort_range<DigitBits>(a.sub(big_g, big_len), scratch, big_len, chunk + 1);
                        big_g = g; big_len = e - g;
                    } else {
                        sort_range<DigitBits>(a.sub(g, e - g), scratch, e - g, chunk + 1);
                    }
                }
                g = e;
            }
            if (big_len < 2) return;
            a = a.sub(big_g, big_len);   // the largest group: iterate instead of recursing
            n = big_len;
            ++chunk;
        }
    }
}

}  // namespace brain_detail

// Sort the view a[0, n). DigitBits <= 0 selects the digit width automatically.
template <int DigitBits, class A>
inline void brainsort_impl(A a) {
    const size_t n = a.size();
    if (n < 2) return;
    if (n <= brain_detail::kInsertionMax) { insertion_sort(a, 0, n); return; }
    if (n > 0xFFFFFFFFull) { merge_sort(a); return; }   // positions and counters are 32-bit
    brain_detail::Scratch<A> scratch;
    brain_detail::sort_range<DigitBits>(a, scratch, n, 0);
}

template <class A>
inline void brainsort_view(A a) { brainsort_impl<0>(a); }

}  // namespace detail
}  // namespace brainsort

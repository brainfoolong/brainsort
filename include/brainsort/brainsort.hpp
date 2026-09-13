// brainsort - a stable, exact sort for keys that map to an ordered integer:
// numbers, strings, dates, enums, pointers and combinations of them.
//
//   #include "brainsort/brainsort.hpp"
//
//   std::vector<int> v = ...;
//   brainsort::sort(v);                                       // by the values
//   brainsort::sort(v.begin(), v.end());
//   brainsort::sort(rows, [](const Row& r) { return r.id; }); // by a key
//   brainsort::sort(rows, [](const Row& r) { return std::pair(r.group, brainsort::desc(r.score)); });
//   brainsort::sort(rows, [](const Row& a, const Row& b) { return a.name < b.name; }); // comparator: merge sort
//
// Every overload is stable: equal keys keep their input order. The sort is
// O(n) passes over the data for fixed-size keys and O(n * key length) for
// strings; it never degrades on any input pattern. Sorted, reversed and
// nearly sorted input is recognised on the elements before anything is
// built and handled in place. A comparator gets a stable merge sort with
// the same handling of ordered input.
//
// Keys: every integral type, bool, the character types, float, double, enums,
// pointers, std::string, std::string_view, C strings, char arrays,
// std::chrono durations and time points, std::pair, std::tuple, std::array
// of keys, and brainsort::desc(key) for a reversed order. Specialise
// brainsort::key_traits<K> for your own key type (see detail/keys.hpp).
//
// Elements: any move-assignable type behind a random-access iterator (vector,
// deque, array, span, C arrays, pointers). Elements are permuted once, after
// the keys were sorted, so element size does not matter.
//
// Floating point: -0.0 and +0.0 compare equal; NaNs have a defined place
// (positive NaN after +inf, negative NaN before -inf), so no input is unsafe.
//
// Memory: about 1.5 records per element while sorting (8 bytes for keys of up
// to 32 bits, 16 bytes for up to 64 bits or a string, more for composite
// keys) plus, for trivially copyable elements, one element per element for
// the final permutation; a comparator sort uses one element per element.
// Up to BRAINSORT_MEMORY_CACHE bytes (32 MiB) of freed blocks are kept for
// the next sort; release_memory() gives them back. If an allocation fails
// the sort completes anyway, through std::stable_sort with the same order;
// nothing throws except the projection or comparator itself. A projection
// that throws during the first pass over the keys leaves the range
// unchanged; one that throws later, or a comparator that throws, leaves
// every element in the range in an unspecified order.
//
// Limits: at most 2^32 - 1 elements per call (larger ranges go to
// std::stable_sort with the same order); strings of 2^32 bytes or more
// likewise. Thread safe: the only shared state is the block cache, under
// its own mutex.
#pragma once

#include "brainsort/detail/algorithm.hpp"
#include "brainsort/detail/compsort.hpp"
#include "brainsort/detail/config.hpp"
#include "brainsort/detail/infer.hpp"
#include "brainsort/detail/keys.hpp"
#include "brainsort/detail/records.hpp"
#include "brainsort/detail/traits.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <new>
#include <optional>
#include <ranges>
#include <type_traits>
#include <utility>

namespace brainsort {
namespace detail {

// Below this many elements the elements themselves are insertion sorted by
// key: no allocation, and cheaper than building records.
constexpr size_t kSmallSort = 32;
// Elements up to this size are sorted in place on nearly sorted input; larger
// ones through the records, where moving only the out-of-place elements is
// cheaper than rewriting every element once.
constexpr size_t kElementRouteMax = 16;
// Below this many elements the key a comparator compares is not inferred:
// the sample, the records and the check cost more than the comparison sort
// of a small range saves (level at about 5,000 elements).
constexpr size_t kInferMin = 4096;

// Thrown internally when a key cannot be represented as a record.
struct unrepresentable_key {};

template <class It, class Proj>
using proj_result_t = std::invoke_result_t<Proj&, std::iter_reference_t<It>>;
template <class It, class Proj>
using key_of_t = std::remove_cvref_t<proj_result_t<It, Proj>>;

// Apply the permutation the sorted records describe: out[i] = in[index(rec[i])].
// In place, following cycles, with moves; works for any move-assignable type.
template <class It, class Rec>
inline void permute_cycles(It first, Rec* rec, size_t n) {
    using T = std::iter_value_t<It>;
    for (size_t i = 0; i < n; ++i) {
        if (index_of(rec[i]) == i) continue;
        T      tmp = std::move(first[static_cast<std::ptrdiff_t>(i)]);
        size_t j   = i;
        for (;;) {
            const size_t k = index_of(rec[j]);
            set_index(rec[j], static_cast<uint32_t>(j));
            if (k == i) { first[static_cast<std::ptrdiff_t>(j)] = std::move(tmp); break; }
            first[static_cast<std::ptrdiff_t>(j)] = std::move(first[static_cast<std::ptrdiff_t>(k)]);
            j = k;
        }
    }
}

// The same through a gather buffer: sequential writes and independent loads,
// which is faster for trivially copyable elements. Falls back to the cycle
// walk if the buffer cannot be allocated. When the input was nearly sorted
// (`sparse`), most elements are already in their final place and the cycle
// walk moves only the others, so it is taken if at most an eighth are out
// of place.
template <class Alloc, class It, class Rec>
inline void permute(It first, Rec* rec, size_t n, bool sparse) {
    using T = std::iter_value_t<It>;
    if (sparse) {
        size_t moved = 0;
        for (size_t i = 0; i < n; ++i) moved += index_of(rec[i]) != i;
        if (moved <= n / 8) { permute_cycles(first, rec, n); return; }
    }
    if constexpr (std::is_trivially_copyable_v<T> && alignof(T) <= alignof(std::max_align_t)) {
        try {
            Buf<T, Alloc> tmp(n);
            T* t = tmp.data();
            for (size_t i = 0; i < n; ++i) t[i] = first[static_cast<std::ptrdiff_t>(index_of(rec[i]))];
            for (size_t i = 0; i < n; ++i) first[static_cast<std::ptrdiff_t>(i)] = t[i];
            return;
        } catch (const std::bad_alloc&) {
        }
    }
    permute_cycles(first, rec, n);
}

// Holds the previous key of a scan: a pointer when the projection returns a
// reference, the value itself when it returns one.
template <class It, class Proj>
struct KeyHolder {
    using R = proj_result_t<It, Proj>;
    using K = key_of_t<It, Proj>;
    static constexpr bool by_ref = std::is_reference_v<R>;
    std::conditional_t<by_ref, const K*, K> v;
    explicit KeyHolder(R&& r) : v(store(static_cast<R&&>(r))) {}
    void assign(R&& r) { v = store(static_cast<R&&>(r)); }
    const K& get() const { if constexpr (by_ref) return *v; else return v; }
private:
    static auto store(R&& r) { if constexpr (by_ref) return &static_cast<const K&>(r); else return K(static_cast<R&&>(r)); }
};

// Compare two keys in the order the records would sort them. Floating-point
// keys are compared as values: the same order as their radix form except for
// NaN, which is flagged so the caller can leave such input to the records.
template <class K>
BRAINSORT_ALWAYS_INLINE int scan_compare(const K& a, const K& b, bool& nan) {
    if constexpr (std::is_floating_point_v<K>) {
        nan |= (a != a) | (b != b);
        return a < b ? -1 : (b < a ? 1 : 0);
    } else {
        return compare_keys<false, K>(a, b);
    }
}

// What one pass over the keys found, before any record is built or memory
// allocated. The pass counts descents and ascents and stops as soon as the
// counts prove the input unordered (a few hundred elements into random
// input), so an expensive projection is called few times on such input.
enum class Shape : int { unordered, sorted, reversed, nearly_sorted };
struct Prescan {
    Shape  shape    = Shape::unordered;
    size_t descents = 0;
    size_t ascents  = 0;   // as seen by the pass: none are counted before the first descent
};
constexpr size_t kPrescanBlock = 256;
inline void classify(Prescan& r, size_t n, size_t desc, size_t asc) {
    r.descents = desc;
    r.ascents  = asc;
    if (desc == 0) r.shape = Shape::sorted;
    else if (asc == 0) r.shape = Shape::reversed;
    else if (desc <= n / 16) r.shape = Shape::nearly_sorted;
}

// A plain array of 32- or 64-bit numbers: eight or four elements per vector
// compared against their predecessors, the compare bits counted; NaN is
// flagged from the vector compare too.
template <class K> constexpr bool vector_prescan_v =
    std::is_same_v<K, int32_t> || std::is_same_v<K, uint32_t> || std::is_same_v<K, int64_t> ||
    std::is_same_v<K, uint64_t> || std::is_same_v<K, float> || std::is_same_v<K, double>;
#ifdef BRAINSORT_X86_64
template <class K> BRAINSORT_TARGET_AVX2 inline __m256i vld(const K* q) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q)); }
template <class K> BRAINSORT_TARGET_AVX2 inline unsigned vgt(__m256i x, __m256i y) {   // bit e set where element e of x > of y
    if constexpr (std::is_same_v<K, float>)
        return static_cast<unsigned>(_mm256_movemask_ps(_mm256_cmp_ps(_mm256_castsi256_ps(x), _mm256_castsi256_ps(y), _CMP_GT_OQ)));
    else if constexpr (std::is_same_v<K, double>)
        return static_cast<unsigned>(_mm256_movemask_pd(_mm256_cmp_pd(_mm256_castsi256_pd(x), _mm256_castsi256_pd(y), _CMP_GT_OQ)));
    else if constexpr (sizeof(K) == 4) {
        if constexpr (std::is_unsigned_v<K>) { const __m256i s = _mm256_set1_epi32(INT32_MIN); x = _mm256_xor_si256(x, s); y = _mm256_xor_si256(y, s); }
        return static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(x, y))));
    } else {
        if constexpr (std::is_unsigned_v<K>) { const __m256i s = _mm256_set1_epi64x(INT64_MIN); x = _mm256_xor_si256(x, s); y = _mm256_xor_si256(y, s); }
        return static_cast<unsigned>(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpgt_epi64(x, y))));
    }
}
template <class K> BRAINSORT_TARGET_AVX2 inline bool vnan(__m256i x) {
    if constexpr (std::is_same_v<K, float>) return _mm256_movemask_ps(_mm256_cmp_ps(_mm256_castsi256_ps(x), _mm256_castsi256_ps(x), _CMP_UNORD_Q)) != 0;
    else if constexpr (std::is_same_v<K, double>) return _mm256_movemask_pd(_mm256_cmp_pd(_mm256_castsi256_pd(x), _mm256_castsi256_pd(x), _CMP_UNORD_Q)) != 0;
    else return false;
}
template <class K>
BRAINSORT_TARGET_AVX2 inline Prescan prescan_avx2(const K* p, size_t n) {
    constexpr size_t V = 32 / sizeof(K);
    Prescan r;
    size_t desc = 0, asc = 0, i = 1, next_check = kPrescanBlock;
    bool   nan = false;
    if constexpr (std::is_floating_point_v<K>) nan = p[0] != p[0];
    for (; i + V <= n; i += V) {
        const __m256i cur = vld(p + i), prev = vld(p + i - 1);
        desc += static_cast<size_t>(std::popcount(vgt<K>(prev, cur)));
        asc  += static_cast<size_t>(std::popcount(vgt<K>(cur, prev)));
        nan |= vnan<K>(cur);
        if (i >= next_check) {
            next_check += kPrescanBlock;
            if (asc > 0 && desc > (i >> 3) + 64) return r;
        }
    }
    for (; i < n; ++i) {
        const int c = scan_compare<K>(p[i - 1], p[i], nan);
        desc += c > 0;
        asc  += c < 0;
    }
    if (nan) return r;
    r.descents = desc;
    r.ascents  = asc;
    if (desc == 0) r.shape = Shape::sorted;
    else if (asc == 0) r.shape = Shape::reversed;
    else if (desc <= n / 16) r.shape = Shape::nearly_sorted;
    return r;
}
#endif

// Whether the pairs before `upto` hold an ascent.
template <class It, class Proj>
inline bool prefix_ascent(It first, size_t upto, Proj& proj) {
    using K = key_of_t<It, Proj>;
    bool nan = false;
    KeyHolder<It, Proj> prev(std::invoke(proj, first[0]));
    for (size_t i = 1; i < upto; ++i) {
        KeyHolder<It, Proj> cur(std::invoke(proj, first[static_cast<std::ptrdiff_t>(i)]));
        if (scan_compare<K>(prev.get(), cur.get(), nan) < 0) return true;
        prev = std::move(cur);
    }
    return false;
}

// One pass over the keys in three phases: the longest non-descending prefix
// costs one compare and one branch per pair, so does the non-ascending run
// after the first descent, and only from the first ascent after that are
// the pairs counted. The prefix's ascents are looked up only when the
// bail-out or the classification asks for them, so every decision is the
// one a plain count would make.
template <class It, class Proj>
inline Prescan prescan(It first, size_t n, Proj& proj) {
    using K = key_of_t<It, Proj>;
#ifdef BRAINSORT_X86_64
    if constexpr (std::is_same_v<Proj, std::identity> && std::contiguous_iterator<It> && vector_prescan_v<K>) {
        if (have_avx2()) return prescan_avx2<K>(std::to_address(first), n);
    }
#endif
    Prescan r;
    bool    nan = false;
    KeyHolder<It, Proj> prev(std::invoke(proj, first[0]));
    size_t i = 1;
    for (; i < n; ++i) {
        KeyHolder<It, Proj> cur(std::invoke(proj, first[static_cast<std::ptrdiff_t>(i)]));
        const int c = scan_compare<K>(prev.get(), cur.get(), nan);
        prev = std::move(cur);
        if (c > 0) break;
    }
    if (i == n) {
        if (nan) return r;   // NaN: the records know its place, the value compare does not
        r.shape = Shape::sorted;
        return r;
    }
    size_t desc = 1, j = i + 1;
    int    prefix = -1;   // whether the prefix has an ascent, once asked
    const auto prefix_asc = [&]() { if (prefix < 0) prefix = prefix_ascent(first, i, proj) ? 1 : 0; return prefix > 0; };
    for (; j < n; ++j) {
        KeyHolder<It, Proj> cur(std::invoke(proj, first[static_cast<std::ptrdiff_t>(j)]));
        const int c = scan_compare<K>(prev.get(), cur.get(), nan);
        prev = std::move(cur);
        if (c < 0) break;
        desc += c > 0;
        // Unordered for certain: the displaced-element route gives up once
        // the displaced elements exceed scanned/8 + 64, and every descent
        // displaces at least one element.
        if ((j & (kPrescanBlock - 1)) == 0 && desc > (j >> 3) + 64 && prefix_asc()) return r;
    }
    if (j == n) {
        if (nan) return r;
        classify(r, n, desc, prefix_asc() ? 1 : 0);
        return r;
    }
    size_t asc = 1;
    for (++j; j < n; ++j) {
        KeyHolder<It, Proj> cur(std::invoke(proj, first[static_cast<std::ptrdiff_t>(j)]));
        const int c = scan_compare<K>(prev.get(), cur.get(), nan);
        prev = std::move(cur);
        desc += c > 0;
        asc  += c < 0;
        if ((j & (kPrescanBlock - 1)) == 0 && desc > (j >> 3) + 64) return r;
    }
    if (nan) return r;
    classify(r, n, desc, asc);
    return r;
}

// Reverse a non-increasing range in place with each group of equal keys put
// back into input order, so the result is stable. Strictly decreasing input
// (every pair a descent) has no groups.
template <class It, class Proj>
inline void reverse_stable(It first, size_t n, Proj& proj, const Prescan& s) {
    using K = key_of_t<It, Proj>;
    std::reverse(first, first + static_cast<std::ptrdiff_t>(n));
    if (s.descents == n - 1) return;
    bool   nan = false;
    size_t i = 0;
    while (i < n) {
        size_t j = i + 1;
        {
            KeyHolder<It, Proj> ki(std::invoke(proj, first[static_cast<std::ptrdiff_t>(i)]));
            while (j < n) {
                KeyHolder<It, Proj> kj(std::invoke(proj, first[static_cast<std::ptrdiff_t>(j)]));
                if (scan_compare<K>(ki.get(), kj.get(), nan) != 0) break;
                ++j;
            }
        }
        if (j - i > 1) std::reverse(first + static_cast<std::ptrdiff_t>(i), first + static_cast<std::ptrdiff_t>(j));
        i = j;
    }
}

// The three-way order of two elements, by a key projection or by a
// comparator, for the routes that run on the elements themselves.
template <class T, class Proj>
struct ProjOrder {
    Proj* proj;
    int operator()(const T& a, const T& b) const {
        using K = std::remove_cvref_t<std::invoke_result_t<Proj&, const T&>>;
        bool nan = false;
        return scan_compare<K>(std::invoke(*proj, a), std::invoke(*proj, b), nan);
    }
};
template <class T, class Comp>
struct CompOrder {
    Comp* comp;
    int operator()(const T& a, const T& b) const { return (*comp)(b, a) ? 1 : ((*comp)(a, b) ? -1 : 0); }
};

// The elements themselves as the array the core's displaced-element route
// sorts, for nearly sorted input: no records, no permutation, the few
// displaced elements are pulled out, sorted and merged back in place.
// Trivially copyable elements behind a contiguous iterator only.
template <class T, class Order, class Alloc>
class ElemView {
public:
    using value_type = T;
    using hooks      = NoHooks;
    using alloc      = Alloc;
    static constexpr bool counted = false;
    ElemView(T* p, size_t n, Order order) : p_(p), n_(n), order_(order) {}
    size_t size() const noexcept { return n_; }
    T*     data() const noexcept { return p_; }
    T      get(size_t i) const { return p_[i]; }
    void   set(size_t i, const T& v) const { p_[i] = v; }
    bool   less(const T& a, const T& b) const { return order_(a, b) < 0; }
    int    compare(const T& a, const T& b) const { return order_(a, b); }
    template <class U> static U* alloc_array(size_t n) { return static_cast<U*>(Alloc::allocate(n * sizeof(U))); }
    template <class U> static void free_array(U* p, size_t n) noexcept { Alloc::deallocate(p, n * sizeof(U)); }
private:
    T*     p_;
    size_t n_;
    Order  order_;
};

template <class T> constexpr bool element_route_v = std::is_trivially_copyable_v<T> && std::is_default_constructible_v<T>;

template <class Alloc, class It, class Order>
inline bool sort_displaced_elements(It first, size_t n, Order order) {
    using T = std::iter_value_t<It>;
    if constexpr (std::contiguous_iterator<It> && element_route_v<T>) {
        try {
            return brain_detail::sort_displaced(ElemView<T, Order, Alloc>(std::to_address(first), n, order), n);
        } catch (const std::bad_alloc&) {
            return false;   // nothing was moved: the route restores the range before it gives up
        }
    } else {
        (void)first; (void)n; (void)order;
        return false;
    }
}

// Whether the pairs before `upto` hold an ascent, by the comparator.
template <class It, class Comp>
inline bool prefix_ascent_comp(It first, size_t upto, Comp& comp) {
    for (size_t i = 1; i < upto; ++i)
        if (comp(first[static_cast<std::ptrdiff_t>(i - 1)], first[static_cast<std::ptrdiff_t>(i)])) return true;
    return false;
}

// The prescan of the comparator overloads: the non-descending prefix and
// the strictly descending run after the first descent cost one call of the
// `less` comparator and one branch per pair; ties and the pairs after the
// first ascent are counted. The bail-out a plain count would have taken
// inside the run is replayed after it, so every decision is the count's.
template <class It, class Comp>
inline Prescan prescan_comp(It first, size_t n, Comp& comp) {
    Prescan r;
    size_t i = 1;
    for (; i < n; ++i)
        if (comp(first[static_cast<std::ptrdiff_t>(i)], first[static_cast<std::ptrdiff_t>(i - 1)])) break;
    if (i == n) {
        r.shape = Shape::sorted;
        return r;
    }
    int prefix = -1;   // whether the prefix has an ascent, once asked
    const auto prefix_asc = [&]() { if (prefix < 0) prefix = prefix_ascent_comp(first, i, comp) ? 1 : 0; return prefix > 0; };
    size_t j = i + 1;
    for (; j < n; ++j)
        if (!comp(first[static_cast<std::ptrdiff_t>(j)], first[static_cast<std::ptrdiff_t>(j - 1)])) break;
    for (size_t b = (i + kPrescanBlock) & ~(kPrescanBlock - 1); b < j; b += kPrescanBlock) {
        if (b - i + 1 > (b >> 3) + 64) {
            if (prefix_asc()) return r;
            break;
        }
    }
    size_t desc = j - i;
    if (j == n) {
        classify(r, n, desc, prefix_asc() ? 1 : 0);
        return r;
    }
    for (; j < n; ++j) {
        const auto& prev = first[static_cast<std::ptrdiff_t>(j - 1)];
        const auto& cur  = first[static_cast<std::ptrdiff_t>(j)];
        if (comp(cur, prev)) ++desc;
        else if (comp(prev, cur)) break;
        if ((j & (kPrescanBlock - 1)) == 0 && desc > (j >> 3) + 64 && prefix_asc()) return r;
    }
    if (j == n) {
        classify(r, n, desc, prefix_asc() ? 1 : 0);
        return r;
    }
    size_t asc = 1;
    for (++j; j < n; ++j) {
        const auto& prev = first[static_cast<std::ptrdiff_t>(j - 1)];
        const auto& cur  = first[static_cast<std::ptrdiff_t>(j)];
        const bool  d    = comp(cur, prev);
        desc += d;
        asc  += !d && comp(prev, cur);
        if ((j & (kPrescanBlock - 1)) == 0 && desc > (j >> 3) + 64) return r;
    }
    classify(r, n, desc, asc);
    return r;
}

template <class It, class Comp>
inline void reverse_stable_comp(It first, size_t n, Comp& comp, const Prescan& s) {
    std::reverse(first, first + static_cast<std::ptrdiff_t>(n));
    if (s.descents == n - 1) return;
    size_t i = 0;
    while (i < n) {
        size_t j = i + 1;
        while (j < n && !comp(first[static_cast<std::ptrdiff_t>(i)], first[static_cast<std::ptrdiff_t>(j)]) &&
               !comp(first[static_cast<std::ptrdiff_t>(j)], first[static_cast<std::ptrdiff_t>(i)]))
            ++j;
        if (j - i > 1) std::reverse(first + static_cast<std::ptrdiff_t>(i), first + static_cast<std::ptrdiff_t>(j));
        i = j;
    }
}

template <class It, class Proj, class Less>
inline void small_sort(It first, size_t n, Less& less) {
    using T = std::iter_value_t<It>;
    for (size_t i = 1; i < n; ++i) {
        T      v = std::move(first[static_cast<std::ptrdiff_t>(i)]);
        size_t j = i;
        while (j > 0 && less(v, first[static_cast<std::ptrdiff_t>(j - 1)])) {
            first[static_cast<std::ptrdiff_t>(j)] = std::move(first[static_cast<std::ptrdiff_t>(j - 1)]);
            --j;
        }
        first[static_cast<std::ptrdiff_t>(j)] = std::move(v);
    }
}

// A range whose elements are 64-bit keys that invert from their radix form
// (int64, uint64, double, a pointer): the keys themselves are sorted, 8
// bytes per element and no index, and written back from their radix form.
// The two zeros of a double share a key and the sort keeps them in input
// order, so the negative ones are put back by their rank among the zeros.
template <class Alloc, class It, class K>
inline void sort_keys_only(It first, size_t n) {
    constexpr bool dbl = std::is_same_v<K, double>;
    Buf<Key64, Alloc> keys(n);
    Key64* k = keys.data();
    brain_detail::Scratch<View<Key64, Alloc>> scratch;   // the sorted keys may end up in its buffer
    size_t neg_zeros = 0;
    for (size_t i = 0; i < n; ++i) {
        const K& v = first[static_cast<std::ptrdiff_t>(i)];
        k[i] = key64_of<K>(v);
        if constexpr (dbl) neg_zeros += is_negative_zero(v);
    }
    std::optional<Buf<uint32_t, Alloc>> ranks;   // of the negative zeros among the zeros, in input order
    if constexpr (dbl) {
        if (neg_zeros) {
            uint32_t* r = ranks.emplace(neg_zeros).data();
            size_t    z = 0, j = 0;
            for (size_t i = 0; i < n; ++i) {
                const double d = first[static_cast<std::ptrdiff_t>(i)];
                if (d == 0.0) {
                    if (is_negative_zero(d)) r[j++] = static_cast<uint32_t>(z);
                    ++z;
                }
            }
        }
    }
    if (!brainsort_impl<0>(View<Key64, Alloc>(k, n), scratch, true)) k = scratch.buffer();
    if constexpr (dbl) {
        for (size_t i = 0; i < n; ++i) first[static_cast<std::ptrdiff_t>(i)] = double_of(k[i]);
        if (neg_zeros) {
            // the zeros are the keys of signed value 0: the first is found by bisection
            size_t lo = 0, hi = n;
            while (lo < hi) {
                const size_t mid = lo + (hi - lo) / 2;
                if (k[mid].key < 0) lo = mid + 1; else hi = mid;
            }
            const uint32_t* r = ranks->data();
            for (size_t j = 0; j < neg_zeros; ++j) first[static_cast<std::ptrdiff_t>(lo + r[j])] = -0.0;
        }
    } else {
        for (size_t i = 0; i < n; ++i) first[static_cast<std::ptrdiff_t>(i)] = key_of<K>(k[i]);
    }
}

// Sort [first, last) by proj(element), through records of type Rec. Returns
// false, with the range untouched, if the keys could not be represented.
template <class Alloc, class It, class Proj>
inline bool sort_records(It first, size_t n, Proj& proj, bool sparse) {
    using T   = std::iter_value_t<It>;
    using R   = proj_result_t<It, Proj>;
    using K   = key_of_t<It, Proj>;
    using Rec = record_t<K>;
    constexpr bool materialise = !std::is_reference_v<R> && kParts<K>.owning;   // keys are temporaries that own their bytes
    constexpr bool self_keyed  = std::is_same_v<Proj, std::identity> && std::is_same_v<K, T>;
    if constexpr (self_keyed && std::is_same_v<Rec, Rec64> && (exact_single_v<K> || std::is_same_v<K, double>)) {
        sort_keys_only<Alloc, It, K>(first, n);
        return true;
    }

    Buf<Rec, Alloc> recs(n);
    Rec* rec = recs.data();
    brain_detail::Scratch<View<Rec, Alloc>> scratch;   // the sorted records may end up in its buffer
    if constexpr (materialise) {
        Buf<K, Alloc> keys(n);
        for (size_t i = 0; i < n; ++i) keys.emplace(std::invoke(proj, first[static_cast<std::ptrdiff_t>(i)]));
        for (size_t i = 0; i < n; ++i)
            if (!build_record<K>(rec[i], keys[i], static_cast<uint32_t>(i))) return false;
        if (!brainsort_impl<0>(View<Rec, Alloc>(rec, n), scratch, true)) rec = scratch.buffer();
    } else {
        for (size_t i = 0; i < n; ++i) {
            decltype(auto) k = std::invoke(proj, first[static_cast<std::ptrdiff_t>(i)]);
            if (!build_record<K>(rec[i], k, static_cast<uint32_t>(i))) return false;
        }
        if (!brainsort_impl<0>(View<Rec, Alloc>(rec, n), scratch, true)) rec = scratch.buffer();
    }
    // The elements are the keys and the key inverts: write the sorted keys back.
    if constexpr (self_keyed && exact_single_v<K> && std::is_same_v<Rec, Rec32>) {
        for (size_t i = 0; i < n; ++i) first[static_cast<std::ptrdiff_t>(i)] = key_of<K>(rec[i]);
    } else {
        permute<Alloc>(first, rec, n, sparse);
    }
    return true;
}

template <class Alloc, class It, class Proj>
inline void sort_by_key_impl(It first, It last, Proj proj) {
    using T = std::iter_value_t<It>;
    static_assert(std::random_access_iterator<It>, "brainsort::sort needs random-access iterators (vector, deque, array, span, pointers)");
    static_assert(std::is_lvalue_reference_v<std::iter_reference_t<It>>,
                  "brainsort::sort needs iterators that dereference to lvalues (std::vector<bool> is not supported)");
    static_assert(std::is_invocable_v<Proj&, std::iter_reference_t<It>>,
                  "brainsort::sort: the key projection must accept an element (const T&)");
    using K = key_of_t<It, Proj>;
    static_assert(is_key_v<K>,
                  "brainsort::sort: this key type has no adapter. Sort by a key with sort_by_key(first, last, proj), "
                  "pass a comparator to sort_with(first, last, comp), or specialise brainsort::key_traits<K>.");
    static_assert(shape_ok_v<K>, "brainsort::sort: a composite key must have between 1 and 32 leaves");
    static_assert(std::is_move_constructible_v<T> && std::is_move_assignable_v<T>,
                  "brainsort::sort: elements must be move-constructible and move-assignable");

    if (first == last) return;
    const size_t n = static_cast<size_t>(last - first);
    if (n < 2) return;

    auto less = [&proj](const T& a, const T& b) {
        return compare_keys<false, K>(std::invoke(proj, a), std::invoke(proj, b)) < 0;
    };
    if (n <= kSmallSort) { small_sort<It, Proj>(first, n, less); return; }

    // Element types whose moves may throw cannot be permuted safely in
    // place; ranges too long for 32-bit indices cannot be recorded.
    constexpr bool safe_moves = std::is_nothrow_move_constructible_v<T> && std::is_nothrow_move_assignable_v<T>;
    bool nearly = false;
    if (safe_moves) {
        const Prescan s = prescan(first, n, proj);
        if (s.shape == Shape::sorted) return;
        if (s.shape == Shape::reversed) { reverse_stable(first, n, proj, s); return; }
        nearly = s.shape == Shape::nearly_sorted;
        // Small elements are sorted in place by the displaced-element route;
        // large ones go through the records and a sparse permutation, which
        // moves only the elements that are out of place.
        if (nearly && sizeof(T) <= kElementRouteMax && sort_displaced_elements<Alloc>(first, n, ProjOrder<T, Proj>{&proj})) return;
    }
    if (safe_moves && n <= 0xFFFFFFFFull && n <= (~size_t(0)) / sizeof(record_t<K>) / 4) {
        bool done = false;
        try {
            done = sort_records<Alloc>(first, n, proj, nearly);
        } catch (const std::bad_alloc&) {
            done = false;   // the range is untouched: sort it by comparison instead
        }
        if (done) return;
    }
    std::stable_sort(first, last, less);
}

// A comparator on the elements an index array points to: the three-way
// order the displaced-element route wants, and the plain `less` the
// comparison sort wants.
template <class T, class Comp>
struct IndexOrder {
    const T* p;
    Comp*    comp;
    int  operator()(uint32_t a, uint32_t b) const { return (*comp)(p[b], p[a]) ? 1 : ((*comp)(p[a], p[b]) ? -1 : 0); }
};
template <class T, class Comp>
struct IndexLess {
    const T* p;
    Comp*    comp;
    bool operator()(uint32_t a, uint32_t b) const { return (*comp)(p[a], p[b]); }
};

// Apply the permutation idx describes: out[i] = in[idx[i]]. When the input
// was nearly sorted most elements are already in place, and the cycle walk
// moves only the others; otherwise through a gather buffer, with the cycle
// walk as the fallback when it cannot be allocated.
template <class Alloc, class T>
inline void permute_indices(T* p, uint32_t* idx, size_t n, bool sparse) {
    auto cycles = [&] {
        for (size_t i = 0; i < n; ++i) {
            if (idx[i] == i) continue;
            const T tmp = p[i];
            size_t  j   = i;
            for (;;) {
                const size_t k = idx[j];
                idx[j] = static_cast<uint32_t>(j);
                if (k == i) { p[j] = tmp; break; }
                p[j] = p[k];
                j    = k;
            }
        }
    };
    if (sparse) {
        size_t moved = 0;
        for (size_t i = 0; i < n; ++i) moved += idx[i] != i;
        if (moved <= n / 8) { cycles(); return; }
    }
    try {
        Buf<T, Alloc> tmp(n);
        T* t = tmp.data();
        for (size_t i = 0; i < n; ++i) t[i] = p[idx[i]];
        std::memcpy(static_cast<void*>(p), t, n * sizeof(T));
        return;
    } catch (const std::bad_alloc&) {
    }
    cycles();
}

// The comparator sort of elements larger than kElementRouteMax bytes: the
// indices 0..n-1 are sorted by the order of the elements they point to (the
// displaced-element route when the input is nearly sorted, the comparison
// sort otherwise), and the elements are permuted once at the end. The
// passes move 4-byte indices instead of the elements, and a comparator
// that throws leaves the elements untouched. Without `fallback` only the
// displaced-element route is tried: false, range untouched, when it gives up.
template <class Alloc, class T, class Comp>
inline bool sort_with_indices(T* p, size_t n, Comp& comp, bool nearly, bool fallback) {
    Buf<uint32_t, Alloc> idx(n);
    uint32_t* const i = idx.data();
    for (size_t k = 0; k < n; ++k) i[k] = static_cast<uint32_t>(k);
    if (!nearly || !sort_displaced_elements<Alloc>(i, n, IndexOrder<T, Comp>{p, &comp})) {
        if (!fallback) return false;
        IndexLess<T, Comp> less{p, &comp};
        stable_comparison_sort<Alloc>(i, n, less);
    }
    permute_indices<Alloc>(p, i, n, nearly);
    return true;
}

// The comparator overloads: sorted, reversed and nearly sorted input are
// handled on the elements like the key overloads do; then the key the
// comparator compares is inferred (detail/infer.hpp) and, when a window of
// the element agrees with it, the range is sorted by that key and verified;
// everything else goes to the comparison sort of detail/compsort.hpp, on
// the elements themselves up to kElementRouteMax bytes and through an index
// array beyond. Elements that are not trivially copyable, or not behind a
// contiguous iterator, go to std::stable_sort.
template <class Alloc, class It, class Comp>
inline void sort_with_impl(It first, It last, Comp comp) {
    using T = std::iter_value_t<It>;
    static_assert(std::random_access_iterator<It>, "brainsort::sort_with needs random-access iterators");
    static_assert(std::is_invocable_r_v<bool, Comp&, const T&, const T&>, "brainsort::sort_with: the comparator must accept two elements");
    if (first == last) return;
    const size_t n = static_cast<size_t>(last - first);
    if (n < 2) return;
    if constexpr (std::contiguous_iterator<It> && element_route_v<T>) {
        T* const p = std::to_address(first);
        if (n <= kSmallSort) { comp_detail::insertion_run(p, n, comp); return; }
        const Prescan s = prescan_comp(first, n, comp);
        if (s.shape == Shape::sorted) return;
        if (s.shape == Shape::reversed) { reverse_stable_comp(first, n, comp, s); return; }
        const bool nearly  = s.shape == Shape::nearly_sorted;
        const bool indexed = n <= 0xFFFFFFFFull;
        try {
            // nearly sorted input: the displaced elements, in place or on indices
            if constexpr (sizeof(T) <= kElementRouteMax) {
                if (nearly && sort_displaced_elements<Alloc>(first, n, CompOrder<T, Comp>{&comp})) return;
            } else {
                if (nearly && indexed && sort_with_indices<Alloc>(p, n, comp, true, false)) return;
            }
            if (indexed && n >= kInferMin && sort_inferred<Alloc>(p, n, comp)) return;
            if constexpr (sizeof(T) <= kElementRouteMax) {
                stable_comparison_sort<Alloc>(p, n, comp);
                return;
            } else {
                if (indexed) { sort_with_indices<Alloc>(p, n, comp, false, true); return; }
                stable_comparison_sort<Alloc>(p, n, comp);
                return;
            }
        } catch (const std::bad_alloc&) {
        }
    }
    std::stable_sort(first, last, std::move(comp));
}

template <class F, class T>
constexpr bool is_comparator_v = std::is_invocable_r_v<bool, F&, const T&, const T&>;
template <class F, class T>
constexpr bool is_projection_v = std::is_invocable_v<F&, const T&> && !std::is_invocable_v<F&, const T&, const T&>;

}  // namespace detail

// ---- public API ------------------------------------------------------------------------

// Sort by a key computed from each element.
template <std::random_access_iterator It, class Proj>
inline void sort_by_key(It first, It last, Proj proj) {
    detail::sort_by_key_impl<detail::DefaultAlloc>(first, last, std::move(proj));
}
template <std::ranges::random_access_range R, class Proj>
inline void sort_by_key(R&& r, Proj proj) {
    sort_by_key(std::ranges::begin(r), std::ranges::end(r), std::move(proj));
}

// Sort with an arbitrary comparator: stable, with the same handling of
// sorted, reversed and nearly sorted input as the key overloads. On
// trivially copyable elements the key the comparator compares is inferred
// from a sample of its answers, checked after the sort, and used for the
// radix sort when it holds; a comparator that is not the order of a window
// of the element gets the library's comparison sort.
template <std::random_access_iterator It, class Comp>
inline void sort_with(It first, It last, Comp comp) {
    detail::sort_with_impl<detail::DefaultAlloc>(first, last, std::move(comp));
}
template <std::ranges::random_access_range R, class Comp>
inline void sort_with(R&& r, Comp comp) {
    sort_with(std::ranges::begin(r), std::ranges::end(r), std::move(comp));
}

// Sort by the elements themselves.
template <std::random_access_iterator It>
inline void sort(It first, It last) {
    detail::sort_by_key_impl<detail::DefaultAlloc>(first, last, std::identity{});
}
template <std::ranges::random_access_range R>
inline void sort(R&& r) {
    sort(std::ranges::begin(r), std::ranges::end(r));
}

// Sort with a projection or a comparator, whichever `f` is.
template <std::random_access_iterator It, class F>
inline void sort(It first, It last, F f) {
    using T = std::iter_value_t<It>;
    static_assert(detail::is_comparator_v<F, T> || std::is_invocable_v<F&, const T&>,
                  "brainsort::sort: the third argument must be a key projection f(element) or a comparator f(a, b)");
    static_assert(!(detail::is_comparator_v<F, T> && std::is_invocable_v<F&, const T&>),
                  "brainsort::sort: this callable accepts one and two arguments; call sort_by_key or sort_with to say which");
    if constexpr (detail::is_comparator_v<F, T>) sort_with(first, last, std::move(f));
    else sort_by_key(first, last, std::move(f));
}
template <std::ranges::random_access_range R, class F>
inline void sort(R&& r, F f) {
    sort(std::ranges::begin(r), std::ranges::end(r), std::move(f));
}

// Free the blocks the library keeps for reuse (see BRAINSORT_MEMORY_CACHE in
// detail/traits.hpp). Never needed for correctness.
inline void release_memory() noexcept { detail::memory_cache().release(); }

// The same names as the standard library, for a drop-in replacement: every
// brainsort sort is stable.
template <std::random_access_iterator It>
inline void stable_sort(It first, It last) { sort(first, last); }
template <std::random_access_iterator It, class F>
inline void stable_sort(It first, It last, F f) { sort(first, last, std::move(f)); }
template <std::ranges::random_access_range R>
inline void stable_sort(R&& r) { sort(std::forward<R>(r)); }
template <std::ranges::random_access_range R, class F>
inline void stable_sort(R&& r, F f) { sort(std::forward<R>(r), std::move(f)); }

}  // namespace brainsort

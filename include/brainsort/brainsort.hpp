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
//   brainsort::sort(rows, [](const Row& a, const Row& b) { return a.name < b.name; }); // comparator: std::stable_sort
//
// Every overload is stable: equal keys keep their input order. The sort is
// O(n) passes over the data for fixed-size keys and O(n * key length) for
// strings; it never degrades on any input pattern.
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
// the final permutation. If an allocation fails the sort completes anyway,
// through std::stable_sort with the same order; nothing throws except the
// projection itself, and if that throws the range is unchanged.
//
// Limits: at most 2^32 - 1 elements per call (larger ranges go to
// std::stable_sort with the same order); strings of 2^32 bytes or more
// likewise. Thread safe: no shared mutable state.
#pragma once

#include "brainsort/detail/algorithm.hpp"
#include "brainsort/detail/config.hpp"
#include "brainsort/detail/keys.hpp"
#include "brainsort/detail/records.hpp"
#include "brainsort/detail/traits.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <new>
#include <ranges>
#include <type_traits>
#include <utility>

namespace brainsort {
namespace detail {

// Below this many elements the elements themselves are insertion sorted by
// key: no allocation, and cheaper than building records.
constexpr size_t kSmallSort = 32;

// An array of n objects of type U in memory from Alloc. For trivially
// constructible U it is raw storage; otherwise objects are constructed one
// by one with emplace() and destroyed in the destructor.
template <class U, class Alloc>
class Buf {
public:
    explicit Buf(size_t n) : n_(n), built_(0) {
        p_ = static_cast<U*>(Alloc::allocate(n * sizeof(U)));
        if constexpr (std::is_trivially_default_constructible_v<U>) built_ = n;
    }
    ~Buf() {
        if constexpr (!std::is_trivially_destructible_v<U>)
            for (size_t i = built_; i-- > 0;) p_[i].~U();
        Alloc::deallocate(p_, n_ * sizeof(U));
    }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    template <class... Args> U& emplace(Args&&... args) {
        U* u = ::new (static_cast<void*>(p_ + built_)) U(std::forward<Args>(args)...);
        ++built_;
        return *u;
    }
    U*     data() const noexcept { return p_; }
    size_t size() const noexcept { return n_; }
    U&     operator[](size_t i) const noexcept { return p_[i]; }
private:
    U*     p_;
    size_t n_;
    size_t built_;
};

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
// walk if the buffer cannot be allocated.
template <class Alloc, class It, class Rec>
inline void permute(It first, Rec* rec, size_t n) {
    using T = std::iter_value_t<It>;
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

// Already sorted or reversed input, recognised on the elements before any
// record is built or memory allocated: one pass that stops at the first pair
// that rules both out (a few elements into unordered input). Returns true if
// the range is sorted on return. Reversed input (non-increasing) is reversed
// in place with each group of equal keys put back into input order, so the
// result is stable; the groups are found in a scan that runs before anything
// moves, so a projection that throws leaves the range unchanged.
template <class It, class Proj>
inline bool sort_if_monotone(It first, size_t n, Proj& proj) {
    using K = key_of_t<It, Proj>;
    size_t desc = 0, asc = 0;
    {
        KeyHolder<It, Proj> prev(std::invoke(proj, first[0]));
        for (size_t i = 1; i < n; ++i) {
            KeyHolder<It, Proj> cur(std::invoke(proj, first[static_cast<std::ptrdiff_t>(i)]));
            const int c = compare_keys<false, K>(prev.get(), cur.get());
            desc += c > 0;
            asc  += c < 0;
            if (desc != 0 && asc != 0) return false;
            prev = std::move(cur);
        }
    }
    if (desc == 0) return true;   // non-decreasing: nothing to do
    // Non-increasing: reverse the whole range, then each tie group back.
    std::reverse(first, first + static_cast<std::ptrdiff_t>(n));
    size_t i = 0;
    while (i < n) {
        size_t j = i + 1;
        {
            KeyHolder<It, Proj> ki(std::invoke(proj, first[static_cast<std::ptrdiff_t>(i)]));
            while (j < n) {
                KeyHolder<It, Proj> kj(std::invoke(proj, first[static_cast<std::ptrdiff_t>(j)]));
                if (compare_keys<false, K>(ki.get(), kj.get()) != 0) break;
                ++j;
            }
        }
        if (j - i > 1) std::reverse(first + static_cast<std::ptrdiff_t>(i), first + static_cast<std::ptrdiff_t>(j));
        i = j;
    }
    return true;
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

// Sort [first, last) by proj(element), through records of type Rec. Returns
// false, with the range untouched, if the keys could not be represented.
template <class Alloc, class It, class Proj>
inline bool sort_records(It first, size_t n, Proj& proj) {
    using T   = std::iter_value_t<It>;
    using R   = proj_result_t<It, Proj>;
    using K   = key_of_t<It, Proj>;
    using Rec = record_t<K>;
    constexpr bool materialise = !std::is_reference_v<R> && kParts<K>.owning;   // keys are temporaries that own their bytes

    Buf<Rec, Alloc> recs(n);
    Rec* rec = recs.data();
    if constexpr (materialise) {
        Buf<K, Alloc> keys(n);
        for (size_t i = 0; i < n; ++i) keys.emplace(std::invoke(proj, first[static_cast<std::ptrdiff_t>(i)]));
        for (size_t i = 0; i < n; ++i)
            if (!build_record<K>(rec[i], keys[i], static_cast<uint32_t>(i))) return false;
        brainsort_impl<0>(View<Rec, Alloc>(rec, n));
    } else {
        for (size_t i = 0; i < n; ++i) {
            decltype(auto) k = std::invoke(proj, first[static_cast<std::ptrdiff_t>(i)]);
            if (!build_record<K>(rec[i], k, static_cast<uint32_t>(i))) return false;
        }
        brainsort_impl<0>(View<Rec, Alloc>(rec, n));
    }
    // The elements are the keys and the key inverts: write the sorted keys back.
    if constexpr (std::is_same_v<Proj, std::identity> && exact_single_v<K> && std::is_same_v<K, T> &&
                  (std::is_same_v<Rec, Rec32> || std::is_same_v<Rec, Rec64>)) {
        for (size_t i = 0; i < n; ++i) first[static_cast<std::ptrdiff_t>(i)] = key_of<K>(rec[i]);
    } else {
        permute<Alloc>(first, rec, n);
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
    if (safe_moves && sort_if_monotone(first, n, proj)) return;
    if (safe_moves && n <= 0xFFFFFFFFull && n <= (~size_t(0)) / sizeof(record_t<K>) / 4) {
        bool done = false;
        try {
            done = sort_records<Alloc>(first, n, proj);
        } catch (const std::bad_alloc&) {
            done = false;   // the range is untouched: sort it by comparison instead
        }
        if (done) return;
    }
    std::stable_sort(first, last, less);
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

// Sort with an arbitrary comparator: stable, but by comparison
// (std::stable_sort), not by brainsort.
template <std::random_access_iterator It, class Comp>
inline void sort_with(It first, It last, Comp comp) {
    std::stable_sort(first, last, std::move(comp));
}
template <std::ranges::random_access_range R, class Comp>
inline void sort_with(R&& r, Comp comp) {
    std::stable_sort(std::ranges::begin(r), std::ranges::end(r), std::move(comp));
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

// Free the memory this thread's sorts keep for reuse (see
// BRAINSORT_MEMORY_CACHE in detail/traits.hpp). Never needed for
// correctness; the memory is released when the thread ends.
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

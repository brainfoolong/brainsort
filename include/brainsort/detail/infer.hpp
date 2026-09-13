// brainsort: key inference for the comparator overloads.
//
// A comparator is a black box, but a trivially copyable element is not.
// The sort guesses that the comparator is the order of some aligned window
// of the element's bytes, read as an integer or a float, ascending or
// descending. Every such window is tested against a strided sample of
// adjacent-pair outcomes of the comparator; a window that agrees with all
// of them is sorted by the record path, the result is gathered into a
// buffer and verified with one sequential comparator pass. Equal runs are
// the comparator's equality classes, and sorting each by original index
// restores stability even when the window is finer than the comparator.
// Only a comparator that disagrees in direction somewhere fails the guess;
// the range is then still untouched and goes to the comparison sort.
#pragma once
#include "brainsort/detail/algorithm.hpp"
#include "brainsort/detail/config.hpp"
#include "brainsort/detail/keys.hpp"
#include "brainsort/detail/records.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace brainsort {
namespace detail {
namespace infer_detail {

// Adjacent pairs the comparator is asked about before a window is chosen.
constexpr size_t kSamplePairs = 2048;

enum class Kind : uint8_t { i8, u8, i16, u16, i32, u32, f32, i64, u64, f64 };
// Widest first: among the windows that agree with the sample the widest wins.
constexpr Kind kKinds[10] = {Kind::i64, Kind::u64, Kind::f64, Kind::i32, Kind::u32, Kind::f32, Kind::i16, Kind::u16, Kind::i8, Kind::u8};
constexpr size_t kind_size(Kind k) {
    switch (k) {
        case Kind::i8: case Kind::u8: return 1;
        case Kind::i16: case Kind::u16: return 2;
        case Kind::i32: case Kind::u32: case Kind::f32: return 4;
        default: return 8;
    }
}
constexpr bool kind_wide(Kind k) { return kind_size(k) == 8; }
constexpr bool kind_float(Kind k) { return k == Kind::f32 || k == Kind::f64; }
constexpr bool kind_signed(Kind k) { return k == Kind::i8 || k == Kind::i16 || k == Kind::i32 || k == Kind::i64; }

template <class U> inline U load(const unsigned char* q) noexcept {
    U u;
    std::memcpy(&u, q, sizeof u);
    return u;
}

// A candidate key: a window of the element read as `kind`, possibly descending.
struct Cand {
    size_t off  = 0;
    Kind   kind = Kind::i64;
    bool   desc = false;

    // The order-preserving radix value of the window of the element at p: 32
    // bits wide for the narrow kinds, 64 for the wide ones.
    uint64_t radix(const unsigned char* p) const noexcept {
        const unsigned char* q = p + off;
        uint64_t r;
        switch (kind) {
            case Kind::i8:  r = static_cast<uint32_t>(static_cast<int32_t>(load<int8_t>(q))) ^ 0x80000000u; break;
            case Kind::u8:  r = load<uint8_t>(q); break;
            case Kind::i16: r = static_cast<uint32_t>(static_cast<int32_t>(load<int16_t>(q))) ^ 0x80000000u; break;
            case Kind::u16: r = load<uint16_t>(q); break;
            case Kind::i32: r = static_cast<uint32_t>(load<int32_t>(q)) ^ 0x80000000u; break;
            case Kind::u32: r = load<uint32_t>(q); break;
            case Kind::f32: r = key_traits<float>::to_radix(load<float>(q)); break;
            case Kind::i64: return finish64(static_cast<uint64_t>(load<int64_t>(q)) ^ 0x8000000000000000ull);
            case Kind::u64: return finish64(load<uint64_t>(q));
            default:        return finish64(key_traits<double>::to_radix(load<double>(q)));
        }
        return desc ? static_cast<uint32_t>(~static_cast<uint32_t>(r)) : r;
    }
    uint64_t finish64(uint64_t r) const noexcept { return desc ? ~r : r; }

    // A float window is believed only if no sampled value is a denormal:
    // integers read as floats are denormals, floats in use never are.
    template <class T>
    bool float_plausible(const T* v, size_t n, size_t stride) const noexcept {
        const unsigned char* p = reinterpret_cast<const unsigned char*>(v);
        for (size_t i = 0; i < n; i += stride) {
            const unsigned char* q = p + i * sizeof(T) + off;
            bool denormal;
            if (kind == Kind::f32) {
                const uint32_t b = load<uint32_t>(q);
                denormal = (b & 0x7F800000u) == 0 && (b & 0x007FFFFFu) != 0;
            } else {
                const uint64_t b = load<uint64_t>(q);
                denormal = (b & 0x7FF0000000000000ull) == 0 && (b & 0x000FFFFFFFFFFFFFull) != 0;
            }
            if (denormal) return false;
        }
        return true;
    }
};

// The three-way outcome of a `less` comparator on a pair.
template <class T, class Comp>
inline int order3(const T& a, const T& b, Comp& comp) {
    return comp(a, b) ? -1 : (comp(b, a) ? 1 : 0);
}

// The window that agrees with every sampled comparator outcome, if any;
// among several, the widest, then a plausible float, then signed, then
// unsigned.
template <class Alloc, class T, class Comp>
inline bool infer(const T* v, size_t n, Comp& comp, Cand& best) {
    constexpr size_t sz = sizeof(T);
    if (n < 2) return false;
    const size_t pairs  = std::min(kSamplePairs, n - 1);
    const size_t stride = (n - 1) / pairs;
    struct Outcome { uint32_t i; int o; };
    Buf<Outcome, Alloc> outcomes(pairs);   // pair (i, i + 1) for i = k * stride
    bool any = false;
    for (size_t k = 0; k < pairs; ++k) {
        const size_t i = k * stride;
        outcomes[k] = {static_cast<uint32_t>(i), order3(v[i], v[i + 1], comp)};
        any |= outcomes[k].o != 0;
    }
    if (!any) return false;
    const unsigned char* p = reinterpret_cast<const unsigned char*>(v);
    unsigned best_score = 0;
    bool     found      = false;
    for (Kind kind : kKinds) {
        const size_t ks = kind_size(kind);
        if (ks > sz) continue;
        for (size_t off = 0; off + ks <= sz; off += ks) {
            for (bool desc : {false, true}) {
                const Cand c{off, kind, desc};
                bool agrees = true;
                for (size_t k = 0; agrees && k < pairs; ++k) {
                    const uint64_t a = c.radix(p + outcomes[k].i * sz), b = c.radix(p + (outcomes[k].i + 1) * sz);
                    agrees = (a < b ? -1 : (a > b ? 1 : 0)) == outcomes[k].o;
                }
                if (!agrees) continue;
                const unsigned score = static_cast<unsigned>(ks) * 10 +
                                       (kind_float(kind) ? (c.float_plausible(v, n, std::max<size_t>(stride, 1)) ? 3u : 0u) : (kind_signed(kind) ? 2u : 1u));
                if (!found || score > best_score) { best = c; best_score = score; found = true; }
            }
        }
    }
    return found;
}

// Sorts v[0,n) by the candidate through records of type Rec, verifies the
// gathered result with the comparator and restores stability inside equal
// runs. False, with the range untouched, when the comparator disagrees with
// the candidate somewhere.
template <class Alloc, class Rec, class T, class Comp>
inline bool sort_verified(T* v, size_t n, const Cand& c, Comp& comp) {
    constexpr size_t sz = sizeof(T);
    const unsigned char* p = reinterpret_cast<const unsigned char*>(v);
    Buf<Rec, Alloc> recs(n);
    Rec* rec = recs.data();
    for (size_t i = 0; i < n; ++i) {
        const uint64_t r = c.radix(p + i * sz);
        if constexpr (std::is_same_v<Rec, Rec64>) rec[i] = Rec64{static_cast<int64_t>(r ^ 0x8000000000000000ull), static_cast<uint32_t>(i), 0};
        else rec[i] = Rec32{static_cast<int32_t>(static_cast<uint32_t>(r) ^ 0x80000000u), static_cast<uint32_t>(i)};
    }
    brain_detail::Scratch<View<Rec, Alloc>> scratch;
    if (!brainsort_impl<0>(View<Rec, Alloc>(rec, n), scratch, true)) rec = scratch.buffer();
    Buf<T, Alloc> tmp(n);
    T* t = tmp.data();
    for (size_t i = 0; i < n; ++i) t[i] = v[rec[i].idx];
    // The equal run [s, e) is a class of the comparator: its elements go
    // into index order.
    auto fix_run = [&](size_t s, size_t e) {
        bool ordered = true;
        for (size_t i = s + 1; ordered && i < e; ++i) ordered = rec[i - 1].idx < rec[i].idx;
        if (ordered) return;
        std::sort(rec + s, rec + e, [](const Rec& a, const Rec& b) { return a.idx < b.idx; });
        for (size_t i = s; i < e; ++i) t[i] = v[rec[i].idx];
    };
    size_t s = 0;
    for (size_t i = 1; i < n; ++i) {
        if (comp(t[i - 1], t[i])) {            // less: the common case, one call
            if (i - s > 1) fix_run(s, i);
            s = i;
        } else if (comp(t[i], t[i - 1])) {     // greater: the guess was wrong
            return false;
        }
    }
    if (n - s > 1) fix_run(s, n);
    std::memcpy(static_cast<void*>(v), t, n * sizeof(T));
    return true;
}

}  // namespace infer_detail

// The comparator sort with key inference: true when the range was sorted
// through an inferred window, false (range untouched) when no window agrees
// with the comparator, or when memory ran out.
template <class Alloc, class T, class Comp>
inline bool sort_inferred(T* v, size_t n, Comp& comp) {
    infer_detail::Cand c;
    try {
        if (!infer_detail::infer<Alloc>(v, n, comp, c)) return false;
        if (infer_detail::kind_wide(c.kind)) return infer_detail::sort_verified<Alloc, Rec64>(v, n, c, comp);
        return infer_detail::sort_verified<Alloc, Rec32>(v, n, c, comp);
    } catch (const std::bad_alloc&) {
        return false;
    }
}

}  // namespace detail
}  // namespace brainsort

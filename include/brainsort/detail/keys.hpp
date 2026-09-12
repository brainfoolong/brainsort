// brainsort: key adapters.
//
// brainsort::key_traits<K> tells the library how to turn a key of type K
// into something the radix engine can sort. Three kinds of key exist:
//
//   fixed      an order-preserving unsigned integer of radix_bits bits
//                using radix_type = uint32_t | uint64_t;
//                static constexpr int  radix_bits;      1..64
//                static constexpr bool exact;           from_radix exists and inverts to_radix
//                static radix_type to_radix(const K&);  values < 2^radix_bits
//                static K from_radix(radix_type);       only if exact
//   bytes      a byte string, compared lexicographically as unsigned bytes
//                static constexpr bool owning;          the key object owns the bytes
//                static std::string_view bytes(const K&);
//   composite  a fixed sequence of keys, compared lexicographically
//                static constexpr size_t size;
//                template <size_t I> static decltype(auto) get(const K&);
//
// plus brainsort::descending<K>, which reverses the order of any key.
//
// Adapters are provided for every integral type (bool and the character
// types included), float and double, enums, pointers, std::string,
// std::string_view, const char*, char arrays, std::chrono durations and time
// points, std::pair, std::tuple and std::array. Specialise key_traits for
// your own key type; see the README.
//
// All keys of one sort must have the same shape, so the shape is a property
// of the key type alone and is computed at compile time here.
#pragma once

#include "brainsort/detail/traits.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace brainsort {

enum class key_kind { fixed, bytes, composite, descending };

// The customisation point. Undefined for types that are not keys.
template <class K, class Enable = void> struct key_traits;

// Reverses the order of a key: sort(v, [](auto& r) { return desc(r.score); }).
template <class K> struct descending { K key; };
template <class K> constexpr descending<std::remove_cvref_t<K>> desc(K&& k) {
    return descending<std::remove_cvref_t<K>>{std::forward<K>(k)};
}

// ---- built-in adapters --------------------------------------------------------------

// Integers, bool and the character types. Signed values are order-preserved
// by flipping the sign bit.
template <class K>
struct key_traits<K, std::enable_if_t<std::is_integral_v<K>>> {
    static constexpr key_kind kind       = key_kind::fixed;
    static constexpr int      radix_bits = std::is_same_v<K, bool> ? 1 : static_cast<int>(sizeof(K) * 8);
    using radix_type = std::conditional_t<(radix_bits <= 32), uint32_t, uint64_t>;
    static constexpr bool     exact      = true;
    static radix_type to_radix(K k) noexcept {
        if constexpr (std::is_same_v<K, bool>) {
            return k ? 1u : 0u;
        } else {
            using U = std::make_unsigned_t<K>;
            U u = static_cast<U>(k);
            if constexpr (std::is_signed_v<K>) u ^= static_cast<U>(U(1) << (radix_bits - 1));
            return static_cast<radix_type>(u);
        }
    }
    static K from_radix(radix_type r) noexcept {
        if constexpr (std::is_same_v<K, bool>) {
            return r != 0;
        } else {
            using U = std::make_unsigned_t<K>;
            U u = static_cast<U>(r);
            if constexpr (std::is_signed_v<K>) u ^= static_cast<U>(U(1) << (radix_bits - 1));
            return static_cast<K>(u);
        }
    }
};

// float and double: IEEE-754 order as an unsigned key. A positive value maps
// to bits | 2^(w-1), a negative one to 2^(w-1) - magnitude, so -0.0 and +0.0
// both map to 2^(w-1) and compare equal, as operator< says. NaNs get a
// defined place: a NaN with the sign bit clear sorts after +infinity, one
// with the sign bit set before -infinity (ordered by payload), which makes
// the order total and the sort safe on any input.
template <>
struct key_traits<float> {
    static constexpr key_kind kind       = key_kind::fixed;
    static constexpr int      radix_bits = 32;
    using radix_type = uint32_t;
    static constexpr bool     exact      = false;
    static uint32_t to_radix(float f) noexcept {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof bits);
        const uint32_t sign = 0x80000000u;
        return (bits & sign) ? sign - (bits & ~sign) : (bits | sign);
    }
};
template <>
struct key_traits<double> {
    static constexpr key_kind kind       = key_kind::fixed;
    static constexpr int      radix_bits = 64;
    using radix_type = uint64_t;
    static constexpr bool     exact      = false;
    static uint64_t to_radix(double d) noexcept {
        uint64_t bits;
        std::memcpy(&bits, &d, sizeof bits);
        const uint64_t sign = 0x8000000000000000ull;
        return (bits & sign) ? sign - (bits & ~sign) : (bits | sign);
    }
};

// Enumerations: by their underlying integer.
template <class K>
struct key_traits<K, std::enable_if_t<std::is_enum_v<K>>> {
    using U    = std::underlying_type_t<K>;
    using base = key_traits<U>;
    static constexpr key_kind kind       = key_kind::fixed;
    static constexpr int      radix_bits = base::radix_bits;
    using radix_type = typename base::radix_type;
    static constexpr bool     exact      = true;
    static radix_type to_radix(K k) noexcept { return base::to_radix(static_cast<U>(k)); }
    static K from_radix(radix_type r) noexcept { return static_cast<K>(base::from_radix(r)); }
};

namespace detail {
template <class P> constexpr bool is_char_pointer_v =
    std::is_pointer_v<P> && std::is_same_v<std::remove_cv_t<std::remove_pointer_t<P>>, char>;
}

// Pointers (other than char pointers, which are C strings): by address.
template <class K>
struct key_traits<K, std::enable_if_t<std::is_pointer_v<K> && !detail::is_char_pointer_v<K>>> {
    static constexpr key_kind kind       = key_kind::fixed;
    static constexpr int      radix_bits = static_cast<int>(sizeof(void*) * 8);
    using radix_type = std::conditional_t<(radix_bits <= 32), uint32_t, uint64_t>;
    static constexpr bool     exact      = true;
    static radix_type to_radix(K k) noexcept { return static_cast<radix_type>(reinterpret_cast<uintptr_t>(k)); }
    static K from_radix(radix_type r) noexcept { return reinterpret_cast<K>(static_cast<uintptr_t>(r)); }
};

// Strings: std::string, std::string_view (char and char8_t), C strings and
// char arrays, all compared as unsigned bytes like std::string does.
template <class Tr, class Al>
struct key_traits<std::basic_string<char, Tr, Al>> {
    static constexpr key_kind kind   = key_kind::bytes;
    static constexpr bool     owning = true;
    static std::string_view bytes(const std::basic_string<char, Tr, Al>& s) noexcept { return {s.data(), s.size()}; }
};
template <class Tr, class Al>
struct key_traits<std::basic_string<char8_t, Tr, Al>> {
    static constexpr key_kind kind   = key_kind::bytes;
    static constexpr bool     owning = true;
    static std::string_view bytes(const std::basic_string<char8_t, Tr, Al>& s) noexcept {
        return {reinterpret_cast<const char*>(s.data()), s.size()};
    }
};
template <class Tr>
struct key_traits<std::basic_string_view<char, Tr>> {
    static constexpr key_kind kind   = key_kind::bytes;
    static constexpr bool     owning = false;
    static std::string_view bytes(std::basic_string_view<char, Tr> s) noexcept { return {s.data(), s.size()}; }
};
template <class Tr>
struct key_traits<std::basic_string_view<char8_t, Tr>> {
    static constexpr key_kind kind   = key_kind::bytes;
    static constexpr bool     owning = false;
    static std::string_view bytes(std::basic_string_view<char8_t, Tr> s) noexcept {
        return {reinterpret_cast<const char*>(s.data()), s.size()};
    }
};
template <class K>
struct key_traits<K, std::enable_if_t<detail::is_char_pointer_v<K>>> {
    static constexpr key_kind kind   = key_kind::bytes;
    static constexpr bool     owning = false;
    static std::string_view bytes(K s) noexcept { return s ? std::string_view(s) : std::string_view(); }
};
template <size_t N>
struct key_traits<char[N]> {
    static constexpr key_kind kind   = key_kind::bytes;
    static constexpr bool     owning = false;
    static std::string_view bytes(const char (&s)[N]) noexcept {
        size_t len = 0;
        while (len < N && s[len] != '\0') ++len;
        return {s, len};
    }
};

// std::chrono durations and time points: by their tick count.
template <class Rep, class Period>
struct key_traits<std::chrono::duration<Rep, Period>> {
    using D    = std::chrono::duration<Rep, Period>;
    using base = key_traits<Rep>;
    static constexpr key_kind kind       = key_kind::fixed;
    static constexpr int      radix_bits = base::radix_bits;
    using radix_type = typename base::radix_type;
    static constexpr bool     exact      = base::exact;
    static radix_type to_radix(const D& d) noexcept { return base::to_radix(d.count()); }
    static D from_radix(radix_type r) noexcept { return D(base::from_radix(r)); }
};
template <class Clock, class Dur>
struct key_traits<std::chrono::time_point<Clock, Dur>> {
    using TP   = std::chrono::time_point<Clock, Dur>;
    using base = key_traits<Dur>;
    static constexpr key_kind kind       = key_kind::fixed;
    static constexpr int      radix_bits = base::radix_bits;
    using radix_type = typename base::radix_type;
    static constexpr bool     exact      = base::exact;
    static radix_type to_radix(const TP& t) noexcept { return base::to_radix(t.time_since_epoch()); }
    static TP from_radix(radix_type r) noexcept { return TP(base::from_radix(r)); }
};

// Composite keys: pair, tuple, array. Compared lexicographically, member by
// member, exactly like their operator<.
template <class A, class B>
struct key_traits<std::pair<A, B>> {
    static constexpr key_kind kind = key_kind::composite;
    static constexpr size_t   size = 2;
    template <size_t I> static decltype(auto) get(const std::pair<A, B>& p) noexcept { return std::get<I>(p); }
};
template <class... Ts>
struct key_traits<std::tuple<Ts...>> {
    static constexpr key_kind kind = key_kind::composite;
    static constexpr size_t   size = sizeof...(Ts);
    template <size_t I> static decltype(auto) get(const std::tuple<Ts...>& t) noexcept { return std::get<I>(t); }
};
template <class K, size_t N>
struct key_traits<std::array<K, N>> {
    static constexpr key_kind kind = key_kind::composite;
    static constexpr size_t   size = N;
    template <size_t I> static const K& get(const std::array<K, N>& a) noexcept { return a[I]; }
};

// descending<K>: the same key, reversed.
template <class K>
struct key_traits<descending<K>> {
    static constexpr key_kind kind = key_kind::descending;
    using inner = K;
    static const K& inner_ref(const descending<K>& d) noexcept { return d.key; }
};

// ---- shape of a key type ------------------------------------------------------------
namespace detail {

template <class K, class = void> struct has_key_traits : std::false_type {};
template <class K> struct has_key_traits<K, std::void_t<decltype(key_traits<K>::kind)>> : std::true_type {};
template <class K> constexpr bool is_key_v = has_key_traits<std::remove_cvref_t<K>>::value;

template <class K, size_t I>
using part_t = std::remove_cvref_t<decltype(key_traits<K>::template get<I>(std::declval<const K&>()))>;

// One leaf of a key: a fixed part of `bits` bits or a byte string, either
// possibly reversed.
struct PartInfo {
    bool bytes = false;
    int  bits  = 0;
    bool desc  = false;
};
constexpr int kMaxParts = 32;
struct PartList {
    PartInfo p[kMaxParts] = {};
    int      n      = 0;
    bool     owning = false;   // some bytes part owns its storage (std::string)
};

template <class K> constexpr void collect_parts(PartList& L, bool desc) {
    using KT = key_traits<K>;
    if constexpr (KT::kind == key_kind::fixed) {
        if (L.n < kMaxParts) L.p[L.n] = PartInfo{false, KT::radix_bits, desc};
        ++L.n;
    } else if constexpr (KT::kind == key_kind::bytes) {
        if (L.n < kMaxParts) L.p[L.n] = PartInfo{true, 0, desc};
        ++L.n;
        L.owning = L.owning || KT::owning;
    } else if constexpr (KT::kind == key_kind::descending) {
        collect_parts<typename KT::inner>(L, !desc);
    } else {
        [&]<size_t... I>(std::index_sequence<I...>) {
            (collect_parts<part_t<K, I>>(L, desc), ...);
        }(std::make_index_sequence<KT::size>{});
    }
}
template <class K> constexpr PartList parts_of() {
    PartList L;
    collect_parts<K>(L, false);
    return L;
}
template <class K> constexpr PartList kParts = parts_of<K>();

template <class K> constexpr bool shape_ok_v = kParts<K>.n >= 1 && kParts<K>.n <= kMaxParts;
template <class K> constexpr bool all_fixed_v = [] {
    const PartList& L = kParts<K>;
    for (int i = 0; i < L.n; ++i) if (L.p[i].bytes) return false;
    return true;
}();
template <class K> constexpr int total_fixed_bits_v = [] {
    const PartList& L = kParts<K>;
    int b = 0;
    for (int i = 0; i < L.n; ++i) b += L.p[i].bits;
    return b;
}();
// A single fixed key whose adapter can invert the radix: the sorted keys can
// be written back directly when the elements are the keys.
template <class K> constexpr bool exact_single_v = [] {
    if constexpr (key_traits<K>::kind == key_kind::fixed) return key_traits<K>::exact;
    else return false;
}();

// Visit the leaves of a key in order: f(leaf, std::bool_constant<desc>).
template <class K, bool Desc, class F>
inline void for_each_part(const K& k, F& f) {
    using KT = key_traits<K>;
    if constexpr (KT::kind == key_kind::fixed || KT::kind == key_kind::bytes) {
        f(k, std::bool_constant<Desc>{});
    } else if constexpr (KT::kind == key_kind::descending) {
        for_each_part<typename KT::inner, !Desc>(KT::inner_ref(k), f);
    } else {
        [&]<size_t... I>(std::index_sequence<I...>) {
            (for_each_part<part_t<K, I>, Desc>(KT::template get<I>(k), f), ...);
        }(std::make_index_sequence<KT::size>{});
    }
}

// The order of the records, as a three-way compare of two keys: the
// comparator of the small-range insertion sort and of the fallbacks, so every
// path orders exactly alike (NaN and signed zeros included).
inline int normalize3(int c) noexcept { return c < 0 ? -1 : (c > 0 ? 1 : 0); }
template <bool Desc, class K>
inline int compare_keys(const K& a, const K& b) {
    using KT = key_traits<K>;
    if constexpr (KT::kind == key_kind::fixed) {
        const auto ra = KT::to_radix(a), rb = KT::to_radix(b);
        const int  c  = ra < rb ? -1 : (ra > rb ? 1 : 0);
        return Desc ? -c : c;
    } else if constexpr (KT::kind == key_kind::bytes) {
        const std::string_view sa = KT::bytes(a), sb = KT::bytes(b);
        const size_t m = sa.size() < sb.size() ? sa.size() : sb.size();
        int c = m ? std::memcmp(sa.data(), sb.data(), m) : 0;
        if (c == 0) c = sa.size() < sb.size() ? -1 : (sa.size() > sb.size() ? 1 : 0);
        c = normalize3(c);
        return Desc ? -c : c;
    } else if constexpr (KT::kind == key_kind::descending) {
        return compare_keys<!Desc, typename KT::inner>(KT::inner_ref(a), KT::inner_ref(b));
    } else {
        int c = 0;
        [&]<size_t... I>(std::index_sequence<I...>) {
            ((c == 0 ? (c = compare_keys<Desc, part_t<K, I>>(KT::template get<I>(a), KT::template get<I>(b))) : 0), ...);
        }(std::make_index_sequence<KT::size>{});
        return c;
    }
}

}  // namespace detail
}  // namespace brainsort

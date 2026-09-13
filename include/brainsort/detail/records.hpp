// brainsort: the sort records.
//
// The public API never sorts the elements of the caller directly. It builds
// an array of small records, each holding the radix form of one key and the
// original index, sorts those with the core algorithm, and then permutes the
// elements once. That gives every element type, whatever its size or
// copyability, the same vectorised paths, and it means the caller's data is
// untouched until the final permutation.
//
// Four record types cover every key shape:
//   Rec32     fixed keys of up to 32 bits in total (8 bytes, AVX2 scout)
//   Rec64     fixed keys of up to 64 bits in total (16 bytes, AVX2 scout)
//   StrRec    a single byte string (16 bytes, chunked)
//   CompRec   anything else: a sequence of fixed and byte-string parts
// Several fixed keys are packed into one radix value (a pair<int, int> is
// one 64-bit key), so composite keys of integers stay on the fast paths.
#pragma once

#include "brainsort/detail/keys.hpp"
#include "brainsort/detail/traits.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <type_traits>
#include <utility>

namespace brainsort {
namespace detail {

// ---- fixed-key records ------------------------------------------------------------
// The key is stored as the signed integer whose signed order equals the radix
// order (radix ^ sign bit), which is what the vector compares expect.
struct Rec32 {
    int32_t  key;
    uint32_t idx;
};
struct Rec64 {
    int64_t  key;
    uint32_t idx;
    uint32_t pad;
};
static_assert(sizeof(Rec32) == 8 && sizeof(Rec64) == 16);

// ---- string record ----------------------------------------------------------------
template <bool Desc>
struct StrRec {
    const char* ptr;
    uint32_t    len;
    uint32_t    idx;
};

// ---- composite record -------------------------------------------------------------
// Kinds: 0 = fixed part (64-bit radix), 1 = byte string, 2 = byte string reversed.
struct BytesSlot {
    const char* ptr;
    uint32_t    len;
    uint32_t    pad;
};
template <int... Kinds>
struct CompRec {
    static constexpr int    kNParts  = sizeof...(Kinds);
    static constexpr int    kinds[]  = {Kinds...};
    static constexpr int    n_fixed  = ((Kinds == 0 ? 1 : 0) + ...);
    static constexpr int    n_bytes  = kNParts - n_fixed;
    uint64_t  f[n_fixed > 0 ? n_fixed : 1];
    BytesSlot b[n_bytes > 0 ? n_bytes : 1];
    uint32_t  idx;
    uint32_t  pad;
};

}  // namespace detail

// ---- element traits of the records ---------------------------------------------------
template <> struct elem_traits<detail::Rec32> {
    using T = detail::Rec32;
    static constexpr bool     chunked = false;
    static constexpr SimdKind simd    = SimdKind::i32;
    using key_type = uint32_t;
    static bool     less(T a, T b) { return a.key < b.key; }
    static int      compare(T a, T b) { return a.key < b.key ? -1 : (b.key < a.key ? 1 : 0); }
    static int      compare_from(T a, T b, int) { return compare(a, b); }
    static key_type radix_key(T a, int) { return static_cast<uint32_t>(a.key) ^ 0x80000000u; }
    static bool     chunk_ends(T, int) { return true; }
};
template <> struct elem_traits<detail::Rec64> {
    using T = detail::Rec64;
    static constexpr bool     chunked = false;
    static constexpr SimdKind simd    = SimdKind::i64;
    using key_type = uint64_t;
    static bool     less(T a, T b) { return a.key < b.key; }
    static int      compare(T a, T b) { return a.key < b.key ? -1 : (b.key < a.key ? 1 : 0); }
    static int      compare_from(T a, T b, int) { return compare(a, b); }
    static key_type radix_key(T a, int) { return static_cast<uint64_t>(a.key) ^ 0x8000000000000000ull; }
    static bool     chunk_ends(T, int) { return true; }
};
template <bool Desc> struct elem_traits<detail::StrRec<Desc>> {
    using T = detail::StrRec<Desc>;
    static constexpr bool     chunked = true;
    static constexpr SimdKind simd    = SimdKind::none;
    using key_type = uint64_t;
    BRAINSORT_ALWAYS_INLINE static int compare_from(T a, T b, int chunk) {
        const int c = detail::str_compare_from(a.ptr, a.len, b.ptr, b.len, static_cast<uint32_t>(chunk) * detail::kStrChunkBytes);
        return Desc ? -c : c;
    }
    static int      compare(T a, T b) { return compare_from(a, b, 0); }
    static bool     less(T a, T b) { return compare_from(a, b, 0) < 0; }
    static key_type radix_key(T a, int chunk) {
        return Desc ? detail::str_chunk_key_desc(a.ptr, a.len, chunk) : detail::str_chunk_key(a.ptr, a.len, chunk);
    }
    static bool chunk_ends(T a, int chunk) { return detail::str_chunk_ends(a.len, chunk); }
};
template <int... Kinds> struct elem_traits<detail::CompRec<Kinds...>> {
    using T = detail::CompRec<Kinds...>;
    static constexpr bool     chunked = true;
    static constexpr SimdKind simd    = SimdKind::none;
    using key_type = uint64_t;
    static constexpr int kN = T::kNParts;

    // Slot index of each part within f[] or b[].
    struct Layout {
        int slot[kN > 0 ? kN : 1] = {};
        constexpr Layout() {
            int fi = 0, bi = 0;
            for (int p = 0; p < kN; ++p) slot[p] = T::kinds[p] == 0 ? fi++ : bi++;
        }
    };
    static constexpr Layout kLayout{};

    // Chunks of part p of element a: one for a fixed part, len / 7 + 1 for a string.
    static int chunks_of(const T& a, int p) {
        return T::kinds[p] == 0 ? 1 : static_cast<int>(a.b[kLayout.slot[p]].len / detail::kStrChunkBytes) + 1;
    }
    // The part that owns global chunk `chunk` of element a, and the chunk index within it.
    static int locate(const T& a, int chunk, int& local) {
        int p = 0;
        for (; p < kN; ++p) {
            const int nc = chunks_of(a, p);
            if (chunk < nc) { local = chunk; return p; }
            chunk -= nc;
        }
        local = 0;
        return kN;   // past the end: cannot happen for a chunk the core asks about
    }
    static uint64_t part_chunk_key(const T& a, int p, int local) {
        const int kind = T::kinds[p];
        if (kind == 0) return a.f[kLayout.slot[p]];
        const detail::BytesSlot& s = a.b[kLayout.slot[p]];
        return kind == 2 ? detail::str_chunk_key_desc(s.ptr, s.len, local) : detail::str_chunk_key(s.ptr, s.len, local);
    }
    static key_type radix_key(const T& a, int chunk) {
        int local;
        const int p = locate(a, chunk, local);
        return p < kN ? part_chunk_key(a, p, local) : 0;
    }
    static bool chunk_ends(const T& a, int chunk) {
        int local;
        const int p = locate(a, chunk, local);
        if (p >= kN - 1) {
            if (p >= kN) return true;
            return T::kinds[p] == 0 || local == chunks_of(a, p) - 1;
        }
        return false;
    }
    static int compare_part(const T& a, const T& b, int p, uint32_t off) {
        const int kind = T::kinds[p];
        if (kind == 0) {
            const uint64_t x = a.f[kLayout.slot[p]], y = b.f[kLayout.slot[p]];
            return x < y ? -1 : (x > y ? 1 : 0);
        }
        const detail::BytesSlot& sa = a.b[kLayout.slot[p]];
        const detail::BytesSlot& sb = b.b[kLayout.slot[p]];
        const int c = detail::str_compare_from(sa.ptr, sa.len, sb.ptr, sb.len, off);
        return kind == 2 ? -c : c;
    }
    // Compare from chunk `chunk` on: a and b agree on every earlier chunk, so
    // the chunk lies in the same part of both and the parts before it tie.
    static int compare_from(const T& a, const T& b, int chunk) {
        int local;
        int p = locate(a, chunk, local);
        if (p >= kN) return 0;
        int c = compare_part(a, b, p, static_cast<uint32_t>(local) * detail::kStrChunkBytes);
        for (++p; c == 0 && p < kN; ++p) c = compare_part(a, b, p, 0);
        return c;
    }
    static int  compare(const T& a, const T& b) { return compare_from(a, b, 0); }
    static bool less(const T& a, const T& b) { return compare_from(a, b, 0) < 0; }
};

namespace detail {

// ---- choosing the record type for a key type -------------------------------------------
constexpr int kind_code(const PartInfo& p) { return !p.bytes ? 0 : (p.desc ? 2 : 1); }

template <class K, size_t... I>
auto make_comp_rec(std::index_sequence<I...>) -> CompRec<kind_code(kParts<K>.p[I])...>;

template <class K, class = void> struct record_for;
template <class K>
struct record_for<K, std::enable_if_t<shape_ok_v<K>>> {
    static constexpr const PartList& L = kParts<K>;
    using type = std::conditional_t<
        all_fixed_v<K> && total_fixed_bits_v<K> <= 32, Rec32,
        std::conditional_t<
            all_fixed_v<K> && total_fixed_bits_v<K> <= 64, Rec64,
            std::conditional_t<
                (L.n == 1 && L.p[0].bytes && !L.p[0].desc), StrRec<false>,
                std::conditional_t<
                    (L.n == 1 && L.p[0].bytes && L.p[0].desc), StrRec<true>,
                    decltype(make_comp_rec<K>(std::make_index_sequence<static_cast<size_t>(L.n)>{}))>>>>;
};
template <class K> using record_t = typename record_for<K>::type;

// ---- building a record from a key --------------------------------------------------------
// Returns false if a key cannot be represented (a string of 2^32 bytes or
// more); the caller then sorts by comparison instead.
template <class Rec> struct RecordBuilder;

template <> struct RecordBuilder<Rec32> {
    uint32_t acc = 0;
    template <class P, bool D> void operator()(const P& p, std::bool_constant<D>) {
        using PT = key_traits<P>;
        uint32_t r = static_cast<uint32_t>(PT::to_radix(p));
        if constexpr (D) r = static_cast<uint32_t>((PT::radix_bits >= 32 ? ~uint32_t(0) : ((uint32_t(1) << PT::radix_bits) - 1)) - r);
        acc = PT::radix_bits >= 32 ? r : ((acc << PT::radix_bits) | r);
    }
    bool finish(Rec32& rec, uint32_t idx) const {
        rec.key = static_cast<int32_t>(acc ^ 0x80000000u);
        rec.idx = idx;
        return true;
    }
};
template <> struct RecordBuilder<Rec64> {
    uint64_t acc = 0;
    template <class P, bool D> void operator()(const P& p, std::bool_constant<D>) {
        using PT = key_traits<P>;
        uint64_t r = static_cast<uint64_t>(PT::to_radix(p));
        if constexpr (D) r = (PT::radix_bits >= 64 ? ~uint64_t(0) : ((uint64_t(1) << PT::radix_bits) - 1)) - r;
        acc = PT::radix_bits >= 64 ? r : ((acc << PT::radix_bits) | r);
    }
    bool finish(Rec64& rec, uint32_t idx) const {
        rec.key = static_cast<int64_t>(acc ^ 0x8000000000000000ull);
        rec.idx = idx;
        rec.pad = 0;
        return true;
    }
};
template <bool Desc> struct RecordBuilder<StrRec<Desc>> {
    const char* ptr = nullptr;
    size_t      len = 0;
    template <class P, bool D> void operator()(const P& p, std::bool_constant<D>) {
        const std::string_view s = key_traits<P>::bytes(p);
        ptr = s.data();
        len = s.size();
    }
    bool finish(StrRec<Desc>& rec, uint32_t idx) const {
        if (len > 0xFFFFFFFFull) return false;
        rec.ptr = ptr ? ptr : "";
        rec.len = static_cast<uint32_t>(len);
        rec.idx = idx;
        return true;
    }
};
template <int... Kinds> struct RecordBuilder<CompRec<Kinds...>> {
    using Rec = CompRec<Kinds...>;
    Rec  rec{};
    int  fi = 0, bi = 0;
    bool ok = true;
    template <class P, bool D> void operator()(const P& p, std::bool_constant<D>) {
        using PT = key_traits<P>;
        if constexpr (PT::kind == key_kind::fixed) {
            uint64_t r = static_cast<uint64_t>(PT::to_radix(p));
            if constexpr (D) r = (PT::radix_bits >= 64 ? ~uint64_t(0) : ((uint64_t(1) << PT::radix_bits) - 1)) - r;
            rec.f[fi++] = r;
        } else {
            const std::string_view s = PT::bytes(p);
            if (s.size() > 0xFFFFFFFFull) ok = false;
            rec.b[bi].ptr = s.data() ? s.data() : "";
            rec.b[bi].len = static_cast<uint32_t>(s.size());
            rec.b[bi].pad = 0;
            ++bi;
        }
    }
    bool finish(Rec& out, uint32_t idx) {
        rec.idx = idx;
        rec.pad = 0;
        out = rec;
        return ok;
    }
};

template <class K, class Rec>
inline bool build_record(Rec& rec, const K& key, uint32_t idx) {
    RecordBuilder<Rec> b;
    for_each_part<K, false>(key, b);
    return b.finish(rec, idx);
}

// The original index of a sorted record.
inline uint32_t index_of(const Rec32& r) noexcept { return r.idx; }
inline uint32_t index_of(const Rec64& r) noexcept { return r.idx; }
template <bool D> inline uint32_t index_of(const StrRec<D>& r) noexcept { return r.idx; }
template <int... Ks> inline uint32_t index_of(const CompRec<Ks...>& r) noexcept { return r.idx; }
inline void set_index(Rec32& r, uint32_t i) noexcept { r.idx = i; }
inline void set_index(Rec64& r, uint32_t i) noexcept { r.idx = i; }
template <bool D> inline void set_index(StrRec<D>& r, uint32_t i) noexcept { r.idx = i; }
template <int... Ks> inline void set_index(CompRec<Ks...>& r, uint32_t i) noexcept { r.idx = i; }

// The key of a sorted fixed record, for the write-back of self-keyed elements.
template <class K> inline K key_of(const Rec32& r) noexcept {
    return key_traits<K>::from_radix(static_cast<typename key_traits<K>::radix_type>(static_cast<uint32_t>(r.key) ^ 0x80000000u));
}
template <class K> inline K key_of(const Rec64& r) noexcept {
    return key_traits<K>::from_radix(static_cast<typename key_traits<K>::radix_type>(static_cast<uint64_t>(r.key) ^ 0x8000000000000000ull));
}

// A double is written back from its record as well: the radix transform is
// a bijection on every bit pattern except that -0.0 and +0.0 share a key,
// so the record's spare word remembers a negative zero.
inline void note_negative_zero(Rec64& r, double d) noexcept {
    r.pad = d == 0.0 && std::signbit(d) ? 1u : 0u;
}
inline double double_of(const Rec64& r) noexcept {
    const uint64_t sign = 0x8000000000000000ull;
    const uint64_t u    = static_cast<uint64_t>(r.key) ^ sign;
    uint64_t bits       = u & sign ? (u & ~sign) : (sign | (sign - u));
    if (r.pad) bits = sign;   // -0.0
    double d;
    std::memcpy(&d, &bits, sizeof d);
    return d;
}

}  // namespace detail
}  // namespace brainsort

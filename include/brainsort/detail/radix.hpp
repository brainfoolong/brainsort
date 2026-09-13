// brainsort: stable LSD radix sort core. A pluggable key function maps an
// element to an unsigned 32- or 64-bit value whose digits are distributed.
//
// The plain lsd_radix builds the histograms for every pass in one read pass
// over the input and skips passes in which all elements land in the same
// bucket; the fused variant below it (what brainsort uses) builds the next
// histogram while scattering the current pass.
#pragma once
#include "brainsort/detail/traits.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace brainsort {
namespace detail {
namespace radix_detail {

constexpr int kMaxPasses = 8;   // 64-bit keys with 8-bit digits

struct DigitPlan {
    int passes = 0;
    int shift[kMaxPasses] = {};
    int bits[kMaxPasses] = {};
    int max_bits = 0;
    size_t width() const { return size_t(1) << max_bits; }
};

// Split `total_bits` into the fewest digits of at most `max_digit_bits`,
// as evenly as possible (32 bits / 11 -> 11,11,10; 20 / 11 -> 10,10).
inline DigitPlan make_plan(int total_bits, int max_digit_bits) {
    DigitPlan p;
    if (total_bits <= 0) return p;
    p.passes = (total_bits + max_digit_bits - 1) / max_digit_bits;
    const int base = total_bits / p.passes, extra = total_bits % p.passes;
    int sh = 0;
    for (int i = 0; i < p.passes; ++i) {
        const int b = base + (i < extra ? 1 : 0);
        p.bits[i]  = b;
        p.shift[i] = sh;
        sh += b;
        p.max_bits = std::max(p.max_bits, b);
    }
    return p;
}

// Digit width for an array of n elements: a table of ~n entries is the sweet
// spot on current CPUs (fewer passes beat smaller tables), bounded to 8..16.
inline int digit_bits_for(size_t n) {
    int b = 0;
    while ((size_t(1) << (b + 1)) <= n) ++b;
    return b < 8 ? 8 : (b > 16 ? 16 : b);
}

template <class K> constexpr int key_bits() { return static_cast<int>(sizeof(K)) * 8; }

// Key functors: element -> unsigned key for the given chunk, through the
// view type A so the counted path can tally the string bytes a chunk
// extraction loads. Each also maps an already computed radix key (raw) so a
// caller that needs the raw key as well computes it once.
template <class A> struct FullKey {          // the full radix key
    using T = typename A::value_type;
    using K = typename A::key_type;
    int chunk;
    K operator()(T x) const { return A::key(x, chunk); }
    K raw(K k) const { return k; }
};
template <class A> struct ShiftKey {         // contiguous varying-bit range, shifted down
    using T = typename A::value_type;
    using K = typename A::key_type;
    int chunk, low;
    K operator()(T x) const { return A::key(x, chunk) >> low; }
    K raw(K k) const { return k >> low; }
};
// Portable PEXT: the bits of k that `mask` selects, compressed to the bottom,
// one step per mask bit. The counted path uses it on every CPU so the numbers
// it records do not depend on BMI2; the timed path uses the instruction.
template <class K> inline K pext_soft(K k, K mask) noexcept {
    K out = 0, bit = 1;
    for (K m = mask; m != 0; m &= m - 1, bit <<= 1)
        if (k & (m & (~m + 1))) out |= bit;
    return out;
}
template <class A> struct PextKeySoft {      // only the varying bits, compressed to the bottom (portable)
    using T = typename A::value_type;
    using K = typename A::key_type;
    int chunk;
    K   mask;
    K operator()(T x) const { return raw(A::key(x, chunk)); }
    K raw(K k) const { return pext_soft(k, mask); }
};
#ifdef BRAINSORT_X86_64
template <class A> struct PextKey {          // only the varying bits, compressed to the bottom (BMI2)
    using T = typename A::value_type;
    using K = typename A::key_type;
    int chunk;
    K   mask;
    BRAINSORT_TARGET_BMI2 K operator()(T x) const { return raw(A::key(x, chunk)); }
    BRAINSORT_TARGET_BMI2 K raw(K k) const {
        if constexpr (sizeof(K) == 8) return static_cast<K>(_pext_u64(k, mask));
        else return static_cast<K>(_pext_u32(k, mask));
    }
};
#endif

// Turn counts into exclusive prefix sums and flag passes where every element
// shares one digit.
template <class Acc>
inline void radix_prefix(uint32_t* h, size_t n, const DigitPlan& plan, bool* trivial) {
    const size_t width = plan.width();
    for (int p = 0; p < plan.passes; ++p) {
        uint32_t*    hp      = h + static_cast<size_t>(p) * width;
        const size_t buckets = size_t(1) << plan.bits[p];
        if constexpr (Acc::counted) Acc::hooks::on_table_sweep(hp, buckets, sizeof(uint32_t), true, true);
        uint32_t sum = 0;
        trivial[p]   = false;
        for (size_t d = 0; d < buckets; ++d) {
            const uint32_t cnt = hp[d];
            if (cnt == n) trivial[p] = true;
            hp[d] = sum;
            sum += cnt;
        }
    }
}

// Run the scatter passes. Result always ends up in src.
template <class Acc, class KeyFn>
BRAINSORT_ALWAYS_INLINE void radix_scatter(Acc src, Acc dst, size_t n, const DigitPlan& plan, KeyFn key,
                                           uint32_t* h, const bool* trivial) {
    Acc  s = src, d = dst;
    bool in_src = true;
    for (int p = 0; p < plan.passes; ++p) {
        if (trivial[p]) { if constexpr (Acc::counted) ++Acc::hooks::stats().passes_skipped; continue; }
        if constexpr (Acc::counted) ++Acc::hooks::stats().radix_passes;
        uint32_t*      hp    = h + static_cast<size_t>(p) * plan.width();
        const int      shift = plan.shift[p];
        const uint32_t mask  = (1u << plan.bits[p]) - 1;
        for (size_t i = 0; i < n; ++i) {
            const auto     e  = s.get(i);
            const uint32_t dg = static_cast<uint32_t>(key(e) >> shift) & mask;
            if constexpr (Acc::counted) Acc::hooks::on_table_rw(hp + dg, sizeof(uint32_t));
            d.set(hp[dg]++, e);
        }
        std::swap(s, d);
        in_src = !in_src;
    }
    if (!in_src)
        for (size_t i = 0; i < n; ++i) src.set(i, dst.get(i));
}

// Stable LSD radix sort of src[0,n) with dst as scratch of the same size,
// for a compile-time pass count so the histogram loop is fully unrolled.
template <int Passes, class Acc, class KeyFn>
BRAINSORT_ALWAYS_INLINE void lsd_radix_fixed(Acc src, Acc dst, size_t n, const DigitPlan& plan, KeyFn key) {
    const size_t width = plan.width();
    AuxRaw<uint32_t, Acc> hist(static_cast<size_t>(Passes) * width);
    uint32_t* h = hist.data();
    std::memset(h, 0, Passes * width * sizeof(uint32_t));
    if constexpr (Acc::counted) Acc::hooks::on_table_sweep(h, Passes * width, sizeof(uint32_t), false, true);
    int shift[Passes];
    uint32_t mask[Passes];
    for (int p = 0; p < Passes; ++p) { shift[p] = plan.shift[p]; mask[p] = (1u << plan.bits[p]) - 1; }
    for (size_t i = 0; i < n; ++i) {
        const auto u = key(src.get(i));
        for (int p = 0; p < Passes; ++p) {
            const size_t slot = static_cast<size_t>(p) * width + (static_cast<uint32_t>(u >> shift[p]) & mask[p]);
            if constexpr (Acc::counted) Acc::hooks::on_table_rw(h + slot, sizeof(uint32_t));
            ++h[slot];
        }
    }
    bool trivial[kMaxPasses];
    radix_prefix<Acc>(h, n, plan, trivial);
    radix_scatter(src, dst, n, plan, key, h, trivial);
}

// Runtime pass count: dispatch to the unrolled variants. Always inlined so a
// BMI2-targeted key functor is inlined into the context of the caller.
template <class Acc, class KeyFn>
BRAINSORT_ALWAYS_INLINE void lsd_radix(Acc src, Acc dst, size_t n, const DigitPlan& plan, KeyFn key) {
    if (plan.passes == 0 || n < 2) return;
    switch (plan.passes) {
        case 1:  lsd_radix_fixed<1>(src, dst, n, plan, key); break;
        case 2:  lsd_radix_fixed<2>(src, dst, n, plan, key); break;
        case 3:  lsd_radix_fixed<3>(src, dst, n, plan, key); break;
        case 4:  lsd_radix_fixed<4>(src, dst, n, plan, key); break;
        case 5:  lsd_radix_fixed<5>(src, dst, n, plan, key); break;
        case 6:  lsd_radix_fixed<6>(src, dst, n, plan, key); break;
        case 7:  lsd_radix_fixed<7>(src, dst, n, plan, key); break;
        default: lsd_radix_fixed<8>(src, dst, n, plan, key); break;
    }
}

// ---------------------------------------------------------------------------
// Fused LSD radix (used by brainsort). Differences from lsd_radix above:
//   * the histogram of pass p+1 is built while pass p scatters, so only two
//     count tables are live at any time regardless of the pass count, and
//     the initial histogram pass fills one table instead of `passes`;
//   * passes whose digit does not vary (known from the key mask) are skipped
//     up front, so the pass parity is known in advance;
//   * the counter type is a template parameter: 16-bit counters when
//     n < 65536 halve the tables and their cache footprint;
//   * the prefix sums are vectorised (AVX2) when available;
//   * the caller says where the result must end up (src or dst), so a part
//     that already sits in the scratch buffer is sorted straight out of it.
// `keymask` is the OR of (key XOR key0) in the domain of `key`, i.e. the bits
// that vary at all.
// ---------------------------------------------------------------------------

// Exclusive prefix sums in place. Sizes are powers of two >= 256.
template <class Cnt>
inline void prefix_scalar(Cnt* h, size_t n) {
    Cnt sum = 0;
    for (size_t b = 0; b < n; ++b) { const Cnt c = h[b]; h[b] = sum; sum = static_cast<Cnt>(sum + c); }
}
#ifdef BRAINSORT_X86_64
BRAINSORT_TARGET_AVX2 inline void prefix_avx2(uint16_t* h, size_t n) {
    const __m256i last = _mm256_set1_epi16(0x0F0E);   // shuffle: broadcast the top element of each 128-bit lane
    __m256i carry = _mm256_setzero_si256();
    for (size_t i = 0; i < n; i += 16) {
        const __m256i in = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(h + i));
        __m256i x = in;
        x = _mm256_add_epi16(x, _mm256_slli_si256(x, 2));
        x = _mm256_add_epi16(x, _mm256_slli_si256(x, 4));
        x = _mm256_add_epi16(x, _mm256_slli_si256(x, 8));
        __m256i lo = _mm256_shuffle_epi8(x, last);
        lo = _mm256_permute2x128_si256(lo, lo, 0x08);        // low-lane total into the high lane, zero below
        x = _mm256_add_epi16(x, lo);
        const __m256i incl = _mm256_add_epi16(x, carry);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(h + i), _mm256_sub_epi16(incl, in));
        carry = _mm256_permute4x64_epi64(_mm256_shuffle_epi8(incl, last), 0xFF);
    }
}
BRAINSORT_TARGET_AVX2 inline void prefix_avx2(uint32_t* h, size_t n) {
    __m256i carry = _mm256_setzero_si256();
    for (size_t i = 0; i < n; i += 8) {
        const __m256i in = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(h + i));
        __m256i x = in;
        x = _mm256_add_epi32(x, _mm256_slli_si256(x, 4));
        x = _mm256_add_epi32(x, _mm256_slli_si256(x, 8));
        __m256i lo = _mm256_shuffle_epi32(x, 0xFF);
        lo = _mm256_permute2x128_si256(lo, lo, 0x08);
        x = _mm256_add_epi32(x, lo);
        const __m256i incl = _mm256_add_epi32(x, carry);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(h + i), _mm256_sub_epi32(incl, in));
        carry = _mm256_permute4x64_epi64(_mm256_shuffle_epi32(incl, 0xFF), 0xFF);
    }
}
#endif
template <class Cnt>
inline void prefix_sums(Cnt* h, size_t n) {
#ifdef BRAINSORT_X86_64
    // The vector versions handle 16 (uint16) or 8 (uint32) entries per step;
    // n is a power of two, so n >= 16 means whole steps only.
    if (n >= 16 && have_avx2()) { prefix_avx2(h, n); return; }
#endif
    prefix_scalar(h, n);
}

// The live passes of a plan for a key mask: passes whose digit varies.
inline int live_passes(const DigitPlan& plan, uint64_t keymask, int* order) {
    int live = 0;
    for (int p = 0; p < plan.passes; ++p) {
        const uint64_t dm = (uint64_t(1) << plan.bits[p]) - 1;
        if (((keymask >> plan.shift[p]) & dm) != 0) order[live++] = p;
    }
    return live;
}

// Scatter passes order[q0..live) of s[0,n) with d as the other buffer.
// tab[q0 & 1] must hold the raw counts of pass order[q0]; the counts of each
// following pass are built while the previous one scatters, so the two
// tables alternate. The data ends up in s or d as `result_in_src` asks, or,
// when `free` is set, wherever the last pass left it; `ended_in_src` says
// where that is either way.
template <class Cnt, class Acc, class KeyFn>
BRAINSORT_ALWAYS_INLINE void radix_passes(Acc s, Acc d, size_t n, const DigitPlan& plan, KeyFn key,
                                          const int* order, int live, int q0, Cnt* const* tab,
                                          bool result_in_src, bool free, bool& ended_in_src) {
    const size_t width  = plan.width();
    bool         in_src = true;
    for (int q = q0; q < live; ++q) {
        const int      p     = order[q];
        Cnt*           h     = tab[q & 1];
        const int      shift = plan.shift[p];
        const uint32_t mask  = (1u << plan.bits[p]) - 1;
        if constexpr (Acc::counted) { ++Acc::hooks::stats().radix_passes; Acc::hooks::on_table_sweep(h, size_t(1) << plan.bits[p], sizeof(Cnt), true, true); }
        prefix_sums(h, size_t(1) << plan.bits[p]);
        if (q + 1 < live) {
            const int      p2     = order[q + 1];
            Cnt*           h2     = tab[(q + 1) & 1];
            const int      shift2 = plan.shift[p2];
            const uint32_t mask2  = (1u << plan.bits[p2]) - 1;
            std::memset(h2, 0, width * sizeof(Cnt));
            if constexpr (Acc::counted) Acc::hooks::on_table_sweep(h2, width, sizeof(Cnt), false, true);
            for (size_t i = 0; i < n; ++i) {
                const auto     e   = s.get(i);
                const auto     u   = key(e);
                const uint32_t dg  = static_cast<uint32_t>(u >> shift) & mask;
                const uint32_t dg2 = static_cast<uint32_t>(u >> shift2) & mask2;
                if constexpr (Acc::counted) { Acc::hooks::on_table_rw(h + dg, sizeof(Cnt)); Acc::hooks::on_table_rw(h2 + dg2, sizeof(Cnt)); }
                d.set(h[dg]++, e);
                ++h2[dg2];
            }
        } else {
            for (size_t i = 0; i < n; ++i) {
                const auto     e  = s.get(i);
                const uint32_t dg = static_cast<uint32_t>(key(e) >> shift) & mask;
                if constexpr (Acc::counted) Acc::hooks::on_table_rw(h + dg, sizeof(Cnt));
                d.set(h[dg]++, e);
            }
        }
        std::swap(s, d);
        in_src = !in_src;
    }
    if (free) { ended_in_src = in_src; return; }
    if (in_src != result_in_src)
        for (size_t i = 0; i < n; ++i) d.set(i, s.get(i));
    ended_in_src = result_in_src;
}

// Fused LSD radix of src[0,n) with dst[0,n) as the other buffer. `store` is
// caller-provided histogram memory of `store_n` Cnt entries, reused across
// calls so the count tables are allocated once instead of per sort; the
// 16-bit digit tables are 256-512 KiB, which the allocator would otherwise
// mmap and munmap on every call. When `store` is too small (or null) the
// tables fall back to a local allocation.
template <class Cnt, class Acc, class KeyFn>
BRAINSORT_ALWAYS_INLINE void lsd_radix_fused(Acc src, Acc dst, size_t n, const DigitPlan& plan,
                                             KeyFn key, uint64_t keymask, bool result_in_src,
                                             void* store = nullptr, size_t store_n = 0) {
    int order[kMaxPasses];
    const int live = n >= 2 ? live_passes(plan, keymask, order) : 0;
    if (live == 0) {
        if (!result_in_src)
            for (size_t i = 0; i < n; ++i) dst.set(i, src.get(i));
        return;
    }
    const size_t width = plan.width();
    const size_t need  = live > 1 ? 2 * width : width;
    AuxRaw<Cnt, Acc> local(store && store_n >= need ? 0 : need);
    Cnt* base = (store && store_n >= need) ? static_cast<Cnt*>(store) : local.data();
    Cnt* tab[2] = {base, live > 1 ? base + width : base};
    if constexpr (Acc::counted) Acc::hooks::stats().passes_skipped += static_cast<uint64_t>(plan.passes - live);
    {   // histogram of the first live pass only
        const int      shift = plan.shift[order[0]];
        const uint32_t mask  = (1u << plan.bits[order[0]]) - 1;
        std::memset(tab[0], 0, width * sizeof(Cnt));
        if constexpr (Acc::counted) Acc::hooks::on_table_sweep(tab[0], width, sizeof(Cnt), false, true);
        for (size_t i = 0; i < n; ++i) {
            const uint32_t dg = static_cast<uint32_t>(key(src.get(i)) >> shift) & mask;
            if constexpr (Acc::counted) Acc::hooks::on_table_rw(tab[0] + dg, sizeof(Cnt));
            ++tab[0][dg];
        }
    }
    bool ended;
    radix_passes<Cnt>(src, dst, n, plan, key, order, live, 0, tab, result_in_src, false, ended);
}

}  // namespace radix_detail
}  // namespace detail
}  // namespace brainsort

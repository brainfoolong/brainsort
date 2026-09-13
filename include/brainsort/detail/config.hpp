// brainsort: compiler and platform adaptation.
//
// Everything the algorithm needs beyond standard C++20 goes through this
// header: the SIMD target attributes, CPU feature detection, byte swaps and
// a few diagnostics pragmas. GCC, Clang and MSVC are supported on x86-64;
// every other compiler or architecture gets the portable scalar code, which
// is the same code the vector paths fall back to for their tails.
//
// Define BRAINSORT_NO_SIMD to compile the scalar code on x86-64 as well
// (the test suite builds once each way).
#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>

#define BRAINSORT_VERSION_MAJOR 0
#define BRAINSORT_VERSION_MINOR 3
#define BRAINSORT_VERSION_PATCH 0
#define BRAINSORT_VERSION "0.4.0"

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(BRAINSORT_NO_SIMD)
#define BRAINSORT_X86_64 1
#include <immintrin.h>
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

#if defined(__GNUC__) || defined(__clang__)
#define BRAINSORT_TARGET_AVX2 __attribute__((target("avx2")))
#define BRAINSORT_TARGET_BMI2 __attribute__((target("bmi2")))
#define BRAINSORT_ALWAYS_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
// MSVC compiles AVX2/BMI2 intrinsics without a per-function target; the
// functions are only called after the run-time CPU check.
#define BRAINSORT_TARGET_AVX2
#define BRAINSORT_TARGET_BMI2
#define BRAINSORT_ALWAYS_INLINE __forceinline
#else
#define BRAINSORT_TARGET_AVX2
#define BRAINSORT_TARGET_BMI2
#define BRAINSORT_ALWAYS_INLINE inline
#endif

#if defined(__GNUC__) && !defined(__clang__)
#define BRAINSORT_DIAG_PUSH_MAYBE_UNINIT \
    _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
#define BRAINSORT_DIAG_POP _Pragma("GCC diagnostic pop")
#else
#define BRAINSORT_DIAG_PUSH_MAYBE_UNINIT
#define BRAINSORT_DIAG_POP
#endif

namespace brainsort {
namespace detail {

inline uint64_t bswap64(uint64_t x) noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
    return _byteswap_uint64(x);
#else
    return __builtin_bswap64(x);
#endif
}
inline uint32_t bswap32(uint32_t x) noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
    return _byteswap_ulong(x);
#else
    return __builtin_bswap32(x);
#endif
}

template <class K> inline int popcount(K k) noexcept {
    if constexpr (sizeof(K) == 8) return std::popcount(static_cast<uint64_t>(k));
    else return std::popcount(static_cast<uint32_t>(k));
}
template <class K> inline int ctz(K k) noexcept {   // k != 0
    if constexpr (sizeof(K) == 8) return std::countr_zero(static_cast<uint64_t>(k));
    else return std::countr_zero(static_cast<uint32_t>(k));
}
template <class K> inline int highest_bit(K k) noexcept {   // index of the top set bit, k != 0
    if constexpr (sizeof(K) == 8) return 63 - std::countl_zero(static_cast<uint64_t>(k));
    else return 31 - std::countl_zero(static_cast<uint32_t>(k));
}

// ---- CPU features ------------------------------------------------------------
struct CpuFeatures {
    bool avx2 = false;
    bool bmi2 = false;
};

#ifdef BRAINSORT_X86_64
inline void cpuid(unsigned leaf, unsigned sub, unsigned& a, unsigned& b, unsigned& c, unsigned& d) noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
    int r[4];
    __cpuidex(r, static_cast<int>(leaf), static_cast<int>(sub));
    a = static_cast<unsigned>(r[0]); b = static_cast<unsigned>(r[1]);
    c = static_cast<unsigned>(r[2]); d = static_cast<unsigned>(r[3]);
#else
    __cpuid_count(leaf, sub, a, b, c, d);
#endif
}
inline uint64_t xgetbv0() noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
    return _xgetbv(0);
#else
    unsigned eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<uint64_t>(edx) << 32) | eax;
#endif
}
#endif

inline CpuFeatures detect_cpu() noexcept {
    CpuFeatures f;
#ifdef BRAINSORT_X86_64
    unsigned a, b, c, d;
    cpuid(0, 0, a, b, c, d);
    const unsigned max_leaf = a;
    if (max_leaf < 7) return f;
    cpuid(1, 0, a, b, c, d);
    const bool osxsave = (c >> 27) & 1, avx = (c >> 28) & 1;
    cpuid(7, 0, a, b, c, d);
    const bool avx2 = (b >> 5) & 1, bmi2 = (b >> 8) & 1;
    // AVX2 needs the OS to save the YMM state (XCR0 bits 1 and 2); BMI2 needs nothing.
    const bool ymm = osxsave && avx && (xgetbv0() & 6) == 6;
    f.avx2 = avx2 && ymm;
    f.bmi2 = bmi2;
#endif
    return f;
}
inline bool have_avx2() noexcept {
    static const bool v = detect_cpu().avx2;
    return v;
}
inline bool have_bmi2() noexcept {
    static const bool v = detect_cpu().bmi2;
    return v;
}

// ---- cache sizes ---------------------------------------------------------------
// The data cache sizes of the core the sort runs on, read once from the CPU
// (cpuid leaf 4 on Intel, leaf 0x8000001D on AMD). They decide when a part
// is too large for its radix passes to run in cache and how large the
// buckets of the MSD scatter should be. Where nothing can be read (another
// architecture, a virtual machine that reports no caches) the fallbacks are
// those of a common desktop core.
struct CacheSizes {
    size_t l1d = size_t(32) << 10;
    size_t l2  = size_t(1) << 20;
    size_t l3  = size_t(32) << 20;
};

inline CacheSizes detect_caches() noexcept {
    CacheSizes c;
#ifdef BRAINSORT_X86_64
    unsigned a, b, cc, d;
    cpuid(0, 0, a, b, cc, d);
    const unsigned max_leaf = a;
    cpuid(0x80000000u, 0, a, b, cc, d);
    bool ext = false;   // AMD: the cache leaf lives in the extended range
    if (a >= 0x8000001Du) {
        cpuid(0x80000001u, 0, a, b, cc, d);
        ext = (cc >> 22) & 1;   // topology extensions
    }
    if (!ext && max_leaf < 4) return c;
    size_t l1d = 0, l2 = 0, l3 = 0;
    for (unsigned i = 0; i < 16; ++i) {
        cpuid(ext ? 0x8000001Du : 4u, i, a, b, cc, d);
        const unsigned type = a & 0x1F;   // 1 data, 2 instruction, 3 unified
        if (type == 0) break;
        const unsigned level = (a >> 5) & 7;
        const size_t   size  = (((b >> 22) & 0x3FF) + 1) * (((b >> 12) & 0x3FF) + 1) * ((b & 0xFFF) + 1) * (static_cast<size_t>(cc) + 1);
        if (type == 2) continue;
        if (level == 1) l1d = size;
        else if (level == 2) l2 = size;
        else if (level == 3) l3 = size;
    }
    if (l1d) c.l1d = l1d;
    if (l2) c.l2 = l2;
    if (l3) c.l3 = l3;
#endif
    return c;
}
inline const CacheSizes& cache_sizes() noexcept {
    static const CacheSizes c = detect_caches();
    return c;
}

}  // namespace detail
}  // namespace brainsort

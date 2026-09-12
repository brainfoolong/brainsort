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
#define BRAINSORT_VERSION_MINOR 2
#define BRAINSORT_VERSION_PATCH 0
#define BRAINSORT_VERSION "0.2.0"

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

}  // namespace detail
}  // namespace brainsort

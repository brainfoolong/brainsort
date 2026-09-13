// brainsort: the element contract, the array view, the allocation and
// instrumentation hooks, and the scratch buffer types the algorithm uses.
//
// The algorithm is written against a view type A with this interface:
//
//   using value_type = T;            the element type (trivially copyable)
//   using traits     = elem_traits<T>;
//   using key_type   = traits::key_type;
//   using hooks      = ...;          instrumentation (NoHooks for the library)
//   static constexpr bool counted;   true only on an instrumented build
//   A(T* p, size_t n); size(); data(); get(i); set(i, v); swap(i, j); sub(off, len)
//   less(a, b); compare(a, b); compare_from(a, b, chunk); static key(x, chunk)
//   template <class U> static U* alloc_array(size_t n);          throws std::bad_alloc
//   template <class U> static void free_array(U* p, size_t n) noexcept;
//
// elem_traits<T> describes an element type to the algorithm:
//
//   static constexpr bool     chunked;   the key has several chunks (strings)
//   static constexpr SimdKind simd;      layout promise for the vector paths
//   using key_type = uint32_t | uint64_t;
//   static bool     less(T, T);  static int compare(T, T);
//   static int      compare_from(T, T, int chunk);   compare from chunk `chunk` on
//   static key_type radix_key(T, int chunk);         order-preserving unsigned key of that chunk
//   static bool     chunk_ends(T, int chunk);        no chunk after this one
//
// SimdKind::i32 promises sizeof(T) == 8 with a signed 32-bit key whose signed
// order equals the element order in bytes 0-3; i64 promises sizeof(T) == 16
// and a signed 64-bit key in bytes 0-7; f64 the same with an IEEE double;
// k64 promises sizeof(T) == 8 and that the element is the signed 64-bit key.
#pragma once

#include "brainsort/detail/config.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <new>

namespace brainsort {

enum class SimdKind { none, i32, i64, f64, k64 };

template <class T> struct elem_traits;

namespace detail {

// ---- instrumentation hooks ----------------------------------------------------
// Every call sits under `if constexpr (A::counted)`, so on the library view
// nothing here is ever instantiated; the benchmark supplies a hooks type that
// records the same calls.
struct NoHooks {
    struct DepthScope {};
    struct Stats {
        uint64_t giveups = 0, splits = 0, split_retries = 0, part_sorts = 0;
        uint64_t dict_tries = 0, dict_hits = 0, plan_retries = 0;
        uint64_t radix_passes = 0, passes_skipped = 0;
        void note_route(int) {}
    };
    static Stats& stats() { static Stats s; return s; }
    static void on_read(const void*, size_t) {}
    static void on_write(const void*, size_t) {}
    static void on_table_read(const void*, size_t) {}
    static void on_table_write(const void*, size_t) {}
    static void on_table_rw(const void*, size_t) {}
    static void on_table_sweep(const void*, size_t, size_t, bool, bool) {}
    template <class T> static void on_key_chunk(const T&, int) {}
};

// ---- allocation ------------------------------------------------------------------
// A sort allocates a few large blocks (the records, the scratch buffer, the
// permutation buffer) and frees them at the end. Fresh pages from the system
// are the most expensive part of sorting a hundred thousand elements: every
// page is faulted in on first touch, which costs as much as the sorting
// itself. So the library keeps the largest blocks its sorts have freed, up
// to BRAINSORT_MEMORY_CACHE bytes in total (default 32 MiB; 0 disables the
// cache), and the next sort takes them back warm. The cache is shared by
// every thread under a mutex that is taken a few times per sort; it holds
// its blocks until brainsort::release_memory() is called or the process
// ends. Each block carries its capacity in a small header, so a block that
// served a smaller request is still known in full when it comes back.
#ifndef BRAINSORT_MEMORY_CACHE
#define BRAINSORT_MEMORY_CACHE (size_t(32) << 20)
#endif

class MemoryCache {
public:
    static constexpr size_t kMinBlock = size_t(64) << 10;   // smaller blocks go to the heap directly
    static constexpr size_t kHeader   = 16;                  // keeps the alignment of ::operator new
    static constexpr int    kSlots    = 8;                   // cached free blocks

    void* allocate(size_t bytes) {
        {   // best fit among the cached blocks
            std::lock_guard<std::mutex> lock(mutex_);
            int best = -1;
            for (int i = 0; i < n_; ++i)
                if (blocks_[i].cap >= bytes && (best < 0 || blocks_[i].cap < blocks_[best].cap)) best = i;
            if (best >= 0) {
                const Block b = blocks_[best];
                blocks_[best]  = blocks_[--n_];
                cached_ -= b.cap;
                return b.p;
            }
        }
        char* raw = static_cast<char*>(::operator new(bytes + kHeader));
        std::memcpy(raw, &bytes, sizeof bytes);
        return raw + kHeader;
    }
    void deallocate(void* p) noexcept {
        size_t cap;
        std::memcpy(&cap, static_cast<char*>(p) - kHeader, sizeof cap);
        void* victim = nullptr;   // freed outside the lock
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (cap > limit_) {
                victim = p;
            } else {
                // Make room: drop the smallest cached blocks while the total
                // would exceed the limit; a full cache keeps the larger block.
                while (n_ > 0 && cached_ + cap > limit_) drop(smallest());
                if (n_ == kSlots) {
                    const int s = smallest();
                    if (blocks_[s].cap >= cap) victim = p;
                    else drop(s);
                }
                if (!victim) { blocks_[n_++] = Block{p, cap}; cached_ += cap; }
            }
        }
        if (victim) release_block(victim);
    }
    void release() noexcept {
        Block  held[kSlots];
        int    n;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            n = n_;
            for (int i = 0; i < n; ++i) held[i] = blocks_[i];
            n_ = 0;
            cached_ = 0;
        }
        for (int i = 0; i < n; ++i) release_block(held[i].p);
    }

private:
    struct Block { void* p; size_t cap; };
    static void release_block(void* p) noexcept { ::operator delete(static_cast<char*>(p) - kHeader); }
    int smallest() const noexcept {
        int s = 0;
        for (int i = 1; i < n_; ++i) if (blocks_[i].cap < blocks_[s].cap) s = i;
        return s;
    }
    void drop(int i) noexcept {   // under the lock
        cached_ -= blocks_[i].cap;
        release_block(blocks_[i].p);
        blocks_[i] = blocks_[--n_];
    }

    std::mutex mutex_;
    Block      blocks_[kSlots] = {};
    int        n_              = 0;
    size_t     cached_         = 0;
    size_t     limit_          = BRAINSORT_MEMORY_CACHE;
};

// No destructor runs at exit: a sort still running on another thread at that
// point keeps working, and the process returns the memory anyway.
inline MemoryCache& memory_cache() {
    static MemoryCache* const cache = new MemoryCache;
    return *cache;
}

struct DefaultAlloc {
    static void* allocate(size_t bytes) {
        if (bytes < MemoryCache::kMinBlock || BRAINSORT_MEMORY_CACHE == 0) return ::operator new(bytes ? bytes : 1);
        return memory_cache().allocate(bytes);
    }
    static void deallocate(void* p, size_t bytes) noexcept {
        if (bytes < MemoryCache::kMinBlock || BRAINSORT_MEMORY_CACHE == 0) { ::operator delete(p); return; }
        memory_cache().deallocate(p);
    }
};

// ---- the array view of the library ------------------------------------------------
template <class T, class Alloc = DefaultAlloc>
class View {
public:
    using value_type = T;
    using traits     = elem_traits<T>;
    using key_type   = typename traits::key_type;
    using hooks      = NoHooks;
    using alloc      = Alloc;
    static constexpr bool counted = false;

    View() : p_(nullptr), n_(0) {}
    View(T* p, size_t n) : p_(p), n_(n) {}

    size_t size() const { return n_; }
    T*     data() const { return p_; }
    T    get(size_t i) const { assert(i < n_); return p_[i]; }
    void set(size_t i, T v) const { assert(i < n_); p_[i] = v; }
    bool less(T a, T b) const { return traits::less(a, b); }
    int  compare(T a, T b) const { return traits::compare(a, b); }
    BRAINSORT_ALWAYS_INLINE int compare_from(T a, T b, int chunk) const { return traits::compare_from(a, b, chunk); }
    static key_type key(T x, int chunk) { return traits::radix_key(x, chunk); }
    void swap(size_t i, size_t j) const { T t = get(i); set(i, get(j)); set(j, t); }
    View sub(size_t off, size_t len) const { assert(off + len <= n_); return View(p_ + off, len); }

    template <class U> static U* alloc_array(size_t n) {
        return static_cast<U*>(Alloc::allocate(n * sizeof(U)));
    }
    template <class U> static void free_array(U* p, size_t n) noexcept { Alloc::deallocate(p, n * sizeof(U)); }

private:
    T*     p_;
    size_t n_;
};

// ---- scratch buffers -----------------------------------------------------------------
// A is the view type of the elements being sorted: it supplies the allocator
// and the hooks, and its counted flag says whether accesses are recorded.

// A buffer of elements whose accesses count like those of the main array.
template <class T, class A>
class AuxBuffer {
public:
    explicit AuxBuffer(size_t n) : p_(A::template alloc_array<T>(n)), n_(n) {}
    ~AuxBuffer() { if (p_) A::template free_array<T>(p_, n_); }
    AuxBuffer(const AuxBuffer&) = delete;
    AuxBuffer& operator=(const AuxBuffer&) = delete;

    A      arr() const { return A(p_, n_); }
    size_t size() const { return n_; }

    // Drop current contents and reallocate with a new capacity.
    void resize_discard(size_t n) {
        A::template free_array<T>(p_, n_);
        p_ = nullptr;
        n_ = 0;
        p_ = A::template alloc_array<T>(n);
        n_ = n;
    }

private:
    T*     p_;
    size_t n_;
};

// A buffer of plain values (indices, positions) whose accesses are counted
// like element accesses, so index-based algorithms hide nothing.
template <class T, class A>
class AuxVec {
public:
    static constexpr bool counted = A::counted;
    explicit AuxVec(size_t n) : p_(A::template alloc_array<T>(n)), n_(n) {}
    ~AuxVec() { A::template free_array<T>(p_, n_); }
    AuxVec(const AuxVec&) = delete;
    AuxVec& operator=(const AuxVec&) = delete;
    size_t size() const { return n_; }
    T get(size_t i) const { assert(i < n_); if constexpr (counted) A::hooks::on_read(p_ + i, sizeof(T)); return p_[i]; }
    void set(size_t i, T v) const { assert(i < n_); if constexpr (counted) A::hooks::on_write(p_ + i, sizeof(T)); p_[i] = v; }
    // Lightweight view with the same interface (copyable, non-owning).
    struct View {
        using value_type = T;
        static constexpr bool counted = A::counted;
        T* p; size_t n;
        size_t size() const { return n; }
        T get(size_t i) const { assert(i < n); if constexpr (counted) A::hooks::on_read(p + i, sizeof(T)); return p[i]; }
        void set(size_t i, T v) const { assert(i < n); if constexpr (counted) A::hooks::on_write(p + i, sizeof(T)); p[i] = v; }
    };
    View view() const { return View{p_, n_}; }
private:
    T*     p_;
    size_t n_;
};

// Raw scratch (histograms, tables): tracked as memory only. The algorithms
// report its accesses through the hooks on their counted path.
template <class T, class A>
class AuxRaw {
public:
    explicit AuxRaw(size_t n) : p_(A::template alloc_array<T>(n)), n_(n) {}
    ~AuxRaw() { A::template free_array<T>(p_, n_); }
    AuxRaw(const AuxRaw&) = delete;
    AuxRaw& operator=(const AuxRaw&) = delete;
    T* data() const { return p_; }
    size_t size() const { return n_; }
private:
    T*     p_;
    size_t n_;
};

// ---- small shared helpers ------------------------------------------------------------
template <class A>
inline void insertion_sort(A a, size_t lo, size_t hi) {
    using T = typename A::value_type;
    for (size_t i = lo + 1; i < hi; ++i) {
        T v = a.get(i);
        size_t j = i;
        while (j > lo) {
            T p = a.get(j - 1);
            if (!a.less(v, p)) break;
            a.set(j, p);
            --j;
        }
        a.set(j, v);
    }
}

// Requires an element <= every element of [lo,hi) to sit at a[lo-1].
template <class A>
inline void unguarded_insertion_sort(A a, size_t lo, size_t hi) {
    using T = typename A::value_type;
    for (size_t i = lo; i < hi; ++i) {
        T v = a.get(i);
        size_t j = i;
        T p = a.get(j - 1);
        while (a.less(v, p)) {
            a.set(j, p);
            --j;
            p = a.get(j - 1);
        }
        a.set(j, v);
    }
}

// Forward copy: safe for overlapping ranges when dst <= src.
template <class A>
inline void copy_forward(A src, size_t s, A dst, size_t d, size_t len) {
    for (size_t i = 0; i < len; ++i) dst.set(d + i, src.get(s + i));
}
// Backward copy: safe for overlapping ranges when dst >= src.
template <class A>
inline void copy_backward(A src, size_t s, A dst, size_t d, size_t len) {
    for (size_t i = len; i-- > 0;) dst.set(d + i, src.get(s + i));
}

template <class A>
inline void reverse_range(A a, size_t lo, size_t hi) {
    while (lo + 1 < hi) {
        --hi;
        a.swap(lo, hi);
        ++lo;
    }
}

inline size_t floor_log2(size_t n) {
    size_t r = 0;
    while (n > 1) { n >>= 1; ++r; }
    return r;
}

// ---- string chunk keys -----------------------------------------------------------------
// A string key is consumed in 7-byte chunks. Chunk c is bytes [7c, 7c+7)
// big-endian in the top 56 bits of the key, and the number of valid bytes in
// this chunk (0..7) in the low byte. Lexicographic order of the chunk keys
// equals lexicographic order of the strings, and a shorter string sorts
// before a longer one with the same prefix. A string of length L has
// L / 7 + 1 chunks: the last one has fewer than 7 valid bytes.
constexpr int      kStrChunkBytes = 7;
constexpr uint64_t kStrKeyMask    = ~uint64_t(0xFF);

inline uint64_t str_chunk_key(const char* ptr, uint32_t len, int chunk) noexcept {
    const uint32_t off = static_cast<uint32_t>(chunk) * kStrChunkBytes;
    uint64_t w = 0;
    if (off + 8 <= len) {                         // fast path: one 8-byte load
        std::memcpy(&w, ptr + off, 8);
        return (bswap64(w) & kStrKeyMask) | static_cast<uint64_t>(kStrChunkBytes);
    }
    if (off >= len) return 0;
    const uint32_t valid = len - off;             // 1..7 bytes left; this runs once per element per pass
    const unsigned char* q = reinterpret_cast<const unsigned char*>(ptr + off);
    if (valid >= 4) {                             // two overlapping 4-byte loads, branch-free assembly
        uint32_t hi, lo;
        std::memcpy(&hi, q, 4);
        std::memcpy(&lo, q + valid - 4, 4);
        w = (static_cast<uint64_t>(bswap32(hi)) << 32) |
            (static_cast<uint64_t>(bswap32(lo)) << (8 * (8 - valid)));
        return (w & kStrKeyMask) | valid;
    }
    for (uint32_t i = 0; i < valid; ++i) w = (w << 8) | q[i];
    w <<= 8 * (kStrChunkBytes - valid);
    return (w << 8) | valid;
}
// The same chunk for a key that sorts in descending order: the bytes are
// inverted and the valid count is reversed, so a tie on the bytes puts the
// longer string first.
inline uint64_t str_chunk_key_desc(const char* ptr, uint32_t len, int chunk) noexcept {
    const uint64_t k = str_chunk_key(ptr, len, chunk);
    return (~k & kStrKeyMask) | (static_cast<uint64_t>(kStrChunkBytes) - (k & 0xFF));
}
inline bool str_chunk_ends(uint32_t len, int chunk) noexcept {
    return static_cast<uint32_t>(chunk) * kStrChunkBytes + kStrChunkBytes > len;
}
// Three-way compare starting at byte `off` (both strings are known to agree
// on the first `off` bytes when off > 0). memcmp on purpose: several inlined
// variants were measured with hardware counters and every one of them
// tripled the branch mispredictions on keys of mixed length, because the
// compiler turned the length dispatch into data-dependent branches; the
// result of memcmp feeds a conditional move instead.
BRAINSORT_ALWAYS_INLINE int str_compare_from(const char* pa, uint32_t la0, const char* pb, uint32_t lb0, uint32_t off) noexcept {
    const uint32_t la = la0 > off ? la0 - off : 0;
    const uint32_t lb = lb0 > off ? lb0 - off : 0;
    const uint32_t m  = la < lb ? la : lb;
    const int c = std::memcmp(pa + off, pb + off, m);
    if (c != 0) return c;
    return la < lb ? -1 : (la > lb ? 1 : 0);
}

}  // namespace detail
}  // namespace brainsort

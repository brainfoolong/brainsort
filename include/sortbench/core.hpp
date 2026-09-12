// Core types shared by every algorithm and by the benchmark harness.
//
// The algorithms themselves live in the brainsort library
// (include/brainsort/); this header supplies what the benchmark adds on top:
// the four element types with their original-index `id`, the instrumented
// array view Array<T, Counted>, and the deterministic trace.
//
// Every algorithm is written against Array<T, Counted>, a tiny view type that
// exposes get()/set()/less(). With Counted == true every element read, element
// write and key comparison is tallied in g_counters; with Counted == false the
// calls compile down to plain loads/stores/compares. The benchmark measures
// time/instructions with the uncounted variant and access counts with the
// counted variant, so instrumentation never distorts the timing numbers.
//
// Element types are described to the library by brainsort::elem_traits<T>
// (compare, radix key, chunking, SIMD layout) and to the harness by
// sb::KeyTraits<T>, which adds the type name and where the original index
// lives (for verification).
//
// Beyond the three basic counters the counted path feeds g_trace, which
// records everything about a run that is a pure function of (algorithm,
// input): histogram-table traffic, string key bytes, comparison-outcome
// flips, a hash of the access sequence, and the misses of a fixed cache
// model driven by allocation-relative addresses. None of it depends on
// clocks, real addresses or the machine, so two runs anywhere agree bit for
// bit; the tests hold every algorithm to a golden file of these numbers.
#pragma once

#include "brainsort/detail/traits.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <type_traits>

namespace sb {

// ---- element types ---------------------------------------------------------
// In every type `id` is the original position of the element and is used to
// verify that the output is a permutation of the input and (for stable sorts)
// that stability holds. It is never used by the algorithms.

struct Item {              // 8 bytes: int32 key
    int32_t  key;
    uint32_t id;
};
struct DblItem {           // 16 bytes: double key (a price, a measurement, a timestamp)
    double   key;
    uint32_t id;
    uint32_t pad;
};
struct I64Item {           // 16 bytes: int64 key (a database id, a nanosecond timestamp)
    int64_t  key;
    uint32_t id;
    uint32_t pad;
};
struct StrItem {           // 16 bytes: string key (pointer + length, like std::string_view)
    const char* ptr;
    uint32_t    len;
    uint32_t    id;
};

inline bool operator==(Item a, Item b)       { return a.key == b.key && a.id == b.id; }
inline bool operator==(DblItem a, DblItem b) { return a.key == b.key && a.id == b.id; }
inline bool operator==(I64Item a, I64Item b) { return a.key == b.key && a.id == b.id; }
inline bool operator==(StrItem a, StrItem b) {
    return a.len == b.len && a.id == b.id && std::memcmp(a.ptr, b.ptr, a.len) == 0;
}

}  // namespace sb

// ---- element traits for the library ------------------------------------------
// radix_key(x, chunk): an unsigned integer such that, for two elements
// sharing all chunks < c, radix_key(., c) orders them like less() does, with
// equal keys meaning "undecided" (chunked types) or "equal" (exact types).
// chunk_ends(x, chunk): true if the element has no chunk after `chunk`.
namespace brainsort {

template <> struct elem_traits<sb::Item> {
    using T = sb::Item;
    static constexpr bool     chunked = false;
    static constexpr SimdKind simd    = SimdKind::i32;
    using key_type = uint32_t;
    static bool     less(T a, T b) { return a.key < b.key; }
    static int      compare(T a, T b) { return a.key < b.key ? -1 : (b.key < a.key ? 1 : 0); }
    static int      compare_from(T a, T b, int) { return compare(a, b); }
    static key_type radix_key(T a, int) { return static_cast<uint32_t>(a.key) ^ 0x80000000u; }
    static bool     chunk_ends(T, int) { return true; }
};

template <> struct elem_traits<sb::DblItem> {
    using T = sb::DblItem;
    static constexpr bool     chunked = false;
    static constexpr SimdKind simd    = SimdKind::f64;
    using key_type = uint64_t;
    static bool     less(T a, T b) { return a.key < b.key; }
    static int      compare(T a, T b) { return a.key < b.key ? -1 : (b.key < a.key ? 1 : 0); }
    static int      compare_from(T a, T b, int) { return compare(a, b); }
    // IEEE-754 order for non-NaN values as an unsigned key: a positive double
    // maps to bits | 2^63, a negative one to 2^63 - magnitude. Subtracting the
    // magnitude (rather than the textbook ~bits) keeps the trailing zero bits
    // of a negative mantissa zero in the key too, so data that crosses zero
    // has a narrow varying-bit mask instead of all 64 bits varying. -0.0 and
    // +0.0 both map to 2^63, so the key agrees with less(): the two zeros
    // compare equal, and a stable sort must keep them in input order.
    static key_type radix_key(T a, int) {
        uint64_t bits;
        std::memcpy(&bits, &a.key, sizeof bits);
        const uint64_t sign = 0x8000000000000000ull;
        return (bits & sign) ? sign - (bits & ~sign) : (bits | sign);
    }
    static bool chunk_ends(T, int) { return true; }
};

template <> struct elem_traits<sb::I64Item> {
    using T = sb::I64Item;
    static constexpr bool     chunked = false;
    static constexpr SimdKind simd    = SimdKind::i64;
    using key_type = uint64_t;
    static bool     less(T a, T b) { return a.key < b.key; }
    static int      compare(T a, T b) { return a.key < b.key ? -1 : (b.key < a.key ? 1 : 0); }
    static int      compare_from(T a, T b, int) { return compare(a, b); }
    static key_type radix_key(T a, int) { return static_cast<uint64_t>(a.key) ^ 0x8000000000000000ull; }
    static bool     chunk_ends(T, int) { return true; }
};

template <> struct elem_traits<sb::StrItem> {
    using T = sb::StrItem;
    static constexpr bool     chunked = true;
    static constexpr SimdKind simd    = SimdKind::none;
    using key_type = uint64_t;
    static constexpr int kChunkBytes = detail::kStrChunkBytes;
    static uint32_t chunk_offset(int chunk) { return static_cast<uint32_t>(chunk) * kChunkBytes; }
    BRAINSORT_ALWAYS_INLINE static int compare_from(T a, T b, int chunk) {
        return detail::str_compare_from(a.ptr, a.len, b.ptr, b.len, chunk_offset(chunk));
    }
    static int      compare(T a, T b) { return compare_from(a, b, 0); }
    static bool     less(T a, T b) { return compare_from(a, b, 0) < 0; }
    static key_type radix_key(T a, int chunk) { return detail::str_chunk_key(a.ptr, a.len, chunk); }
    static bool     chunk_ends(T a, int chunk) { return detail::str_chunk_ends(a.len, chunk); }
};

}  // namespace brainsort

namespace sb {

using SimdKind = ::brainsort::SimdKind;

// ---- harness traits ---------------------------------------------------------
// The library traits plus what the harness needs: a name, the original index,
// and for strings the byte-counting helpers of the trace.
template <class T> struct KeyTraits;

template <> struct KeyTraits<Item> : ::brainsort::elem_traits<Item> {
    static constexpr const char* name = "int32";
    static uint32_t id(Item a) { return a.id; }
    static void     set_id(Item& a, uint32_t v) { a.id = v; }
    static uint32_t chunk_offset(int) { return 0; }
};
template <> struct KeyTraits<DblItem> : ::brainsort::elem_traits<DblItem> {
    static constexpr const char* name = "double";
    static uint32_t id(DblItem a) { return a.id; }
    static void     set_id(DblItem& a, uint32_t v) { a.id = v; }
    static uint32_t chunk_offset(int) { return 0; }
};
template <> struct KeyTraits<I64Item> : ::brainsort::elem_traits<I64Item> {
    static constexpr const char* name = "int64";
    static uint32_t id(I64Item a) { return a.id; }
    static void     set_id(I64Item& a, uint32_t v) { a.id = v; }
    static uint32_t chunk_offset(int) { return 0; }
};
template <> struct KeyTraits<StrItem> : ::brainsort::elem_traits<StrItem> {
    static constexpr const char* name = "string";
    static uint32_t id(StrItem a) { return a.id; }
    static void     set_id(StrItem& a, uint32_t v) { a.id = v; }
    // Bytes of each string a compare from `off` looks at: up to and including
    // the first differing byte, or the whole common length. Counted path only.
    static uint32_t examined(StrItem a, StrItem b, uint32_t off) {
        const uint32_t la = a.len > off ? a.len - off : 0;
        const uint32_t lb = b.len > off ? b.len - off : 0;
        const uint32_t m  = la < lb ? la : lb;
        uint32_t i = 0;
        while (i < m && a.ptr[off + i] == b.ptr[off + i]) ++i;
        return i < m ? i + 1 : m;
    }
    // Bytes radix_key(a, chunk) loads from the string.
    static uint32_t chunk_bytes(StrItem a, int chunk) {
        const uint32_t off = chunk_offset(chunk);
        if (off + 8 <= a.len) return 8;
        return off < a.len ? a.len - off : 0;
    }
};

// ---- access counters --------------------------------------------------------
struct Counters {
    uint64_t reads = 0;     // element loads from the main array or any aux buffer
    uint64_t writes = 0;    // element stores to the main array or any aux buffer
    uint64_t compares = 0;  // key comparisons
    void reset() { *this = Counters{}; }
};
inline Counters g_counters;

// ---- auxiliary (heap) memory tracking ----------------------------------------
// All scratch memory an algorithm needs goes through the tracked allocator, so
// the exact peak number of extra bytes is known independently of the OS.
struct AuxStats {
    size_t current_bytes = 0;
    size_t peak_bytes = 0;
    size_t allocations = 0;
    void reset() { *this = AuxStats{}; }
};
inline AuxStats g_aux;

// ---- algorithm telemetry ---------------------------------------------------------
// Deterministic facts about how a run went, filled in on the counted path
// only: recursion depth, fallbacks, and for brainsort which routes ran and
// what the radix planner did.
struct AlgoStats {
    uint32_t depth = 0, max_depth = 0;   // live and maximum recursion depth (timsort: run-stack height)
    uint64_t fallbacks = 0;              // introsort/pdqsort: heapsort takeovers
    uint64_t bad_parts = 0;              // pdqsort: highly unbalanced partitions (shuffled)
    // brainsort routes: 1 sorted, 2 reverse, 3 few runs, 4 displaced, 5 radix.
    int      first_route = 0;            // the route the top-level range took
    uint64_t route[6] = {};
    uint64_t giveups = 0;                // route 4 (displaced) started and gave up
    uint64_t splits = 0, split_retries = 0;
    uint64_t part_sorts = 0;             // partition sort for 2-4 distinct keys
    uint64_t dict_tries = 0, dict_hits = 0;
    uint64_t plan_retries = 0;           // a speculative plan failed verification
    uint64_t radix_passes = 0;           // scatter passes actually run
    uint64_t passes_skipped = 0;         // planned passes whose digit did not vary
    void reset() { *this = AlgoStats{}; }
    void note_route(int r) { ++route[r]; if (!first_route) first_route = r; }
    void enter() { if (++depth > max_depth) max_depth = depth; }
    void leave() { --depth; }
};
inline AlgoStats g_stats;

template <bool C> struct DepthScope {
    DepthScope()  { if constexpr (C) g_stats.enter(); }
    ~DepthScope() { if constexpr (C) g_stats.leave(); }
    DepthScope(const DepthScope&) = delete;
    DepthScope& operator=(const DepthScope&) = delete;
};

// ---- deterministic trace -----------------------------------------------------
// A fixed cache model: 64-byte lines, true LRU, an L1 of 32 KiB 8-way and an
// L2 of 1 MiB 16-way, probed only on an L1 miss. Addresses are not real:
// every buffer (the array, each scratch allocation, the string pool) gets its
// own base at a 4 GiB boundary in the order it was registered, so the miss
// counts are a property of the access pattern and the model, not of where
// the allocator happened to put things.
struct CacheModel {
    struct Level {
        uint32_t  sets = 0, ways = 0;
        uint64_t* tag = nullptr;
        uint64_t* stamp = nullptr;
        void init(uint32_t s, uint32_t w) {
            sets = s; ways = w;
            tag   = static_cast<uint64_t*>(std::malloc(sizeof(uint64_t) * s * w));
            stamp = static_cast<uint64_t*>(std::malloc(sizeof(uint64_t) * s * w));
            if (!tag || !stamp) throw std::bad_alloc();
            clear();
        }
        void clear() {
            for (size_t i = 0; i < size_t(sets) * ways; ++i) { tag[i] = ~uint64_t(0); stamp[i] = 0; }
        }
        bool access(uint64_t line, uint64_t now) {
            const size_t s = static_cast<size_t>(line & (sets - 1)) * ways;
            uint64_t*    t = tag + s;
            uint64_t*    a = stamp + s;
            uint32_t victim = 0;
            uint64_t oldest = ~uint64_t(0);
            for (uint32_t w = 0; w < ways; ++w) {
                if (t[w] == line) { a[w] = now; return true; }
                if (a[w] < oldest) { oldest = a[w]; victim = w; }
            }
            t[victim] = line;
            a[victim] = now;
            return false;
        }
    };
    Level    l1, l2;
    uint64_t now = 0;
    uint64_t l1_misses = 0, l2_misses = 0;
    bool     ready = false;

    void reset() {
        if (!ready) { l1.init(64, 8); l2.init(1024, 16); ready = true; }
        else { l1.clear(); l2.clear(); }
        now = 0; l1_misses = 0; l2_misses = 0;
    }
    void touch(uint64_t va, size_t bytes) {
        const uint64_t last = (va + bytes - 1) >> 6;
        for (uint64_t line = va >> 6; line <= last; ++line) {
            ++now;
            if (!l1.access(line, now)) {
                ++l1_misses;
                if (!l2.access(line, now)) ++l2_misses;
            }
        }
    }
};

struct Trace {
    static constexpr size_t   kMaxRegions = 256;
    static constexpr uint64_t kFnvOffset  = 1469598103934665603ull;
    static constexpr uint64_t kFnvPrime   = 1099511628211ull;

    bool active = false;   // a counted run is in progress: hooks record
    bool model  = false;   // also feed the cache model and the trace hash (slower)

    uint64_t table_reads = 0, table_writes = 0, table_bytes = 0;   // histogram / index table traffic
    uint64_t key_bytes = 0;      // string key bytes loaded by compares and chunk extraction
    uint64_t cmp_flips = 0;      // comparisons whose outcome differed from the previous one
    uint64_t hash = kFnvOffset;  // FNV-1a over the whole event sequence
    uint64_t unmapped = 0;       // model events on addresses outside every registered buffer
    bool     have_cmp = false, last_cmp = false;

    struct Region { uintptr_t lo = 0, hi = 0; uint64_t vbase = 0; size_t bytes = 0; bool heap = false; bool used = false; };
    Region     regions[kMaxRegions];
    size_t     n_regions = 0;    // high-water mark of used slots
    uint64_t   next_seq  = 0;
    size_t     last      = 0;    // last region hit (accesses cluster)
    CacheModel cache;

    // Start recording. `array` is the element array being sorted; every
    // scratch allocation made from now on becomes a region as well.
    void begin(const void* array, size_t bytes, bool with_model) {
        table_reads = table_writes = table_bytes = key_bytes = cmp_flips = unmapped = 0;
        hash = kFnvOffset; have_cmp = last_cmp = false;
        for (auto& r : regions) r = Region{};
        n_regions = 0; next_seq = 0; last = 0;
        model = with_model;
        if (model) cache.reset();
        active = true;
        add_region(array, bytes, false);
    }
    void end() { active = false; model = false; }

    void add_region(const void* p, size_t bytes, bool heap) {
        size_t i = 0;
        while (i < n_regions && regions[i].used) ++i;
        if (i == kMaxRegions) return;
        if (i == n_regions) ++n_regions;
        const uintptr_t lo = reinterpret_cast<uintptr_t>(p);
        regions[i] = Region{lo, lo + (bytes ? bytes : 1), (++next_seq) << 32, bytes, heap, true};
    }
    // Returns the region's size (0 if the pointer started no such region).
    size_t remove_region(const void* p, bool heap) {
        const uintptr_t lo = reinterpret_cast<uintptr_t>(p);
        for (size_t i = 0; i < n_regions; ++i)
            if (regions[i].used && regions[i].lo == lo && regions[i].heap == heap) {
                const size_t b = regions[i].bytes;
                regions[i].used = false;
                return b;
            }
        return 0;
    }
    bool locate(const void* p, uint64_t& va) {
        const uintptr_t a = reinterpret_cast<uintptr_t>(p);
        const Region& r = regions[last];
        if (r.used && a >= r.lo && a < r.hi) { va = r.vbase + (a - r.lo); return true; }
        for (size_t i = 0; i < n_regions; ++i) {
            const Region& q = regions[i];
            if (q.used && a >= q.lo && a < q.hi) { last = i; va = q.vbase + (a - q.lo); return true; }
        }
        return false;
    }
    bool mapped(const void* p) { uint64_t va; return locate(p, va); }

    void mix(uint64_t w) { hash = (hash ^ w) * kFnvPrime; }
    void event(const void* p, size_t bytes, uint64_t op) {
        uint64_t va;
        if (!locate(p, va)) { ++unmapped; return; }
        mix((va << 3) | op);
        cache.touch(va, bytes);
    }

    // Element traffic on the array and the counted scratch buffers.
    void on_read(const void* p, size_t bytes)  { ++g_counters.reads;  if (model) event(p, bytes, 1); }
    void on_write(const void* p, size_t bytes) { ++g_counters.writes; if (model) event(p, bytes, 2); }
    // The upstream element wrapper cannot tell a buffer from a temporary: an
    // address inside a registered buffer is memory, anything else is a local.
    void on_move(const void* src, const void* dst, size_t bytes) {
        if (mapped(src)) on_read(src, bytes);
        if (mapped(dst)) on_write(dst, bytes);
    }
    void on_operand(const void* p, size_t bytes) { if (mapped(p)) on_read(p, bytes); }
    // Histogram and index tables (raw scratch): one entry.
    void on_table_read(const void* p, size_t bytes)  { ++table_reads;  table_bytes += bytes; if (model) event(p, bytes, 3); }
    void on_table_write(const void* p, size_t bytes) { ++table_writes; table_bytes += bytes; if (model) event(p, bytes, 4); }
    void on_table_rw(const void* p, size_t bytes)    { on_table_read(p, bytes); on_table_write(p, bytes); }
    // A whole table swept sequentially (memset, prefix sums).
    void on_table_sweep(const void* p, size_t entries, size_t entry_bytes, bool read, bool write) {
        if (read)  { table_reads  += entries; table_bytes += entries * entry_bytes; }
        if (write) { table_writes += entries; table_bytes += entries * entry_bytes; }
        if (model && entries) event(p, entries * entry_bytes, read && write ? 6 : (read ? 3 : 4));
    }
    void on_key(const void* p, size_t bytes) { if (!bytes) return; key_bytes += bytes; if (model) event(p, bytes, 5); }
    template <class T> void on_key_compare(const T& a, const T& b, uint32_t off) {
        if constexpr (KeyTraits<T>::chunked) {
            const uint32_t ex = KeyTraits<T>::examined(a, b, off);
            on_key(a.ptr + off, ex);
            on_key(b.ptr + off, ex);
        }
    }
    template <class T> void on_key_chunk(const T& a, int chunk) {
        if constexpr (KeyTraits<T>::chunked) on_key(a.ptr + KeyTraits<T>::chunk_offset(chunk), KeyTraits<T>::chunk_bytes(a, chunk));
    }
    void on_compare(bool r) {
        ++g_counters.compares;
        if (have_cmp && r != last_cmp) ++cmp_flips;
        have_cmp = true; last_cmp = r;
        if (model) mix(0x100 | uint64_t(r));
    }
    template <class T> bool counted_less(const T& a, const T& b) {
        const bool r = KeyTraits<T>::less(a, b);
        on_key_compare(a, b, 0);
        on_compare(r);
        return r;
    }
    template <class T> int counted_compare_from(const T& a, const T& b, int chunk) {
        const uint32_t off = KeyTraits<T>::chunk_offset(chunk);
        const int c = KeyTraits<T>::compare_from(a, b, chunk);
        on_key_compare(a, b, off);
        on_compare(c < 0);
        return c;
    }
    // Global operator new / delete (see trace_alloc.hpp): scratch that the
    // upstream code allocates is tracked exactly like ours.
    void on_alloc(void* p, size_t bytes) {
        if (!active) return;
        add_region(p, bytes, true);
        g_aux.current_bytes += bytes;
        if (g_aux.current_bytes > g_aux.peak_bytes) g_aux.peak_bytes = g_aux.current_bytes;
        ++g_aux.allocations;
    }
    void on_free(void* p) {
        if (!active) return;
        g_aux.current_bytes -= remove_region(p, true);
    }
};
inline Trace g_trace;

// The hooks type the counted view hands to the library: every event the
// algorithms report goes to g_trace / g_stats.
struct TraceHooks {
    using DepthScope = sb::DepthScope<true>;
    static AlgoStats& stats() { return g_stats; }
    static void on_read(const void* p, size_t b)        { g_trace.on_read(p, b); }
    static void on_write(const void* p, size_t b)       { g_trace.on_write(p, b); }
    static void on_table_read(const void* p, size_t b)  { g_trace.on_table_read(p, b); }
    static void on_table_write(const void* p, size_t b) { g_trace.on_table_write(p, b); }
    static void on_table_rw(const void* p, size_t b)    { g_trace.on_table_rw(p, b); }
    static void on_table_sweep(const void* p, size_t entries, size_t entry_bytes, bool read, bool write) {
        g_trace.on_table_sweep(p, entries, entry_bytes, read, write);
    }
    template <class T> static void on_key_chunk(const T& a, int chunk) { g_trace.on_key_chunk(a, chunk); }
};

// GCC 13 reports `p` as maybe-uninitialised at the add_region call when
// this is inlined into a try/catch (DispBuf::grow); it cannot be, malloc
// either returns or the throw leaves.
BRAINSORT_DIAG_PUSH_MAYBE_UNINIT
template <class T>
inline T* aux_alloc_t(size_t n) {
    const size_t bytes = n * sizeof(T);
    g_aux.current_bytes += bytes;
    if (g_aux.current_bytes > g_aux.peak_bytes) g_aux.peak_bytes = g_aux.current_bytes;
    ++g_aux.allocations;
    void* p = std::malloc(bytes ? bytes : 1);
    if (!p) throw std::bad_alloc();
    if (g_trace.active) g_trace.add_region(p, bytes, false);
    return static_cast<T*>(p);
}
BRAINSORT_DIAG_POP
template <class T>
inline void aux_free_t(T* p, size_t n) {
    g_aux.current_bytes -= n * sizeof(T);
    if (g_trace.active) g_trace.remove_region(p, false);
    std::free(p);
}

// ---- array view --------------------------------------------------------------
template <class T, bool Counted>
class Array {
public:
    using value_type = T;
    using traits     = ::brainsort::elem_traits<T>;
    using key_type   = typename traits::key_type;
    using hooks      = std::conditional_t<Counted, TraceHooks, ::brainsort::detail::NoHooks>;
    static constexpr bool counted = Counted;

    Array() : p_(nullptr), n_(0) {}
    Array(T* p, size_t n) : p_(p), n_(n) {}

    size_t size() const { return n_; }
    T*     data() const { return p_; }

    T get(size_t i) const {
        assert(i < n_);
        if constexpr (Counted) g_trace.on_read(p_ + i, sizeof(T));
        return p_[i];
    }
    void set(size_t i, T v) const {
        assert(i < n_);
        if constexpr (Counted) g_trace.on_write(p_ + i, sizeof(T));
        p_[i] = v;
    }
    bool less(T a, T b) const {
        if constexpr (Counted) return g_trace.counted_less(a, b);
        else return traits::less(a, b);
    }
    // Three-way compares count as one comparison (one comparator call).
    int compare(T a, T b) const {
        if constexpr (Counted) return g_trace.counted_compare_from(a, b, 0);
        else return traits::compare(a, b);
    }
    BRAINSORT_ALWAYS_INLINE int compare_from(T a, T b, int chunk) const {
        if constexpr (Counted) return g_trace.counted_compare_from(a, b, chunk);
        else return traits::compare_from(a, b, chunk);
    }
    // The radix key of an element; on the counted path the string bytes the
    // extraction loads are tallied.
    static key_type key(T x, int chunk) {
        if constexpr (Counted) g_trace.on_key_chunk(x, chunk);
        return traits::radix_key(x, chunk);
    }
    void swap(size_t i, size_t j) const {
        T t = get(i);
        set(i, get(j));
        set(j, t);
    }
    // Sub-view over [off, off+len).
    Array sub(size_t off, size_t len) const {
        assert(off + len <= n_);
        return Array(p_ + off, len);
    }

    // Scratch memory for the algorithms: through the tracked allocator.
    template <class U> static U* alloc_array(size_t n) { return aux_alloc_t<U>(n); }
    template <class U> static void free_array(U* p, size_t n) noexcept { aux_free_t<U>(p, n); }

private:
    T*     p_;
    size_t n_;
};

// The scratch buffer types of the library, bound to the counted or uncounted view.
template <class T, bool Counted> using AuxBuffer = ::brainsort::detail::AuxBuffer<T, Array<T, Counted>>;
template <class T, bool Counted> using AuxVec    = ::brainsort::detail::AuxVec<T, Array<T, Counted>>;

// ---- small shared helpers ------------------------------------------------------
using ::brainsort::detail::insertion_sort;
using ::brainsort::detail::unguarded_insertion_sort;
using ::brainsort::detail::copy_forward;
using ::brainsort::detail::copy_backward;
using ::brainsort::detail::reverse_range;
using ::brainsort::detail::floor_log2;

}  // namespace sb

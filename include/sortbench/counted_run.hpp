// One counted run of an algorithm on a generated input, and everything
// deterministic that comes out of it: the access counts, the table and key
// traffic, the cache-model misses, the telemetry, a fingerprint of the output
// order and a hash of the whole access sequence. The benchmark child, the
// --counts-only mode and the golden-file test all go through this function,
// so they cannot disagree about what is measured.
#pragma once
#include "sortbench/core.hpp"
#include "sortbench/datasets.hpp"
#include "sortbench/registry.hpp"
#include "sortbench/verify.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace sb {

struct DetRecord {
    bool        ok = false;
    std::string error;
    uint64_t reads = 0, writes = 0, compares = 0;
    uint64_t table_reads = 0, table_writes = 0, table_bytes = 0;
    uint64_t key_bytes = 0, cmp_flips = 0;
    uint64_t traffic_bytes = 0;          // (reads + writes) * element size + table bytes + key bytes
    uint64_t l1_misses = 0, l2_misses = 0, unmapped = 0;
    uint64_t aux_peak_bytes = 0, aux_allocs = 0;
    uint32_t max_depth = 0;
    uint64_t fallbacks = 0, bad_parts = 0;
    int      first_route = 0;
    uint64_t route[6] = {};
    uint64_t giveups = 0, splits = 0, split_retries = 0, part_sorts = 0, dict_tries = 0, dict_hits = 0;
    uint64_t plan_retries = 0, radix_passes = 0, passes_skipped = 0;
    uint64_t order_hash = 0, trace_hash = 0;
};

inline const char* route_name(int r) {
    static const char* const names[] = {"-", "sorted", "reverse", "runs", "displaced", "radix"};
    return r >= 0 && r <= 5 ? names[r] : "?";
}
inline int route_index(const std::string& s) {
    for (int r = 0; r <= 5; ++r) if (s == route_name(r)) return r;
    return 0;
}

// The deterministic columns, in output order. Every one is a pure function
// of (algorithm, input); the golden test compares all of them.
inline const char* const kDetColumns[] = {
    "reads", "writes", "compares", "table_reads", "table_writes", "table_bytes", "key_bytes", "cmp_flips",
    "traffic_bytes", "l1_misses", "l2_misses", "unmapped", "aux_peak_bytes", "aux_allocs",
    "max_depth", "fallbacks", "bad_parts", "route", "r_sorted", "r_reverse", "r_runs", "r_displaced", "r_radix",
    "giveups", "splits", "split_retries", "part_sorts", "dict_tries", "dict_hits", "plan_retries",
    "radix_passes", "passes_skipped", "order_hash", "trace_hash",
};
constexpr size_t kDetColumnCount = sizeof(kDetColumns) / sizeof(kDetColumns[0]);

inline std::string hex64(uint64_t v) {
    char buf[20];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(v));
    return buf;
}

// Values in kDetColumns order.
inline std::vector<std::string> det_values(const DetRecord& r) {
    auto u = [](uint64_t v) { return std::to_string(static_cast<unsigned long long>(v)); };
    return {
        u(r.reads), u(r.writes), u(r.compares), u(r.table_reads), u(r.table_writes), u(r.table_bytes), u(r.key_bytes), u(r.cmp_flips),
        u(r.traffic_bytes), u(r.l1_misses), u(r.l2_misses), u(r.unmapped), u(r.aux_peak_bytes), u(r.aux_allocs),
        u(r.max_depth), u(r.fallbacks), u(r.bad_parts), route_name(r.first_route),
        u(r.route[1]), u(r.route[2]), u(r.route[3]), u(r.route[4]), u(r.route[5]),
        u(r.giveups), u(r.splits), u(r.split_retries), u(r.part_sorts), u(r.dict_tries), u(r.dict_hits), u(r.plan_retries),
        u(r.radix_passes), u(r.passes_skipped), hex64(r.order_hash), hex64(r.trace_hash),
    };
}

// Run the counted variant once. `model` also drives the cache model and the
// trace hash (several times slower; the benchmark and the golden test use
// it, the small correctness checks do not). On return `work` holds the
// output. The record's `ok` is false if the output failed verification or
// scratch memory leaked.
template <class T>
inline DetRecord counted_run(const AlgoInfo<T>& algo, const Dataset<T>& data, const std::vector<T>& ref, bool model,
                             std::vector<T>& work) {
    const size_t n = data.items.size();
    work = data.items;   // allocated before the trace starts, so it is not scratch
    g_counters.reset();
    g_aux.reset();
    g_stats.reset();
    g_trace.begin(work.data(), n * sizeof(T), model);
    if (data.pool) g_trace.add_region(data.pool->data(), data.pool->size(), false);
    algo.run_counted(work.data(), n);
    g_trace.end();

    DetRecord r;
    r.reads = g_counters.reads; r.writes = g_counters.writes; r.compares = g_counters.compares;
    r.table_reads = g_trace.table_reads; r.table_writes = g_trace.table_writes; r.table_bytes = g_trace.table_bytes;
    r.key_bytes = g_trace.key_bytes; r.cmp_flips = g_trace.cmp_flips;
    r.traffic_bytes = (r.reads + r.writes) * sizeof(T) + r.table_bytes + r.key_bytes;
    r.l1_misses = g_trace.cache.l1_misses; r.l2_misses = g_trace.cache.l2_misses; r.unmapped = g_trace.unmapped;
    r.aux_peak_bytes = g_aux.peak_bytes; r.aux_allocs = g_aux.allocations;
    r.max_depth = g_stats.max_depth; r.fallbacks = g_stats.fallbacks; r.bad_parts = g_stats.bad_parts;
    r.first_route = g_stats.first_route;
    for (int i = 0; i < 6; ++i) r.route[i] = g_stats.route[i];
    r.giveups = g_stats.giveups; r.splits = g_stats.splits; r.split_retries = g_stats.split_retries;
    r.part_sorts = g_stats.part_sorts; r.dict_tries = g_stats.dict_tries; r.dict_hits = g_stats.dict_hits;
    r.plan_retries = g_stats.plan_retries; r.radix_passes = g_stats.radix_passes; r.passes_skipped = g_stats.passes_skipped;
    r.trace_hash = g_trace.hash;
    // Fingerprint of the output order: for an unstable sort this pins down
    // how it ordered equal keys, which no verification checks.
    uint64_t h = Trace::kFnvOffset;
    for (size_t i = 0; i < n; ++i) h = (h ^ KeyTraits<T>::id(work[i])) * Trace::kFnvPrime;
    r.order_hash = h;

    std::string err;
    if (g_aux.current_bytes != 0) { r.error = "aux_memory_leak"; return r; }
    if (!verify_sorted(work, ref, algo.stable, err)) { r.error = "counted_pass_failed:" + err; return r; }
    r.ok = true;
    return r;
}

}  // namespace sb

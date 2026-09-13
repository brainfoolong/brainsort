// Correctness tests for every registered algorithm, on every element type,
// on both the counted and the uncounted code path. Built with assertions
// enabled so the Array view's bounds checks are active.
//
// The last test holds every algorithm to results/counts.csv, the golden file
// of deterministic numbers (counts, table and key traffic, cache-model
// misses, telemetry, order and trace fingerprints) at the benchmark size and
// seed. Any change in what an algorithm does shows up there, timing or not;
// regenerate the file deliberately with `sortbench --counts-only`.
#include "sortbench/core.hpp"
#include "sortbench/counted_run.hpp"
#include "sortbench/datasets.hpp"
#include "sortbench/registry.hpp"
#include "sortbench/trace_alloc.hpp"
#include "sortbench/verify.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace sb;

namespace {

int g_failures = 0;
int g_checks   = 0;

void fail(const std::string& msg) {
    ++g_failures;
    std::printf("FAIL: %s\n", msg.c_str());
}

template <class T>
void check_one(const AlgoInfo<T>& algo, const std::string& ds, size_t n, uint64_t seed) {
    const Dataset<T>      data  = generate_dataset<T>(ds, n, seed);
    const std::vector<T>& input = data.items;
    const std::vector<T>  ref   = make_reference(input);
    const std::string where = std::string(KeyTraits<T>::name) + "/" + algo.name + " on " + ds + " n=" +
                              std::to_string(n) + " seed=" + std::to_string(seed);
    std::string err;

    // Uncounted path.
    std::vector<T> raw = input;
    algo.run_raw(raw.data(), n);
    ++g_checks;
    if (!verify_sorted(raw, ref, algo.stable, err)) fail(where + " [raw]: " + err);

    // Counted path: must produce the identical result, must not leak scratch
    // memory, and must have done plausible amounts of work.
    std::vector<T>  counted;
    const DetRecord det = counted_run(algo, data, ref, false, counted);
    ++g_checks;
    if (!det.ok) fail(where + " [counted]: " + det.error);
    if (counted != raw) fail(where + ": counted and raw variants produced different orderings");
    if (algo.counts_accesses && n > 1) {
        if (det.reads == 0)
            fail(where + ": no reads counted on the instrumented path");
        if (det.writes == 0 && counted != input)
            fail(where + ": output differs from input but no writes were counted");
        // A comparison sort cannot decide sortedness with fewer than n-1 compares.
        if (algo.comparison_sort && det.compares < n - 1)
            fail(where + ": fewer than n-1 comparisons (" + std::to_string(det.compares) + ")");
    }
}

template <class T>
void test_type() {
    const size_t sizes[] = {0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 23, 24, 25, 31, 32, 33, 47, 48, 63, 64, 65,
                            100, 127, 128, 129, 200, 255, 256, 257, 500, 511, 512, 513, 1000, 1023, 1024,
                            1025, 2000, 4096, 5000, 10000};
    const uint64_t seeds[] = {1, 2, 3};
    const std::string type = KeyTraits<T>::name;

    for (const auto& algo : algorithms<T>()) {
        const int failures_before = g_failures;
        for (const auto& ds : kDatasets) {
            if (!dataset_applies(ds.name, type)) continue;
            for (size_t n : sizes)
                for (uint64_t s : seeds) check_one<T>(algo, ds.name, n, s);
            // The benchmark size itself, one seed.
            check_one<T>(algo, ds.name, 100000, 20260912);
        }
        std::printf("%-8s %-16s %s\n", type.c_str(), algo.name, g_failures == failures_before ? "ok" : "FAILED");
    }
}

// The verifier must reject a wrong answer, and string keys must order by
// prefix rules and stay stable under ties.
void test_verifier() {
    std::string err;
    std::vector<Item> bad = make_reference(generate_dataset<Item>("random", 100, 1).items);
    std::swap(bad[3], bad[4]);
    ++g_checks;
    if (verify_sorted(bad, make_reference(generate_dataset<Item>("random", 100, 1).items), false, err))
        fail("verifier accepted an unsorted array");
    std::vector<Item> dup = make_reference(generate_dataset<Item>("few_unique", 100, 1).items);
    size_t i = 0;
    while (i + 1 < dup.size() && dup[i].key != dup[i + 1].key) ++i;
    std::swap(dup[i], dup[i + 1]);  // same keys, swapped ids: unstable but sorted
    ++g_checks;
    if (verify_sorted(dup, make_reference(generate_dataset<Item>("few_unique", 100, 1).items), true, err))
        fail("verifier accepted an unstable result as stable");
    if (!verify_sorted(dup, make_reference(generate_dataset<Item>("few_unique", 100, 1).items), false, err))
        fail("verifier rejected a valid unstable result: " + err);

    // Short strings, prefixes of each other, ties, and one long string that
    // needs several key chunks.
    static const char* words[] = {"abab", "abc", "ab", "abab", "b", "abab", "abababababababababababababab", "abababababababababababababab", "", "abababababababababababababaa"};
    std::vector<StrItem> s(std::size(words));
    for (size_t k = 0; k < s.size(); ++k) { s[k].ptr = words[k]; s[k].len = static_cast<uint32_t>(std::strlen(words[k])); s[k].id = static_cast<uint32_t>(k); }
    std::vector<StrItem> ref = make_reference(s);
    for (int rep = 0; rep < 40; ++rep) {   // repeat so the array exceeds the insertion-sort threshold
        std::vector<StrItem> big;
        for (int r = 0; r <= rep; ++r) for (const auto& x : s) { big.push_back(x); big.back().id = static_cast<uint32_t>(big.size() - 1); }
        std::vector<StrItem> bref = make_reference(big);
        std::vector<StrItem> out = big;
        find_algorithm<StrItem>("brainsort")->run_raw(out.data(), out.size());
        ++g_checks;
        if (!verify_sorted(out, bref, true, err)) { fail("brainsort on short strings (rep " + std::to_string(rep) + "): " + err); break; }
    }
}

// Adversarial and edge-case inputs for brainsort specifically: the cases that
// the generated datasets do not reach but that have historically broken
// sorting implementations (signed zeros and infinities, integer extremes,
// inputs that drive each route into its give-up/retry path, the counter-type
// size boundary, and strings that are prefixes of one another).
template <class T>
void check_brain(const std::string& name, std::vector<T> in) {
    const std::vector<T> ref = make_reference(in);
    std::string err;
    std::vector<T> raw = in;
    find_algorithm<T>("brainsort")->run_raw(raw.data(), raw.size());
    ++g_checks;
    if (!verify_sorted(raw, ref, true, err)) fail("brainsort adversarial " + name + " [raw]: " + err);
    std::vector<T> counted = in;
    g_counters.reset();
    g_aux.reset();
    find_algorithm<T>("brainsort")->run_counted(counted.data(), counted.size());
    ++g_checks;
    if (!verify_sorted(counted, ref, true, err)) fail("brainsort adversarial " + name + " [counted]: " + err);
    if (g_aux.current_bytes != 0) fail("brainsort adversarial " + name + ": scratch leaked");
    if (counted != raw) fail("brainsort adversarial " + name + ": counted vs raw differ");
}

void test_brainsort_adversarial() {
    std::mt19937_64 rng(20260912);
    // Doubles: -0.0 must sort as equal to +0.0 (the radix total-order key folds
    // them together); infinities and huge magnitudes at both ends.
    {
        std::vector<DblItem> v;
        const double pool[] = {-0.0, 0.0, 1.0, -1.0, INFINITY, -INFINITY, 1e300, -1e300, 42.0, 42.0};
        for (int i = 0; i < 6000; ++i) v.push_back({pool[rng() % 10], static_cast<uint32_t>(i), 0});
        check_brain("double signed-zero/inf", v);
    }
    {   // alternating -0.0 / +0.0: stability must hold across the equal zeros
        std::vector<DblItem> v;
        for (int i = 0; i < 4000; ++i) v.push_back({(i & 1) ? -0.0 : 0.0, static_cast<uint32_t>(i), 0});
        check_brain("double zero-stability", v);
    }
    {   // int32 / int64 extremes: the sign-flip radix key must keep order
        std::vector<Item> v;
        const int32_t ex[] = {INT32_MIN, INT32_MAX, 0, -1, 1};
        for (int i = 0; i < 5000; ++i) v.push_back({ex[rng() % 5], static_cast<uint32_t>(i)});
        check_brain("int32 extremes", v);
        std::vector<I64Item> w;
        for (int i = 0; i < 5000; ++i) w.push_back({static_cast<int64_t>(rng()), static_cast<uint32_t>(i), 0});
        w.push_back({INT64_MIN, static_cast<uint32_t>(w.size()), 0});
        w.push_back({INT64_MAX, static_cast<uint32_t>(w.size()), 0});
        check_brain("int64 full-range", w);
    }
    {   // route 3 (displaced): long sorted body then a reversed tail, and a
        // bursty pattern that makes the route give up and hand off to radix.
        std::vector<Item> v;
        for (int i = 0; i < 90000; ++i) v.push_back({i, static_cast<uint32_t>(i)});
        for (int i = 0; i < 10000; ++i) v.push_back({10000 - i, static_cast<uint32_t>(90000 + i)});
        check_brain("route3 reversed-tail", v);
        std::vector<Item> b;
        for (int blk = 0; blk < 100; ++blk) {
            for (int i = 0; i < 1000; ++i) b.push_back({blk * 1000 + i, 0});
            b.push_back({-1000000, 0});
        }
        for (size_t i = 0; i < b.size(); ++i) b[i].id = static_cast<uint32_t>(i);
        check_brain("route3 bursty-giveup", b);
    }
    {   // route 4 split: degenerate pivot (one side empty) and strong skew in
        // both directions, which forces the split's buffer-overflow retry.
        std::vector<Item> v;
        for (int i = 0; i < 50000; ++i) v.push_back({0, static_cast<uint32_t>(i)});
        for (int i = 0; i < 50000; ++i) v.push_back({static_cast<int32_t>(rng()), static_cast<uint32_t>(50000 + i)});
        check_brain("split degenerate", v);
        for (int dir = 0; dir < 2; ++dir) {
            std::vector<Item> s;
            for (int i = 0; i < 99000; ++i) s.push_back({dir ? 1000000 + static_cast<int32_t>(rng() % 1000) : static_cast<int32_t>(rng() % 1000), static_cast<uint32_t>(i)});
            for (int i = 0; i < 1000; ++i) s.push_back({dir ? static_cast<int32_t>(rng() % 100) : 1000000 + static_cast<int32_t>(rng() % 100), static_cast<uint32_t>(99000 + i)});
            check_brain(dir ? "split skew-high" : "split skew-low", s);
        }
    }
    {   // sizes straddling the radix counter-type boundary (65536) and route
        // thresholds
        for (size_t n : {size_t{33}, size_t{1024}, size_t{1025}, size_t{65535}, size_t{65536}, size_t{65537}, size_t{131072}}) {
            std::vector<Item> v;
            for (size_t i = 0; i < n; ++i) v.push_back({static_cast<int32_t>(rng()), static_cast<uint32_t>(i)});
            check_brain("size-boundary n=" + std::to_string(n), v);
        }
    }
    {   // strings that are prefixes of each other, empties, and long keys that
        // need several chunks, amplified past the insertion-sort threshold
        std::vector<std::string> pool;
        for (int i = 0; i < 2500; ++i) {
            std::string s(rng() % 40, 'a');
            for (char& c : s) c = static_cast<char>('a' + rng() % 3);
            pool.push_back(s);
        }
        pool.push_back("");
        pool.push_back("");
        std::vector<char> buf;
        std::vector<std::pair<size_t, size_t>> span;
        for (auto& s : pool) { span.push_back({buf.size(), s.size()}); buf.insert(buf.end(), s.begin(), s.end()); }
        std::vector<StrItem> v;
        for (int rep = 0; rep < 50; ++rep)
            for (size_t i = 0; i < pool.size(); ++i)
                v.push_back({buf.data() + span[i].first, static_cast<uint32_t>(span[i].second), static_cast<uint32_t>(v.size())});
        check_brain("string prefixes/empty/chunks", v);
    }
    {   // NaN breaks the comparator's strict-weak-ordering contract and so the
        // output order is out of scope - but it must never read or write out of
        // bounds (all brainsort loops are bounded by index, not by the
        // comparator). Run it with the bounds-check assertions active and
        // confirm no scratch leaks; the order is not verified.
        const double pool[] = {NAN, -NAN, 0.0, -0.0, 1.0, -1.0, INFINITY, -INFINITY};
        std::vector<DblItem> v;
        for (int i = 0; i < 20000; ++i) v.push_back({pool[rng() % 8], static_cast<uint32_t>(i), 0});
        g_counters.reset();
        g_aux.reset();
        std::vector<DblItem> raw = v;
        find_algorithm<DblItem>("brainsort")->run_raw(raw.data(), raw.size());
        find_algorithm<DblItem>("brainsort")->run_counted(v.data(), v.size());
        ++g_checks;
        if (g_aux.current_bytes != 0) fail("brainsort NaN: scratch leaked");
    }
}

// ---- golden file of deterministic numbers ---------------------------------
// results/counts.csv is written by `sortbench --counts-only --all-types
// --all-datasets`. Every row is recomputed here and every
// deterministic column must match exactly. std::stable_sort comes from the
// toolchain's standard library, and gfx::timsort and the Boost.Sort sorts
// run on its std::vector and algorithms, so those may legitimately differ
// between standard libraries (libstdc++, libc++, MSVC) and their versions;
// a mismatch there is reported but is not a failure.
// One dataset per algorithm is also run twice in a row: the two records must
// be identical, which catches any dependence on real addresses.
std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : line) {
        if (c == ',') { out.push_back(cur); cur.clear(); }
        else if (c != '\r') cur += c;
    }
    out.push_back(cur);
    return out;
}

bool toolchain_owned(const std::string& algo) { return algo == "std::stable_sort" || algo == "gfx::timsort" || algo == "boost::spinsort" || algo == "boost::flat_stable_sort"; }

template <class T>
void golden_type(const std::vector<std::map<std::string, std::string>>& rows, int& checked, int& mismatched, int& warned) {
    const std::string type = KeyTraits<T>::name;
    std::map<std::string, bool> twice_done;
    for (const auto& row : rows) {
        if (row.at("type") != type || row.at("ok") != "1") continue;
        const std::string ds = row.at("dataset"), an = row.at("algorithm");
        const AlgoInfo<T>* algo = find_algorithm<T>(an);
        if (!algo || !dataset_exists(ds) || !dataset_applies(ds, type)) continue;
        const size_t   n    = std::strtoull(row.at("n").c_str(), nullptr, 10);
        const uint64_t seed = std::strtoull(row.at("seed").c_str(), nullptr, 10);
        const Dataset<T>     data = generate_dataset<T>(ds, n, seed);
        const std::vector<T> ref  = make_reference(data.items);
        std::vector<T>       out;
        const DetRecord      det  = counted_run(*algo, data, ref, true, out);
        ++checked;
        ++g_checks;
        if (!det.ok) { fail(type + "/" + an + " on " + ds + " golden run: " + det.error); continue; }
        const std::vector<std::string> vals = det_values(det);
        std::string diff;
        for (size_t i = 0; i < kDetColumnCount; ++i) {
            auto it = row.find(kDetColumns[i]);
            if (it == row.end()) continue;   // an older golden file without this column
            if (it->second != vals[i]) diff += std::string(diff.empty() ? "" : ", ") + kDetColumns[i] + " " + it->second + " -> " + vals[i];
        }
        if (!diff.empty()) {
            if (toolchain_owned(an)) { ++warned; std::printf("WARN: %s/%s on %s differs from results/counts.csv (standard library?): %s\n", type.c_str(), an.c_str(), ds.c_str(), diff.c_str()); }
            else { ++mismatched; fail(type + "/" + an + " on " + ds + " differs from results/counts.csv: " + diff); }
        }
        if (!twice_done[an]) {
            twice_done[an] = true;
            std::vector<T> out2;
            const DetRecord det2 = counted_run(*algo, data, ref, true, out2);
            ++g_checks;
            if (det_values(det2) != vals) fail(type + "/" + an + " on " + ds + ": two identical counted runs gave different numbers (address dependence)");
        }
    }
}

size_t g_golden_max_n = 100000;

void test_golden() {
#ifdef SB_SOURCE_DIR
    const std::string path = std::string(SB_SOURCE_DIR) + "/results/counts.csv";
#else
    const std::string path = "results/counts.csv";
#endif
    std::ifstream in(path);
    if (!in) { std::printf("golden: %s not found, skipped (make it with: sortbench --counts-only --all-types --all-datasets --csv results/counts.csv)\n", path.c_str()); return; }
    std::string line;
    if (!std::getline(in, line)) return;
    const std::vector<std::string> header = split_csv(line);
    std::vector<std::map<std::string, std::string>> rows;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const std::vector<std::string> f = split_csv(line);
        std::map<std::string, std::string> row;
        for (size_t i = 0; i < header.size() && i < f.size(); ++i) row[header[i]] = f[i];
        rows.push_back(row);
    }
    // The rows above g_golden_max_n (the million-element rows by default) are
    // skipped: recomputing them takes ten minutes. --golden-max-n raises it.
    size_t skipped = 0;
    {
        std::vector<std::map<std::string, std::string>> kept;
        for (auto& row : rows) {
            if (std::strtoull(row.at("n").c_str(), nullptr, 10) > g_golden_max_n) ++skipped;
            else kept.push_back(std::move(row));
        }
        rows.swap(kept);
    }
    int checked = 0, mismatched = 0, warned = 0;
    golden_type<Item>(rows, checked, mismatched, warned);
    golden_type<DblItem>(rows, checked, mismatched, warned);
    golden_type<I64Item>(rows, checked, mismatched, warned);
    golden_type<StrItem>(rows, checked, mismatched, warned);
    std::printf("golden: %d rows of %s recomputed, %d mismatches, %d toolchain warnings, %zu rows above n = %zu skipped\n", checked, path.c_str(), mismatched, warned, skipped, g_golden_max_n);
}

}  // namespace

int main(int argc, char** argv) {
    bool golden_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--golden") golden_only = true;
        else if (a == "--golden-max-n" && i + 1 < argc) g_golden_max_n = std::strtoull(argv[++i], nullptr, 10);
        else { std::fprintf(stderr, "usage: sortbench_tests [--golden] [--golden-max-n N]\n"); return 2; }
    }
    if (!golden_only) {
        test_type<Item>();
        test_type<DblItem>();
        test_type<I64Item>();
        test_type<StrItem>();
        test_verifier();
        test_brainsort_adversarial();
    }
    test_golden();
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

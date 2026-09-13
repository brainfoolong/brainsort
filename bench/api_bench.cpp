// The public API against the standard library and pdqsort, on plain
// containers: what a user of brainsort::sort gets, including the cost of
// building the records and permuting the elements.
//
//   brainsort_api_bench                 n = 100k, 1M and 10M, 5 repetitions
//   brainsort_api_bench --quick         n = 100k, 3 repetitions (the CI smoke run)
//   brainsort_api_bench --max-n 1000000 stop at one million elements
//   brainsort_api_bench --host TEXT     describe the machine in the stamp
//
// Prints a Markdown document: a line for humans, the run stamp of
// sortbench/stamp.hpp as an HTML comment (scripts/website.py reads it), then
// one table: median wall time in ms for brainsort::sort, std::sort,
// std::stable_sort and pdqsort (the branchless partition for arithmetic
// keys, as pdqsort.h selects on its own) per element type, dataset and
// size, and the ratio of brainsort to the fastest of the others. The last
// rows call brainsort::sort with a comparator, its comparison sort.
#include "brainsort/brainsort.hpp"
#include "pdqsort/pdqsort.h"
#include "sortbench/stamp.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct Row {          // a 64-byte record sorted by its first field
    int64_t key;
    char    payload[56];
};

template <class T> struct Gen;
template <> struct Gen<int32_t> {
    static int32_t random(std::mt19937_64& r) { return static_cast<int32_t>(r()); }
    static int32_t from(int64_t v) { return static_cast<int32_t>(v); }
    static const char* name() { return "int32_t"; }
};
template <> struct Gen<int64_t> {
    static int64_t random(std::mt19937_64& r) { return static_cast<int64_t>(r()); }
    static int64_t from(int64_t v) { return v; }
    static const char* name() { return "int64_t"; }
};
template <> struct Gen<double> {
    static double random(std::mt19937_64& r) { return std::uniform_real_distribution<double>(-1e6, 1e6)(r); }
    static double from(int64_t v) { return static_cast<double>(v) * 0.25; }
    static const char* name() { return "double"; }
};
template <> struct Gen<std::string> {
    static std::string random(std::mt19937_64& r) {
        std::string s(3 + r() % 10, 'a');
        for (char& c : s) c = static_cast<char>('a' + r() % 26);
        return s;
    }
    static std::string from(int64_t v) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%012lld", static_cast<long long>(v + (1ll << 40)));
        return buf;
    }
    static const char* name() { return "std::string"; }
};
template <> struct Gen<Row> {
    static Row random(std::mt19937_64& r) { Row x{}; x.key = static_cast<int64_t>(r()); return x; }
    static Row from(int64_t v) { Row x{}; x.key = v; return x; }
    static const char* name() { return "64-byte struct by int64"; }
};

template <class T>
std::vector<T> make(const std::string& ds, size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<T> v;
    v.reserve(n);
    if (ds == "random")            for (size_t i = 0; i < n; ++i) v.push_back(Gen<T>::random(rng));
    else if (ds == "sorted")       for (size_t i = 0; i < n; ++i) v.push_back(Gen<T>::from(static_cast<int64_t>(i)));
    else if (ds == "reverse")      for (size_t i = 0; i < n; ++i) v.push_back(Gen<T>::from(static_cast<int64_t>(n - i)));
    else if (ds == "few_unique")   for (size_t i = 0; i < n; ++i) v.push_back(Gen<T>::from(static_cast<int64_t>(rng() % 100)));
    else if (ds == "nearly_sorted") {
        for (size_t i = 0; i < n; ++i) v.push_back(Gen<T>::from(static_cast<int64_t>(i)));
        for (size_t k = 0; k < n / 100; ++k) std::swap(v[rng() % n], v[rng() % n]);
    }
    return v;
}

template <class T> struct Less {
    bool operator()(const T& a, const T& b) const { return a < b; }
};
template <> struct Less<Row> {
    bool operator()(const Row& a, const Row& b) const { return a.key < b.key; }
};

template <class T, class F>
double median_ms(const std::vector<T>& input, int reps, F&& sorter) {
    std::vector<double> t;
    std::vector<T>      work(input.size());
    for (int r = 0; r < reps; ++r) {
        std::copy(input.begin(), input.end(), work.begin());
        const auto t0 = std::chrono::steady_clock::now();
        sorter(work);
        const auto t1 = std::chrono::steady_clock::now();
        t.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        if (!std::is_sorted(work.begin(), work.end(), Less<T>{})) { std::fprintf(stderr, "not sorted!\n"); std::exit(1); }
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

// `comparator`: brainsort::sort is called with the comparator the others
// use, which takes the comparison sort of the library instead of the radix.
template <class T>
void bench_type(const std::vector<size_t>& sizes, int reps, bool comparator = false) {
    for (size_t n : sizes) {
        if (std::is_same_v<T, std::string> && n > 1000000) continue;   // 10M strings: memory
        if (std::is_same_v<T, Row> && n > 1000000) continue;
        for (const char* ds : {"random", "sorted", "reverse", "nearly_sorted", "few_unique"}) {
            const std::vector<T> in = make<T>(ds, n, 20260912);
            const double bs = median_ms(in, reps, [comparator](std::vector<T>& v) {
                if (comparator) brainsort::sort(v, Less<T>{});
                else if constexpr (std::is_same_v<T, Row>) brainsort::sort(v, [](const Row& r) { return r.key; });
                else brainsort::sort(v);
            });
            const double ss = median_ms(in, reps, [](std::vector<T>& v) { std::sort(v.begin(), v.end(), Less<T>{}); });
            const double st = median_ms(in, reps, [](std::vector<T>& v) { std::stable_sort(v.begin(), v.end(), Less<T>{}); });
            const double pd = median_ms(in, reps, [](std::vector<T>& v) {
                if constexpr (std::is_arithmetic_v<T>) pdqsort_branchless(v.begin(), v.end());
                else pdqsort(v.begin(), v.end(), Less<T>{});
            });
            const double best = std::min({ss, st, pd});
            std::printf("| %s%s | %zu | %s | %.3f | %.3f | %.3f | %.3f | %.2fx |\n", Gen<T>::name(), comparator ? " by comparator" : "", n, ds, bs, ss, st, pd, best / bs);
            std::fflush(stdout);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool quick = false;
    size_t max_n = 10000000;
    std::string host;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--quick") quick = true;
        else if (a == "--max-n" && i + 1 < argc) max_n = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--host" && i + 1 < argc) host = argv[++i];
        else { std::fprintf(stderr, "usage: brainsort_api_bench [--quick] [--max-n N] [--host TEXT]\n"); return 2; }
    }
    std::vector<size_t> sizes;
    for (size_t n : {size_t{100000}, size_t{1000000}, size_t{10000000}})
        if (n <= max_n && (!quick || n == 100000)) sizes.push_back(n);
    const int reps = quick ? 3 : 5;
    const sb::RunStamp stamp = sb::RunStamp::now();
    std::printf("%s%s. brainsort::sort on a plain std::vector against std::sort, std::stable_sort and pdqsort; median of %d runs, wall ms. Generated by brainsort_api_bench.\n",
                stamp.summary().c_str(), host.empty() ? "" : (", " + host).c_str(), reps);
    std::printf("<!-- stamp {\n  \"host\": %s,\n%s,\n  \"reps\": %d\n} -->\n\n", sb::json_str(host).c_str(), stamp.json_fields().c_str(), reps);
    std::printf("| type | n | dataset | brainsort::sort | std::sort | std::stable_sort | pdqsort | vs best |\n");
    std::printf("|---|---:|---|---:|---:|---:|---:|---:|\n");
    std::fflush(stdout);
    bench_type<int32_t>(sizes, reps);
    bench_type<int64_t>(sizes, reps);
    bench_type<double>(sizes, reps);
    bench_type<std::string>(sizes, reps);
    bench_type<Row>(sizes, reps);
    bench_type<int32_t>(sizes, reps, true);
    bench_type<Row>(sizes, reps, true);
    return 0;
}

// The single-header distribution must compile on its own and sort correctly.
#include "brainsort.hpp"

#include <algorithm>
#include <cstdio>
#include <random>
#include <string>
#include <utility>
#include <vector>

int main() {
    std::mt19937_64 rng(1);
    int failures = 0;

    std::vector<int> v(100000);
    for (auto& x : v) x = static_cast<int>(rng());
    std::vector<int> w = v;
    brainsort::sort(v);
    std::sort(w.begin(), w.end());
    if (v != w) { std::puts("FAIL: vector<int>"); ++failures; }

    std::vector<std::string> s(50000);
    for (auto& x : s) { x.resize(3 + rng() % 10); for (auto& c : x) c = static_cast<char>('a' + rng() % 26); }
    std::vector<std::string> t = s;
    brainsort::sort(s);
    std::stable_sort(t.begin(), t.end());
    if (s != t) { std::puts("FAIL: vector<string>"); ++failures; }

    struct Row { int group; double score; std::string name; };
    std::vector<Row> rows(20000);
    for (auto& r : rows) { r.group = static_cast<int>(rng() % 50); r.score = static_cast<double>(rng() % 1000); r.name = std::to_string(rng() % 100); }
    std::vector<Row> ref = rows;
    brainsort::sort(rows, [](const Row& r) { return std::pair(r.group, brainsort::desc(r.score)); });
    std::stable_sort(ref.begin(), ref.end(), [](const Row& a, const Row& b) { return a.group != b.group ? a.group < b.group : a.score > b.score; });
    for (size_t i = 0; i < rows.size(); ++i)
        if (rows[i].group != ref[i].group || rows[i].score != ref[i].score || rows[i].name != ref[i].name) { std::puts("FAIL: rows"); ++failures; break; }

    std::printf("single header: %s (brainsort %s)\n", failures ? "FAILED" : "ok", BRAINSORT_VERSION);
    return failures ? 1 : 0;
}

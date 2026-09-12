// Fuzz target for the public API. Built with -fsanitize=fuzzer (and
// BRAINSORT_LIBFUZZER defined) it is a libFuzzer entry point; built as a
// plain program it feeds itself N random inputs (the CTest smoke run).
//
// The input bytes choose a key type and the elements; the result is checked
// for order, stability and permutation against an independent comparison,
// and any failure aborts.
#include "brainsort/brainsort.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace {

template <class K> struct Tagged {
    K        key;
    uint32_t id;
};

template <class K, class Less>
void check(const std::vector<Tagged<K>>& out, size_t n, Less less, const char* what) {
    std::vector<unsigned char> seen(n, 0);
    for (size_t i = 0; i < n; ++i) {
        if (out[i].id >= n || seen[out[i].id]) { std::fprintf(stderr, "%s: not a permutation\n", what); std::abort(); }
        seen[out[i].id] = 1;
        if (i == 0) continue;
        if (less(out[i].key, out[i - 1].key)) { std::fprintf(stderr, "%s: not sorted at %zu\n", what, i); std::abort(); }
        if (!less(out[i - 1].key, out[i].key) && out[i - 1].id > out[i].id) { std::fprintf(stderr, "%s: not stable at %zu\n", what, i); std::abort(); }
    }
}

template <class K> void run(std::vector<Tagged<K>> v, const char* what) {
    const size_t n = v.size();
    brainsort::sort(v, [](const Tagged<K>& t) -> const K& { return t.key; });
    check(v, n, [](const K& a, const K& b) { return a < b; }, what);
}

void fuzz_one(const uint8_t* data, size_t size) {
    if (size == 0) return;
    const uint8_t mode = data[0] % 6;
    ++data;
    --size;
    switch (mode) {
        case 0: {   // int32 keys from 4-byte chunks
            std::vector<Tagged<int32_t>> v;
            for (size_t i = 0; i + 4 <= size; i += 4) { int32_t k; std::memcpy(&k, data + i, 4); v.push_back({k, static_cast<uint32_t>(v.size())}); }
            run(std::move(v), "int32");
            break;
        }
        case 1: {   // 8-bit keys: many duplicates
            std::vector<Tagged<uint8_t>> v;
            for (size_t i = 0; i < size; ++i) v.push_back({data[i], static_cast<uint32_t>(i)});
            run(std::move(v), "uint8");
            break;
        }
        case 2: {   // strings: split at zero bytes and at long runs
            std::vector<Tagged<std::string>> v;
            std::string cur;
            for (size_t i = 0; i < size; ++i) {
                if (data[i] == 0 || cur.size() >= 40) { v.push_back({cur, static_cast<uint32_t>(v.size())}); cur.clear(); }
                else cur.push_back(static_cast<char>(data[i]));
            }
            v.push_back({cur, static_cast<uint32_t>(v.size())});
            run(std::move(v), "string");
            break;
        }
        case 3: {   // doubles from 8-byte chunks, NaNs included: checked against the documented total order
            std::vector<Tagged<double>> v;
            for (size_t i = 0; i + 8 <= size; i += 8) { double k; std::memcpy(&k, data + i, 8); v.push_back({k, static_cast<uint32_t>(v.size())}); }
            const size_t n = v.size();
            brainsort::sort(v, [](const Tagged<double>& t) { return t.key; });
            check(v, n, [](double a, double b) { return brainsort::key_traits<double>::to_radix(a) < brainsort::key_traits<double>::to_radix(b); }, "double");
            break;
        }
        case 4: {   // composite: (int16, string, descending<uint32>)
            using K = std::tuple<int16_t, std::string, brainsort::descending<uint32_t>>;
            std::vector<Tagged<K>> v;
            for (size_t i = 0; i + 8 <= size; i += 8) {
                int16_t a; uint32_t c;
                std::memcpy(&a, data + i, 2);
                std::memcpy(&c, data + i + 2, 4);
                std::string s(reinterpret_cast<const char*>(data + i + 6), data[i + 6] % 3);
                v.push_back({K{a, s, brainsort::desc(c)}, static_cast<uint32_t>(v.size())});
            }
            const size_t n = v.size();
            brainsort::sort(v, [](const Tagged<K>& t) -> const K& { return t.key; });
            check(v, n, [](const K& x, const K& y) {
                return std::tuple(std::get<0>(x), std::get<1>(x)) != std::tuple(std::get<0>(y), std::get<1>(y))
                           ? std::tuple(std::get<0>(x), std::get<1>(x)) < std::tuple(std::get<0>(y), std::get<1>(y))
                           : std::get<2>(x).key > std::get<2>(y).key;
            }, "composite");
            break;
        }
        default: {   // int64 with structure: mostly sorted with a few edits
            std::vector<Tagged<int64_t>> v;
            for (size_t i = 0; i < size; ++i) v.push_back({static_cast<int64_t>(i) * 1000 + (data[i] % 7 == 0 ? -static_cast<int64_t>(data[i]) * 100000 : 0), static_cast<uint32_t>(i)});
            run(std::move(v), "int64 structured");
            break;
        }
    }
}

}  // namespace

#ifdef BRAINSORT_LIBFUZZER
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    fuzz_one(data, size);
    return 0;
}
#else
int main(int argc, char** argv) {
    const int iterations = argc > 1 ? std::atoi(argv[1]) : 2000;
    std::mt19937_64 rng(20260912);
    for (int it = 0; it < iterations; ++it) {
        const size_t size = static_cast<size_t>(rng() % (it % 10 == 0 ? 20000 : 600));
        std::vector<uint8_t> bytes(size);
        for (auto& b : bytes) b = static_cast<uint8_t>(rng() % (it % 3 == 0 ? 4 : 256));
        fuzz_one(bytes.data(), bytes.size());
    }
    std::printf("fuzz smoke: %d inputs ok\n", iterations);
    return 0;
}
#endif

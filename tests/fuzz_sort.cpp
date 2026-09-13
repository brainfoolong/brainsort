// Fuzz target for the public API. Built with -fsanitize=fuzzer (and
// BRAINSORT_LIBFUZZER defined) it is a libFuzzer entry point; built as a
// plain program it feeds itself N random inputs (the CTest smoke run).
//
// The input bytes choose a key type and the elements; the result is checked
// for order, stability and permutation against an independent comparison,
// and any failure aborts.
//
// One mode hands sort_with a comparator that is not an order at all (random
// per call, or a hash of the pair) or an order the key inference cannot
// express as a byte window: there only the permutation is checked, and, for
// the orders, that the result is sorted and stable by them.
//
// A leading size byte repeats the element bytes, so that a short fuzz input
// reaches the routes that start at thousands of elements (the key inference,
// the scatter beyond the cache, the index route) without the fuzzer having
// to produce the bytes.
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

// Elements for the comparator overloads, every byte initialised: 16 bytes,
// sorted in place, and 32 bytes, sorted through indices.
struct Row  { uint64_t key; uint32_t id; uint32_t pad; };
struct Wide { uint64_t key; uint32_t id; uint32_t pad[5]; };
static_assert(sizeof(Row) == 16 && sizeof(Wide) == 32);

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

uint64_t mix(uint64_t x) {
    uint64_t z = x + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// The comparators that break the contract, or the inference. Flavours 0
// and 1 are not orders: only the permutation can be checked. Flavours 2 and
// 3 are total orders that no byte window of the element expresses, so an
// inferred window that the sample agreed with has to fail the verification
// pass.
template <class E> struct Adversary {
    uint8_t  flavour;
    uint64_t state;
    uint64_t threshold;
    bool operator()(const E& a, const E& b) {
        switch (flavour % 4) {
            case 0:   // random per call: asymmetric, and a different answer next time
                state = mix(state);
                return state % 3 == 0;
            case 1: {   // a hash of the pair: antisymmetric and repeatable, not transitive
                const uint32_t lo = std::min(a.id, b.id), hi = std::max(a.id, b.id);
                const uint64_t h  = mix(state ^ (static_cast<uint64_t>(lo) << 32 | hi)) % 7;
                const int      o  = h == 0 ? 0 : h < 4 ? -1 : 1;
                return (a.id > b.id ? -o : o) < 0;
            }
            case 2:   // ascending below the threshold, descending above it
                return a.key > threshold && b.key > threshold ? b.key < a.key : a.key < b.key;
            default:   // the order of the key times an odd constant
                return a.key * 0x9E3779B97F4A7C15ull < b.key * 0x9E3779B97F4A7C15ull;
        }
    }
};

// Every element exactly once; for a comparator that is an order, sorted and
// stable by it.
template <class E>
void check_elems(const std::vector<E>& out, size_t n, Adversary<E>* less, const char* what) {
    std::vector<unsigned char> seen(n, 0);
    for (const E& e : out) {
        if (e.id >= n || seen[e.id]) { std::fprintf(stderr, "%s: not a permutation\n", what); std::abort(); }
        seen[e.id] = 1;
    }
    if (less == nullptr) return;
    for (size_t i = 1; i < n; ++i) {
        if ((*less)(out[i], out[i - 1])) { std::fprintf(stderr, "%s: not sorted at %zu\n", what, i); std::abort(); }
        if (!(*less)(out[i - 1], out[i]) && out[i - 1].id > out[i].id) { std::fprintf(stderr, "%s: not stable at %zu\n", what, i); std::abort(); }
    }
}

template <class E>
void adversarial(const uint8_t* data, size_t size, uint8_t flavour, size_t repeat) {
    uint64_t seed = 0;
    for (size_t i = 0; i < 8 && i < size; ++i) seed = seed << 8 | data[i];
    const uint64_t threshold = (size > 8 ? data[8] : 0) * 256ull;
    std::vector<E> v;
    for (size_t r = 0; r < repeat; ++r)
        for (size_t i = 0; i + 2 <= size; i += 2) {
            E e{};
            e.key = static_cast<uint64_t>(data[i] | data[i + 1] << 8);
            e.id  = static_cast<uint32_t>(v.size());
            v.push_back(e);
        }
    const size_t n = v.size();
    Adversary<E> comp{flavour, mix(seed), threshold};
    brainsort::sort_with(v, comp);
    char what[80];
    std::snprintf(what, sizeof what, "adversarial comparator, flavour %d, %zu bytes", flavour % 4, sizeof(E));
    check_elems(v, n, flavour % 4 >= 2 ? &comp : nullptr, what);
}

// The element bytes are repeated this many times: 1 for most inputs, up to
// 64 for one in eight, so an 8 KiB fuzz input reaches 100,000 elements.
size_t repeats(uint8_t b) { return b % 8 != 0 ? 1 : 1 + (b / 8) % 64; }

void fuzz_one(const uint8_t* data, size_t size) {
    if (size < 2) return;
    const uint8_t mode   = data[0] % 7;
    const size_t  repeat = repeats(data[1]);
    data += 2;
    size -= 2;
    switch (mode) {
        case 0: {   // int32 keys from 4-byte chunks
            std::vector<Tagged<int32_t>> v;
            for (size_t r = 0; r < repeat; ++r)
                for (size_t i = 0; i + 4 <= size; i += 4) { int32_t k; std::memcpy(&k, data + i, 4); v.push_back({k, static_cast<uint32_t>(v.size())}); }
            run(std::move(v), "int32");
            break;
        }
        case 1: {   // 8-bit keys: many duplicates
            std::vector<Tagged<uint8_t>> v;
            for (size_t r = 0; r < repeat; ++r)
                for (size_t i = 0; i < size; ++i) v.push_back({data[i], static_cast<uint32_t>(v.size())});
            run(std::move(v), "uint8");
            break;
        }
        case 2: {   // strings: split at zero bytes and at long runs
            std::vector<Tagged<std::string>> v;
            std::string cur;
            for (size_t r = 0; r < repeat; ++r)
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
            for (size_t r = 0; r < repeat; ++r)
                for (size_t i = 0; i + 8 <= size; i += 8) { double k; std::memcpy(&k, data + i, 8); v.push_back({k, static_cast<uint32_t>(v.size())}); }
            const size_t n = v.size();
            brainsort::sort(v, [](const Tagged<double>& t) { return t.key; });
            check(v, n, [](double a, double b) { return brainsort::key_traits<double>::to_radix(a) < brainsort::key_traits<double>::to_radix(b); }, "double");
            break;
        }
        case 4: {   // composite: (int16, string, descending<uint32>)
            using K = std::tuple<int16_t, std::string, brainsort::descending<uint32_t>>;
            std::vector<Tagged<K>> v;
            for (size_t r = 0; r < repeat; ++r)
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
        case 5: {   // int64 with structure: mostly sorted with a few edits
            std::vector<Tagged<int64_t>> v;
            for (size_t r = 0; r < repeat; ++r)
                for (size_t i = 0; i < size; ++i) v.push_back({static_cast<int64_t>(v.size()) * 1000 + (data[i] % 7 == 0 ? -static_cast<int64_t>(data[i]) * 100000 : 0), static_cast<uint32_t>(v.size())});
            run(std::move(v), "int64 structured");
            break;
        }
        default: {   // a comparator that is not an order, or an order the inference cannot express
            if (size == 0) return;
            const uint8_t flavour = data[0] % 4, wide = data[0] / 4 % 2;
            ++data;
            --size;
            if (wide) adversarial<Wide>(data, size, flavour, repeat);
            else      adversarial<Row>(data, size, flavour, repeat);
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

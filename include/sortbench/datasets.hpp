// Input distributions for every element type. Every generator is
// deterministic for a given seed and assigns id = original index so the
// harness can verify permutation/stability.
#pragma once
#include "sortbench/core.hpp"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace sb {

struct DatasetInfo {
    const char* name;
    const char* description;
};

// The first five are the default benchmark set; the rest are used by the
// tests and can be selected with --dataset / --all-datasets.
inline const DatasetInfo kDatasets[] = {
    {"random",        "uniformly random keys"},
    {"sorted",        "already sorted ascending"},
    {"reverse",       "sorted descending"},
    {"nearly_sorted", "sorted, then 1% of positions swapped at random"},
    {"few_unique",    "random keys drawn from only 100 distinct values"},
    {"all_equal",     "every key identical"},
    {"runs",          "concatenation of ascending runs of random length (16..2000)"},
    {"organ_pipe",    "ascending then descending"},
    {"small_range",   "random keys from 4 distinct values"},
    {"sawtooth",      "ascending sawtooth with period 1000"},
    {"prefixed",      "keys sharing a long common prefix (ids like customer-00012345)"},
    {"sparse_bits",   "int32 only: random keys with only bits 0-3 and 28-31 varying"},
};
constexpr size_t kDefaultDatasetCount = 5;

inline const char* const kTypeNames[] = {"int32", "double", "int64", "string"};

// A generated input. For string elements the characters live in `pool`.
template <class T>
struct Dataset {
    std::vector<T>                     items;
    std::shared_ptr<std::vector<char>> pool;
};

namespace ds_detail {

// The standard distributions are implementation-defined: libc++ draws a
// different sequence from the same engine than libstdc++ (and MSVC has its
// own), so the datasets would differ between toolchains and with them the
// golden counts. These two are written out, and are the same everywhere.
// A uniform integer in [lo, hi], by rejection on the masked low bits of the
// 64-bit draw (at most two draws on average).
inline uint64_t uniform_u64(std::mt19937_64& rng, uint64_t lo, uint64_t hi) {
    const uint64_t range = hi - lo;
    if (range == ~uint64_t(0)) return rng();
    uint64_t mask = range;
    mask |= mask >> 1; mask |= mask >> 2; mask |= mask >> 4; mask |= mask >> 8; mask |= mask >> 16; mask |= mask >> 32;
    for (;;) {
        const uint64_t r = rng() & mask;
        if (r <= range) return lo + r;
    }
}
inline int64_t uniform_i64(std::mt19937_64& rng, int64_t lo, int64_t hi) {
    return static_cast<int64_t>(uniform_u64(rng, 0, static_cast<uint64_t>(hi) - static_cast<uint64_t>(lo)) + static_cast<uint64_t>(lo));
}
// A uniform double in [lo, hi): the top 53 bits of the draw scaled to [0, 1).
inline double uniform_real(std::mt19937_64& rng, double lo, double hi) {
    const double u = static_cast<double>(rng() >> 11) * 0x1.0p-53;
    return lo + (hi - lo) * u;
}

// Per-type key generation: a random key, a key from a small integer
// (monotone, distinct for distinct integers), and a "prefixed" key.
template <class T> struct KeyGen;

template <> struct KeyGen<Item> {
    using key = int32_t;
    static key random(std::mt19937_64& rng) { return static_cast<int32_t>(static_cast<uint32_t>(rng())); }
    static key from_int(int64_t v) { return static_cast<int32_t>(v); }
    static key prefixed(std::mt19937_64& rng) { return static_cast<int32_t>(0x40000000u | (rng() & 0xFFFFFu)); }
    static key sparse_bits(std::mt19937_64& rng) { const uint32_t r = rng() & 0xFF; return static_cast<int32_t>(((r >> 4) << 28) | (r & 0xF)); }
    static void assign(Item& it, key k) { it.key = k; }
};
template <> struct KeyGen<DblItem> {
    using key = double;
    static key random(std::mt19937_64& rng) { return uniform_real(rng, -1e6, 1e6); }
    static key from_int(int64_t v) { return static_cast<double>(v) * 0.25; }
    static key prefixed(std::mt19937_64& rng) { return 1e9 + uniform_real(rng, 0, 1e6); }
    static key sparse_bits(std::mt19937_64& rng) { return from_int(KeyGen<Item>::sparse_bits(rng)); }
    static void assign(DblItem& it, key k) { it.key = k; it.pad = 0; }
};
template <> struct KeyGen<I64Item> {
    using key = int64_t;
    static key random(std::mt19937_64& rng) { return static_cast<int64_t>(rng()); }
    static key from_int(int64_t v) { return v; }
    static key prefixed(std::mt19937_64& rng) { return static_cast<int64_t>(0x5A5A000000000000ull | (rng() & 0xFFFFFu)); }
    static key sparse_bits(std::mt19937_64& rng) { return KeyGen<Item>::sparse_bits(rng); }
    static void assign(I64Item& it, key k) { it.key = k; it.pad = 0; }
};
template <> struct KeyGen<StrItem> {
    using key = std::string;
    static key random(std::mt19937_64& rng) {   // a lowercase word, 3..12 letters
        const size_t len = 3 + rng() % 10;
        std::string s(len, 'a');
        for (char& c : s) c = static_cast<char>('a' + rng() % 26);
        return s;
    }
    static key from_int(int64_t v) {   // zero-padded decimal, monotone in v
        char buf[32];
        std::snprintf(buf, sizeof buf, "%012llu", static_cast<unsigned long long>(v + (1ll << 40)));
        return buf;
    }
    static key prefixed(std::mt19937_64& rng) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "customer-%08u", static_cast<unsigned>(rng() % 100000000u));
        return buf;
    }
    static key sparse_bits(std::mt19937_64& rng) { return from_int(KeyGen<Item>::sparse_bits(rng)); }
};

template <class T>
inline std::vector<typename KeyGen<T>::key> generate_keys(const std::string& name, size_t n, uint64_t seed) {
    using G = KeyGen<T>;
    using K = typename G::key;
    std::mt19937_64 rng(seed);
    std::vector<K> v(n);
    auto random_all = [&] { for (auto& k : v) k = G::random(rng); };

    if (name == "random") {
        random_all();
    } else if (name == "sorted") {
        random_all();
        std::sort(v.begin(), v.end());
    } else if (name == "reverse") {
        random_all();
        std::sort(v.begin(), v.end());
        std::reverse(v.begin(), v.end());
    } else if (name == "nearly_sorted") {
        random_all();
        std::sort(v.begin(), v.end());
        if (n > 1) {
            const size_t swaps = std::max<size_t>(1, n / 100);
            for (size_t s = 0; s < swaps; ++s) {
                const size_t i = static_cast<size_t>(uniform_u64(rng, 0, n - 1));
                const size_t j = static_cast<size_t>(uniform_u64(rng, 0, n - 1));
                std::swap(v[i], v[j]);
            }
        }
    } else if (name == "few_unique") {
        std::vector<K> pool(100);
        for (auto& k : pool) k = G::random(rng);
        for (auto& k : v) k = pool[rng() % 100];
    } else if (name == "all_equal") {
        for (auto& k : v) k = G::from_int(42);
    } else if (name == "runs") {
        size_t i = 0;
        while (i < n) {
            size_t  l   = std::min(static_cast<size_t>(uniform_u64(rng, 16, 2000)), n - i);
            int64_t key = uniform_i64(rng, -100000, 100000);
            for (size_t k = 0; k < l; ++k) { v[i + k] = G::from_int(key); key += static_cast<int64_t>(uniform_u64(rng, 0, 10)); }
            i += l;
        }
    } else if (name == "organ_pipe") {
        const size_t half = n / 2;
        for (size_t i = 0; i < n; ++i) v[i] = G::from_int(static_cast<int64_t>(i < half ? i : n - i));
    } else if (name == "small_range") {
        for (auto& k : v) k = G::from_int(static_cast<int64_t>(rng() % 4));
    } else if (name == "sawtooth") {
        for (size_t i = 0; i < n; ++i) v[i] = G::from_int(static_cast<int64_t>(i % 1000));
    } else if (name == "prefixed") {
        for (auto& k : v) k = G::prefixed(rng);
    } else if (name == "sparse_bits") {
        for (auto& k : v) k = G::sparse_bits(rng);
    } else {
        throw std::runtime_error("unknown dataset: " + name);
    }
    return v;
}

template <class T>
inline Dataset<T> build(const std::vector<typename KeyGen<T>::key>& keys) {
    Dataset<T> d;
    d.items.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        KeyGen<T>::assign(d.items[i], keys[i]);
        KeyTraits<T>::set_id(d.items[i], static_cast<uint32_t>(i));
    }
    return d;
}
template <>
inline Dataset<StrItem> build<StrItem>(const std::vector<std::string>& keys) {
    Dataset<StrItem> d;
    size_t total = 0;
    for (const auto& s : keys) total += s.size();
    d.pool = std::make_shared<std::vector<char>>(total + 1);
    d.items.resize(keys.size());
    size_t off = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
        std::memcpy(d.pool->data() + off, keys[i].data(), keys[i].size());
        d.items[i].ptr = d.pool->data() + off;
        d.items[i].len = static_cast<uint32_t>(keys[i].size());
        d.items[i].id  = static_cast<uint32_t>(i);
        off += keys[i].size();
    }
    return d;
}

}  // namespace ds_detail

template <class T>
inline Dataset<T> generate_dataset(const std::string& name, size_t n, uint64_t seed) {
    return ds_detail::build<T>(ds_detail::generate_keys<T>(name, n, seed));
}

inline bool dataset_exists(const std::string& name) {
    for (const auto& d : kDatasets) if (name == d.name) return true;
    return false;
}
inline bool type_exists(const std::string& name) {
    for (const char* t : kTypeNames) if (name == t) return true;
    return false;
}
// sparse_bits only means something for int32 keys.
inline bool dataset_applies(const std::string& name, const std::string& type) {
    return name != "sparse_bits" || type == "int32";
}

}  // namespace sb

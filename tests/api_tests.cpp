// Tests of the public brainsort API: every built-in key type, composite and
// descending keys, user-defined key traits, every container and iterator
// kind, non-trivial and move-only elements, the documented floating-point
// order, allocation failure at every allocation point, exceptions from the
// projection, thread safety, and a few large inputs.
//
// Every result is checked for order (by an independently written natural
// order of the key type), for stability (original indices increasing within
// equal keys) and for being a permutation of the input.
#include "brainsort/brainsort.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <new>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;
int g_checks   = 0;

void fail(const std::string& msg) {
    ++g_failures;
    std::printf("FAIL: %s\n", msg.c_str());
}
#define CHECK(cond, msg)                                              \
    do {                                                              \
        ++g_checks;                                                   \
        if (!(cond)) fail(std::string(msg) + " [" #cond "]");         \
    } while (0)

// ---- the natural order of a key type, written independently of the library ----
template <class K> struct is_descending : std::false_type {};
template <class K> struct is_descending<brainsort::descending<K>> : std::true_type {};
template <class K, class = void> struct is_tuple_like : std::false_type {};
template <class K> struct is_tuple_like<K, std::void_t<decltype(std::tuple_size<K>::value)>> : std::true_type {};

template <class K> struct Natural {
    static bool less(const K& a, const K& b) {
        if constexpr (is_descending<K>::value) {
            return Natural<decltype(a.key)>::less(b.key, a.key);
        } else if constexpr (is_tuple_like<K>::value) {
            return lex(a, b, std::make_index_sequence<std::tuple_size<K>::value>{});
        } else if constexpr (std::is_floating_point_v<K>) {
            // The documented total order: -NaN < -inf < ... < -0 = +0 < ... < +inf < +NaN.
            return rank(a) < rank(b);
        } else if constexpr (std::is_same_v<K, const char*> || std::is_same_v<K, char*>) {
            return std::string_view(a) < std::string_view(b);
        } else {
            return a < b;
        }
    }
    template <size_t... I> static bool lex(const K& a, const K& b, std::index_sequence<I...>) {
        int c = 0;
        ((c == 0 ? (c = cmp<std::tuple_element_t<I, K>>(std::get<I>(a), std::get<I>(b))) : 0), ...);
        return c < 0;
    }
    template <class P> static int cmp(const P& x, const P& y) {
        if (Natural<P>::less(x, y)) return -1;
        if (Natural<P>::less(y, x)) return 1;
        return 0;
    }
    static long double rank(K x) {   // a strictly order-preserving image of the documented order (non-NaN)
        if (std::isnan(x)) return std::signbit(x) ? -INFINITY : INFINITY;   // NaNs: handled by the dedicated test
        return static_cast<long double>(x);
    }
};

template <class K> struct Tagged {
    K        key;
    uint32_t id;
};

template <class K>
bool verify(const std::vector<Tagged<K>>& out, size_t n, std::string& err) {
    if (out.size() != n) { err = "size changed"; return false; }
    std::vector<unsigned char> seen(n, 0);
    for (size_t i = 0; i < n; ++i) {
        if (out[i].id >= n || seen[out[i].id]) { err = "not a permutation at " + std::to_string(i); return false; }
        seen[out[i].id] = 1;
        if (i > 0) {
            if (Natural<K>::less(out[i].key, out[i - 1].key)) { err = "not sorted at " + std::to_string(i); return false; }
            if (!Natural<K>::less(out[i - 1].key, out[i].key) && out[i - 1].id > out[i].id) {
                err = "not stable at " + std::to_string(i);
                return false;
            }
        }
    }
    return true;
}

// ---- key generators ---------------------------------------------------------------------
// make(v): a key that is monotone in v, so the integer patterns below carry
// over to every key type; random(): a full-entropy key. Strings are stored
// in a pool that outlives the sort (string_view and const char* keys point
// into it).
struct Pool {
    std::deque<std::string> strings;
    const std::string& add(std::string s) { strings.push_back(std::move(s)); return strings.back(); }
};
inline int g_pointer_pool[4096];

enum class Color : int8_t { red = -3, green = 0, blue = 5, black = 100 };
enum Plain { P0, P1, P2, P3, P4 };

struct UserFixed { uint16_t major, minor; };   // ordered by (major, minor): a user key_traits
struct UserBytes { std::string tag; };         // ordered by tag: a user bytes key
}  // namespace

namespace brainsort {
template <> struct key_traits<UserFixed> {
    static constexpr key_kind kind       = key_kind::fixed;
    static constexpr int      radix_bits = 32;
    using radix_type = uint32_t;
    static constexpr bool     exact      = false;
    static uint32_t to_radix(const UserFixed& u) noexcept { return (uint32_t(u.major) << 16) | u.minor; }
};
template <> struct key_traits<UserBytes> {
    static constexpr key_kind kind   = key_kind::bytes;
    static constexpr bool     owning = true;
    static std::string_view bytes(const UserBytes& u) noexcept { return u.tag; }
};
}  // namespace brainsort

namespace {
inline bool operator<(const UserFixed& a, const UserFixed& b) { return std::tie(a.major, a.minor) < std::tie(b.major, b.minor); }
inline bool operator<(const UserBytes& a, const UserBytes& b) { return a.tag < b.tag; }

std::string padded(int64_t v) {   // zero-padded decimal, monotone in v
    char buf[32];
    std::snprintf(buf, sizeof buf, "%020lld", static_cast<long long>(v + (1ll << 62)));
    return buf;
}
std::string word(std::mt19937_64& rng) {
    std::string s(3 + rng() % 10, 'a');
    for (char& c : s) c = static_cast<char>('a' + rng() % 26);
    return s;
}

template <class K, class = void> struct GenImpl;

// Integers, bool, character types.
template <class K> struct GenImpl<K, std::enable_if_t<std::is_integral_v<K>>> {
    static K make(int64_t v, Pool&, std::mt19937_64&) {
        if constexpr (std::is_same_v<K, bool>) return (v & 1) != 0;
        else return static_cast<K>(v);
    }
    static K random(Pool&, std::mt19937_64& rng) {
        if constexpr (std::is_same_v<K, bool>) return (rng() & 1) != 0;
        else return static_cast<K>(rng());
    }
};
template <class K> struct GenImpl<K, std::enable_if_t<std::is_floating_point_v<K>>> {
    static K make(int64_t v, Pool&, std::mt19937_64&) { return static_cast<K>(v) * K(0.25); }
    static K random(Pool&, std::mt19937_64& rng) {
        return static_cast<K>(std::uniform_real_distribution<double>(-1e6, 1e6)(rng));
    }
};
template <> struct GenImpl<Color> {
    static Color make(int64_t v, Pool&, std::mt19937_64&) {
        static const Color c[4] = {Color::red, Color::green, Color::blue, Color::black};
        return c[((v % 4) + 4) % 4];
    }
    static Color random(Pool& p, std::mt19937_64& rng) { return make(static_cast<int64_t>(rng() % 4), p, rng); }
};
template <> struct GenImpl<Plain> {
    static Plain make(int64_t v, Pool&, std::mt19937_64&) { return static_cast<Plain>(((v % 5) + 5) % 5); }
    static Plain random(Pool& p, std::mt19937_64& rng) { return make(static_cast<int64_t>(rng() % 5), p, rng); }
};
template <> struct GenImpl<int*> {
    static int* make(int64_t v, Pool&, std::mt19937_64&) { return g_pointer_pool + (((v % 4096) + 4096) % 4096); }
    static int* random(Pool& p, std::mt19937_64& rng) { return make(static_cast<int64_t>(rng() % 4096), p, rng); }
};
template <> struct GenImpl<std::string> {
    static std::string make(int64_t v, Pool&, std::mt19937_64&) { return padded(v); }
    static std::string random(Pool&, std::mt19937_64& rng) { return word(rng); }
};
template <> struct GenImpl<std::string_view> {
    static std::string_view make(int64_t v, Pool& p, std::mt19937_64&) { return p.add(padded(v)); }
    static std::string_view random(Pool& p, std::mt19937_64& rng) { return p.add(word(rng)); }
};
template <> struct GenImpl<const char*> {
    static const char* make(int64_t v, Pool& p, std::mt19937_64&) { return p.add(padded(v)).c_str(); }
    static const char* random(Pool& p, std::mt19937_64& rng) { return p.add(word(rng)).c_str(); }
};
template <class Rep, class Per> struct GenImpl<std::chrono::duration<Rep, Per>> {
    using D = std::chrono::duration<Rep, Per>;
    static D make(int64_t v, Pool& p, std::mt19937_64& r) { return D(GenImpl<Rep>::make(v, p, r)); }
    static D random(Pool& p, std::mt19937_64& r) { return D(GenImpl<Rep>::random(p, r)); }
};
template <class C, class D> struct GenImpl<std::chrono::time_point<C, D>> {
    using TP = std::chrono::time_point<C, D>;
    static TP make(int64_t v, Pool& p, std::mt19937_64& r) { return TP(GenImpl<D>::make(v, p, r)); }
    static TP random(Pool& p, std::mt19937_64& r) { return TP(GenImpl<D>::random(p, r)); }
};
template <class K> struct GenImpl<brainsort::descending<K>> {
    static brainsort::descending<K> make(int64_t v, Pool& p, std::mt19937_64& r) { return {GenImpl<K>::make(-v, p, r)}; }
    static brainsort::descending<K> random(Pool& p, std::mt19937_64& r) { return {GenImpl<K>::random(p, r)}; }
};
template <> struct GenImpl<UserFixed> {
    static UserFixed make(int64_t v, Pool&, std::mt19937_64&) {
        const uint64_t u = static_cast<uint64_t>(v + (1ll << 40));
        return {static_cast<uint16_t>((u / 1000) & 0xFFFF), static_cast<uint16_t>(u % 1000)};
    }
    static UserFixed random(Pool&, std::mt19937_64& rng) { return {static_cast<uint16_t>(rng()), static_cast<uint16_t>(rng())}; }
};
template <> struct GenImpl<UserBytes> {
    static UserBytes make(int64_t v, Pool&, std::mt19937_64&) { return {padded(v)}; }
    static UserBytes random(Pool&, std::mt19937_64& rng) { return {word(rng)}; }
};
// Composite keys: v is split into digits, most significant first, so the
// lexicographic order of the parts follows v.
template <class K, size_t I> using part_t = std::tuple_element_t<I, K>;
template <class K> struct CompGen {
    static constexpr size_t N = std::tuple_size<K>::value;
    template <size_t... I> static K make_seq(int64_t v, Pool& p, std::mt19937_64& r, std::index_sequence<I...>) {
        const int64_t u = v < 0 ? -v : v;   // non-negative, so truncating division keeps the order
        int64_t div[N];
        int64_t d = 1;
        for (size_t i = N; i-- > 0;) { div[i] = d; d *= 1000; }
        return K{GenImpl<part_t<K, I>>::make((u / div[I]) % 1000, p, r)...};
    }
    template <size_t... I> static K random_seq(Pool& p, std::mt19937_64& r, std::index_sequence<I...>) {
        return K{GenImpl<part_t<K, I>>::random(p, r)...};
    }
    static K make(int64_t v, Pool& p, std::mt19937_64& r) { return make_seq(v, p, r, std::make_index_sequence<N>{}); }
    static K random(Pool& p, std::mt19937_64& r) { return random_seq(p, r, std::make_index_sequence<N>{}); }
};
template <class A, class B> struct GenImpl<std::pair<A, B>> : CompGen<std::pair<A, B>> {};
template <class... Ts> struct GenImpl<std::tuple<Ts...>> : CompGen<std::tuple<Ts...>> {};
template <class K, size_t N> struct GenImpl<std::array<K, N>> : CompGen<std::array<K, N>> {};

// ---- input patterns -------------------------------------------------------------------
const char* const kPatterns[] = {"random", "sorted", "reverse", "nearly_sorted", "few_unique", "all_equal",
                                 "runs", "organ_pipe", "small_range", "sawtooth"};

template <class K>
std::vector<Tagged<K>> make_input(const std::string& pattern, size_t n, uint64_t seed, Pool& pool) {
    std::mt19937_64 rng(seed);
    std::vector<int64_t> v(n);
    if (pattern == "sorted")        for (size_t i = 0; i < n; ++i) v[i] = static_cast<int64_t>(i);
    else if (pattern == "reverse")  for (size_t i = 0; i < n; ++i) v[i] = static_cast<int64_t>(n - 1 - i);
    else if (pattern == "nearly_sorted") {
        for (size_t i = 0; i < n; ++i) v[i] = static_cast<int64_t>(i);
        for (size_t k = 0; k < n / 100 + 1 && n > 1; ++k) std::swap(v[rng() % n], v[rng() % n]);
    } else if (pattern == "few_unique") for (auto& x : v) x = static_cast<int64_t>(rng() % 100);
    else if (pattern == "all_equal")    for (auto& x : v) x = 42;
    else if (pattern == "runs") {
        size_t i = 0;
        while (i < n) {
            const size_t len = 16 + rng() % 2000;
            int64_t base = static_cast<int64_t>(rng() % 100000);
            for (size_t k = 0; k < len && i < n; ++k, ++i) v[i] = base + static_cast<int64_t>(k);
        }
    } else if (pattern == "organ_pipe") { for (size_t i = 0; i < n; ++i) v[i] = static_cast<int64_t>(i < n / 2 ? i : n - i); }
    else if (pattern == "small_range")  for (auto& x : v) x = static_cast<int64_t>(rng() % 4);
    else if (pattern == "sawtooth")     for (size_t i = 0; i < n; ++i) v[i] = static_cast<int64_t>(i % 1000);
    std::vector<Tagged<K>> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (pattern == "random") out.push_back({GenImpl<K>::random(pool, rng), static_cast<uint32_t>(i)});
        else out.push_back({GenImpl<K>::make(v[i], pool, rng), static_cast<uint32_t>(i)});
    }
    return out;
}

// ---- the main matrix ---------------------------------------------------------------------
template <class K>
void test_key_type(const char* name, size_t max_n) {
    const size_t sizes[] = {0, 1, 2, 3, 5, 8, 16, 31, 32, 33, 64, 100, 255, 256, 257, 1000, 1023, 1024, 1025,
                            4096, 10000, 65535, 65536, 65537, 100000};
    const int failures_before = g_failures;
    for (const char* pattern : kPatterns) {
        for (size_t n : sizes) {
            if (n > max_n) continue;
            for (uint64_t seed : {1u, 2u}) {
                if (seed == 2 && n > 10000) continue;
                Pool pool;
                const std::vector<Tagged<K>> in = make_input<K>(pattern, n, seed * 7919 + n, pool);
                const std::string where = std::string(name) + "/" + pattern + " n=" + std::to_string(n) + " seed=" + std::to_string(seed);
                std::string err;
                // Projection returning a reference.
                std::vector<Tagged<K>> a = in;
                brainsort::sort(a, [](const Tagged<K>& t) -> const K& { return t.key; });
                CHECK(verify(a, n, err), where + " [by ref]: " + err);
                // Projection returning by value (owning strings are then materialised).
                std::vector<Tagged<K>> b = in;
                brainsort::sort_by_key(b.begin(), b.end(), [](const Tagged<K>& t) { return t.key; });
                CHECK(verify(b, n, err), where + " [by value]: " + err);
                CHECK(std::equal(a.begin(), a.end(), b.begin(), [](const Tagged<K>& x, const Tagged<K>& y) { return x.id == y.id; }),
                      where + ": the two projection forms disagree");
            }
        }
    }
    std::printf("%-40s %s\n", name, g_failures == failures_before ? "ok" : "FAILED");
}

// Sorting a vector of keys directly must agree with std::stable_sort under
// the natural order of the type (which is operator< for these types).
template <class K>
void test_self_keyed(const char* name, size_t max_n) {
    const int failures_before = g_failures;
    for (const char* pattern : kPatterns) {
        for (size_t n : {size_t{0}, size_t{1}, size_t{7}, size_t{33}, size_t{1000}, size_t{4097}, size_t{100000}}) {
            if (n > max_n) continue;
            Pool pool;
            const std::vector<Tagged<K>> in = make_input<K>(pattern, n, n + 17, pool);
            std::vector<K> v, w;
            for (const auto& t : in) { v.push_back(t.key); w.push_back(t.key); }
            brainsort::sort(v);
            std::stable_sort(w.begin(), w.end(), [](const K& x, const K& y) { return Natural<K>::less(x, y); });
            // Trivially copyable keys are compared bit for bit (so -0.0 and +0.0 stay apart).
            auto identical = [](const K& x, const K& y) {
                if constexpr (std::is_trivially_copyable_v<K>) return std::memcmp(&x, &y, sizeof(K)) == 0;
                else return !(Natural<K>::less(x, y) || Natural<K>::less(y, x));
            };
            bool same = v.size() == w.size();
            for (size_t i = 0; same && i < v.size(); ++i) same = identical(v[i], w[i]);
            CHECK(same, std::string(name) + "/" + pattern + " n=" + std::to_string(n) + ": brainsort::sort(vector) differs from std::stable_sort");
            // The iterator form and the stable_sort alias must do the same.
            std::vector<K> x, y;
            for (const auto& t : in) { x.push_back(t.key); y.push_back(t.key); }
            brainsort::sort(x.begin(), x.end());
            brainsort::stable_sort(y);
            bool same2 = true;
            for (size_t i = 0; same2 && i < v.size(); ++i) same2 = identical(v[i], x[i]) && identical(v[i], y[i]);
            CHECK(same2, std::string(name) + "/" + pattern + " n=" + std::to_string(n) + ": iterator or stable_sort form differs");
        }
    }
    std::printf("%-40s %s\n", (std::string("vector<") + name + ">").c_str(), g_failures == failures_before ? "ok" : "FAILED");
}

// ---- the documented floating-point order --------------------------------------------------
template <class F>
void test_float_order(const char* name) {
    using U = std::conditional_t<sizeof(F) == 4, uint32_t, uint64_t>;
    const int failures_before = g_failures;
    std::mt19937_64 rng(99);
    auto from_bits = [](U u) { F f; std::memcpy(&f, &u, sizeof f); return f; };
    const U quiet = sizeof(F) == 4 ? 0x7FC00000u : U(0x7FF8000000000000ull);
    const U sign  = sizeof(F) == 4 ? 0x80000000u : U(0x8000000000000000ull);
    std::vector<F> pool = {F(0.0), F(-0.0), F(1.0), F(-1.0), F(INFINITY), F(-INFINITY), F(1e30), F(-1e30),
                           from_bits(quiet), from_bits(quiet | sign), from_bits(quiet | 1), from_bits(quiet | sign | 1),
                           std::numeric_limits<F>::denorm_min(), -std::numeric_limits<F>::denorm_min(),
                           std::numeric_limits<F>::max(), std::numeric_limits<F>::lowest()};
    for (size_t n : {size_t{16}, size_t{40}, size_t{5000}, size_t{70000}}) {
        std::vector<Tagged<F>> v;
        for (size_t i = 0; i < n; ++i) v.push_back({pool[rng() % pool.size()], static_cast<uint32_t>(i)});
        brainsort::sort(v, [](const Tagged<F>& t) { return t.key; });
        bool ok = true;
        std::vector<unsigned char> seen(n, 0);
        for (size_t i = 0; i < n; ++i) {
            if (v[i].id >= n || seen[v[i].id]) ok = false;
            seen[v[i].id] = 1;
            if (i == 0) continue;
            const F a = v[i - 1].key, b = v[i].key;
            // Documented order: -NaN, -inf, negatives, zeros (equal), positives, +inf, +NaN.
            auto cls = [](F x) {
                if (std::isnan(x)) return std::signbit(x) ? 0 : 6;
                if (std::isinf(x)) return x < 0 ? 1 : 5;
                if (x == 0) return 3;
                return x < 0 ? 2 : 4;
            };
            const int ca = cls(a), cb = cls(b);
            if (ca > cb) ok = false;
            if (ca == cb && (ca == 2 || ca == 4) && b < a) ok = false;
            if (ca == cb && ca == 3 && v[i - 1].id > v[i].id) ok = false;   // -0 and +0 are equal: stable
            if (ca == cb && (ca == 0 || ca == 6)) {   // NaNs: by their payload, stable when equal
                U ua, ub;
                std::memcpy(&ua, &a, sizeof ua);
                std::memcpy(&ub, &b, sizeof ub);
                const U ma = ua & ~sign, mb = ub & ~sign;
                if (ca == 6 && ma > mb) ok = false;
                if (ca == 0 && ma < mb) ok = false;
                if (ma == mb && v[i - 1].id > v[i].id) ok = false;
            }
        }
        CHECK(ok, std::string(name) + " NaN/zero/inf order at n=" + std::to_string(n));
    }
    std::printf("%-40s %s\n", (std::string(name) + " total order").c_str(), g_failures == failures_before ? "ok" : "FAILED");
}

// ---- plain 64-bit keys ------------------------------------------------------------------------
// The keys-only route: the sorted keys are written back bit for bit, the
// two zeros of a double in input order. NaNs are covered by test_float_order.
template <class K>
void check_plain(const std::vector<K>& v, const std::string& ctx) {
    std::vector<K> a = v, b = v;
    brainsort::sort(a);
    std::stable_sort(b.begin(), b.end(), [](const K& x, const K& y) { return Natural<K>::less(x, y); });
    bool same = a.size() == b.size();
    for (size_t i = 0; same && i < a.size(); ++i) same = std::memcmp(&a[i], &b[i], sizeof(K)) == 0;
    CHECK(same, ctx + " n=" + std::to_string(v.size()) + ": brainsort::sort differs from std::stable_sort");
}
void test_plain_keys(size_t max_n) {
    const int failures_before = g_failures;
    std::mt19937_64 rng(5);
    for (size_t n : {size_t{33}, size_t{1000}, size_t{4097}, size_t{100000}}) {
        if (n > max_n) continue;
        std::vector<double> f, zeros, nearly;
        std::vector<uint64_t> u;
        std::vector<int64_t>  few;
        std::vector<size_t>   rev;
        for (size_t i = 0; i < n; ++i) {
            switch (rng() % 12) {
                case 0: case 1: case 2: f.push_back(-0.0); break;
                case 3: case 4: case 5: f.push_back(0.0); break;
                case 6: f.push_back(INFINITY); break;
                case 7: f.push_back(-INFINITY); break;
                case 8: f.push_back(static_cast<double>(rng() % 100) - 50.0); break;
                default: { uint64_t bits = rng(); double d; std::memcpy(&d, &bits, sizeof d); f.push_back(std::isnan(d) ? 1.5 : d); }
            }
            zeros.push_back(i % 3 == 0 ? -0.0 : 0.0);
            nearly.push_back(i % 7 == 0 ? -0.0 : static_cast<double>(i / 4) - static_cast<double>(n / 8));
            u.push_back(rng());
            few.push_back(static_cast<int64_t>(rng() % 7) - 3);
            rev.push_back(n - i);
        }
        check_plain(f, "double with zeros");
        check_plain(zeros, "double all zeros");
        check_plain(nearly, "double nearly sorted with negative zeros");
        check_plain(u, "uint64 random");
        check_plain(few, "int64 few unique");
        check_plain(rev, "size_t reversed");
    }
    {   // pointers, by address (a char pointer would be a C string)
        std::vector<int> ints(5000);
        std::vector<const int*> p, w;
        for (const int& c : ints) p.push_back(&c);
        std::shuffle(p.begin(), p.end(), rng);
        w = p;
        brainsort::sort(p);
        std::sort(w.begin(), w.end());
        CHECK(p == w, "pointers by address");
    }
    std::printf("%-40s %s\n", "plain 64-bit keys", g_failures == failures_before ? "ok" : "FAILED");
}

// ---- containers, iterators and element kinds ------------------------------------------------
struct Row {
    int64_t     id;
    std::string name;
    double      score;
};
struct Big {
    int64_t key;
    char    payload[120];
};
struct alignas(64) Aligned {
    int32_t key;
    int32_t pad[15];
};
struct Throwy {   // moves may throw: sorted by comparison, never permuted in place
    int value;
    Throwy(int v) : value(v) {}
    Throwy(const Throwy& o) : value(o.value) {}
    Throwy(Throwy&& o) : value(o.value) {}
    Throwy& operator=(const Throwy& o) { value = o.value; return *this; }
    Throwy& operator=(Throwy&& o) { value = o.value; return *this; }
};

void test_containers() {
    const int failures_before = g_failures;
    std::mt19937_64 rng(5);
    const size_t n = 20000;

    {   // std::deque (random access, not contiguous)
        std::deque<int> d;
        std::vector<int> v;
        for (size_t i = 0; i < n; ++i) { const int x = static_cast<int>(rng()); d.push_back(x); v.push_back(x); }
        brainsort::sort(d);
        std::sort(v.begin(), v.end());
        CHECK(std::equal(d.begin(), d.end(), v.begin()), "deque<int>");
    }
    {   // std::array, C array, span, pointers
        std::array<uint16_t, 5000> a;
        uint16_t c[5000];
        for (size_t i = 0; i < 5000; ++i) a[i] = c[i] = static_cast<uint16_t>(rng());
        std::vector<uint16_t> ref(a.begin(), a.end());
        std::sort(ref.begin(), ref.end());
        brainsort::sort(a);
        CHECK(std::equal(a.begin(), a.end(), ref.begin()), "std::array<uint16_t>");
        brainsort::sort(c);
        CHECK(std::equal(c, c + 5000, ref.begin()), "C array of uint16_t");
        uint16_t d[5000];
        for (size_t i = 0; i < 5000; ++i) d[i] = ref[4999 - i];
        brainsort::sort(std::span<uint16_t>(d, 5000));
        CHECK(std::equal(d, d + 5000, ref.begin()), "std::span<uint16_t>");
        uint16_t e[5000];
        for (size_t i = 0; i < 5000; ++i) e[i] = ref[4999 - i];
        brainsort::sort(e + 0, e + 5000);
        CHECK(std::equal(e, e + 5000, ref.begin()), "pointer range");
    }
    {   // rows: by an integer, by a string member by reference, by a temporary string, by a descending double
        std::vector<Row> rows;
        for (size_t i = 0; i < n; ++i) rows.push_back({static_cast<int64_t>(rng() % 5000), word(rng), static_cast<double>(rng() % 1000) / 8});
        auto stable_ref = [&](auto less) { std::vector<Row> r = rows; std::stable_sort(r.begin(), r.end(), less); return r; };
        auto same = [](const std::vector<Row>& a, const std::vector<Row>& b) {
            return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](const Row& x, const Row& y) {
                return x.id == y.id && x.name == y.name && x.score == y.score;
            });
        };
        std::vector<Row> r1 = rows;
        brainsort::sort(r1, [](const Row& r) { return r.id; });
        CHECK(same(r1, stable_ref([](const Row& a, const Row& b) { return a.id < b.id; })), "Row by id");
        std::vector<Row> r2 = rows;
        brainsort::sort(r2, [](const Row& r) -> const std::string& { return r.name; });
        CHECK(same(r2, stable_ref([](const Row& a, const Row& b) { return a.name < b.name; })), "Row by name (reference)");
        std::vector<Row> r3 = rows;
        brainsort::sort(r3, [](const Row& r) { return r.name + "!"; });
        CHECK(same(r3, stable_ref([](const Row& a, const Row& b) { return a.name + "!" < b.name + "!"; })), "Row by temporary string");
        std::vector<Row> r4 = rows;
        brainsort::sort(r4, [](const Row& r) { return std::pair(r.id, brainsort::desc(r.score)); });
        CHECK(same(r4, stable_ref([](const Row& a, const Row& b) { return a.id != b.id ? a.id < b.id : a.score > b.score; })), "Row by (id, desc score)");
        std::vector<Row> r5 = rows;
        brainsort::sort(r5, [](const Row& r) { return std::tuple(std::string_view(r.name), r.id); });
        CHECK(same(r5, stable_ref([](const Row& a, const Row& b) { return std::tie(a.name, a.id) < std::tie(b.name, b.id); })), "Row by (name, id)");
        std::vector<Row> r6 = rows;
        brainsort::sort(r6, [](const Row& a, const Row& b) { return a.score < b.score; });   // comparator form
        CHECK(same(r6, stable_ref([](const Row& a, const Row& b) { return a.score < b.score; })), "Row with comparator");
        std::vector<Row> r7 = rows;
        brainsort::sort_with(r7, [](const Row& a, const Row& b) { return a.id > b.id; });
        CHECK(same(r7, stable_ref([](const Row& a, const Row& b) { return a.id > b.id; })), "Row with sort_with");
        std::vector<Row> r8 = rows;
        brainsort::sort_by_key(r8, [](const Row& r) { return brainsort::desc(std::string_view(r.name)); });
        CHECK(same(r8, stable_ref([](const Row& a, const Row& b) { return a.name > b.name; })), "Row by descending string");
    }
    {   // vector<string> sorted directly, and by a char pointer view
        std::vector<std::string> s;
        for (size_t i = 0; i < n; ++i) s.push_back(word(rng));
        std::vector<std::string> ref = s;
        std::stable_sort(ref.begin(), ref.end());
        std::vector<std::string> a = s;
        brainsort::sort(a);
        CHECK(a == ref, "vector<string>");
        std::vector<std::string> b = s;
        brainsort::sort(b, [](const std::string& x) { return x.c_str(); });
        CHECK(b == ref, "vector<string> by c_str()");
    }
    {   // move-only elements: permuted in place through moves
        std::vector<std::unique_ptr<int>> p;
        for (size_t i = 0; i < n; ++i) p.push_back(std::make_unique<int>(static_cast<int>(rng() % 1000)));
        std::vector<int> ref;
        for (const auto& x : p) ref.push_back(*x);
        std::sort(ref.begin(), ref.end());
        brainsort::sort(p, [](const std::unique_ptr<int>& x) { return *x; });
        bool ok = true;
        for (size_t i = 0; i < n; ++i) ok = ok && p[i] && *p[i] == ref[i];
        CHECK(ok, "vector<unique_ptr<int>> by value");
    }
    {   // large and over-aligned elements
        std::vector<Big> big(n);
        for (size_t i = 0; i < n; ++i) { big[i].key = static_cast<int64_t>(rng() % 3000); std::snprintf(big[i].payload, sizeof big[i].payload, "%zu", i); }
        std::vector<Big> ref = big;
        std::stable_sort(ref.begin(), ref.end(), [](const Big& a, const Big& b) { return a.key < b.key; });
        brainsort::sort(big, [](const Big& b) { return b.key; });
        bool ok = true;
        for (size_t i = 0; i < n; ++i) ok = ok && big[i].key == ref[i].key && std::strcmp(big[i].payload, ref[i].payload) == 0;
        CHECK(ok, "120-byte elements");
        std::vector<Aligned> al(n);
        for (size_t i = 0; i < n; ++i) { al[i].key = static_cast<int32_t>(rng() % 3000); al[i].pad[0] = static_cast<int32_t>(i); }
        std::vector<Aligned> aref = al;
        std::stable_sort(aref.begin(), aref.end(), [](const Aligned& a, const Aligned& b) { return a.key < b.key; });
        brainsort::sort(al, [](const Aligned& a) { return a.key; });
        ok = true;
        for (size_t i = 0; i < n; ++i) ok = ok && al[i].key == aref[i].key && al[i].pad[0] == aref[i].pad[0];
        CHECK(ok, "over-aligned elements");
    }
    {   // elements whose moves may throw
        std::vector<Throwy> t;
        for (size_t i = 0; i < n; ++i) t.emplace_back(static_cast<int>(rng() % 100));
        std::vector<int> ref;
        for (const auto& x : t) ref.push_back(x.value);
        std::stable_sort(ref.begin(), ref.end());
        brainsort::sort(t, [](const Throwy& x) { return x.value; });
        bool ok = true;
        for (size_t i = 0; i < n; ++i) ok = ok && t[i].value == ref[i];
        CHECK(ok, "elements with throwing moves");
    }
    {   // tiny ranges
        std::vector<int> e;
        brainsort::sort(e);
        CHECK(e.empty(), "empty");
        std::vector<int> one = {7};
        brainsort::sort(one);
        CHECK(one[0] == 7, "one element");
        std::vector<int> two = {9, 3};
        brainsort::sort(two);
        CHECK(two[0] == 3 && two[1] == 9, "two elements");
    }
    std::printf("%-40s %s\n", "containers and element kinds", g_failures == failures_before ? "ok" : "FAILED");
}

// ---- allocation failure at every allocation point, and throwing projections -------------------
struct FailAlloc {
    static inline size_t count   = 0;
    static inline size_t fail_at = ~size_t(0);
    static inline size_t live    = 0;
    static inline size_t peak    = 0;
    static void* allocate(size_t bytes) {
        if (count++ == fail_at) throw std::bad_alloc();
        void* p = ::operator new(bytes ? bytes : 1);
        if (++live > peak) peak = live;
        return p;
    }
    static void deallocate(void* p, size_t) noexcept {
        if (!p) return;
        --live;
        ::operator delete(p);
    }
};

template <class K>
void alloc_failure_case(const char* name, const std::string& pattern, size_t n) {
    Pool pool;
    const std::vector<Tagged<K>> in = make_input<K>(pattern, n, 31 + n, pool);
    auto proj = [](const Tagged<K>& t) -> const K& { return t.key; };
    // How many allocations does a normal run make?
    FailAlloc::count = 0; FailAlloc::fail_at = ~size_t(0); FailAlloc::live = 0;
    {
        std::vector<Tagged<K>> v = in;
        brainsort::detail::sort_by_key_impl<FailAlloc>(v.begin(), v.end(), proj);
        std::string err;
        CHECK(verify(v, n, err), std::string(name) + "/" + pattern + " baseline: " + err);
    }
    const size_t total = FailAlloc::count;
    CHECK(FailAlloc::live == 0, std::string(name) + "/" + pattern + ": scratch leaked");
    for (size_t k = 0; k <= total; ++k) {
        FailAlloc::count = 0; FailAlloc::fail_at = k; FailAlloc::live = 0;
        std::vector<Tagged<K>> v = in;
        brainsort::detail::sort_by_key_impl<FailAlloc>(v.begin(), v.end(), proj);
        std::string err;
        CHECK(verify(v, n, err), std::string(name) + "/" + pattern + " allocation " + std::to_string(k) + " of " + std::to_string(total) + " failed: " + err);
        CHECK(FailAlloc::live == 0, std::string(name) + "/" + pattern + " allocation " + std::to_string(k) + " failed: scratch leaked");
    }
}

// The keys-only route of a plain vector of 64-bit keys: the key array, the
// scratch, and for doubles the ranks of the negative zeros; each failure
// falls back to std::stable_sort and the result is the same, bit for bit.
template <class K>
void alloc_failure_plain(const char* name, const std::vector<K>& in) {
    std::vector<K> w = in;
    std::stable_sort(w.begin(), w.end(), [](const K& x, const K& y) { return Natural<K>::less(x, y); });
    auto check = [&](const std::vector<K>& v, const std::string& ctx) {
        bool same = v.size() == w.size();
        for (size_t i = 0; same && i < v.size(); ++i) same = std::memcmp(&v[i], &w[i], sizeof(K)) == 0;
        CHECK(same, ctx);
    };
    FailAlloc::count = 0; FailAlloc::fail_at = ~size_t(0); FailAlloc::live = 0;
    {
        std::vector<K> v = in;
        brainsort::detail::sort_by_key_impl<FailAlloc>(v.begin(), v.end(), std::identity{});
        check(v, std::string(name) + " baseline");
    }
    const size_t total = FailAlloc::count;
    CHECK(FailAlloc::live == 0, std::string(name) + ": scratch leaked");
    CHECK(total > 0, std::string(name) + ": the baseline did not allocate");
    for (size_t k = 0; k <= total; ++k) {
        FailAlloc::count = 0; FailAlloc::fail_at = k; FailAlloc::live = 0;
        std::vector<K> v = in;
        brainsort::detail::sort_by_key_impl<FailAlloc>(v.begin(), v.end(), std::identity{});
        check(v, std::string(name) + " allocation " + std::to_string(k) + " of " + std::to_string(total) + " failed");
        CHECK(FailAlloc::live == 0, std::string(name) + " allocation " + std::to_string(k) + " failed: scratch leaked");
    }
}

void test_allocation_failure() {
    const int failures_before = g_failures;
    {
        std::mt19937_64 rng(5);
        std::vector<double> d;
        for (size_t i = 0; i < 5000; ++i) d.push_back(i % 5 == 0 ? -0.0 : (i % 5 == 1 ? 0.0 : static_cast<double>(rng() % 1000) - 500.0));
        alloc_failure_plain<double>("double with zeros", d);
        std::vector<uint64_t> u;
        for (size_t i = 0; i < 5000; ++i) u.push_back(rng());
        alloc_failure_plain<uint64_t>("uint64 random", u);
    }
    alloc_failure_case<int32_t>("int32", "random", 5000);
    alloc_failure_case<int32_t>("int32", "nearly_sorted", 20000);
    alloc_failure_case<int32_t>("int32", "few_unique", 5000);
    alloc_failure_case<int64_t>("int64", "random", 5000);
    alloc_failure_case<double>("double", "organ_pipe", 5000);
    alloc_failure_case<std::string>("string", "random", 3000);
    alloc_failure_case<std::string>("string", "few_unique", 3000);
    alloc_failure_case<std::pair<int64_t, int64_t>>("pair<int64,int64>", "random", 2000);
    // A projection that throws leaves the range untouched and leaks nothing.
    {
        std::vector<Row> rows;
        std::mt19937_64 rng(3);
        for (size_t i = 0; i < 4000; ++i) rows.push_back({static_cast<int64_t>(rng() % 500), word(rng), 0.0});
        const std::vector<Row> before = rows;
        for (size_t at : {size_t{0}, size_t{1}, size_t{2500}, size_t{3999}}) {
            size_t calls = 0;
            bool   thrown = false;
            FailAlloc::count = 0; FailAlloc::fail_at = ~size_t(0); FailAlloc::live = 0;
            try {
                brainsort::detail::sort_by_key_impl<FailAlloc>(rows.begin(), rows.end(), [&](const Row& r) {
                    if (calls++ == at) throw std::runtime_error("projection failed");
                    return r.name + "x";   // owning temporary: materialised, then the throw
                });
            } catch (const std::runtime_error&) {
                thrown = true;
            }
            CHECK(thrown, "throwing projection propagates");
            CHECK(FailAlloc::live == 0, "throwing projection leaks nothing");
            bool same = true;
            for (size_t i = 0; i < rows.size(); ++i) same = same && rows[i].id == before[i].id && rows[i].name == before[i].name;
            CHECK(same, "throwing projection leaves the range unchanged");
        }
    }
    std::printf("%-40s %s\n", "allocation failure and exceptions", g_failures == failures_before ? "ok" : "FAILED");
}

// ---- the comparator overloads ------------------------------------------------------------
// The same checks as for keys, on every pattern and size: the merge sort,
// the in-place routes for ordered input, the small sort, the std::stable_sort
// path for non-trivial elements, and a comparator that throws.
struct Wide {   // a trivially copyable 64-byte element
    int32_t  key;
    uint32_t id;
    char     payload[56];
};
template <class T>
void comparator_case(const char* name, size_t max_n) {
    const int failures_before = g_failures;
    auto less = [](const T& a, const T& b) { return a.key < b.key; };
    for (const char* pattern : kPatterns) {
        for (size_t n : {size_t{0}, size_t{1}, size_t{2}, size_t{16}, size_t{17}, size_t{32}, size_t{33}, size_t{100}, size_t{255},
                         size_t{1000}, size_t{4097}, size_t{10000}, size_t{100000}}) {
            if (n > max_n) continue;
            Pool pool;
            const std::vector<Tagged<int32_t>> keys = make_input<int32_t>(pattern, n, 5 * n + 3, pool);
            std::vector<T> in;
            for (const auto& k : keys) { T t{}; t.key = k.key; t.id = k.id; in.push_back(t); }
            const std::string where = std::string(name) + "/" + pattern + " n=" + std::to_string(n);
            auto check = [&](const std::vector<T>& out, const char* form) {
                std::vector<Tagged<int32_t>> tagged;
                for (const auto& t : out) tagged.push_back({t.key, t.id});
                std::string err;
                CHECK(verify(tagged, n, err), where + " [" + form + "]: " + err);
            };
            std::vector<T> a = in;
            brainsort::sort_with(a.begin(), a.end(), less);
            check(a, "sort_with");
            std::vector<T> b = in;
            brainsort::sort(b, less);
            check(b, "sort(range, comp)");
            std::vector<T> c = in;
            brainsort::stable_sort(c, less);
            check(c, "stable_sort(range, comp)");
        }
    }
    std::printf("%-40s %s\n", name, g_failures == failures_before ? "ok" : "FAILED");
}
void test_comparator(size_t big) {
    comparator_case<Tagged<int32_t>>("comparator, 8-byte elements", big);
    comparator_case<Wide>("comparator, 64-byte elements", big / 10);
    const int failures_before = g_failures;
    {   // non-trivial elements and a non-contiguous container: std::stable_sort
        Pool pool;
        std::vector<Tagged<std::string>> v = make_input<std::string>("random", 5000, 9, pool);
        brainsort::sort(v, [](const Tagged<std::string>& a, const Tagged<std::string>& b) { return a.key < b.key; });
        std::string err;
        CHECK(verify(v, v.size(), err), "comparator on string elements: " + err);
        std::deque<Tagged<int32_t>> d;
        for (const auto& t : make_input<int32_t>("nearly_sorted", 5000, 10, pool)) d.push_back(t);
        brainsort::sort(d, [](const Tagged<int32_t>& a, const Tagged<int32_t>& b) { return a.key < b.key; });
        std::vector<Tagged<int32_t>> dv(d.begin(), d.end());
        CHECK(verify(dv, dv.size(), err), "comparator on a deque: " + err);
    }
    {   // a comparator that throws leaves every element in the range
        Pool pool;
        for (const char* pattern : {"random", "nearly_sorted", "reverse"}) {
            const std::vector<Tagged<int32_t>> in = make_input<int32_t>(pattern, 3000, 11, pool);
            std::vector<uint32_t> ids;
            for (const auto& t : in) ids.push_back(t.id);
            std::sort(ids.begin(), ids.end());
            for (size_t at : {size_t{0}, size_t{1}, size_t{100}, size_t{2999}, size_t{7000}, size_t{20000}}) {
                std::vector<Tagged<int32_t>> v = in;
                size_t calls = 0;
                bool   thrown = false;
                try {
                    brainsort::sort(v, [&](const Tagged<int32_t>& a, const Tagged<int32_t>& b) {
                        if (calls++ == at) throw std::runtime_error("comparator failed");
                        return a.key < b.key;
                    });
                } catch (const std::runtime_error&) {
                    thrown = true;
                }
                std::vector<uint32_t> got;
                for (const auto& t : v) got.push_back(t.id);
                std::sort(got.begin(), got.end());
                CHECK(got == ids, std::string("throwing comparator on ") + pattern + " at call " + std::to_string(at) + " keeps every element");
                if (!thrown) {
                    std::string err;
                    CHECK(verify(v, v.size(), err), std::string("comparator that did not throw on ") + pattern + ": " + err);
                }
            }
        }
    }
    {   // a projection that throws on the in-place route for nearly sorted input keeps every element
        Pool pool;
        const std::vector<Tagged<int32_t>> in = make_input<int32_t>("nearly_sorted", 20000, 12, pool);
        std::vector<uint32_t> ids;
        for (const auto& t : in) ids.push_back(t.id);
        std::sort(ids.begin(), ids.end());
        for (size_t at : {size_t{0}, size_t{5000}, size_t{20000}, size_t{20500}, size_t{25000}, size_t{40000}, size_t{41000}}) {
            std::vector<Tagged<int32_t>> v = in;
            size_t calls = 0;
            bool   thrown = false;
            const bool during_scan = at < in.size();
            try {
                brainsort::sort(v, [&](const Tagged<int32_t>& t) {
                    if (calls++ == at) throw std::runtime_error("projection failed");
                    return t.key;
                });
            } catch (const std::runtime_error&) {
                thrown = true;
            }
            std::vector<uint32_t> got;
            for (const auto& t : v) got.push_back(t.id);
            if (thrown && during_scan) {
                bool same = true;
                for (size_t i = 0; i < v.size(); ++i) same = same && v[i].id == in[i].id;
                CHECK(same, "projection throwing during the first scan leaves the range unchanged (call " + std::to_string(at) + ")");
            }
            std::sort(got.begin(), got.end());
            CHECK(got == ids, "projection throwing at call " + std::to_string(at) + " keeps every element");
            if (!thrown) { std::string err; CHECK(verify(v, v.size(), err), "projection that did not throw: " + err); }
        }
    }
    {   // the memory cache can be released at any time
        Pool pool;
        std::vector<Tagged<int64_t>> v = make_input<int64_t>("random", 200000, 13, pool);
        brainsort::sort(v, [](const Tagged<int64_t>& t) { return t.key; });
        brainsort::release_memory();
        std::vector<Tagged<int64_t>> w = make_input<int64_t>("random", 200000, 14, pool);
        brainsort::sort(w, [](const Tagged<int64_t>& t) { return t.key; });
        brainsort::release_memory();
        std::string err;
        CHECK(verify(v, v.size(), err) && verify(w, w.size(), err), "release_memory between sorts: " + err);
    }
    std::printf("%-40s %s\n", "comparator paths and exceptions", g_failures == failures_before ? "ok" : "FAILED");
}

// ---- threads -----------------------------------------------------------------------------
void test_threads() {
    const int failures_before = g_failures;
    std::vector<std::thread> ts;
    std::vector<int>         ok(8, 0);
    for (int t = 0; t < 8; ++t) {
        ts.emplace_back([t, &ok] {
            std::mt19937_64 rng(1000 + t);
            Pool pool;
            std::vector<Tagged<int32_t>> a = make_input<int32_t>("random", 200000, 10 + t, pool);
            std::vector<Tagged<std::string>> b = make_input<std::string>("random", 30000, 20 + t, pool);
            std::vector<Tagged<double>> c = make_input<double>("nearly_sorted", 100000, 30 + t, pool);
            brainsort::sort(a, [](const Tagged<int32_t>& x) { return x.key; });
            brainsort::sort(b, [](const Tagged<std::string>& x) -> const std::string& { return x.key; });
            brainsort::sort(c, [](const Tagged<double>& x) { return x.key; });
            std::string err;
            ok[t] = verify(a, a.size(), err) && verify(b, b.size(), err) && verify(c, c.size(), err);
        });
    }
    for (auto& t : ts) t.join();
    for (int t = 0; t < 8; ++t) CHECK(ok[t], "thread " + std::to_string(t));
    std::printf("%-40s %s\n", "8 threads", g_failures == failures_before ? "ok" : "FAILED");
}

// ---- large inputs -------------------------------------------------------------------------
void test_large() {
    const int failures_before = g_failures;
    {
        std::mt19937_64 rng(77);
        std::vector<int32_t> v(10'000'000);
        for (auto& x : v) x = static_cast<int32_t>(rng());
        std::vector<int32_t> w = v;
        brainsort::sort(v);
        std::sort(w.begin(), w.end());
        CHECK(v == w, "10M int32");
    }
    {
        Pool pool;
        std::vector<Tagged<int64_t>> v = make_input<int64_t>("random", 2'000'000, 5, pool);
        brainsort::sort(v, [](const Tagged<int64_t>& t) { return t.key; });
        std::string err;
        CHECK(verify(v, v.size(), err), "2M int64: " + err);
    }
    {
        Pool pool;
        std::vector<Tagged<double>> v = make_input<double>("sawtooth", 1'000'000, 6, pool);
        brainsort::sort(v, [](const Tagged<double>& t) { return t.key; });
        std::string err;
        CHECK(verify(v, v.size(), err), "1M double: " + err);
    }
    {   // 16-byte records beyond the cache: the MSD scatter on both halves
        Pool pool;
        std::vector<Tagged<double>> v = make_input<double>("random", 5'000'000, 8, pool);
        brainsort::sort(v, [](const Tagged<double>& t) { return t.key; });
        std::string err;
        CHECK(verify(v, v.size(), err), "5M double: " + err);
        std::vector<double> d;
        for (const auto& t : v) d.push_back(t.key);
        std::mt19937_64 rng(9);
        std::shuffle(d.begin(), d.end(), rng);
        for (size_t i = 0; i < d.size(); i += 97) d[i] = -0.0;   // written back bit for bit
        std::vector<double> e = d;
        brainsort::sort(d);
        std::stable_sort(e.begin(), e.end());
        CHECK(d.size() == e.size() && std::memcmp(d.data(), e.data(), d.size() * sizeof(double)) == 0, "5M double values, negative zeros kept");
    }
    {   // the comparison sort beyond the small sizes
        std::mt19937_64 rng(77);
        std::vector<Wide> v(2'000'000);
        for (size_t i = 0; i < v.size(); ++i) { v[i].key = static_cast<int32_t>(rng() % 1000000); v[i].id = static_cast<uint32_t>(i); }
        brainsort::sort(v, [](const Wide& a, const Wide& b) { return a.key < b.key; });
        std::vector<Tagged<int32_t>> tagged;
        for (const auto& t : v) tagged.push_back({t.key, t.id});
        std::string err;
        CHECK(verify(tagged, tagged.size(), err), "2M 64-byte elements by comparator: " + err);
    }
    {
        Pool pool;
        std::vector<Tagged<std::string>> v = make_input<std::string>("random", 1'000'000, 7, pool);
        brainsort::sort(v, [](const Tagged<std::string>& t) -> const std::string& { return t.key; });
        std::string err;
        CHECK(verify(v, v.size(), err), "1M string: " + err);
    }
    std::printf("%-40s %s\n", "large inputs", g_failures == failures_before ? "ok" : "FAILED");
}

// ---- random shapes ---------------------------------------------------------------------------
void test_random_shapes() {
    const int failures_before = g_failures;
    std::mt19937_64 rng(2026);
    for (int it = 0; it < 400; ++it) {
        const size_t n = rng() % 3000;
        const char*  pattern = kPatterns[rng() % std::size(kPatterns)];
        Pool pool;
        std::string err;
        std::vector<Tagged<int32_t>> a = make_input<int32_t>(pattern, n, rng(), pool);
        brainsort::sort(a, [](const Tagged<int32_t>& t) { return t.key; });
        CHECK(verify(a, n, err), std::string("random shape int32 ") + pattern + " n=" + std::to_string(n) + ": " + err);
        std::vector<Tagged<std::string>> b = make_input<std::string>(pattern, n, rng(), pool);
        brainsort::sort(b, [](const Tagged<std::string>& t) -> const std::string& { return t.key; });
        CHECK(verify(b, n, err), std::string("random shape string ") + pattern + " n=" + std::to_string(n) + ": " + err);
        using C = std::tuple<int16_t, std::string_view, brainsort::descending<uint32_t>>;
        std::vector<Tagged<C>> c = make_input<C>(pattern, n, rng(), pool);
        brainsort::sort(c, [](const Tagged<C>& t) -> const C& { return t.key; });
        CHECK(verify(c, n, err), std::string("random shape composite ") + pattern + " n=" + std::to_string(n) + ": " + err);
    }
    std::printf("%-40s %s\n", "random shapes", g_failures == failures_before ? "ok" : "FAILED");
}

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && std::string(argv[1]) == "--quick";
    const size_t big = quick ? 10000 : 100000, mid = quick ? 4096 : 10000;
    std::printf("brainsort %s, %s\n", BRAINSORT_VERSION,
#ifdef BRAINSORT_X86_64
                brainsort::detail::have_avx2() ? "x86-64 with AVX2" : "x86-64 without AVX2"
#else
                "scalar build"
#endif
    );

    test_key_type<int8_t>("int8_t", mid);
    test_key_type<uint8_t>("uint8_t", mid);
    test_key_type<int16_t>("int16_t", mid);
    test_key_type<uint16_t>("uint16_t", mid);
    test_key_type<int32_t>("int32_t", big);
    test_key_type<uint32_t>("uint32_t", mid);
    test_key_type<int64_t>("int64_t", big);
    test_key_type<uint64_t>("uint64_t", mid);
    test_key_type<bool>("bool", mid);
    test_key_type<char>("char", mid);
    test_key_type<wchar_t>("wchar_t", mid);
    test_key_type<char16_t>("char16_t", mid);
    test_key_type<char32_t>("char32_t", mid);
    test_key_type<float>("float", mid);
    test_key_type<double>("double", big);
    test_key_type<Color>("enum class Color : int8_t", mid);
    test_key_type<Plain>("unscoped enum", mid);
    test_key_type<int*>("int*", mid);
    test_key_type<std::string>("std::string", big);
    test_key_type<std::string_view>("std::string_view", mid);
    test_key_type<const char*>("const char*", mid);
    test_key_type<std::chrono::milliseconds>("chrono::milliseconds", mid);
    test_key_type<std::chrono::duration<double>>("chrono::duration<double>", mid);
    test_key_type<std::chrono::system_clock::time_point>("chrono::system_clock::time_point", mid);
    test_key_type<std::pair<int32_t, int32_t>>("pair<int32,int32> (packed)", mid);
    test_key_type<std::tuple<int8_t, uint16_t, int32_t>>("tuple<int8,uint16,int32> (packed)", mid);
    test_key_type<std::tuple<uint8_t, bool>>("tuple<uint8,bool> (packed 32)", mid);
    test_key_type<std::tuple<int64_t, int64_t>>("tuple<int64,int64> (composite)", mid);
    test_key_type<std::pair<int32_t, std::string>>("pair<int32,string>", mid);
    test_key_type<std::tuple<std::string_view, int32_t>>("tuple<string_view,int32>", mid);
    test_key_type<std::tuple<std::string, std::string>>("tuple<string,string>", mid);
    test_key_type<std::array<int32_t, 3>>("array<int32,3> (composite)", mid);
    test_key_type<brainsort::descending<int32_t>>("descending<int32>", mid);
    test_key_type<brainsort::descending<double>>("descending<double>", mid);
    test_key_type<brainsort::descending<std::string>>("descending<string>", mid);
    test_key_type<std::pair<int32_t, brainsort::descending<double>>>("pair<int32,desc<double>>", mid);
    test_key_type<brainsort::descending<std::pair<int32_t, std::string>>>("desc<pair<int32,string>>", mid);
    test_key_type<std::tuple<double, std::string, brainsort::descending<uint8_t>>>("tuple<double,string,desc<uint8>>", mid);
    test_key_type<std::tuple<std::pair<int32_t, int32_t>, std::tuple<std::string>>>("nested tuple", mid);
    test_key_type<UserFixed>("user fixed key_traits", mid);
    test_key_type<UserBytes>("user bytes key_traits", mid);

    test_self_keyed<int32_t>("int32_t", big);
    test_self_keyed<uint64_t>("uint64_t", mid);
    test_self_keyed<int8_t>("int8_t", mid);
    test_self_keyed<float>("float", mid);
    test_self_keyed<double>("double", big);
    test_self_keyed<std::string>("std::string", mid);
    test_self_keyed<std::pair<int16_t, int16_t>>("pair<int16,int16>", mid);
    test_self_keyed<std::chrono::nanoseconds>("chrono::nanoseconds", mid);

    test_float_order<float>("float");
    test_float_order<double>("double");
    test_plain_keys(big);
    test_containers();
    test_allocation_failure();
    test_comparator(big);
    test_threads();
    test_random_shapes();
    if (!quick) test_large();

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

// Output verification. The reference is a std::stable_sort of the input, so
//   * a stable algorithm must reproduce it exactly (keys and ids),
//   * an unstable algorithm must reproduce the key sequence exactly and its
//     ids must be a permutation of 0..n-1 (i.e. no element lost/duplicated).
#pragma once
#include "sortbench/core.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace sb {

template <class T>
inline std::vector<T> make_reference(const std::vector<T>& input) {
    std::vector<T> ref = input;
    std::stable_sort(ref.begin(), ref.end(), [](const T& a, const T& b) { return KeyTraits<T>::less(a, b); });
    return ref;
}

template <class T>
inline bool verify_sorted(const std::vector<T>& out, const std::vector<T>& ref,
                          bool must_be_stable, std::string& err) {
    using KT = KeyTraits<T>;
    if (out.size() != ref.size()) { err = "size changed"; return false; }
    const size_t n = out.size();
    if (must_be_stable) {
        for (size_t i = 0; i < n; ++i) {
            if (out[i] != ref[i]) {
                err = "mismatch at index " + std::to_string(i) + " (id " + std::to_string(KT::id(out[i])) +
                      " vs expected id " + std::to_string(KT::id(ref[i])) + ") - not sorted or not stable";
                return false;
            }
        }
        return true;
    }
    std::vector<unsigned char> seen(n, 0);
    for (size_t i = 0; i < n; ++i) {
        if (KT::less(out[i], ref[i]) || KT::less(ref[i], out[i])) {
            err = "key mismatch at index " + std::to_string(i) + " (id " + std::to_string(KT::id(out[i])) +
                  " vs expected id " + std::to_string(KT::id(ref[i])) + ")";
            return false;
        }
        const uint32_t id = KT::id(out[i]);
        if (id >= n || seen[id]) {
            err = "id " + std::to_string(id) + " duplicated or out of range at index " +
                  std::to_string(i) + " - output is not a permutation of the input";
            return false;
        }
        seen[id] = 1;
    }
    return true;
}

}  // namespace sb

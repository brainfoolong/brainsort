// The plain LSD radix baselines (radix11, radix16), built on the radix core
// of the library (include/brainsort/detail/radix.hpp). They are the building
// blocks brainsort is made of, run as a reference, not as opponents.
#pragma once
#include "brainsort/detail/radix.hpp"
#include "sortbench/core.hpp"

namespace sb {

// Plain textbook LSD radix sort on the full key with DigitBits-bit digits.
// Only meaningful for exact (non-chunked) key types.
template <int DigitBits, class A>
inline void radix_sort(A a) {
    using namespace ::brainsort::detail::radix_detail;
    using T = typename A::value_type;
    static_assert(!KeyTraits<T>::chunked, "plain radix needs an exact key");
    const size_t n = a.size();
    if (n < 2) return;
    AuxBuffer<T, A::counted> tmp(n);
    lsd_radix(a, tmp.arr(), n, make_plan(key_bits<typename A::key_type>(), DigitBits), FullKey<A>{0});
}

template <class A> inline void radix_sort11(A a) { radix_sort<11>(a); }
template <class A> inline void radix_sort16(A a) { radix_sort<16>(a); }

}  // namespace sb

// brainsort - the candidate algorithm developed in this repository.
//
// The implementation lives in the library, include/brainsort/detail/algorithm.hpp,
// and is documented there; this header binds it to the array view of the
// benchmark so it is measured through exactly the code the library ships.
#pragma once
#include "brainsort/detail/algorithm.hpp"
#include "sortbench/core.hpp"

namespace sb {

// DigitBits <= 0 selects the width automatically.
template <int DigitBits, class A>
inline void brainsort_impl(A a) { ::brainsort::detail::brainsort_impl<DigitBits>(a); }

template <class A>
inline void brainsort(A a) { ::brainsort::detail::brainsort_impl<0>(a); }

}  // namespace sb

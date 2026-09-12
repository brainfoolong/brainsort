// Classic top-down merge sort with a full-size scratch buffer: the
// implementation is in the library (include/brainsort/detail/mergesort.hpp),
// where it also serves as the fallback for ranges of 2^32 elements or more.
// O(n log n) always, stable, n extra elements of memory.
#pragma once
#include "brainsort/detail/mergesort.hpp"
#include "sortbench/core.hpp"

namespace sb {

using ::brainsort::detail::merge_sort;

}  // namespace sb

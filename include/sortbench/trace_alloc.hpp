// Replacement global operator new / delete that report to g_trace, so the
// scratch memory the upstream code allocates (std::stable_sort's temporary
// buffer, gfx::timsort's vectors) is tracked exactly like our own: peak
// bytes, allocation count, and a region for the cache model. Outside a
// counted run the hook is one branch on a global flag.
//
// Include this header in exactly one translation unit per executable (the
// replacement functions must be defined once and must not be inline).
#pragma once
#include "sortbench/core.hpp"

#include <cstdlib>
#include <new>

namespace sb::trace_alloc_detail {
inline void* alloc(std::size_t n) {
    void* p = std::malloc(n ? n : 1);
    if (p) g_trace.on_alloc(p, n);
    return p;
}
inline void release(void* p) {
    if (!p) return;
    g_trace.on_free(p);
    std::free(p);
}
}  // namespace sb::trace_alloc_detail

void* operator new(std::size_t n) {
    void* p = sb::trace_alloc_detail::alloc(n);
    if (!p) throw std::bad_alloc();
    return p;
}
void* operator new[](std::size_t n) {
    void* p = sb::trace_alloc_detail::alloc(n);
    if (!p) throw std::bad_alloc();
    return p;
}
void* operator new(std::size_t n, const std::nothrow_t&) noexcept   { return sb::trace_alloc_detail::alloc(n); }
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept { return sb::trace_alloc_detail::alloc(n); }
void operator delete(void* p) noexcept                              { sb::trace_alloc_detail::release(p); }
void operator delete[](void* p) noexcept                            { sb::trace_alloc_detail::release(p); }
void operator delete(void* p, std::size_t) noexcept                 { sb::trace_alloc_detail::release(p); }
void operator delete[](void* p, std::size_t) noexcept               { sb::trace_alloc_detail::release(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept       { sb::trace_alloc_detail::release(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept     { sb::trace_alloc_detail::release(p); }

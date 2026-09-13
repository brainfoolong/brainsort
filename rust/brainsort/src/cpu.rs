//! CPU features and cache sizes: the port of `config.hpp`.
//!
//! On x86-64 the vector kernels are chosen at run time when the `std`
//! feature is on (`is_x86_feature_detected!`); without `std` only the
//! features enabled at compile time (`-C target-feature=+avx2,+bmi2`) are
//! used. `--cfg brainsort_no_simd` compiles the scalar code on x86-64 as
//! well, the twin of the C++ `BRAINSORT_NO_SIMD`.
#![allow(dead_code)]

/// The vector paths are compiled in.
pub const X86_SIMD: bool = cfg!(all(target_arch = "x86_64", not(brainsort_no_simd)));

#[cfg(all(target_arch = "x86_64", not(brainsort_no_simd), feature = "std"))]
mod detect {
    use core::sync::atomic::{AtomicU8, Ordering};
    // 0 unknown, 1 no, 2 yes
    static AVX2: AtomicU8 = AtomicU8::new(0);
    static BMI2: AtomicU8 = AtomicU8::new(0);
    #[inline]
    fn cached(cell: &AtomicU8, probe: fn() -> bool) -> bool {
        match cell.load(Ordering::Relaxed) {
            0 => {
                let v = probe();
                cell.store(if v { 2 } else { 1 }, Ordering::Relaxed);
                v
            }
            v => v == 2,
        }
    }
    /// AVX2 and the OS support for it.
    #[inline]
    pub fn have_avx2() -> bool {
        cached(&AVX2, || std::is_x86_feature_detected!("avx2"))
    }
    /// BMI2.
    #[inline]
    pub fn have_bmi2() -> bool {
        cached(&BMI2, || std::is_x86_feature_detected!("bmi2"))
    }
}
#[cfg(all(target_arch = "x86_64", not(brainsort_no_simd), not(feature = "std")))]
mod detect {
    #[inline(always)]
    pub fn have_avx2() -> bool {
        cfg!(target_feature = "avx2")
    }
    #[inline(always)]
    pub fn have_bmi2() -> bool {
        cfg!(target_feature = "bmi2")
    }
}
#[cfg(not(all(target_arch = "x86_64", not(brainsort_no_simd))))]
mod detect {
    #[inline(always)]
    pub fn have_avx2() -> bool {
        false
    }
    #[inline(always)]
    pub fn have_bmi2() -> bool {
        false
    }
}
#[allow(unused_imports)]
pub use detect::{have_avx2, have_bmi2};

/// The data cache sizes of the core the sort runs on. They decide when a
/// part is too large for its radix passes to run in cache and how large the
/// buckets of the MSD scatter should be. Where nothing can be read (another
/// architecture, a virtual machine that reports no caches) the fallbacks are
/// those of a common desktop core.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheSizes {
    /// First-level data cache.
    pub l1d: usize,
    /// Second level.
    pub l2: usize,
    /// Last level.
    pub l3: usize,
}
impl CacheSizes {
    const DEFAULT: CacheSizes = CacheSizes { l1d: 32 << 10, l2: 1 << 20, l3: 32 << 20 };
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[allow(unused_unsafe)]
fn detect_caches() -> CacheSizes {
    use core::arch::x86_64::__cpuid_count;
    let mut c = CacheSizes::DEFAULT;
    // SAFETY: cpuid exists on every x86-64 CPU.
    let leaf0 = unsafe { __cpuid_count(0, 0) };
    let max_leaf = leaf0.eax;
    let ext0 = unsafe { __cpuid_count(0x8000_0000, 0) };
    let mut ext = false; // AMD: the cache leaf lives in the extended range
    if ext0.eax >= 0x8000_001D {
        let e1 = unsafe { __cpuid_count(0x8000_0001, 0) };
        ext = (e1.ecx >> 22) & 1 == 1; // topology extensions
    }
    if !ext && max_leaf < 4 {
        return c;
    }
    let (mut l1d, mut l2, mut l3) = (0usize, 0usize, 0usize);
    for i in 0..16u32 {
        let r = unsafe { __cpuid_count(if ext { 0x8000_001D } else { 4 }, i) };
        let ty = r.eax & 0x1F; // 1 data, 2 instruction, 3 unified
        if ty == 0 {
            break;
        }
        let level = (r.eax >> 5) & 7;
        let size = (((r.ebx >> 22) & 0x3FF) as usize + 1) * (((r.ebx >> 12) & 0x3FF) as usize + 1) * ((r.ebx & 0xFFF) as usize + 1) * (r.ecx as usize + 1);
        if ty == 2 {
            continue;
        }
        match level {
            1 => l1d = size,
            2 => l2 = size,
            3 => l3 = size,
            _ => {}
        }
    }
    if l1d != 0 {
        c.l1d = l1d;
    }
    if l2 != 0 {
        c.l2 = l2;
    }
    if l3 != 0 {
        c.l3 = l3;
    }
    c
}
#[cfg(any(not(target_arch = "x86_64"), miri))]
fn detect_caches() -> CacheSizes {
    CacheSizes::DEFAULT
}

/// The cache sizes, read once from the CPU.
#[inline]
pub fn cache_sizes() -> CacheSizes {
    use core::sync::atomic::{AtomicUsize, Ordering};
    static L1: AtomicUsize = AtomicUsize::new(0);
    static L2: AtomicUsize = AtomicUsize::new(0);
    static L3: AtomicUsize = AtomicUsize::new(0);
    let l1 = L1.load(Ordering::Relaxed);
    if l1 != 0 {
        return CacheSizes { l1d: l1, l2: L2.load(Ordering::Relaxed), l3: L3.load(Ordering::Relaxed) };
    }
    let c = detect_caches();
    L2.store(c.l2, Ordering::Relaxed);
    L3.store(c.l3, Ordering::Relaxed);
    L1.store(c.l1d, Ordering::Relaxed);
    c
}

//! Route 4 for two to four distinct keys: the partition sort. Four distinct
//! keys (flags, enum codes, quartiles) are common, and a counting radix is a
//! poor fit for them. A stable partition needs no counters: with the
//! distinct sampled keys v0 < v1 < ... the range is split at a middle key
//! exactly like the median split, counting each side's lower key with
//! popcounts on the way; then each side is partitioned at its own key
//! straight into its final place. Two passes and a half instead of three,
//! and n/2 scratch instead of n.
use super::Scratch;
use super::radix_route::{Pivot, sort_split_parts};
use crate::view::{AllocError, Arr, Elem, RadixKey, copy_forward};
#[cfg(all(target_arch = "x86_64", not(brainsort_no_simd)))]
use crate::view::{Hooks, SimdKind};

/// First pass, scalar (also the counted path): stable split of a[0,n) by
/// key >= t2 into a (compacted) and buf[0,cap), with the exact XOR mask,
/// the counts of keys < t1 (the lower key of the < side) and of keys in
/// [t2, t3) (the lower key of the >= side), and `known`, false if some key
/// is none of v[0..3]. Returns false on buffer overflow with the buffered
/// elements folded back into their gap.
#[allow(clippy::too_many_arguments)]
pub fn split2_scalar<V: Arr>(
    a: V,
    buf: V,
    n: usize,
    cap: usize,
    chunk: i32,
    v: &[<V::T as Elem>::Key; 4],
    t1: <V::T as Elem>::Key,
    t2: <V::T as Elem>::Key,
    t3: <V::T as Elem>::Key,
    xm: &mut u64,
    c0_lt: &mut usize,
    c0_ge: &mut usize,
    n_ge: &mut usize,
    known: &mut bool,
) -> bool {
    type K<V> = <<V as Arr>::T as Elem>::Key;
    let k0 = V::key(a.get(0), chunk);
    let mut m = K::<V>::ZERO;
    let mut unknown = false;
    let (mut w, mut b, mut i, mut c1, mut c3) = (0usize, 0usize, 0usize, 0usize, 0usize);
    while i < n {
        let e = a.get(i);
        let k = V::key(e, chunk);
        m = m | (k ^ k0);
        unknown |= (k != v[0]) & (k != v[1]) & (k != v[2]) & (k != v[3]);
        c1 += (k < t1) as usize;
        c3 += (k < t3) as usize;
        if k >= t2 {
            if b == cap {
                break;
            }
            buf.set(b, e);
            b += 1;
        } else {
            a.set(w, e);
            w += 1;
        }
        i += 1;
    }
    *xm = m.to_u64();
    *known = !unknown;
    if i < n {
        copy_forward(buf, 0, a, w, b);
        return false;
    }
    *n_ge = b;
    *c0_lt = c1; // keys < t1 (all of them are < t2)
    *c0_ge = c3 - w; // keys in [t2, t3): (keys < t3) minus (keys < t2), and keys < t2 is w
    true
}

/// Second pass, scalar: stable partition of src[0,m) by key < t into
/// dst[o0, o1) (o1 = o0 + count of the lower key) and dst[o1, end1). Both
/// slots are written every step and only the matching cursor advances; once
/// a class is full the rest is copied plainly, so no junk write lands on a
/// finished element or outside the range.
#[allow(clippy::too_many_arguments)]
pub fn partition2_scalar<V: Arr>(src: V, m: usize, dst: V, o0: usize, o1: usize, end1: usize, chunk: i32, t: <V::T as Elem>::Key) {
    let (mut i, mut c0, mut c1) = (0usize, o0, o1);
    while i < m && c0 < o1 && c1 < end1 {
        let e = src.get(i);
        let g = V::key(e, chunk) >= t;
        dst.set(c0, e);
        dst.set(c1, e);
        c0 += (!g) as usize;
        c1 += g as usize;
        i += 1;
    }
    while i < m {
        let e = src.get(i);
        if V::key(e, chunk) < t {
            dst.set(c0, e);
            c0 += 1;
        } else {
            dst.set(c1, e);
            c1 += 1;
        }
        i += 1;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn split2<V: Arr>(
    a: V,
    buf: V,
    n: usize,
    cap: usize,
    chunk: i32,
    vk: &[<V::T as Elem>::Key; 4],
    t1: <V::T as Elem>::Key,
    t2: <V::T as Elem>::Key,
    t3: <V::T as Elem>::Key,
    xm: &mut u64,
    c0_lt: &mut usize,
    c0_ge: &mut usize,
    n_ge: &mut usize,
    known: &mut bool,
) -> bool {
    #[cfg(all(target_arch = "x86_64", not(brainsort_no_simd)))]
    {
        if !<V::H as Hooks>::COUNTED && matches!(<V::T as Elem>::SIMD, SimdKind::I32 | SimdKind::K64) && chunk == 0 && crate::cpu::have_avx2() {
            // SAFETY: AVX2 was detected; the views hold n and cap elements.
            return unsafe { crate::simd::x86::split2_avx2::<V>(a, buf, n, cap, vk, t1, t2, t3, xm, c0_lt, c0_ge, n_ge, known) };
        }
    }
    split2_scalar(a, buf, n, cap, chunk, vk, t1, t2, t3, xm, c0_lt, c0_ge, n_ge, known)
}
#[allow(clippy::too_many_arguments)]
#[inline]
fn partition2<V: Arr>(src: V, m: usize, dst: V, o0: usize, o1: usize, end1: usize, chunk: i32, t: <V::T as Elem>::Key) {
    #[cfg(all(target_arch = "x86_64", not(brainsort_no_simd)))]
    {
        if !<V::H as Hooks>::COUNTED && matches!(<V::T as Elem>::SIMD, SimdKind::I32 | SimdKind::K64) && chunk == 0 && crate::cpu::have_avx2() {
            // SAFETY: AVX2 was detected; the ranges are inside the views.
            unsafe { crate::simd::x86::partition2_avx2::<V::T>(src.data() as *const V::T, m, dst.data(), o0, o1, end1, t) };
            return;
        }
    }
    partition2_scalar(src, m, dst, o0, o1, end1, chunk, t)
}

/// The partition sort for two to four distinct sampled keys. Returns true
/// when a[0,n) is sorted. Returns false only if the first pass overflowed
/// the buffer (a skewed key distribution); the array is then a stable
/// partial partition and nothing else has changed, so the caller carries
/// on. An unsampled key does not return false: the two sides are finished
/// on the general radix path here.
pub fn partition_sort_few<V: Arr>(a: V, scratch: &mut Scratch<V>, n: usize, chunk: i32, info: &Pivot, have_range: bool, digit_bits: u32) -> Result<bool, AllocError> {
    type K<V> = <<V as Arr>::T as Elem>::Key;
    let d = info.n_distinct;
    let mut vk = [K::<V>::ZERO; 4];
    for j in 0..4 {
        vk[j] = K::<V>::from_u64(info.few[j.min(d - 1)]);
    }
    // Split at v1 for two or three keys (the lower side is then one key and
    // needs no second pass), at v2 for four. The sample must not put clearly
    // more than half above the split, or the buffer would overflow.
    let mid = if d == 4 { 2 } else { 1 };
    let mut above = 0usize;
    for j in mid..d {
        above += info.few_cnt[j];
    }
    if above * 16 > info.n_sample * 9 {
        return Ok(false);
    }
    let t2 = vk[mid];
    let t1 = if mid == 2 { vk[1] } else { vk[0] }; // lower key of the < side (mid == 1: one key, nothing below v0)
    let t3 = if mid + 1 < d { vk[mid + 1] } else { vk[d - 1] }; // lower key of the >= side, or its only key
    let cap = n / 2 + n / 32 + 8;
    let buf = scratch.ensure(a, cap)?;
    let mut xm = 0u64;
    let (mut c0_lt, mut c0_ge, mut n_ge) = (0usize, 0usize, 0usize);
    let mut known = false;
    if !split2(a, buf, n, cap, chunk, &vk, t1, t2, t3, &mut xm, &mut c0_lt, &mut c0_ge, &mut n_ge, &mut known) {
        return Ok(false);
    }
    let n_lt = n - n_ge;
    if !known {
        // an unsampled key: general path per side, the split stands
        sort_split_parts(a, buf, cap, n, n_ge, true, chunk, scratch, xm, info, have_range, info.smin, t2.to_u64(), info.smax, digit_bits)?;
        return Ok(true);
    }
    if mid + 1 < d {
        partition2(buf.sub(0, n_ge), n_ge, a, n_lt, n_lt + c0_ge, n, chunk, t3); // >= side: buffer -> final place
    } else {
        copy_forward(buf, 0, a, n_lt, n_ge); // one key above: just move it
    }
    if mid == 2 {
        // < side has two keys: through the buffer
        let tmp = scratch.ensure(a, n_lt)?;
        partition2(a.sub(0, n_lt), n_lt, tmp, 0, c0_lt, n_lt, chunk, t1);
        copy_forward(tmp, 0, a, 0, n_lt);
    }
    Ok(true)
}

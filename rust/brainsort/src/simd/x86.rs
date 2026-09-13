//! The AVX2 and BMI2 kernels of x86-64. Every function here is called
//! only after the run-time CPU check, exactly as the C++
//! `BRAINSORT_TARGET_AVX2` functions are, and every one has a scalar twin
//! that also handles its tail. The lane orders of the vector compares are
//! documented at each kernel: the scouts leave them permuted rather than
//! spend shuffles on straightening them.
#![allow(clippy::missing_safety_doc, unused_unsafe)]
// The intrinsics are unsafe functions on Rust 1.86 (safe inside a matching
// target_feature function from 1.87 on), hence the explicit unsafe blocks.
use crate::algorithm::radix_route::{HistStore, RadixPlan};
use crate::algorithm::scout::{ScoutResult, commit_now, finish_runs, scout_tail, track_pair};
use crate::algorithm::{COMMIT_BLOCK, Scratch};
use crate::api::{PrescanResult, Shape};
use crate::key::Prescan;
use crate::radix::KeyFn;
use crate::view::{AllocError, Arr, Elem, RadixKey, SimdKind};
use core::arch::x86_64::*;

// ---- the scout ------------------------------------------------------------------------

/// Folds one block of V compare bits into the counts and the run tracking.
/// dm/am have one bit per element of the block; bit bit_of[e] belongs to
/// element i + e. The run tracking only has to look inside a block when the
/// block could end the current run: a descent in a non-decreasing run, an
/// ascent or a tie in a strictly decreasing one.
#[inline(always)]
fn scout_block<const V: usize>(r: &mut ScoutResult, dm: u32, am: u32, i: usize, bit_of: &[u8; 8]) {
    r.descents += dm.count_ones() as usize;
    r.ascents += am.count_ones() as usize;
    if !r.tracking {
        return;
    }
    let full = (1u32 << V) - 1;
    let eq = !(dm | am) & full;
    let walk = if r.cur_dir > 0 {
        dm != 0
    } else if r.cur_dir < 0 {
        (am | eq) != 0
    } else {
        true
    };
    if !walk {
        return;
    }
    let mut e = 0;
    while e < V && r.tracking {
        let b = bit_of[e] as u32;
        track_pair(
            r,
            if (dm >> b) & 1 != 0 {
                -1
            } else if (am >> b) & 1 != 0 {
                1
            } else {
                0
            },
            i + e,
        );
        e += 1;
    }
}

#[inline(always)]
fn skey32<T: Elem>(e: &T) -> i32 {
    (T::radix_key(*e, 0).to_u64() as u32 ^ 0x8000_0000) as i32
}
#[inline(always)]
fn skey64<T: Elem>(e: &T) -> i64 {
    (T::radix_key(*e, 0).to_u64() ^ 0x8000_0000_0000_0000) as i64
}

/// Vectorised scout for int32 keys (8-byte elements), 8 elements per block.
/// The elements are compared in place (keys sit in the even 32-bit lanes),
/// and the two compare results are interleaved with a shift and a blend:
/// bit b of the block mask belongs to element (b >> 1) + 4 * (b & 1).
#[target_feature(enable = "avx2")]
unsafe fn scout_avx2_32<T: Elem>(p: *const T, n: usize) -> ScoutResult {
    const BIT_OF: [u8; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
    let mut r = ScoutResult::default();
    // SAFETY: the caller passes n >= 1 elements of 8 bytes.
    unsafe {
        let vk0 = _mm256_set1_epi32(skey32(&*p));
        let mut vmask = _mm256_setzero_si256();
        let (mut i, mut next_check) = (1usize, COMMIT_BLOCK);
        while i + 8 <= n {
            let c0 = _mm256_loadu_si256(p.add(i) as *const __m256i);
            let c1 = _mm256_loadu_si256(p.add(i + 4) as *const __m256i);
            let q0 = _mm256_loadu_si256(p.add(i - 1) as *const __m256i);
            let q1 = _mm256_loadu_si256(p.add(i + 3) as *const __m256i);
            vmask = _mm256_or_si256(vmask, _mm256_or_si256(_mm256_xor_si256(c0, vk0), _mm256_xor_si256(c1, vk0)));
            let d = _mm256_blend_epi32::<0xAA>(_mm256_cmpgt_epi32(q0, c0), _mm256_slli_epi64::<32>(_mm256_cmpgt_epi32(q1, c1)));
            let u = _mm256_blend_epi32::<0xAA>(_mm256_cmpgt_epi32(c0, q0), _mm256_slli_epi64::<32>(_mm256_cmpgt_epi32(c1, q1)));
            scout_block::<8>(&mut r, _mm256_movemask_ps(_mm256_castsi256_ps(d)) as u32, _mm256_movemask_ps(_mm256_castsi256_ps(u)) as u32, i, &BIT_OF);
            if i >= next_check {
                next_check += COMMIT_BLOCK;
                if commit_now(&r, i + 8) {
                    r.committed = true;
                    break;
                }
            }
            i += 8;
        }
        let mut lanes = [0u64; 4];
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, vmask);
        r.mask = ((lanes[0] | lanes[1] | lanes[2] | lanes[3]) as u32) as u64; // the keys are the low halves
        if r.committed {
            return r; // the bits seen so far: a head start for the radix plan
        }
        scout_tail(&mut r, p, i, n);
        finish_runs(&mut r, n);
        r
    }
}

/// Order-preserving transform of 4 doubles' bit patterns (matches the
/// radix key of a double): negative -> 2^63 - magnitude, otherwise
/// bits | 2^63. Both zeros map to 2^63.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn f64_order_key(x: __m256i) -> __m256i {
    // SAFETY: called within an AVX2 function.
    unsafe {
        let zero = _mm256_setzero_si256();
        let sign = _mm256_set1_epi64x(0x8000_0000_0000_0000u64 as i64);
        let neg = _mm256_cmpgt_epi64(zero, x); // all-ones where sign bit set
        let nk = _mm256_sub_epi64(sign, _mm256_andnot_si256(sign, x));
        _mm256_blendv_epi8(_mm256_or_si256(x, sign), nk, neg)
    }
}
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn gt64<const F64: bool>(x: __m256i, y: __m256i) -> __m256i {
    // SAFETY: called within an AVX2 function.
    unsafe { if F64 { _mm256_castpd_si256(_mm256_cmp_pd::<_CMP_GT_OQ>(_mm256_castsi256_pd(x), _mm256_castsi256_pd(y))) } else { _mm256_cmpgt_epi64(x, y) } }
}

/// Vectorised scout for 16-byte elements with a 64-bit key (double,
/// int64), 8 elements per block: four vectors of two elements each,
/// compared in place (keys in the even 64-bit lanes), each pair of results
/// interleaved with one unpack, so bit b of the block mask belongs to
/// element {0,2,1,3,4,6,5,7}[b].
#[target_feature(enable = "avx2")]
unsafe fn scout_avx2_64<T: Elem, const F64: bool>(p: *const T, n: usize) -> ScoutResult {
    const BIT_OF: [u8; 8] = [0, 2, 1, 3, 4, 6, 5, 7];
    let mut r = ScoutResult::default();
    // SAFETY: the caller passes n >= 1 elements of 16 bytes.
    unsafe {
        // Reference key in the domain the mask is built in: transformed for
        // doubles, raw for int64 (the sign flip cancels in the XOR).
        let k0 = if F64 { T::radix_key(*p, 0).to_u64() } else { T::radix_key(*p, 0).to_u64() ^ 0x8000_0000_0000_0000 };
        let vk0 = _mm256_set1_epi64x(k0 as i64);
        let mut vmask = _mm256_setzero_si256();
        let (mut i, mut next_check) = (1usize, COMMIT_BLOCK);
        while i + 8 <= n {
            let ld = |k: usize| _mm256_loadu_si256(p.add(k) as *const __m256i);
            let (c0, c1, c2, c3) = (ld(i), ld(i + 2), ld(i + 4), ld(i + 6));
            let (q0, q1, q2, q3) = (ld(i - 1), ld(i + 1), ld(i + 3), ld(i + 5));
            let mut keys0 = _mm256_unpacklo_epi64(c0, c1);
            let mut keys1 = _mm256_unpacklo_epi64(c2, c3);
            if F64 {
                keys0 = f64_order_key(keys0);
                keys1 = f64_order_key(keys1);
            }
            vmask = _mm256_or_si256(vmask, _mm256_or_si256(_mm256_xor_si256(keys0, vk0), _mm256_xor_si256(keys1, vk0)));
            let mm = |x: __m256i| _mm256_movemask_pd(_mm256_castsi256_pd(x)) as u32;
            let d0 = mm(_mm256_unpacklo_epi64(gt64::<F64>(q0, c0), gt64::<F64>(q1, c1)));
            let d1 = mm(_mm256_unpacklo_epi64(gt64::<F64>(q2, c2), gt64::<F64>(q3, c3)));
            let u0 = mm(_mm256_unpacklo_epi64(gt64::<F64>(c0, q0), gt64::<F64>(c1, q1)));
            let u1 = mm(_mm256_unpacklo_epi64(gt64::<F64>(c2, q2), gt64::<F64>(c3, q3)));
            scout_block::<8>(&mut r, d0 | (d1 << 4), u0 | (u1 << 4), i, &BIT_OF);
            if i >= next_check {
                next_check += COMMIT_BLOCK;
                if commit_now(&r, i + 8) {
                    r.committed = true;
                    break;
                }
            }
            i += 8;
        }
        let mut lanes = [0u64; 4];
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, vmask);
        r.mask = lanes[0] | lanes[1] | lanes[2] | lanes[3];
        if r.committed {
            return r;
        }
        scout_tail(&mut r, p, i, n);
        finish_runs(&mut r, n);
        r
    }
}

/// The vector scout of an element type with a SIMD layout.
///
/// # Safety
/// AVX2 is available; `p` holds `n >= 1` elements.
#[inline]
pub unsafe fn scout_avx2<T: Elem>(p: *const T, n: usize) -> ScoutResult {
    // SAFETY: the caller's contract.
    unsafe {
        match T::SIMD {
            SimdKind::I32 => scout_avx2_32(p, n),
            SimdKind::F64 => scout_avx2_64::<T, true>(p, n),
            _ => scout_avx2_64::<T, false>(p, n),
        }
    }
}

/// In-place reversal of 8- or 16-byte elements, four or two per vector
/// from each end. Type-agnostic: it moves bytes.
///
/// # Safety
/// AVX2 is available; `p` holds `n` elements of 8 or 16 bytes.
#[target_feature(enable = "avx2")]
pub unsafe fn reverse_avx2<T: Copy>(p: *mut T, n: usize) {
    let v = 32 / core::mem::size_of::<T>();
    let (mut lo, mut hi) = (0usize, n);
    // SAFETY: the caller's contract; every access stays inside [0, n).
    unsafe {
        while hi - lo >= 2 * v {
            let a = _mm256_loadu_si256(p.add(lo) as *const __m256i);
            let b = _mm256_loadu_si256(p.add(hi - v) as *const __m256i);
            let (ra, rb) = if core::mem::size_of::<T>() == 8 {
                (_mm256_permute4x64_epi64::<0b00_01_10_11>(a), _mm256_permute4x64_epi64::<0b00_01_10_11>(b))
            } else {
                (_mm256_permute4x64_epi64::<0b01_00_11_10>(a), _mm256_permute4x64_epi64::<0b01_00_11_10>(b))
            };
            _mm256_storeu_si256(p.add(lo) as *mut __m256i, rb);
            _mm256_storeu_si256(p.add(hi - v) as *mut __m256i, ra);
            lo += v;
            hi -= v;
        }
        while lo + 1 < hi {
            hi -= 1;
            core::ptr::swap(p.add(lo), p.add(hi));
            lo += 1;
        }
    }
}

// ---- the split ----------------------------------------------------------------------------

/// Lane permutations that compress the selected elements of a vector to
/// its front: by mask, 4 elements of 2 lanes, or 2 elements of 4 lanes.
struct CompressLut {
    by4: [[i32; 8]; 16],
    by2: [[i32; 8]; 4],
}
const LUT: CompressLut = {
    let mut by4 = [[0i32; 8]; 16];
    let mut m = 0;
    while m < 16 {
        let mut o = 0;
        let mut e = 0i32;
        while e < 4 {
            if (m >> e) & 1 == 1 {
                by4[m][o] = 2 * e;
                by4[m][o + 1] = 2 * e + 1;
                o += 2;
            }
            e += 1;
        }
        m += 1;
    }
    let mut by2 = [[0i32; 8]; 4];
    let mut m = 0;
    while m < 4 {
        let mut o = 0;
        let mut e = 0i32;
        while e < 2 {
            if (m >> e) & 1 == 1 {
                let mut l = 0i32;
                while l < 4 {
                    by2[m][o] = 4 * e + l;
                    o += 1;
                    l += 1;
                }
            }
            e += 1;
        }
        m += 1;
    }
    CompressLut { by4, by2 }
};

/// Per-kind compare: for a loaded vector, the "< pivot" element mask (bit
/// e for element e); ORs the element keys XOR key0 into vmask.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn lt_mask<T: Elem>(v: __m256i, vpivot: __m256i, vk0: __m256i, vmask: &mut __m256i) -> u32 {
    // SAFETY: called within an AVX2 function.
    unsafe {
        match T::SIMD {
            SimdKind::I32 => {
                *vmask = _mm256_or_si256(*vmask, _mm256_xor_si256(v, vk0));
                let lt = _mm256_cmpgt_epi32(vpivot, v); // key lanes 0,2,4,6
                let lt = _mm256_shuffle_epi32::<0b10_10_00_00>(lt); // copy each key lane over its id lane
                _mm256_movemask_pd(_mm256_castsi256_pd(lt)) as u32 // one bit per element
            }
            SimdKind::I64 => {
                *vmask = _mm256_or_si256(*vmask, _mm256_xor_si256(v, vk0));
                let lt = _mm256_cmpgt_epi64(vpivot, v); // key lanes 0,2
                let m = _mm256_movemask_pd(_mm256_castsi256_pd(lt)) as u32;
                (m & 1) | ((m >> 1) & 2)
            }
            _ => {
                // SAFETY: called within an AVX2 function.
                let sign = _mm256_set1_epi64x(0x8000_0000_0000_0000u64 as i64);
                let k = unsafe { f64_order_key(v) };
                *vmask = _mm256_or_si256(*vmask, _mm256_xor_si256(k, vk0));
                let lt = _mm256_cmpgt_epi64(vpivot, _mm256_xor_si256(k, sign));
                let m = _mm256_movemask_pd(_mm256_castsi256_pd(lt)) as u32;
                (m & 1) | ((m >> 1) & 2)
            }
        }
    }
}
/// Broadcasts a raw radix key the way lt_mask expects a pivot.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn pivot_vec_key<T: Elem>(pk: u64) -> __m256i {
    // SAFETY: called within an AVX2 function.
    unsafe { if T::SIMD == SimdKind::I32 { _mm256_set1_epi32((pk as u32 ^ 0x8000_0000) as i32) } else { _mm256_set1_epi64x((pk ^ 0x8000_0000_0000_0000) as i64) } }
}
/// The reference key for the mask, in the domain the mask is folded in.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn key0_vec<T: Elem>(e: &T) -> __m256i {
    // SAFETY: called within an AVX2 function.
    unsafe {
        match T::SIMD {
            SimdKind::I32 => _mm256_set1_epi32(skey32(e)),
            SimdKind::I64 => _mm256_set1_epi64x(skey64(e)),
            _ => _mm256_set1_epi64x(T::radix_key(*e, 0).to_u64() as i64),
        }
    }
}
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn fold_mask<T: Elem>(vmask: __m256i) -> u64 {
    // SAFETY: called within an AVX2 function.
    unsafe {
        let mut l = [0u64; 4];
        // SAFETY: l has 32 bytes.
        unsafe { _mm256_storeu_si256(l.as_mut_ptr() as *mut __m256i, vmask) };
        if T::SIMD == SimdKind::I32 {
            ((l[0] | l[1] | l[2] | l[3]) as u32) as u64 // int32 keys in the low half of every 64-bit lane
        } else {
            l[0] | l[2] // 64-bit keys in lanes 0 and 2
        }
    }
}
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn compress_store<T>(dst: *mut T, v: __m256i, mask: u32, lanes4: bool) {
    // SAFETY: called within an AVX2 function.
    unsafe {
        let idx = if lanes4 { &LUT.by4[mask as usize] } else { &LUT.by2[mask as usize] };
        // SAFETY: the caller guarantees 32 writable bytes at dst.
        unsafe { _mm256_storeu_si256(dst as *mut __m256i, _mm256_permutevar8x32_epi32(v, _mm256_loadu_si256(idx.as_ptr() as *const __m256i))) };
    }
}

/// Forward split of a[0,n) with AVX2, then the scalar loop for the rest.
/// Elements are compressed to the front of a vector with a permutation
/// looked up by the compare mask, and the whole vector is stored: the lanes
/// past the selected elements are junk that lands on dead space.
///
/// # Safety
/// AVX2 is available; `a` holds `n` elements, `buf` `cap`.
#[target_feature(enable = "avx2")]
pub unsafe fn split_forward_avx2<V: Arr>(a: V, buf: V, n: usize, cap: usize, pivot: V::T, n_ge: &mut usize, mask: Option<&mut u64>) -> bool {
    type K<V> = <<V as Arr>::T as Elem>::Key;
    let v_per = 32 / core::mem::size_of::<V::T>();
    let pa = a.data();
    let pb = buf.data();
    let src = a.data() as *const V::T;
    // SAFETY: the caller's contract; the vector loop keeps b + V <= cap and
    // w <= i so every 32-byte store lands on own slots or dead space.
    unsafe {
        let vp = pivot_vec_key::<V::T>(<V::T as Elem>::radix_key(pivot, 0).to_u64());
        let vk0 = key0_vec::<V::T>(&*src);
        let mut vmask = _mm256_setzero_si256();
        let (mut w, mut b, mut i) = (0usize, 0usize, 0usize);
        while i + v_per <= n && b + v_per <= cap {
            let v = _mm256_loadu_si256(src.add(i) as *const __m256i);
            let lt = lt_mask::<V::T>(v, vp, vk0, &mut vmask);
            let ge = !lt & ((1u32 << v_per) - 1);
            compress_store(pa.add(w), v, lt, v_per == 4);
            compress_store(pb.add(b), v, ge, v_per == 4);
            w += lt.count_ones() as usize;
            b += ge.count_ones() as usize;
            i += v_per;
        }
        let mut m = K::<V>::from_u64(fold_mask::<V::T>(vmask));
        // Scalar rest: same predicate on the keys, same stretch logic.
        let pk = <V::T as Elem>::radix_key(pivot, 0);
        let k0 = <V::T as Elem>::radix_key(*src, 0);
        while i < n {
            let end = n.min(i + (cap - b));
            if end == i {
                break;
            }
            while i < end {
                let e = *src.add(i);
                let k = <V::T as Elem>::radix_key(e, 0);
                let g = (k >= pk) as usize;
                m = m | (k ^ k0);
                *pa.add(w) = e;
                *pb.add(b) = e;
                w += 1 - g;
                b += g;
                i += 1;
            }
        }
        if let Some(mask) = mask {
            *mask |= m.to_u64();
        }
        if i < n {
            crate::view::copy_forward(buf, 0, a, w, b);
            return false;
        }
        *n_ge = b;
        true
    }
}

// ---- the partition sort -----------------------------------------------------------------

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn cmp_domain<T: Elem>(v: __m256i) -> __m256i {
    // SAFETY: called within an AVX2 function.
    unsafe { if T::SIMD == SimdKind::F64 { _mm256_xor_si256(f64_order_key(v), _mm256_set1_epi64x(0x8000_0000_0000_0000u64 as i64)) } else { v } }
}
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn eq_keys<T: Elem>(x: __m256i, y: __m256i) -> __m256i {
    // SAFETY: called within an AVX2 function.
    unsafe { if T::SIMD == SimdKind::I32 { _mm256_cmpeq_epi32(x, y) } else { _mm256_cmpeq_epi64(x, y) } }
}

/// First pass with AVX2, for 8-byte elements (four per vector): the forward
/// split with the class counts and the membership check folded in. Same
/// overflow contract as split2_scalar.
///
/// # Safety
/// AVX2 is available; `a` holds `n` elements, `buf` `cap`.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub unsafe fn split2_avx2<V: Arr>(
    a: V,
    buf: V,
    n: usize,
    cap: usize,
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
    type K<V> = <<V as Arr>::T as Elem>::Key;
    let v_per = 32 / core::mem::size_of::<V::T>();
    let pa = a.data();
    let pb = buf.data();
    let src = a.data() as *const V::T;
    // SAFETY: as split_forward_avx2.
    unsafe {
        let vp1 = pivot_vec_key::<V::T>(t1.to_u64());
        let vp2 = pivot_vec_key::<V::T>(t2.to_u64());
        let vp3 = pivot_vec_key::<V::T>(t3.to_u64());
        let e0 = pivot_vec_key::<V::T>(vk[0].to_u64());
        let e1 = pivot_vec_key::<V::T>(vk[1].to_u64());
        let e2 = pivot_vec_key::<V::T>(vk[2].to_u64());
        let e3 = pivot_vec_key::<V::T>(vk[3].to_u64());
        let vk0 = key0_vec::<V::T>(&*src);
        let (mut vmask, mut junk, mut vbad) = (_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256());
        let (mut w, mut b, mut i, mut c1, mut c3) = (0usize, 0usize, 0usize, 0usize, 0usize);
        while i + v_per <= n && b + v_per <= cap {
            let v = _mm256_loadu_si256(src.add(i) as *const __m256i);
            let lt = lt_mask::<V::T>(v, vp2, vk0, &mut vmask);
            let ge = !lt & ((1u32 << v_per) - 1);
            c1 += lt_mask::<V::T>(v, vp1, vk0, &mut junk).count_ones() as usize;
            c3 += lt_mask::<V::T>(v, vp3, vk0, &mut junk).count_ones() as usize;
            let d = cmp_domain::<V::T>(v);
            let eq = _mm256_or_si256(_mm256_or_si256(eq_keys::<V::T>(d, e0), eq_keys::<V::T>(d, e1)), _mm256_or_si256(eq_keys::<V::T>(d, e2), eq_keys::<V::T>(d, e3)));
            vbad = _mm256_or_si256(vbad, _mm256_andnot_si256(eq, _mm256_set1_epi32(-1)));
            compress_store(pa.add(w), v, lt, v_per == 4);
            compress_store(pb.add(b), v, ge, v_per == 4);
            w += lt.count_ones() as usize;
            b += ge.count_ones() as usize;
            i += v_per;
        }
        let mut m = K::<V>::from_u64(fold_mask::<V::T>(vmask));
        let mut unknown = fold_mask::<V::T>(vbad) != 0;
        let k0 = <V::T as Elem>::radix_key(*src, 0);
        while i < n {
            let end = n.min(i + (cap - b));
            if end == i {
                break;
            }
            while i < end {
                let e = *src.add(i);
                let k = <V::T as Elem>::radix_key(e, 0);
                let g = (k >= t2) as usize;
                m = m | (k ^ k0);
                unknown |= (k != vk[0]) & (k != vk[1]) & (k != vk[2]) & (k != vk[3]);
                c1 += (k < t1) as usize;
                c3 += (k < t3) as usize;
                *pa.add(w) = e;
                *pb.add(b) = e;
                w += 1 - g;
                b += g;
                i += 1;
            }
        }
        *xm = m.to_u64();
        *known = !unknown;
        if i < n {
            crate::view::copy_forward(buf, 0, a, w, b);
            return false;
        }
        *n_ge = b;
        *c0_lt = c1;
        *c0_ge = c3 - w;
        true
    }
}

/// Second pass with AVX2 for 8-byte elements: compress-store to two
/// destinations inside one array. The vector loop runs only while both
/// cursors have a full vector of room; the scalar loops finish.
///
/// # Safety
/// AVX2 is available; `src` holds `m` elements; `[o0, end1)` is inside `dst`.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub unsafe fn partition2_avx2<T: Elem>(src: *const T, m: usize, dst: *mut T, o0: usize, o1: usize, end1: usize, t: T::Key) {
    let v_per = 32 / core::mem::size_of::<T>();
    // SAFETY: the caller's contract; the vector loop keeps both cursors a
    // full vector inside their regions.
    unsafe {
        let vp = pivot_vec_key::<T>(t.to_u64());
        let vk0 = key0_vec::<T>(&*src);
        let mut junk = _mm256_setzero_si256();
        let (mut i, mut c0, mut c1) = (0usize, o0, o1);
        while i + v_per <= m && c0 + v_per <= o1 && c1 + v_per <= end1 {
            let v = _mm256_loadu_si256(src.add(i) as *const __m256i);
            let lt = lt_mask::<T>(v, vp, vk0, &mut junk);
            let ge = !lt & ((1u32 << v_per) - 1);
            compress_store(dst.add(c0), v, lt, v_per == 4);
            compress_store(dst.add(c1), v, ge, v_per == 4);
            c0 += lt.count_ones() as usize;
            c1 += ge.count_ones() as usize;
            i += v_per;
        }
        while i < m && c0 < o1 && c1 < end1 {
            let e = *src.add(i);
            let g = (T::radix_key(e, 0) >= t) as usize;
            *dst.add(c0) = e;
            *dst.add(c1) = e;
            c0 += 1 - g;
            c1 += g;
            i += 1;
        }
        while i < m {
            let e = *src.add(i);
            if T::radix_key(e, 0) < t {
                *dst.add(c0) = e;
                c0 += 1;
            } else {
                *dst.add(c1) = e;
                c1 += 1;
            }
            i += 1;
        }
    }
}

// ---- prefix sums ----------------------------------------------------------------------------

/// Exclusive prefix sums of `n` u32 counters, 8 per step (`n` a multiple of 8).
///
/// # Safety
/// AVX2 is available; `h` holds `n` entries.
#[target_feature(enable = "avx2")]
pub unsafe fn prefix_avx2_u32(h: *mut u32, n: usize) {
    // SAFETY: the caller's contract.
    unsafe {
        let mut carry = _mm256_setzero_si256();
        let mut i = 0;
        while i < n {
            let inp = _mm256_loadu_si256(h.add(i) as *const __m256i);
            let mut x = inp;
            x = _mm256_add_epi32(x, _mm256_slli_si256::<4>(x));
            x = _mm256_add_epi32(x, _mm256_slli_si256::<8>(x));
            let lo = _mm256_shuffle_epi32::<0xFF>(x);
            let lo = _mm256_permute2x128_si256::<0x08>(lo, lo);
            x = _mm256_add_epi32(x, lo);
            let incl = _mm256_add_epi32(x, carry);
            _mm256_storeu_si256(h.add(i) as *mut __m256i, _mm256_sub_epi32(incl, inp));
            carry = _mm256_permute4x64_epi64::<0xFF>(_mm256_shuffle_epi32::<0xFF>(incl));
            i += 8;
        }
    }
}

// ---- PEXT ---------------------------------------------------------------------------------------

/// Only the varying bits, compressed to the bottom (BMI2).
#[derive(Clone, Copy)]
pub struct PextKey<K> {
    chunk: i32,
    mask: K,
}
impl<V: Arr> KeyFn<V> for PextKey<<V::T as Elem>::Key> {
    #[inline(always)]
    fn key(&self, x: V::T) -> <V::T as Elem>::Key {
        <Self as KeyFn<V>>::raw(self, V::key(x, self.chunk))
    }
    #[inline(always)]
    fn raw(&self, k: <V::T as Elem>::Key) -> <V::T as Elem>::Key {
        // SAFETY: the plan allowed PEXT only after BMI2 was detected; the
        // call is inlined into a BMI2 function.
        unsafe {
            if <V::T as Elem>::Key::BITS == 64 {
                RadixKey::from_u64(_pext_u64(k.to_u64(), self.mask.to_u64()))
            } else {
                RadixKey::from_u64(_pext_u32(k.to_u64() as u32, self.mask.to_u64() as u32) as u64)
            }
        }
    }
}

/// The radix under a PEXT plan, compiled for BMI2 so the key functor is
/// inlined with the instruction.
///
/// # Safety
/// BMI2 is available.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "bmi2")]
pub unsafe fn radix_exec_pext<V: Arr>(
    src: V,
    dst: V,
    n: usize,
    chunk: i32,
    p: &RadixPlan,
    domain_mask: u64,
    result_in_src: bool,
    scratch: &mut Scratch<V>,
    xm_out: &mut u64,
    free: bool,
    ended_in_src: &mut bool,
) -> Result<bool, AllocError> {
    type K<V> = <<V as Arr>::T as Elem>::Key;
    crate::algorithm::radix_route::radix_exec(src, dst, n, chunk, p, PextKey { chunk, mask: K::<V>::from_u64(p.pmask) }, domain_mask, result_in_src, scratch, xm_out, free, ended_in_src)
}
#[allow(dead_code)]
fn _unused(_: HistStore) {}

// ---- the prescan of plain slices ----------------------------------------------------------

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn vgt(kind: Prescan, x: __m256i, y: __m256i) -> u32 {
    // SAFETY: called within an AVX2 function.
    unsafe {
        match kind {
            Prescan::F32 => _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_castsi256_ps(x), _mm256_castsi256_ps(y))) as u32,
            Prescan::F64 => _mm256_movemask_pd(_mm256_cmp_pd::<_CMP_GT_OQ>(_mm256_castsi256_pd(x), _mm256_castsi256_pd(y))) as u32,
            Prescan::I32 => _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(x, y))) as u32,
            Prescan::U32 => {
                let s = _mm256_set1_epi32(i32::MIN);
                _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_xor_si256(x, s), _mm256_xor_si256(y, s)))) as u32
            }
            Prescan::I64 => _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpgt_epi64(x, y))) as u32,
            Prescan::U64 => {
                let s = _mm256_set1_epi64x(i64::MIN);
                _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpgt_epi64(_mm256_xor_si256(x, s), _mm256_xor_si256(y, s)))) as u32
            }
            Prescan::None => 0,
        }
    }
}
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn vnan(kind: Prescan, x: __m256i) -> bool {
    // SAFETY: called within an AVX2 function.
    unsafe {
        match kind {
            Prescan::F32 => _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_UNORD_Q>(_mm256_castsi256_ps(x), _mm256_castsi256_ps(x))) != 0,
            Prescan::F64 => _mm256_movemask_pd(_mm256_cmp_pd::<_CMP_UNORD_Q>(_mm256_castsi256_pd(x), _mm256_castsi256_pd(x))) != 0,
            _ => false,
        }
    }
}
/// A plain array of 32- or 64-bit numbers: eight or four elements per
/// vector compared against their predecessors, the compare bits counted;
/// NaN is flagged from the vector compare too, and sends the input to the
/// records, whose order defines the place of NaN.
///
/// # Safety
/// AVX2 is available; `p` holds `n >= 2` elements of the layout `kind`.
#[target_feature(enable = "avx2")]
pub unsafe fn prescan_avx2(kind: Prescan, p: *const u8, n: usize) -> PrescanResult {
    let esz = match kind {
        Prescan::I32 | Prescan::U32 | Prescan::F32 => 4usize,
        _ => 8,
    };
    let v_per = 32 / esz;
    let mut r = PrescanResult::default();
    let (mut desc, mut asc, mut i, mut next_check) = (0usize, 0usize, 1usize, 256usize);
    let mut nan = false;
    // SAFETY: the caller's contract; every load is inside the slice.
    unsafe {
        let scalar = |a: usize, b: usize, nan: &mut bool| -> core::cmp::Ordering {
            // a < b  by the natural order, NaN flagged
            match kind {
                Prescan::I32 => (*(p.add(a * 4) as *const i32)).cmp(&*(p.add(b * 4) as *const i32)),
                Prescan::U32 => (*(p.add(a * 4) as *const u32)).cmp(&*(p.add(b * 4) as *const u32)),
                Prescan::I64 => (*(p.add(a * 8) as *const i64)).cmp(&*(p.add(b * 8) as *const i64)),
                Prescan::U64 => (*(p.add(a * 8) as *const u64)).cmp(&*(p.add(b * 8) as *const u64)),
                Prescan::F32 => {
                    let (x, y) = (*(p.add(a * 4) as *const f32), *(p.add(b * 4) as *const f32));
                    *nan |= x.is_nan() | y.is_nan();
                    x.partial_cmp(&y).unwrap_or(core::cmp::Ordering::Equal)
                }
                _ => {
                    let (x, y) = (*(p.add(a * 8) as *const f64), *(p.add(b * 8) as *const f64));
                    *nan |= x.is_nan() | y.is_nan();
                    x.partial_cmp(&y).unwrap_or(core::cmp::Ordering::Equal)
                }
            }
        };
        if matches!(kind, Prescan::F32 | Prescan::F64) {
            let _ = scalar(0, 0, &mut nan);
        }
        while i + v_per <= n {
            let cur = _mm256_loadu_si256(p.add(i * esz) as *const __m256i);
            let prev = _mm256_loadu_si256(p.add((i - 1) * esz) as *const __m256i);
            desc += vgt(kind, prev, cur).count_ones() as usize;
            asc += vgt(kind, cur, prev).count_ones() as usize;
            nan |= vnan(kind, cur);
            if i >= next_check {
                next_check += 256;
                if asc > 0 && desc > (i >> 3) + 64 {
                    return r;
                }
            }
            i += v_per;
        }
        while i < n {
            let c = scalar(i - 1, i, &mut nan);
            desc += (c == core::cmp::Ordering::Greater) as usize;
            asc += (c == core::cmp::Ordering::Less) as usize;
            i += 1;
        }
    }
    if nan {
        return r;
    }
    r.descents = desc;
    r.ascents = asc;
    r.shape = if desc == 0 {
        Shape::Sorted
    } else if asc == 0 {
        Shape::Reversed
    } else if desc <= n / 16 {
        Shape::NearlySorted
    } else {
        Shape::Unordered
    };
    r
}

//! Key inference for the comparator sort: `sort_by_inferred`.
//!
//! A comparator is a black box, but the elements are not. On a plain
//! element (`PlainBytes`: every byte initialised) the sort guesses that the
//! comparator is the order of some aligned window of the element's bytes,
//! read as an integer or a float, ascending or descending. Every such
//! window is tested against a strided sample of adjacent-pair outcomes of
//! the comparator; a window that agrees with all of them is sorted by the
//! record path, the result is gathered into a buffer and verified with one
//! sequential comparator pass: one call per pair of neighbours with the
//! same window key (the stable sort left them in index order), the
//! three-way outcome where the key changes. A class of the comparator
//! that spans several window keys is sorted by original index, which
//! restores stability even when the window is finer than the comparator.
//! Only a comparator that disagrees in direction somewhere fails the
//! guess; the input is then still untouched and goes to the comparison
//! sort.
use crate::algorithm::{Scratch, brainsort_impl};
use crate::api::{
    Buf, COMPARATOR_ROUTE_MAX, ELEMENT_ROUTE_MAX, INFER_MIN, PrescanResult, SMALL_SORT, Shape, comparison_sort, ord3, prescan_cmp, reverse_stable, small_sort, sort_by_indices, sort_displaced_elements,
};
use crate::key::{f32_radix, f64_radix};
use crate::record::{Rec32, Rec64, Record};
use crate::view::{Alloc, AllocError, NoHooks, View};
use core::cmp::Ordering;
use core::ptr;

/// An element whose every byte is initialised, so that any window of its
/// bytes can be read as an integer or a float: what [`sort_by_inferred`]
/// needs. Implemented for the primitive numbers, `bool`, `char` and arrays
/// of these; a struct opts in with `unsafe impl` once it has no padding
/// (`#[repr(C)]` fields whose sizes add up to the size of the struct, or
/// `#[repr(C, packed)]`).
///
/// # Safety
/// No value of the type may hold a padding or otherwise uninitialised
/// byte.
///
/// [`sort_by_inferred`]: crate::sort_by_inferred
pub unsafe trait PlainBytes: Copy {}
macro_rules! plain {
    ($($t:ty),*) => {$(
        // SAFETY: a primitive number has no padding.
        unsafe impl PlainBytes for $t {}
    )*};
}
plain!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, isize, usize, f32, f64, bool, char);
// SAFETY: an array of plain elements has no padding between them.
unsafe impl<T: PlainBytes, const N: usize> PlainBytes for [T; N] {}

/// Adjacent pairs the comparator is asked about before a window is chosen.
const SAMPLE_PAIRS: usize = 2048;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    I64,
    U64,
    F64,
}
const KINDS: [Kind; 10] = [Kind::I64, Kind::U64, Kind::F64, Kind::I32, Kind::U32, Kind::F32, Kind::I16, Kind::U16, Kind::I8, Kind::U8];
impl Kind {
    fn size(self) -> usize {
        match self {
            Kind::I8 | Kind::U8 => 1,
            Kind::I16 | Kind::U16 => 2,
            Kind::I32 | Kind::U32 | Kind::F32 => 4,
            Kind::I64 | Kind::U64 | Kind::F64 => 8,
        }
    }
    fn wide(self) -> bool {
        self.size() == 8
    }
    fn is_float(self) -> bool {
        matches!(self, Kind::F32 | Kind::F64)
    }
    fn is_signed(self) -> bool {
        matches!(self, Kind::I8 | Kind::I16 | Kind::I32 | Kind::I64)
    }
}

/// A candidate key: a window of the element read as `kind`, possibly descending.
#[derive(Clone, Copy, Debug)]
struct Cand {
    off: usize,
    kind: Kind,
    desc: bool,
}
impl Cand {
    /// The order-preserving radix value of the window of the element at `p`:
    /// 32 bits wide for the narrow kinds, 64 for the wide ones.
    ///
    /// # Safety
    /// `p` points at a plain element of at least `off + kind.size()` bytes.
    #[inline(always)]
    unsafe fn radix(self, p: *const u8) -> u64 {
        // SAFETY: the caller's contract; the reads are unaligned by design.
        let r = unsafe {
            let q = p.add(self.off);
            match self.kind {
                Kind::I8 => ((ptr::read_unaligned(q as *const i8) as i32) as u32) ^ 0x8000_0000,
                Kind::U8 => ptr::read_unaligned(q) as u32,
                Kind::I16 => ((ptr::read_unaligned(q as *const i16) as i32) as u32) ^ 0x8000_0000,
                Kind::U16 => ptr::read_unaligned(q as *const u16) as u32,
                Kind::I32 => (ptr::read_unaligned(q as *const i32) as u32) ^ 0x8000_0000,
                Kind::U32 => ptr::read_unaligned(q as *const u32),
                Kind::F32 => f32_radix(ptr::read_unaligned(q as *const f32)),
                Kind::I64 => return self.finish64((ptr::read_unaligned(q as *const i64) as u64) ^ 0x8000_0000_0000_0000),
                Kind::U64 => return self.finish64(ptr::read_unaligned(q as *const u64)),
                Kind::F64 => return self.finish64(f64_radix(ptr::read_unaligned(q as *const f64))),
            }
        };
        (if self.desc { !r } else { r }) as u64
    }
    #[inline(always)]
    fn finish64(self, r: u64) -> u64 {
        if self.desc { !r } else { r }
    }
    /// A float window is believed only if no sampled value is a denormal:
    /// integers read as floats are denormals, floats in use never are.
    ///
    /// # Safety
    /// As `radix`, for every sampled element.
    unsafe fn float_plausible<T>(self, v: &[T], stride: usize) -> bool {
        let p = v.as_ptr() as *const u8;
        let sz = core::mem::size_of::<T>();
        let mut i = 0;
        while i < v.len() {
            // SAFETY: i < len, the window is inside the element.
            let denormal = unsafe {
                let q = p.add(i * sz + self.off);
                match self.kind {
                    Kind::F32 => {
                        let b = ptr::read_unaligned(q as *const u32);
                        b & 0x7F80_0000 == 0 && b & 0x007F_FFFF != 0
                    }
                    _ => {
                        let b = ptr::read_unaligned(q as *const u64);
                        b & 0x7FF0_0000_0000_0000 == 0 && b & 0x000F_FFFF_FFFF_FFFF != 0
                    }
                }
            };
            if denormal {
                return false;
            }
            i += stride;
        }
        true
    }
}

/// The window that agrees with every sampled comparator outcome, if any;
/// among several, the widest, then a plausible float, then signed, then
/// unsigned.
fn infer<T: PlainBytes, F: FnMut(&T, &T) -> Ordering>(v: &[T], cmp: &mut F) -> Option<Cand> {
    let n = v.len();
    let sz = core::mem::size_of::<T>();
    if sz == 0 || n < 2 {
        return None;
    }
    let pairs = SAMPLE_PAIRS.min(n - 1);
    let stride = (n - 1) / pairs;
    // the sampled outcomes: pair (i, i + 1) for i = k * stride
    let mut outcomes = alloc::vec::Vec::with_capacity(pairs);
    for k in 0..pairs {
        let i = k * stride;
        outcomes.push((i, cmp(&v[i], &v[i + 1])));
    }
    if outcomes.iter().all(|o| o.1 == Ordering::Equal) {
        return None;
    }
    let p = v.as_ptr() as *const u8;
    let mut best: Option<(u32, Cand)> = None;
    for kind in KINDS {
        let ks = kind.size();
        if ks > sz {
            continue;
        }
        if best.is_some_and(|(_, b)| ks < b.kind.size()) {
            break; // a narrower window cannot win
        }
        let mut off = 0;
        while off + ks <= sz {
            for desc in [false, true] {
                let c = Cand { off, kind, desc };
                // SAFETY: every sampled index is below n - 1; the window is inside the element.
                let agrees = outcomes.iter().all(|&(i, o)| unsafe { c.radix(p.add(i * sz)).cmp(&c.radix(p.add((i + 1) * sz))) == o });
                if agrees {
                    // SAFETY: as above.
                    let score = (ks as u32) * 10
                        + if kind.is_float() {
                            if unsafe { c.float_plausible(v, stride.max(1)) } { 3 } else { 0 }
                        } else if kind.is_signed() {
                            2
                        } else {
                            1
                        };
                    if best.is_none_or(|(s, _)| score > s) {
                        best = Some((score, c));
                    }
                }
            }
            off += ks;
        }
    }
    best.map(|(_, c)| c)
}

/// Sorts `v` by the candidate through the records, verifies the gathered
/// result with the comparator and restores stability inside the
/// comparator's classes. Ok(false) with `v` untouched when the comparator
/// disagrees with the candidate somewhere.
fn sort_verified<T: PlainBytes, A: Alloc, R: Record, F: FnMut(&T, &T) -> Ordering>(v: &mut [T], c: Cand, cmp: &mut F, make: impl Fn(u64, u32) -> R) -> Result<bool, AllocError> {
    let n = v.len();
    let sz = core::mem::size_of::<T>();
    let recs = Buf::<R, A>::new(n)?;
    let rec = recs.ptr();
    let p = v.as_ptr() as *const u8;
    for i in 0..n {
        // SAFETY: i < n elements and records.
        unsafe { ptr::write(rec.add(i), make(c.radix(p.add(i * sz)), i as u32)) };
    }
    let mut scratch = Scratch::<View<R, NoHooks, A>>::new();
    let in_src = brainsort_impl(View::<R, NoHooks, A>::new(rec, n), &mut scratch, true, 0)?;
    let rec: *mut R = if in_src { rec } else { scratch.buffer().expect("the result is in the scratch buffer") };
    let tmp = Buf::<T, A>::new(n)?;
    let t = tmp.ptr();
    let first = v.as_mut_ptr();
    // SAFETY: the record indices are a permutation of 0..n; T is Copy, so
    // the gathered copies need no drop and the elements stay valid.
    unsafe {
        for i in 0..n {
            ptr::copy_nonoverlapping(first.add((*rec.add(i)).index() as usize), t.add(i), 1);
        }
        let out = core::slice::from_raw_parts_mut(t, n);
        // A class [s, e) of the comparator that spans several window keys:
        // its elements go into index order.
        let fix_class = |out: &mut [T], s: usize, e: usize| {
            let run = core::slice::from_raw_parts_mut(rec.add(s), e - s);
            run.sort_unstable_by_key(|r| r.index());
            for (k, r) in run.iter().enumerate() {
                ptr::copy_nonoverlapping(first.add(r.index() as usize), out.as_mut_ptr().add(s + k), 1);
            }
        };
        // Neighbours with one window key are in index order already and
        // only have to not descend; where the key changes the outcome
        // decides whether a class ends (less), the guess failed (greater),
        // or a class spans two keys (equal). `s` is the start of the current
        // class while it spans keys, otherwise the last key change that
        // ended a class: the true start is then found by walking back over
        // the same-key pairs.
        let mut s = 0;
        let mut mixed = false;
        for i in 1..n {
            if !mixed && R::compare(*rec.add(i - 1), *rec.add(i)) == 0 {
                if cmp(&out[i - 1], &out[i]) == Ordering::Greater {
                    return Ok(false);
                }
                continue;
            }
            match cmp(&out[i - 1], &out[i]) {
                Ordering::Less => {
                    if mixed {
                        fix_class(out, s, i);
                        mixed = false;
                    }
                    s = i;
                }
                Ordering::Greater => return Ok(false),
                Ordering::Equal => {
                    if !mixed {
                        mixed = true;
                        let mut j = i - 1;
                        while j > s && cmp(&out[j - 1], &out[j]) != Ordering::Less {
                            j -= 1;
                        }
                        s = j;
                    }
                }
            }
        }
        if mixed {
            fix_class(out, s, n);
        }
        ptr::copy_nonoverlapping(t, first, n);
    }
    Ok(true)
}

/// The comparator sort with key inference: the ordered-input routes of
/// `sort_by`, then the inferred window, then the comparison sort.
pub fn sort_by_inferred_impl<T: PlainBytes, A: Alloc, F: FnMut(&T, &T) -> Ordering>(v: &mut [T], mut cmp: F) {
    let n = v.len();
    if n < 2 {
        return;
    }
    if n <= SMALL_SORT {
        small_sort(v, |a, b| cmp(a, b) == Ordering::Less);
        return;
    }
    let s: PrescanResult = prescan_cmp(v, &mut cmp);
    if s.shape == Shape::Sorted {
        return;
    }
    if s.shape == Shape::Reversed {
        reverse_stable(v, &s, |a, b| cmp(a, b) == Ordering::Equal);
        return;
    }
    let nearly = s.shape == Shape::NearlySorted;
    let indexed = n <= u32::MAX as usize;
    let size = core::mem::size_of::<T>();
    // nearly sorted input: the displaced elements, in place or on indices
    if nearly && sort_displaced_elements::<T, A, _>(v, |a, b| ord3(cmp(a, b))) {
        return;
    }
    if nearly && indexed && size > COMPARATOR_ROUTE_MAX && sort_by_indices::<T, A, F>(v, &mut cmp, true, true, false) {
        return;
    }
    if indexed && n >= INFER_MIN {
        if let Some(c) = infer(v, &mut cmp) {
            let done = if c.kind.wide() {
                sort_verified::<T, A, Rec64, F>(v, c, &mut cmp, |r, i| Rec64 { key: (r ^ 0x8000_0000_0000_0000) as i64, idx: i, pad: 0 })
            } else {
                sort_verified::<T, A, Rec32, F>(v, c, &mut cmp, |r, i| Rec32 { key: ((r as u32) ^ 0x8000_0000) as i32, idx: i })
            };
            if let Ok(true) = done {
                return;
            }
        }
    }
    if indexed && size > ELEMENT_ROUTE_MAX && sort_by_indices::<T, A, F>(v, &mut cmp, nearly, false, true) {
        return;
    }
    comparison_sort::<T, A, F>(v, &mut cmp);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::DefaultAlloc;
    use alloc::vec::Vec;

    #[derive(Clone, Copy, Debug, PartialEq)]
    #[repr(C)]
    struct Tagged {
        key: i32,
        idx: u32,
    }
    unsafe impl PlainBytes for Tagged {}

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *seed >> 11
    }
    fn tagged(n: usize, modulus: u64) -> Vec<Tagged> {
        let mut s = 99u64;
        (0..n).map(|i| Tagged { key: (lcg(&mut s) % modulus) as i32 - (modulus / 2) as i32, idx: i as u32 }).collect()
    }
    fn sort<T: PlainBytes, F: FnMut(&T, &T) -> Ordering>(v: &mut [T], cmp: F) {
        sort_by_inferred_impl::<T, DefaultAlloc, F>(v, cmp)
    }
    /// The result of the inferred sort equals the standard library's stable sort.
    fn same_as_std<T: PlainBytes + PartialEq + core::fmt::Debug>(v: &[T], cmp: impl Fn(&T, &T) -> Ordering) {
        let mut a = v.to_vec();
        let mut b = v.to_vec();
        sort(&mut a, &cmp);
        b.sort_by(&cmp);
        assert_eq!(a, b);
    }

    #[test]
    fn exact_key_with_ties_is_stable() {
        same_as_std(&tagged(20_000, 50), |a, b| a.key.cmp(&b.key));
        same_as_std(&tagged(20_000, 1 << 31), |a, b| a.key.cmp(&b.key));
    }
    #[test]
    fn coarse_comparator_is_stable() {
        // the window (the whole key) is finer than the comparator: equal runs go back to input order
        same_as_std(&tagged(20_000, 1 << 31), |a, b| (a.key / 1000).cmp(&(b.key / 1000)));
    }
    #[test]
    fn descending_and_unsigned_and_floats() {
        same_as_std(&tagged(20_000, 1 << 31), |a, b| b.key.cmp(&a.key));
        let mut s = 5u64;
        let u: Vec<u64> = (0..20_000).map(|_| lcg(&mut s) << 20).collect();
        same_as_std(&u, |a, b| a.cmp(b));
        let f: Vec<f64> = (0..20_000).map(|_| (lcg(&mut s) as f64 / (1u64 << 53) as f64 - 0.5) * 1e6).collect();
        same_as_std(&f, |a, b| a.total_cmp(b));
        same_as_std(&f, |a, b| b.partial_cmp(a).unwrap());
    }
    #[test]
    fn a_disagreeing_comparator_falls_back_correctly() {
        let mut v = tagged(20_000, 1 << 31);
        v[10_000].key = 12_345;
        let special = |k: i32| if k == 12_345 { i32::MAX } else { k };
        same_as_std(&v, |a, b| special(a.key).cmp(&special(b.key)));
    }
    #[test]
    fn small_and_ordered_inputs() {
        same_as_std(&tagged(10, 1 << 31), |a, b| a.key.cmp(&b.key));
        let mut sorted = tagged(5_000, 1 << 31);
        sorted.sort_by_key(|t| t.key);
        same_as_std(&sorted, |a, b| a.key.cmp(&b.key));
        sorted.reverse();
        same_as_std(&sorted, |a, b| a.key.cmp(&b.key));
    }
}

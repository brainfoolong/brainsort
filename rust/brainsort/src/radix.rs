//! The stable LSD radix core: digit plans, key functions, prefix sums and
//! the fused scatter passes. The port of `radix.hpp` (the parts brainsort
//! uses).
use crate::view::{Arr, Elem, Hooks, RadixKey, Stat, pext_soft};

/// 64-bit keys with 8-bit digits.
pub const MAX_PASSES: usize = 8;

/// The digits of a radix sort: shifts and widths, low digit first.
#[derive(Clone, Copy, Debug, Default)]
pub struct DigitPlan {
    /// Number of digits.
    pub passes: usize,
    /// Shift of each digit.
    pub shift: [u32; MAX_PASSES],
    /// Width of each digit.
    pub bits: [u32; MAX_PASSES],
    /// The widest digit.
    pub max_bits: u32,
}
impl DigitPlan {
    /// Entries of a count table: 2^max_bits.
    #[inline(always)]
    pub fn width(&self) -> usize {
        1usize << self.max_bits
    }
}

/// Splits `total_bits` into the fewest digits of at most `max_digit_bits`,
/// as evenly as possible (32 bits / 11 -> 11,11,10; 20 / 11 -> 10,10).
pub fn make_plan(total_bits: u32, max_digit_bits: u32) -> DigitPlan {
    let mut p = DigitPlan::default();
    if total_bits == 0 {
        return p;
    }
    p.passes = ((total_bits + max_digit_bits - 1) / max_digit_bits) as usize;
    let base = total_bits / p.passes as u32;
    let extra = (total_bits % p.passes as u32) as usize;
    let mut sh = 0;
    for i in 0..p.passes {
        let b = base + if i < extra { 1 } else { 0 };
        p.bits[i] = b;
        p.shift[i] = sh;
        sh += b;
        p.max_bits = p.max_bits.max(b);
    }
    p
}

/// A key function: element -> unsigned key for one chunk, through the view
/// type so the counted path can tally the string bytes a chunk extraction
/// loads. Each also maps an already computed radix key (`raw`) so a caller
/// that needs the raw key as well computes it once.
pub trait KeyFn<V: Arr>: Copy {
    /// The key of an element.
    fn key(&self, x: V::T) -> <V::T as Elem>::Key;
    /// The key from the raw radix key.
    fn raw(&self, k: <V::T as Elem>::Key) -> <V::T as Elem>::Key;
}
/// The full radix key.
#[derive(Clone, Copy)]
pub struct FullKey {
    /// The chunk.
    pub chunk: i32,
}
impl<V: Arr> KeyFn<V> for FullKey {
    #[inline(always)]
    fn key(&self, x: V::T) -> <V::T as Elem>::Key {
        V::key(x, self.chunk)
    }
    #[inline(always)]
    fn raw(&self, k: <V::T as Elem>::Key) -> <V::T as Elem>::Key {
        k
    }
}
/// The contiguous varying-bit range, shifted down.
#[derive(Clone, Copy)]
pub struct ShiftKey {
    /// The chunk.
    pub chunk: i32,
    /// Bits below `low` are constant.
    pub low: u32,
}
impl<V: Arr> KeyFn<V> for ShiftKey {
    #[inline(always)]
    fn key(&self, x: V::T) -> <V::T as Elem>::Key {
        V::key(x, self.chunk) >> self.low
    }
    #[inline(always)]
    fn raw(&self, k: <V::T as Elem>::Key) -> <V::T as Elem>::Key {
        k >> self.low
    }
}
/// Only the varying bits, compressed to the bottom (portable PEXT).
#[derive(Clone, Copy)]
pub struct PextKeySoft<K> {
    /// The chunk.
    pub chunk: i32,
    /// The varying bits.
    pub mask: K,
}
impl<V: Arr> KeyFn<V> for PextKeySoft<<V::T as Elem>::Key> {
    #[inline(always)]
    fn key(&self, x: V::T) -> <V::T as Elem>::Key {
        pext_soft(V::key(x, self.chunk), self.mask)
    }
    #[inline(always)]
    fn raw(&self, k: <V::T as Elem>::Key) -> <V::T as Elem>::Key {
        pext_soft(k, self.mask)
    }
}
/// Key minus a base (a narrow range that straddles a power of two).
#[derive(Clone, Copy)]
pub struct SubKey<K> {
    /// The chunk.
    pub chunk: i32,
    /// The base.
    pub base: K,
}
impl<V: Arr> KeyFn<V> for SubKey<<V::T as Elem>::Key> {
    #[inline(always)]
    fn key(&self, x: V::T) -> <V::T as Elem>::Key {
        V::key(x, self.chunk).wsub(self.base)
    }
    #[inline(always)]
    fn raw(&self, k: <V::T as Elem>::Key) -> <V::T as Elem>::Key {
        k.wsub(self.base)
    }
}

/// Exclusive prefix sums in place, scalar.
#[inline]
pub fn prefix_scalar(h: *mut u32, n: usize) {
    let mut sum = 0u32;
    for b in 0..n {
        // SAFETY: the caller passes a table of n entries.
        unsafe {
            let c = *h.add(b);
            *h.add(b) = sum;
            sum = sum.wrapping_add(c);
        }
    }
}
/// Exclusive prefix sums in place. Sizes are powers of two >= 256.
#[inline]
pub fn prefix_sums(h: *mut u32, n: usize) {
    #[cfg(all(target_arch = "x86_64", not(brainsort_no_simd)))]
    {
        // The vector version handles 8 entries per step; n is a power of
        // two, so n >= 16 means whole steps only.
        if n >= 16 && crate::cpu::have_avx2() {
            // SAFETY: AVX2 was detected; the table has n entries.
            unsafe { crate::simd::x86::prefix_avx2_u32(h, n) };
            return;
        }
    }
    prefix_scalar(h, n);
}

/// The live passes of a plan for a key mask: passes whose digit varies.
/// Returns how many; `order` receives their indices.
pub fn live_passes(plan: &DigitPlan, keymask: u64, order: &mut [usize; MAX_PASSES]) -> usize {
    let mut live = 0;
    for p in 0..plan.passes {
        let dm = (1u64 << plan.bits[p]) - 1;
        if (keymask >> plan.shift[p]) & dm != 0 {
            order[live] = p;
            live += 1;
        }
    }
    live
}

/// Scatter passes `order[q0..live)` of `s[0,n)` with `d` as the other
/// buffer. `tab[q0 & 1]` must hold the raw counts of pass `order[q0]`; the
/// counts of each following pass are built while the previous one scatters,
/// so the two tables alternate. The data ends up in `s` or `d` as
/// `result_in_src` asks, or, when `free` is set, wherever the last pass left
/// it; the return value says whether that is `s` either way.
///
/// # Safety
/// `tab[0]` and `tab[1]` point at `plan.width()` entries each (the same
/// table when only one pass is live).
#[inline(always)]
pub unsafe fn radix_passes<V: Arr, F: KeyFn<V>>(
    mut s: V,
    mut d: V,
    n: usize,
    plan: &DigitPlan,
    key: F,
    order: &[usize; MAX_PASSES],
    live: usize,
    q0: usize,
    tab: [*mut u32; 2],
    result_in_src: bool,
    free: bool,
) -> bool {
    let width = plan.width();
    let mut in_src = true;
    for q in q0..live {
        let p = order[q];
        let h = tab[q & 1];
        let shift = plan.shift[p];
        let mask = (1u32 << plan.bits[p]) - 1;
        if V::H::COUNTED {
            V::H::stat(Stat::RadixPasses, 1);
            V::H::on_table_sweep(h as *const u8, 1usize << plan.bits[p], 4, true, true);
        }
        prefix_sums(h, 1usize << plan.bits[p]);
        if q + 1 < live {
            let p2 = order[q + 1];
            let h2 = tab[(q + 1) & 1];
            let shift2 = plan.shift[p2];
            let mask2 = (1u32 << plan.bits[p2]) - 1;
            // SAFETY: h2 has width entries.
            unsafe { core::ptr::write_bytes(h2, 0, width) };
            if V::H::COUNTED {
                V::H::on_table_sweep(h2 as *const u8, width, 4, false, true);
            }
            for i in 0..n {
                let e = s.get(i);
                let u = key.key(e);
                let dg = (u >> shift).as_u32() & mask;
                let dg2 = (u >> shift2).as_u32() & mask2;
                if V::H::COUNTED {
                    V::H::on_table_rw(h.wrapping_add(dg as usize) as *const u8, 4);
                    V::H::on_table_rw(h2.wrapping_add(dg2 as usize) as *const u8, 4);
                }
                // SAFETY: dg < 2^bits <= width; dg2 likewise.
                unsafe {
                    let slot = h.add(dg as usize);
                    d.set(*slot as usize, e);
                    *slot += 1;
                    *h2.add(dg2 as usize) += 1;
                }
            }
        } else {
            for i in 0..n {
                let e = s.get(i);
                let dg = (key.key(e) >> shift).as_u32() & mask;
                if V::H::COUNTED {
                    V::H::on_table_rw(h.wrapping_add(dg as usize) as *const u8, 4);
                }
                // SAFETY: dg < width.
                unsafe {
                    let slot = h.add(dg as usize);
                    d.set(*slot as usize, e);
                    *slot += 1;
                }
            }
        }
        core::mem::swap(&mut s, &mut d);
        in_src = !in_src;
    }
    if free {
        return in_src;
    }
    if in_src != result_in_src {
        for i in 0..n {
            d.set(i, s.get(i));
        }
    }
    result_in_src
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    #[test]
    fn plan_splits_evenly() {
        let p = make_plan(32, 11);
        assert_eq!(p.passes, 3);
        assert_eq!(&p.bits[..3], &[11, 11, 10]);
        assert_eq!(&p.shift[..3], &[0, 11, 22]);
        let p = make_plan(20, 11);
        assert_eq!(&p.bits[..2], &[10, 10]);
        assert_eq!(make_plan(0, 11).passes, 0);
    }
    #[test]
    fn prefix_matches_naive() {
        let mut h: Vec<u32> = (0..256).map(|i| (i * 7 % 13) as u32).collect();
        let mut naive = h.clone();
        let mut sum = 0;
        for x in naive.iter_mut() {
            let c = *x;
            *x = sum;
            sum += c;
        }
        prefix_scalar(h.as_mut_ptr(), h.len());
        assert_eq!(h, naive);
    }
    #[test]
    fn pext_soft_compresses() {
        assert_eq!(pext_soft(0b1011_0110u32, 0b1111_0000), 0b1011);
        assert_eq!(pext_soft(0xF0F0u64, 0xFF00), 0xF0);
    }
    #[test]
    fn live_passes_skips_constant_digits() {
        let plan = make_plan(32, 11);
        let mut order = [0usize; MAX_PASSES];
        assert_eq!(live_passes(&plan, 0x0000_0FFF, &mut order), 2);
        assert_eq!(&order[..2], &[0, 1]);
        assert_eq!(live_passes(&plan, 0, &mut order), 0);
    }
}

//! The natural-run merge in front of the comparison sort: the port of the
//! run half of `detail/compsort.hpp`.
//!
//! A comparator gives the radix engine nothing to work with, so `sort_by`
//! falls back to a comparison sort. Before the standard library's stable
//! sort gets the slice, one pass finds the maximal non-descending runs
//! (strictly descending ones are reversed in place, which keeps equal
//! elements in order) and gives up as soon as the runs are too short to
//! pay. Long runs are merged bottom-up between the slice and a buffer of n
//! slots, so an input made of a few sorted pieces costs log(pieces) passes
//! of a merge that takes the next element without a branch on the
//! comparison (the side to take from is selected, not jumped to); a pair
//! of runs already in order is copied instead of merged.
//!
//! The C++ continues with its own stable quicksort when the runs are too
//! short; the crate hands the slice to the standard library's driftsort,
//! which was measured faster than the port of that quicksort on every
//! unordered input (decision 0018).
//!
//! Elements are moved bitwise between the slots of the slice and the
//! buffer and never dropped here: the slice owns them, and every element is
//! in exactly one slot of the slice when the merge returns, or unwinds out
//! of a comparator that panics (the source of the level in flight holds
//! all of them and is copied back, as the C++ catch block does). A
//! comparator that is not a strict weak order gets an unspecified order,
//! every element exactly once: every merge is counted, so no answer can
//! move a cursor past a run or take an element twice.
use crate::api::Buf;
use crate::view::Alloc;
use core::mem::MaybeUninit;
use core::ptr;

/// A slot of the slice or the buffer: an element moved bitwise.
type Slot<T> = MaybeUninit<T>;

/// Below this many elements the runs are not looked for.
pub(crate) const RUNS_MIN: usize = 512;
/// From this run length on, pairs of merges are split rather than paired.
const SPLIT_WIDTH: usize = 2048;

/// The element in a slot.
///
/// # Safety
/// The slot holds an element.
#[inline(always)]
unsafe fn el<'a, T>(p: *const Slot<T>) -> &'a T {
    // SAFETY: the caller's.
    unsafe { (*p).assume_init_ref() }
}
/// Copies `n` slots from `src` to `dst`, non-overlapping.
///
/// # Safety
/// Both ranges are valid and do not overlap.
#[inline(always)]
unsafe fn copy<T>(src: *const Slot<T>, dst: *mut Slot<T>, n: usize) {
    // SAFETY: the caller's.
    unsafe { ptr::copy_nonoverlapping(src, dst, n) }
}

// ---- the merges -------------------------------------------------------------------------------

/// A merge in progress: the two runs and the output cursor.
struct Merge<T> {
    l: *const Slot<T>,
    lend: *const Slot<T>,
    r: *const Slot<T>,
    rend: *const Slot<T>,
    out: *mut Slot<T>,
}
impl<T> Merge<T> {
    /// Copies whichever run is left to the output.
    ///
    /// # Safety
    /// The cursors are inside their runs and out has room.
    unsafe fn finish(&self) {
        // SAFETY: the caller's.
        unsafe {
            if self.l != self.lend {
                copy(self.l, self.out, self.lend.offset_from(self.l) as usize);
            }
            if self.r != self.rend {
                copy(self.r, self.out, self.rend.offset_from(self.r) as usize);
            }
        }
    }
}

/// A single merge from both ends, in counted stretches: the front takes
/// the smaller next element (the left one on a tie), the back the larger
/// last element (the right one on a tie), and the two chains of
/// compare-select-load overlap in the pipeline. A stretch is as many steps
/// as both ends can take without meeting or running a run dry; the tail is
/// merged from the front alone.
///
/// # Safety
/// The runs hold elements and out has room for both.
unsafe fn merge_one<T, F: FnMut(&T, &T) -> bool>(m: &mut Merge<T>, less: &mut F) {
    // SAFETY: every stretch is counted so no cursor leaves its run.
    unsafe {
        let (mut l, mut r, mut le, mut re) = (m.l, m.r, m.lend, m.rend);
        let mut o = m.out;
        let mut oe = o.add(le.offset_from(l) as usize + re.offset_from(r) as usize);
        loop {
            let (nl, nr) = (le.offset_from(l) as usize, re.offset_from(r) as usize);
            let k = nl.min(nr) / 2;
            if k == 0 {
                break;
            }
            for i in 0..k {
                let t = less(el(r), el(l));
                let u = less(el(re.sub(1)), el(le.sub(1)));
                copy(if t { r } else { l }, o.add(i), 1);
                copy(if u { le.sub(1) } else { re.sub(1) }, oe.sub(1 + i), 1);
                r = r.add(t as usize);
                l = l.add(!t as usize);
                le = le.sub(u as usize);
                re = re.sub(!u as usize);
            }
            o = o.add(k);
            oe = oe.sub(k);
        }
        loop {
            let (nl, nr) = (le.offset_from(l) as usize, re.offset_from(r) as usize);
            let k = nl.min(nr);
            if k == 0 {
                break;
            }
            for i in 0..k {
                let t = less(el(r), el(l));
                copy(if t { r } else { l }, o.add(i), 1);
                r = r.add(t as usize);
                l = l.add(!t as usize);
            }
            o = o.add(k);
        }
        *m = Merge { l, lend: le, r, rend: re, out: o };
        m.finish();
    }
}

/// Two merges advanced in lock step from the front, in counted stretches:
/// two independent chains in the pipeline, with fewer live cursors than
/// two merges from both ends would need. The tails go to merge_one.
///
/// # Safety
/// As for merge_one, for both.
unsafe fn merge_two<T, F: FnMut(&T, &T) -> bool>(a: &mut Merge<T>, b: &mut Merge<T>, less: &mut F) {
    // SAFETY: every stretch is counted so no cursor leaves its run.
    unsafe {
        let (mut al, mut ar, mut bl, mut br) = (a.l, a.r, b.l, b.r);
        let (mut ao, mut bo) = (a.out, b.out);
        loop {
            let k = (a.lend.offset_from(al) as usize).min(a.rend.offset_from(ar) as usize).min(b.lend.offset_from(bl) as usize).min(b.rend.offset_from(br) as usize);
            if k == 0 {
                break;
            }
            for i in 0..k {
                let ta = less(el(ar), el(al));
                let tb = less(el(br), el(bl));
                copy(if ta { ar } else { al }, ao.add(i), 1);
                copy(if tb { br } else { bl }, bo.add(i), 1);
                ar = ar.add(ta as usize);
                al = al.add(!ta as usize);
                br = br.add(tb as usize);
                bl = bl.add(!tb as usize);
            }
            ao = ao.add(k);
            bo = bo.add(k);
        }
        a.l = al;
        a.r = ar;
        a.out = ao;
        b.l = bl;
        b.r = br;
        b.out = bo;
        merge_one(a, less);
        merge_one(b, less);
    }
}

/// Splits the merge of L[0,p) and R[0,q) at output position k: the
/// smallest i (j = k - i) such that R[j-1] sorts before L[i], so that
/// L[0,i) and R[0,j) are exactly the first k elements of the stable merge.
/// A binary search of log(n) comparisons.
///
/// # Safety
/// The runs hold elements and k <= p + q.
unsafe fn merge_split<T, F: FnMut(&T, &T) -> bool>(big_l: *const Slot<T>, p: usize, big_r: *const Slot<T>, q: usize, k: usize, less: &mut F) -> usize {
    let (mut lo, mut hi) = (k.saturating_sub(q), k.min(p));
    while lo < hi {
        let i = lo + (hi - lo) / 2;
        // SAFETY: i < p and k - i - 1 < q.
        if !unsafe { less(el(big_r.add(k - i - 1)), el(big_l.add(i))) } {
            lo = i + 1;
        } else {
            hi = i;
        }
    }
    lo
}

/// One level of merging, two independent merges at a time. While the pairs
/// are short, two pairs advance together; once a pair is long, it is split
/// at its middle output position into two halves that advance together.
struct Level<T> {
    pending: Option<Merge<T>>,
}
impl<T> Level<T> {
    fn new() -> Self {
        Level { pending: None }
    }
    /// Merges src[lo,mid) and src[mid,hi), both sorted, into dst[lo,hi).
    ///
    /// # Safety
    /// src[lo,hi) holds elements and dst has room for them.
    unsafe fn add<F: FnMut(&T, &T) -> bool>(&mut self, src: *const Slot<T>, dst: *mut Slot<T>, lo: usize, mid: usize, hi: usize, less: &mut F) {
        // SAFETY: the caller's.
        unsafe {
            if mid == hi || !less(el(src.add(mid)), el(src.add(mid - 1))) {
                // a lone run, or two runs already in order
                copy(src.add(lo), dst.add(lo), hi - lo);
                return;
            }
            let (big_l, big_r) = (src.add(lo), src.add(mid));
            let (p, q) = (mid - lo, hi - mid);
            if p < SPLIT_WIDTH && q < SPLIT_WIDTH {
                let mut m = Merge { l: big_l, lend: big_l.add(p), r: big_r, rend: big_r.add(q), out: dst.add(lo) };
                match self.pending.take() {
                    None => self.pending = Some(m),
                    Some(mut pending) => merge_two(&mut pending, &mut m, less),
                }
                return;
            }
            let k = (hi - lo) / 2;
            let i = merge_split(big_l, p, big_r, q, k, less);
            let j = k - i;
            let mut a = Merge { l: big_l, lend: big_l.add(i), r: big_r, rend: big_r.add(j), out: dst.add(lo) };
            let mut b = Merge { l: big_l.add(i), lend: big_l.add(p), r: big_r.add(j), rend: big_r.add(q), out: dst.add(lo + k) };
            merge_two(&mut a, &mut b, less);
        }
    }
    /// # Safety
    /// As for add.
    unsafe fn finish<F: FnMut(&T, &T) -> bool>(&mut self, less: &mut F) {
        if let Some(mut m) = self.pending.take() {
            // SAFETY: the caller's.
            unsafe { merge_one(&mut m, less) }
        }
    }
}

/// The source of the level in flight, which holds every element: copied
/// back into the slice when it is the buffer, at the end and when a
/// comparison unwinds.
struct Levels<T> {
    a: *mut Slot<T>,
    src: *const Slot<T>,
    n: usize,
}
impl<T> Drop for Levels<T> {
    fn drop(&mut self) {
        if !ptr::eq(self.src, self.a) {
            // SAFETY: src is kept at the level that holds the n elements.
            unsafe { copy(self.src, self.a, self.n) }
        }
    }
}

// ---- natural runs -----------------------------------------------------------------------------

/// The boundaries array of `find_runs` for n elements, minus the closing entry.
pub(crate) const fn max_runs(n: usize) -> usize {
    (n >> 5) + 33
}

/// The boundaries of the maximal non-descending runs of a[0,n) into
/// b[0..=r] (b[0] = 0, b[r] = n); a strictly descending run is reversed in
/// place, which keeps equal elements in their order. Gives up, with false,
/// once the runs seen are too short to be worth merging: more than
/// (i >> 5) + 32 of them after i elements, which on unordered input happens
/// within the first few hundred. b must hold max_runs(n) + 1 entries.
///
/// # Safety
/// a[0,n) holds elements, n fits a u32, and b has max_runs(n) + 1 slots.
unsafe fn find_runs<T, F: FnMut(&T, &T) -> bool>(a: *mut Slot<T>, n: usize, b: *mut u32, r: &mut usize, less: &mut F) -> bool {
    *r = 0;
    // SAFETY: every index stays below n, and the count of runs cannot pass
    // (n >> 5) + 33 before the check returns.
    unsafe {
        *b = 0;
        let mut i = 0;
        while i < n {
            let mut j = i + 1;
            if j < n && less(el(a.add(j)), el(a.add(j - 1))) {
                // strictly descending: each element below the one before
                j += 1;
                while j < n && less(el(a.add(j)), el(a.add(j - 1))) {
                    j += 1;
                }
                core::slice::from_raw_parts_mut(a.add(i), j - i).reverse();
            } else {
                while j < n && !less(el(a.add(j)), el(a.add(j - 1))) {
                    j += 1;
                }
            }
            *r += 1;
            *b.add(*r) = j as u32;
            i = j;
            if *r > (j >> 5) + 32 {
                return false;
            }
        }
    }
    true
}

/// Merges the runs b[0..=r] of a[0,n) bottom-up through buf; the result is
/// in a. b is overwritten level by level.
///
/// # Safety
/// a[0,n) holds elements, buf has n slots, b holds the r + 1 boundaries.
unsafe fn merge_runs<T, F: FnMut(&T, &T) -> bool>(a: *mut Slot<T>, buf: *mut Slot<T>, n: usize, b: *mut u32, mut r: usize, less: &mut F) {
    // SAFETY: the boundaries are ascending and end at n.
    unsafe {
        let mut lv = Levels { a, src: a, n };
        let mut dst = buf;
        while r > 1 {
            let mut level = Level::new();
            let mut w = 0;
            let mut k = 0;
            while k + 1 < r {
                level.add(lv.src, dst, *b.add(k) as usize, *b.add(k + 1) as usize, *b.add(k + 2) as usize, less);
                *b.add(w) = *b.add(k);
                w += 1;
                k += 2;
            }
            if r & 1 == 1 {
                // the last run has no partner
                let last = *b.add(r - 1) as usize;
                copy(lv.src.add(last), dst.add(last), n - last);
                *b.add(w) = last as u32;
                w += 1;
            }
            level.finish(less);
            *b.add(w) = n as u32;
            r = w;
            let s = lv.src;
            lv.src = dst;
            dst = s.cast_mut();
        }
    }
    // the drop copies the result back if it is in the buffer
}

/// Sorts `v` by `less` if it is made of long natural runs: true, and the
/// slice is sorted, stably. False if the runs are too short, if the slice
/// is short, or if the buffer of n slots or the run boundaries cannot be
/// allocated from `A`; then the slice is a permutation of its input with
/// the same stable order (strictly descending runs may have been reversed)
/// for the comparison sort to finish.
pub(crate) fn merge_natural_runs<T, A: Alloc, F: FnMut(&T, &T) -> bool>(v: &mut [T], less: &mut F) -> bool {
    let n = v.len();
    if n < RUNS_MIN || n > u32::MAX as usize {
        return false;
    }
    let Ok(runs) = Buf::<u32, A>::new(max_runs(n) + 1) else {
        return false;
    };
    let a = v.as_mut_ptr().cast::<Slot<T>>();
    let mut r = 0;
    // SAFETY: a holds the n elements of the slice and runs has max_runs(n)
    // + 1 slots.
    if !unsafe { find_runs(a, n, runs.ptr(), &mut r, less) } {
        return false;
    }
    let Ok(buf) = Buf::<Slot<T>, A>::new(n) else {
        return false;
    };
    // SAFETY: buf has n slots; both buffers live for the merge, and every
    // element is back in a slot of the slice when it returns or unwinds.
    unsafe { merge_runs(a, buf.ptr(), n, runs.ptr(), r, less) };
    true
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::memory::DefaultAlloc;
    use alloc::format;
    use alloc::vec::Vec;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Tagged {
        key: u32,
        id: u32,
    }
    fn tag(keys: impl IntoIterator<Item = u32>) -> Vec<Tagged> {
        keys.into_iter().enumerate().map(|(i, key)| Tagged { key, id: i as u32 }).collect()
    }
    fn rng(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed;
        move || {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }
    /// Patterns made of runs, and unordered ones the pass must give up on.
    fn pattern(name: &str, n: usize) -> Vec<Tagged> {
        let mut next = rng(n as u64);
        match name {
            "random" => tag((0..n).map(|_| next() as u32)),
            "few_unique" => tag((0..n).map(|_| (next() % 100) as u32)),
            "all_equal" => tag((0..n).map(|_| 7)),
            "runs" => tag((0..n).map(|i| ((i / 1500) as u32 * 7919 + (i % 1500) as u32 * 3) % 100_000)),
            "runs_with_ties" => tag((0..n).map(|i| ((i / 1500) as u32 * 7919 + (i % 1500) as u32 / 7) % 1000)),
            "organ" => tag((0..n).map(|i| (if i < n / 2 { i } else { n - i }) as u32)),
            "sawtooth" => tag((0..n).map(|i| (i % 1000) as u32)),
            "reversed" => tag((0..n).map(|i| (n - i) as u32)),
            "descending_pieces" => tag((0..n).map(|i| (i / 700) as u32 * 1000 + 700 - (i % 700) as u32)),
            "three_runs" => tag((0..n).map(|i| ((i % (n / 3 + 1)) * 2 + i / (n / 3 + 1)) as u32)),
            _ => unreachable!(),
        }
    }
    const RUN_PATTERNS: [&str; 7] = ["all_equal", "runs", "runs_with_ties", "organ", "sawtooth", "descending_pieces", "three_runs"];
    fn check(v: &[Tagged], n: usize, what: &str) {
        assert_eq!(v.len(), n, "{what}: length");
        let mut ids: Vec<u32> = v.iter().map(|t| t.id).collect();
        ids.sort_unstable();
        assert!(ids.iter().enumerate().all(|(i, &id)| id == i as u32), "{what}: not a permutation");
        assert!(v.windows(2).all(|p| p[0].key < p[1].key || (p[0].key == p[1].key && p[0].id < p[1].id)), "{what}: not sorted stably");
    }
    fn merge(v: &mut [Tagged]) -> bool {
        merge_natural_runs::<Tagged, DefaultAlloc, _>(v, &mut |a, b| a.key < b.key)
    }

    #[test]
    fn merges_long_runs_stably_and_gives_up_on_short_ones() {
        // Miri interprets every comparison: the two large sizes are left out there.
        let sizes: &[usize] = if cfg!(miri) { &[RUNS_MIN, RUNS_MIN + 1, 1023] } else { &[RUNS_MIN, RUNS_MIN + 1, 1023, 4096, 20_000, 70_000] };
        for &n in sizes {
            for p in RUN_PATTERNS {
                let mut v = pattern(p, n);
                assert!(merge(&mut v), "{p} at {n}: made of runs");
                check(&v, n, &format!("{p} at {n}"));
            }
            for p in ["random", "few_unique"] {
                let mut v = pattern(p, n);
                assert!(!merge(&mut v), "{p} at {n}: no runs to merge");
                // a permutation that the comparison sort finishes stably
                let mut ids: Vec<u32> = v.iter().map(|t| t.id).collect();
                ids.sort_unstable();
                assert!(ids.iter().enumerate().all(|(i, &id)| id == i as u32), "{p} at {n}: a permutation");
                v.sort_by_key(|a| a.key);
                check(&v, n, &format!("{p} at {n} after the pass"));
            }
        }
        let mut v = pattern("runs", RUNS_MIN - 1);
        assert!(!merge(&mut v), "below the minimum the pass is skipped");
        assert_eq!(v, pattern("runs", RUNS_MIN - 1));
        let n = if cfg!(miri) { 1024 } else { 10_000 };
        let mut v = pattern("reversed", n);
        assert!(merge(&mut v));
        check(&v, n, "reversed");
    }

    #[test]
    fn a_comparator_that_is_not_an_order_gives_a_permutation() {
        let sizes: &[usize] = if cfg!(miri) { &[600, 1200] } else { &[600, 5000, 30_000] };
        for &n in sizes {
            for p in RUN_PATTERNS {
                for flavour in 0..3 {
                    let mut v = pattern(p, n);
                    let mut next = rng(99);
                    let mut calls = 0usize;
                    let mut less = |_: &Tagged, _: &Tagged| {
                        calls += 1;
                        match flavour {
                            0 => next() & 1 == 0,
                            1 => true,
                            _ => calls > 700, // an order for the run pass, nonsense for the merges
                        }
                    };
                    let _ = merge_natural_runs::<Tagged, DefaultAlloc, _>(&mut v, &mut less);
                    let mut ids: Vec<u32> = v.iter().map(|t| t.id).collect();
                    ids.sort_unstable();
                    assert!(ids.iter().enumerate().all(|(i, &id)| id == i as u32), "{p} at {n}, flavour {flavour}: not a permutation");
                }
            }
        }
    }

    #[test]
    fn a_comparator_that_panics_keeps_every_element() {
        // Every call number in the first few hundred, then a spread up to
        // the last call, on every pattern: the run pass and every level.
        for p in RUN_PATTERNS {
            let n = if cfg!(miri) { 700 } else { 6000 };
            let input = pattern(p, n);
            let calls = std::cell::Cell::new(0usize);
            let mut v = input.clone();
            assert!(merge_natural_runs::<Tagged, DefaultAlloc, _>(&mut v, &mut |a, b| {
                calls.set(calls.get() + 1);
                a.key < b.key
            }));
            let total = calls.get();
            let (first, spread) = if cfg!(miri) { (12, 12) } else { (300, 60) };
            let ats: Vec<usize> = (0..first).chain((0..spread).map(|k| total * k / spread)).chain([total - 1]).collect();
            for at in ats {
                let mut v = input.clone();
                calls.set(0);
                let r = catch_unwind(AssertUnwindSafe(|| {
                    merge_natural_runs::<Tagged, DefaultAlloc, _>(&mut v, &mut |a, b| {
                        let c = calls.get();
                        calls.set(c + 1);
                        if c == at {
                            panic!("comparator failed");
                        }
                        a.key < b.key
                    })
                }));
                assert!(r.is_err(), "{p}: call {at} of {total} happens");
                let mut ids: Vec<u32> = v.iter().map(|t| t.id).collect();
                ids.sort_unstable();
                assert!(ids.iter().enumerate().all(|(i, &id)| id == i as u32), "{p}: comparator panicking at call {at} keeps every element");
            }
        }
    }

    #[test]
    fn elements_with_destructors_move_once() {
        use std::rc::Rc;
        let (n, panic_at) = if cfg!(miri) { (700, 300) } else { (6000, 7000) };
        let input: Vec<Rc<Tagged>> = pattern("runs", n).into_iter().map(Rc::new).collect();
        let mut v = input.clone();
        assert!(merge_natural_runs::<Rc<Tagged>, DefaultAlloc, _>(&mut v, &mut |a, b| a.key < b.key));
        let sorted: Vec<Tagged> = v.iter().map(|t| **t).collect();
        check(&sorted, n, "Rc elements");
        assert!(v.iter().all(|x| Rc::strong_count(x) == 2));
        drop(v);
        let mut v = input.clone();
        let calls = std::cell::Cell::new(0usize);
        let _ = catch_unwind(AssertUnwindSafe(|| {
            merge_natural_runs::<Rc<Tagged>, DefaultAlloc, _>(&mut v, &mut |a, b| {
                let c = calls.get();
                calls.set(c + 1);
                if c == panic_at {
                    panic!("comparator failed");
                }
                a.key < b.key
            })
        }));
        drop(v);
        assert!(input.iter().all(|x| Rc::strong_count(x) == 1), "no element was duplicated or lost");
    }

    #[test]
    fn the_run_pass_gives_up_before_its_array_is_full() {
        assert_eq!(max_runs(0), 33);
        assert_eq!(max_runs(320), 43);
        // r <= (j >> 5) + 33 <= max_runs(n) when the pass returns false
        let mut v = pattern("random", if cfg!(miri) { 2048 } else { 100_000 });
        assert!(!merge(&mut v));
    }
}

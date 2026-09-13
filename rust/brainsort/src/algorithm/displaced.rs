//! Route 3: nearly sorted input. The elements that break the ascending
//! order are pulled out (an outlier-popping rule keeps a long sorted
//! subsequence), the kept elements are compacted in place, only the
//! displaced ones are sorted, and one backward merge puts them back.
//! O(n + k log k) for k displaced elements, memory O(k).
use super::{MAX_POP, RING};
use crate::view::{AllocError, Arr, Hooks, IdxView};
use core::ptr::NonNull;

/// Growable arrays for the displaced elements: the element, its original
/// position, and the height of the kept stack when it was pulled out.
/// Growth failure is reported, not propagated, so the route can restore the
/// array and give up cleanly.
pub struct DispBuf<V: Arr> {
    elem: NonNull<V::T>,
    pos: NonNull<u32>,
    rank: NonNull<u32>,
    cap: usize,
}
impl<V: Arr> DispBuf<V> {
    /// Room for `cap` displaced elements.
    pub fn new(cap: usize) -> Result<Self, AllocError> {
        let elem = V::alloc_array::<V::T>(cap)?;
        let pos = match V::alloc_array::<u32>(cap) {
            Ok(p) => p,
            Err(e) => {
                // SAFETY: just allocated with cap.
                unsafe { V::free_array(elem, cap) };
                return Err(e);
            }
        };
        let rank = match V::alloc_array::<u32>(cap) {
            Ok(p) => p,
            Err(e) => {
                // SAFETY: just allocated with cap.
                unsafe {
                    V::free_array(elem, cap);
                    V::free_array(pos, cap);
                }
                return Err(e);
            }
        };
        Ok(DispBuf { elem, pos, rank, cap })
    }
    /// Capacity.
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.cap
    }
    /// Doubles the capacity; false (nothing changed) if the memory is not there.
    pub fn grow(&mut self) -> bool {
        let ncap = self.cap * 2;
        let Ok(ne) = V::alloc_array::<V::T>(ncap) else { return false };
        let Ok(np) = V::alloc_array::<u32>(ncap) else {
            // SAFETY: just allocated with ncap.
            unsafe { V::free_array(ne, ncap) };
            return false;
        };
        let Ok(nr) = V::alloc_array::<u32>(ncap) else {
            // SAFETY: just allocated with ncap.
            unsafe {
                V::free_array(ne, ncap);
                V::free_array(np, ncap);
            }
            return false;
        };
        // SAFETY: the old arrays hold cap entries, the new ones ncap >= cap.
        unsafe {
            core::ptr::copy_nonoverlapping(self.elem.as_ptr(), ne.as_ptr(), self.cap);
            core::ptr::copy_nonoverlapping(self.pos.as_ptr(), np.as_ptr(), self.cap);
            core::ptr::copy_nonoverlapping(self.rank.as_ptr(), nr.as_ptr(), self.cap);
        }
        self.release();
        self.elem = ne;
        self.pos = np;
        self.rank = nr;
        self.cap = ncap;
        true
    }
    /// Displaced element `i`.
    #[inline(always)]
    pub fn elem(&self, i: usize) -> V::T {
        debug_assert!(i < self.cap);
        if V::H::COUNTED {
            V::H::on_read(self.elem.as_ptr().wrapping_add(i) as *const u8, core::mem::size_of::<V::T>());
        }
        // SAFETY: i < cap.
        unsafe { *self.elem.as_ptr().add(i) }
    }
    /// Records displaced element `i`.
    #[inline(always)]
    pub fn set(&mut self, i: usize, e: V::T, p: u32, r: u32) {
        debug_assert!(i < self.cap);
        if V::H::COUNTED {
            V::H::on_write(self.elem.as_ptr().wrapping_add(i) as *const u8, core::mem::size_of::<V::T>());
            V::H::on_table_write(self.pos.as_ptr().wrapping_add(i) as *const u8, 4);
            V::H::on_table_write(self.rank.as_ptr().wrapping_add(i) as *const u8, 4);
        }
        // SAFETY: i < cap.
        unsafe {
            *self.elem.as_ptr().add(i) = e;
            *self.pos.as_ptr().add(i) = p;
            *self.rank.as_ptr().add(i) = r;
        }
    }
    /// Original position of displaced element `i`.
    #[inline(always)]
    pub fn pos(&self, i: usize) -> u32 {
        if V::H::COUNTED {
            V::H::on_table_read(self.pos.as_ptr().wrapping_add(i) as *const u8, 4);
        }
        // SAFETY: i < cap.
        unsafe { *self.pos.as_ptr().add(i) }
    }
    /// Rank of displaced element `i`.
    #[inline(always)]
    pub fn rank(&self, i: usize) -> u32 {
        if V::H::COUNTED {
            V::H::on_table_read(self.rank.as_ptr().wrapping_add(i) as *const u8, 4);
        }
        // SAFETY: i < cap.
        unsafe { *self.rank.as_ptr().add(i) }
    }
    /// The rank array.
    #[inline(always)]
    pub fn ranks(&mut self) -> *mut u32 {
        self.rank.as_ptr()
    }
    /// The position array.
    #[inline(always)]
    pub fn positions(&self) -> *const u32 {
        self.pos.as_ptr()
    }
    fn release(&mut self) {
        // SAFETY: the arrays were allocated with cap.
        unsafe {
            V::free_array(self.elem, self.cap);
            V::free_array(self.pos, self.cap);
            V::free_array(self.rank, self.cap);
        }
        self.cap = 0;
        self.elem = NonNull::dangling();
        self.pos = NonNull::dangling();
        self.rank = NonNull::dangling();
    }
}
impl<V: Arr> Drop for DispBuf<V> {
    fn drop(&mut self) {
        if self.cap != 0 || self.elem != NonNull::dangling() {
            self.release();
        }
    }
}

/// Undoes the compaction of route 3: a[0,nk) holds the kept elements,
/// `disp` the nd displaced ones with their positions, a[scanned,n) is
/// untouched. Puts every element of [0,scanned) back where it came from.
pub fn restore_displaced<V: Arr>(a: V, disp: &mut DispBuf<V>, nd: usize, nk: usize, scanned: usize) {
    let idx = disp.ranks(); // ranks are not needed any more: reuse as an index array
    // SAFETY: the rank array has at least nd entries.
    let idx = unsafe { core::slice::from_raw_parts_mut(idx, nd) };
    for (t, x) in idx.iter_mut().enumerate() {
        *x = t as u32;
    }
    let pos = disp.positions();
    // SAFETY: positions has nd entries and x < nd.
    idx.sort_unstable_by_key(|&x| unsafe { *pos.add(x as usize) });
    let (mut t, mut k) = (nd, nk);
    let mut p = scanned;
    while p > 0 {
        p -= 1;
        // SAFETY: idx[t-1] < nd.
        if t > 0 && unsafe { *pos.add(idx[t - 1] as usize) } as usize == p {
            t -= 1;
            a.set(p, disp.elem(idx[t] as usize));
        } else {
            k -= 1;
            a.set(p, a.get(k)); // k-1 <= p: not yet overwritten
        }
    }
}

/// The state of route 3, which doubles as the guard that puts every
/// element back if a compare panics (the public API runs this route on the
/// caller's elements, whose key function may panic): during the scan and
/// the index sort the array is restored to its input order; during the
/// final merge the elements that were not merged yet are put back into the
/// gap, so every element stays in the range.
/// The index array of the displaced elements (two halves for the merge
/// sort), owned by the route state so it outlives an unwinding merge.
struct IdxBuf<V: Arr> {
    p: NonNull<u32>,
    n: usize,
    _v: core::marker::PhantomData<V>,
}
impl<V: Arr> Drop for IdxBuf<V> {
    fn drop(&mut self) {
        // SAFETY: allocated with n.
        unsafe { V::free_array(self.p, self.n) };
    }
}
struct Route3<V: Arr> {
    a: V,
    disp: DispBuf<V>,
    idx_buf: Option<IdxBuf<V>>,
    nk: usize,
    nd: usize,
    scanned: usize,
    /// 0: restore on unwind; 1: fill the merge gap on unwind; 2: nothing.
    phase: u8,
    /// Merge phase: the index array, the kept and displaced cursors.
    idx: Option<IdxView<V::H>>,
    ik: usize,
    id: usize,
}
impl<V: Arr> Drop for Route3<V> {
    fn drop(&mut self) {
        match self.phase {
            0 => restore_displaced(self.a, &mut self.disp, self.nd, self.nk, self.scanned),
            1 => {
                // a[0,ik) kept, a[o,n) merged: the unmerged displaced fill the gap
                if let Some(idx) = self.idx {
                    for u in 0..self.id {
                        self.a.set(self.ik + u, self.disp.elem(idx.get(u) as usize));
                    }
                }
            }
            _ => {}
        }
    }
}

/// Route 3. The kept (in-order) elements are compacted to a[0,nk) as the
/// scan goes; displaced elements are moved to a side buffer with their
/// original position. Returns Ok(false), with the array restored to its
/// input order, if too many elements turn out to be displaced.
///
/// Stability without storing every kept position: for a displaced element
/// d let rank(d) be the number of kept elements that precede it in the
/// input. Recording the kept-stack height at the moment d was pulled out
/// and taking the suffix minimum over later pull-outs gives exactly that
/// number, because the stack only ever loses elements from the top.
pub fn sort_displaced<V: Arr>(a: V, n: usize) -> Result<bool, AllocError> {
    let counted = <V::H as Hooks>::COUNTED;
    let limit = n / 8;
    let mut g = Route3::<V> { a, disp: DispBuf::<V>::new(256.min(limit + 1))?, idx_buf: None, nk: 0, nd: 0, scanned: 0, phase: 0, idx: None, ik: 0, id: 0 };
    let mut nk_max = 0usize; // ring entries are valid for stack indices >= nk_max - RING
    let (mut k1, mut k2) = (V::T::default(), V::T::default()); // the last two kept elements
    let mut ring = [0u32; RING]; // original positions of the top kept elements

    // When an element is smaller than the top of the kept subsequence,
    // either it is the outlier or the top few kept elements are (a
    // swapped-in big value, or several in a row). Pop up to MAX_POP kept
    // elements if the element fits right below them; otherwise displace the
    // element itself. The route gives up when displaced elements exceed 1/8
    // of what has been scanned so far (plus slack), so unsuitable input
    // costs a few thousand elements instead of n/8.
    let p = a.data(); // the hot loop works on the raw pointer (counted: through the view)
    // SAFETY (rd/wr): every index passed is below n; the view holds n elements.
    let rd = |i: usize| -> V::T { if counted { a.get(i) } else { unsafe { *p.add(i) } } };
    let wr = |i: usize, v: V::T| {
        if counted { a.set(i, v) } else { unsafe { *p.add(i) = v } }
    };
    let give_up = |g: &mut Route3<V>| -> Result<bool, AllocError> {
        g.phase = 2;
        restore_displaced(g.a, &mut g.disp, g.nd, g.nk, g.scanned);
        Ok(false)
    };
    for i in 0..n {
        g.scanned = i;
        let it = rd(i);
        if g.nk == 0 || !a.less(it, k1) {
            // extends the sorted subsequence
            ring[g.nk & (RING - 1)] = i as u32;
            wr(g.nk, it);
            g.nk += 1;
            k2 = k1;
            k1 = it;
            continue;
        }
        let mut pops = 0usize;
        if g.nk == 1 || !a.less(it, k2) {
            pops = 1;
        } else {
            for j in 2..=MAX_POP {
                // rare path: look deeper
                if g.nk <= j {
                    pops = j;
                    break;
                }
                if !a.less(it, rd(g.nk - 1 - j)) {
                    pops = j;
                    break;
                }
            }
        }
        let budget = limit.min((i >> 3) + 64);
        if g.nd + pops.max(1) > budget + 1 {
            return give_up(&mut g);
        }
        while g.nd + pops.max(1) > g.disp.capacity() {
            if !g.disp.grow() {
                return give_up(&mut g);
            }
        }
        if pops == 0 {
            // this element is the outlier
            let nd = g.nd;
            g.disp.set(nd, it, i as u32, g.nk as u32);
            g.nd += 1;
            continue;
        }
        if g.nk > nk_max {
            nk_max = g.nk;
        }
        if nk_max > RING && g.nk - pops < nk_max - RING {
            return give_up(&mut g);
        }
        for j in 1..=pops {
            let (nd, nk) = (g.nd, g.nk);
            g.disp.set(nd, rd(nk - j), ring[(nk - j) & (RING - 1)], (nk - j) as u32);
            g.nd += 1;
        }
        g.nk -= pops;
        ring[g.nk & (RING - 1)] = i as u32;
        wr(g.nk, it);
        g.nk += 1;
        k1 = it;
        k2 = if g.nk >= 2 { rd(g.nk - 2) } else { V::T::default() };
    }
    g.scanned = n;
    let (nk, nd) = (g.nk, g.nd);
    if nd == 0 {
        g.phase = 2;
        return Ok(true); // cannot happen when descents > 0, but harmless
    }

    // rank(d) = suffix minimum of the recorded stack heights.
    {
        let r = g.disp.ranks();
        // SAFETY: nd <= cap entries.
        unsafe {
            let mut mn = *r.add(nd - 1);
            for t in (0..nd).rev() {
                if *r.add(t) < mn {
                    mn = *r.add(t);
                }
                *r.add(t) = mn;
            }
        }
    }

    // Sort the displaced elements by (key, original position): a stable
    // bottom-up merge sort of an index array (positions break ties, so the
    // input order of the index array does not matter). If the index array
    // cannot be allocated the route gives up like it does on too many
    // displaced elements: the array is restored, nothing is lost.
    let idx_ptr = match V::alloc_array::<u32>(2 * nd) {
        Ok(p) => {
            g.idx_buf = Some(IdxBuf { p, n: 2 * nd, _v: core::marker::PhantomData });
            p
        }
        Err(_) => return give_up(&mut g),
    };
    let idx = IdxView::<V::H>::new(idx_ptr.as_ptr(), 2 * nd);
    for t in 0..nd {
        idx.set(t, t as u32);
    }
    {
        let (mut src, mut dst) = (0usize, nd); // halves of idx_buf
        let mut width = 1;
        while width < nd {
            let mut lo = 0;
            while lo < nd {
                let mid = (lo + width).min(nd);
                let hi = (lo + 2 * width).min(nd);
                let (mut i, mut j, mut o) = (lo, mid, lo);
                while i < mid && j < hi {
                    let (pi, pj) = (idx.get(src + i), idx.get(src + j));
                    // Reads in one fixed order, so the counted numbers are
                    // the same everywhere.
                    let ei = g.disp.elem(pi as usize);
                    let ej = g.disp.elem(pj as usize);
                    let c = a.compare(ej, ei);
                    let mut take_j = c < 0;
                    if c == 0 {
                        let (qi, qj) = (g.disp.pos(pi as usize), g.disp.pos(pj as usize));
                        take_j = qj < qi;
                    }
                    if take_j {
                        idx.set(dst + o, pj);
                        j += 1;
                    } else {
                        idx.set(dst + o, pi);
                        i += 1;
                    }
                    o += 1;
                }
                while i < mid {
                    idx.set(dst + o, idx.get(src + i));
                    o += 1;
                    i += 1;
                }
                while j < hi {
                    idx.set(dst + o, idx.get(src + j));
                    o += 1;
                    j += 1;
                }
                lo += 2 * width;
            }
            core::mem::swap(&mut src, &mut dst);
            width *= 2;
        }
        if src != 0 {
            for t in 0..nd {
                idx.set(t, idx.get(nd + t));
            }
        }
    }

    // Backward merge of the kept elements a[0,nk) and the sorted displaced
    // elements into a[0,n). The write index o = ik + id never overtakes the
    // unread kept elements. On equal keys the kept element goes after the
    // displaced one iff it is not among the rank(d) kept elements that
    // precede d in the input. Two plain less() branches, both predictable
    // on nearly sorted input (the kept element wins most of the time).
    g.phase = 1;
    g.idx = Some(idx);
    g.ik = nk;
    g.id = nd;
    let mut o = n;
    let mut t = idx.get(g.id - 1);
    let mut ed = g.disp.elem(t as usize);
    if g.ik > 0 {
        let mut ek = a.get(g.ik - 1);
        loop {
            let take_kept = if a.less(ed, ek) {
                true
            } else if a.less(ek, ed) {
                false
            } else {
                g.ik - 1 >= g.disp.rank(t as usize) as usize
            };
            if take_kept {
                o -= 1;
                a.set(o, ek);
                g.ik -= 1;
                if g.ik == 0 {
                    break;
                }
                ek = a.get(g.ik - 1);
            } else {
                o -= 1;
                a.set(o, ed);
                g.id -= 1;
                if g.id == 0 {
                    break;
                }
                t = idx.get(g.id - 1);
                ed = g.disp.elem(t as usize);
            }
        }
    }
    g.phase = 2; // no compare from here on
    while g.id > 0 {
        // kept elements exhausted: the rest of the displaced go in front
        o -= 1;
        a.set(o, ed);
        g.id -= 1;
        if g.id == 0 {
            break;
        }
        t = idx.get(g.id - 1);
        ed = g.disp.elem(t as usize);
    }
    Ok(true) // the remaining kept elements are already in place
}

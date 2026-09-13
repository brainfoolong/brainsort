//! The algorithm: the port of `algorithm.hpp`.
//!
//! One linear scout pass over the keys computes the varying-bit mask, the
//! number of descents and ascents, and the monotone runs while there are
//! at most four. That picks the route: sorted (nothing to do), reversed
//! (reverse in place, tie groups restored), at most four runs (merge
//! pairwise), few descents (the displaced-element route), or the radix
//! route. For chunked keys (strings) groups of elements that tie on a chunk
//! are sorted on the next chunk, with the same route selection per group.
//! Stable, exact. Requires n < 2^32 (falls back to merge sort beyond that).
pub mod displaced;
pub mod monotone;
pub mod partition;
pub mod radix_route;
pub mod runs;
pub mod scout;

use crate::view::{AllocError, Arr, AuxBuffer, AuxRaw, DepthScope, Elem, Hooks, Stat, insertion_sort};
use radix_route::HistStore;

/// Below this: insertion sort.
pub const INSERTION_MAX: usize = 32;
/// Route 4 splits into two parts from this size on.
pub const SPLIT_MIN: usize = 1024;
/// Route 2b: <= 2 merge levels.
pub const MAX_RUNS: usize = 4;
/// Route 3: how far back an outlier is looked for.
pub const MAX_POP: usize = 4;
/// Route 3: positions of the top kept elements.
pub const RING: usize = 256;
/// Scout: how often the early-commit test runs.
pub const COMMIT_BLOCK: usize = 1024;
/// Route 4: pivot sample size (at most).
pub const SAMPLE: usize = 512;

/// A 16-byte block standing in for the heap-allocated handle object of the
/// C++ `AuxBuffer` / `AuxRaw` (`new AuxBuffer(...)`): allocated before the
/// array and freed after it, so the allocation sequence of the two
/// libraries, and with it the trace of the counted build, is the same.
struct Handle<V: Arr> {
    p: core::ptr::NonNull<u8>,
    _v: core::marker::PhantomData<V>,
}
impl<V: Arr> Handle<V> {
    const BYTES: usize = 16;
    fn new() -> Result<Self, AllocError> {
        Ok(Handle { p: V::alloc_array::<u8>(Self::BYTES)?, _v: core::marker::PhantomData })
    }
}
impl<V: Arr> Drop for Handle<V> {
    fn drop(&mut self) {
        // SAFETY: allocated with BYTES.
        unsafe { V::free_array(self.p, Self::BYTES) };
    }
}
/// A handle and its array, dropped in the C++ order: the array (the
/// destructor) first, then the handle (operator delete).
struct Owned<V: Arr, B> {
    handle: Handle<V>,
    array: core::mem::ManuallyDrop<B>,
}
impl<V: Arr, B> Owned<V, B> {
    fn new(make: impl FnOnce() -> Result<B, AllocError>) -> Result<Self, AllocError> {
        let handle = Handle::new()?;
        let array = make()?;
        Ok(Owned { handle, array: core::mem::ManuallyDrop::new(array) })
    }
}
impl<V: Arr, B> Drop for Owned<V, B> {
    fn drop(&mut self) {
        // SAFETY: dropped exactly once, before the handle.
        unsafe { core::mem::ManuallyDrop::drop(&mut self.array) };
        let _ = &self.handle;
    }
}
impl<V: Arr, B> core::ops::Deref for Owned<V, B> {
    type Target = B;
    fn deref(&self) -> &B {
        &self.array
    }
}
impl<V: Arr, B> core::ops::DerefMut for Owned<V, B> {
    fn deref_mut(&mut self) -> &mut B {
        &mut self.array
    }
}

/// The element scratch buffer, allocated on first use and sized by the
/// route that needs it, so the routes that need none (sorted, reversed)
/// cost no memory and the others get exactly what they ask for.
pub struct Scratch<V: Arr> {
    buf: Option<Owned<V, AuxBuffer<V>>>,
    hist: Option<Owned<V, AuxRaw<u32, V>>>,
    wc: Option<Owned<V, AuxRaw<u8, V>>>,
}
impl<V: Arr> Default for Scratch<V> {
    fn default() -> Self {
        Self::new()
    }
}
impl<V: Arr> Scratch<V> {
    /// Empty.
    pub fn new() -> Self {
        Scratch { buf: None, hist: None, wc: None }
    }
    /// A view of k elements; grows (discarding contents) if needed.
    pub fn ensure(&mut self, like: V, k: usize) -> Result<V, AllocError> {
        match &mut self.buf {
            None => self.buf = Some(Owned::new(|| AuxBuffer::new(like, k))?),
            Some(b) if b.size() < k => {
                if let Err(e) = b.resize_discard(k) {
                    self.buf = None;
                    return Err(e);
                }
            }
            _ => {}
        }
        Ok(self.buf.as_ref().unwrap().arr().sub(0, k))
    }
    /// Capacity of the element buffer.
    pub fn capacity(&self) -> usize {
        self.buf.as_ref().map_or(0, |b| b.size())
    }
    /// The element buffer, if any.
    pub fn buffer(&self) -> Option<*mut V::T> {
        self.buf.as_ref().map(|b| b.data())
    }
    /// The reusable histogram arena (u32 counters), grown to the largest
    /// table a radix pass has asked for and kept for the rest of the sort.
    pub fn hist(&mut self, entries: usize) -> Result<HistStore, AllocError> {
        match &self.hist {
            Some(h) if h.size() >= entries => {}
            _ => {
                self.hist = None;
                self.hist = Some(Owned::new(|| AuxRaw::new(entries))?);
            }
        }
        let h = self.hist.as_ref().unwrap();
        Ok(HistStore { p: h.data() })
    }
    /// The write-combining buffers of the MSD scatter, likewise.
    pub fn wc(&mut self, bytes: usize) -> Result<*mut u8, AllocError> {
        match &self.wc {
            Some(w) if w.size() >= bytes => {}
            _ => {
                self.wc = None;
                self.wc = Some(Owned::new(|| AuxRaw::new(bytes))?);
            }
        }
        Ok(self.wc.as_ref().unwrap().data())
    }
}

/// Sorts a[0,n) given that all its elements share the key chunks before
/// `chunk`. Groups that tie on a chunk are sorted on the next chunk; all
/// but the largest group recurse, the largest one loops, so the recursion
/// depth is at most log2(n). With `free` (fixed keys only) the result may
/// stay in the scratch buffer; `ended_in_src` says whether it is in a.
pub fn sort_range<V: Arr>(mut a: V, scratch: &mut Scratch<V>, mut n: usize, mut chunk: i32, free: bool, ended_in_src: &mut bool, digit_bits: u32) -> Result<(), AllocError> {
    let _depth = DepthScope::<V::H>::new();
    *ended_in_src = true;
    loop {
        if n < 2 {
            return Ok(());
        }
        if n <= INSERTION_MAX {
            insertion_sort(a, 0, n);
            return Ok(());
        }
        let s = scout::scout(a, n, chunk);
        if !s.committed {
            if s.descents == 0 {
                // route 1: sorted
                if V::H::COUNTED {
                    V::H::note_route(1);
                }
                return Ok(());
            }
            if s.ascents == 0 {
                // route 2: non-increasing
                if V::H::COUNTED {
                    V::H::note_route(2);
                }
                monotone::stable_reverse(a, n, &s);
                return Ok(());
            }
            if s.tracking {
                // route 2b: at most MAX_RUNS runs
                if V::H::COUNTED {
                    V::H::note_route(3);
                }
                return runs::sort_few_runs(a, scratch, n, &s);
            }
            if s.descents <= n / 16 {
                // route 3
                if V::H::COUNTED {
                    V::H::note_route(4);
                }
                if displaced::sort_displaced(a, n)? {
                    return Ok(());
                }
                if V::H::COUNTED {
                    V::H::stat(Stat::Giveups, 1);
                }
            }
        }
        if V::H::COUNTED {
            V::H::note_route(5);
        }
        // "Unordered" gates the route-4 split: committed means the scout
        // bailed on high disorder; otherwise a quarter of the adjacent pairs
        // descending is disorder enough that a split's cache cost is worth
        // paying.
        let unordered = s.committed || s.descents * 4 >= n;
        radix_route::radix_route(a, scratch, n, &mut chunk, s.mask, s.has_mask, unordered, free && !<V::T as Elem>::CHUNKED, ended_in_src, digit_bits)?; // route 4

        if !<V::T as Elem>::CHUNKED {
            return Ok(());
        }
        // Elements tying on this chunk form contiguous groups; each group
        // that has not reached the end of its keys is sorted on the next
        // chunk.
        let (mut g, mut big_g, mut big_len) = (0usize, 0usize, 0usize);
        while g < n {
            let eg = a.get(g);
            let k = V::key(eg, chunk);
            let mut e = g + 1;
            while e < n && V::key(a.get(e), chunk) == k {
                e += 1;
            }
            if e - g > 1 && !<V::T as Elem>::chunk_ends(eg, chunk) {
                if e - g > big_len {
                    if big_len > 1 {
                        sort_range(a.sub(big_g, big_len), scratch, big_len, chunk + 1, false, ended_in_src, digit_bits)?;
                    }
                    big_g = g;
                    big_len = e - g;
                } else {
                    sort_range(a.sub(g, e - g), scratch, e - g, chunk + 1, false, ended_in_src, digit_bits)?;
                }
            }
            g = e;
        }
        if big_len < 2 {
            return Ok(());
        }
        a = a.sub(big_g, big_len); // the largest group: iterate instead of recursing
        n = big_len;
        chunk += 1;
    }
}

/// Sorts the view a[0, n). `digit_bits` 0 selects the digit width
/// automatically. With `free` the sorted elements may be left in the
/// scratch buffer instead of being copied back to a (a one-pass radix then
/// saves that copy); the return value says whether they are in a, and
/// `scratch.buffer()` holds them otherwise.
pub fn brainsort_impl<V: Arr>(a: V, scratch: &mut Scratch<V>, free: bool, digit_bits: u32) -> Result<bool, AllocError> {
    let n = a.size();
    if n < 2 {
        return Ok(true);
    }
    if n <= INSERTION_MAX {
        insertion_sort(a, 0, n);
        return Ok(true);
    }
    if n > 0xFFFF_FFFF {
        crate::mergesort::merge_sort(a)?; // positions and counters are 32-bit
        return Ok(true);
    }
    let mut ended_in_src = true;
    sort_range(a, scratch, n, 0, free, &mut ended_in_src, digit_bits)?;
    Ok(ended_in_src)
}

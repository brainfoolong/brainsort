//! The library's allocator: the global allocator behind a cache of freed
//! blocks.
//!
//! A sort allocates a few large blocks (the records, the scratch buffer, the
//! permutation buffer) and frees them at the end. Fresh pages from the
//! system are the most expensive part of sorting a hundred thousand
//! elements: every page is faulted in on first touch, which costs as much as
//! the sorting itself. So the library keeps the largest blocks its sorts
//! have freed, up to a limit (32 MiB by default; 0 disables the cache), and
//! the next sort takes them back warm. The cache is shared by every thread
//! under a lock that is taken a few times per sort; it holds its blocks
//! until [`release_memory`] is called or the process ends. Each block
//! carries its capacity in a small header, so a block that served a smaller
//! request is still known in full when it comes back.
//!
//! Allocation never aborts: a failed request is reported and the sort
//! completes by comparison.
use crate::view::{Alloc, AllocError};
use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, Ordering};

const MIN_BLOCK: usize = 64 << 10; // smaller blocks go to the heap directly
const HEADER: usize = 16; // keeps the 16-byte alignment
const SLOTS: usize = 8; // cached free blocks
const ALIGN: usize = 16;

static LIMIT: AtomicUsize = AtomicUsize::new(32 << 20);

#[derive(Clone, Copy)]
struct Block {
    p: *mut u8, // the data pointer (header before it)
    cap: usize,
}
// SAFETY: the pointers name heap blocks owned by the cache.
unsafe impl Send for Block {}

struct Cache {
    blocks: [Block; SLOTS],
    n: usize,
    cached: usize,
}
impl Cache {
    const fn new() -> Self {
        Cache { blocks: [Block { p: core::ptr::null_mut(), cap: 0 }; SLOTS], n: 0, cached: 0 }
    }
    fn smallest(&self) -> usize {
        let mut s = 0;
        for i in 1..self.n {
            if self.blocks[i].cap < self.blocks[s].cap {
                s = i;
            }
        }
        s
    }
    /// Removes slot `i`; returns its block for the caller to free outside the lock.
    fn drop_slot(&mut self, i: usize) -> Block {
        let b = self.blocks[i];
        self.cached -= b.cap;
        self.n -= 1;
        self.blocks[i] = self.blocks[self.n];
        b
    }
}

// The lock: std's mutex when available, a spin lock otherwise (the critical
// sections are a few dozen instructions long).
#[cfg(feature = "std")]
mod lock {
    use super::Cache;
    use std::sync::{Mutex, MutexGuard};
    static CACHE: Mutex<Cache> = Mutex::new(Cache::new());
    pub fn lock() -> MutexGuard<'static, Cache> {
        CACHE.lock().unwrap_or_else(|e| e.into_inner())
    }
}
#[cfg(not(feature = "std"))]
mod lock {
    use super::Cache;
    use core::cell::UnsafeCell;
    use core::sync::atomic::{AtomicBool, Ordering};
    struct Spin {
        held: AtomicBool,
        cache: UnsafeCell<Cache>,
    }
    // SAFETY: access goes through the lock.
    unsafe impl Sync for Spin {}
    static CACHE: Spin = Spin { held: AtomicBool::new(false), cache: UnsafeCell::new(Cache::new()) };
    pub(super) struct Guard;
    impl core::ops::Deref for Guard {
        type Target = Cache;
        fn deref(&self) -> &Cache {
            // SAFETY: the lock is held.
            unsafe { &*CACHE.cache.get() }
        }
    }
    impl core::ops::DerefMut for Guard {
        fn deref_mut(&mut self) -> &mut Cache {
            // SAFETY: the lock is held and the guard is unique.
            unsafe { &mut *CACHE.cache.get() }
        }
    }
    impl Drop for Guard {
        fn drop(&mut self) {
            CACHE.held.store(false, Ordering::Release);
        }
    }
    pub fn lock() -> Guard {
        while CACHE.held.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
            core::hint::spin_loop();
        }
        Guard
    }
}

#[inline]
fn raw_alloc(bytes: usize) -> Result<NonNull<u8>, AllocError> {
    let layout = core::alloc::Layout::from_size_align(bytes.max(1), ALIGN).map_err(|_| AllocError)?;
    // SAFETY: the layout has a non-zero size.
    NonNull::new(unsafe { alloc::alloc::alloc(layout) }).ok_or(AllocError)
}
/// # Safety
/// `p` came from `raw_alloc(bytes)`.
#[inline]
unsafe fn raw_free(p: NonNull<u8>, bytes: usize) {
    // SAFETY: the same layout as raw_alloc built.
    unsafe { alloc::alloc::dealloc(p.as_ptr(), core::alloc::Layout::from_size_align_unchecked(bytes.max(1), ALIGN)) }
}
/// Frees a cached block by its data pointer.
/// # Safety
/// `b` was created by `allocate` with `cap` in its header.
unsafe fn release_block(b: Block) {
    // SAFETY: the header precedes the data pointer.
    unsafe { raw_free(NonNull::new_unchecked(b.p.sub(HEADER)), b.cap + HEADER) }
}

/// The library's allocator.
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultAlloc;

impl Alloc for DefaultAlloc {
    fn allocate(bytes: usize) -> Result<NonNull<u8>, AllocError> {
        if bytes < MIN_BLOCK {
            return raw_alloc(bytes);
        }
        {
            // best fit among the cached blocks
            let mut c = lock::lock();
            let mut best = usize::MAX;
            for i in 0..c.n {
                if c.blocks[i].cap >= bytes && (best == usize::MAX || c.blocks[i].cap < c.blocks[best].cap) {
                    best = i;
                }
            }
            if best != usize::MAX {
                let b = c.drop_slot_keep(best);
                // SAFETY: cached blocks are non-null.
                return Ok(unsafe { NonNull::new_unchecked(b.p) });
            }
        }
        let raw = raw_alloc(bytes.checked_add(HEADER).ok_or(AllocError)?)?;
        // SAFETY: the block has HEADER bytes before the data.
        unsafe {
            core::ptr::write_unaligned(raw.as_ptr() as *mut usize, bytes);
            Ok(NonNull::new_unchecked(raw.as_ptr().add(HEADER)))
        }
    }
    unsafe fn deallocate(p: NonNull<u8>, bytes: usize) {
        if bytes < MIN_BLOCK {
            // SAFETY: allocated by raw_alloc(bytes).
            return unsafe { raw_free(p, bytes) };
        }
        // SAFETY: the header holds the capacity.
        let cap = unsafe { core::ptr::read_unaligned(p.as_ptr().sub(HEADER) as *const usize) };
        let me = Block { p: p.as_ptr(), cap };
        let limit = LIMIT.load(Ordering::Relaxed);
        let mut victims: [Option<Block>; SLOTS + 1] = [None; SLOTS + 1];
        let mut nv = 0;
        {
            let mut c = lock::lock();
            if cap > limit {
                victims[nv] = Some(me);
                nv += 1;
            } else {
                // Make room: drop the smallest cached blocks while the total
                // would exceed the limit; a full cache keeps the larger block.
                while c.n > 0 && c.cached + cap > limit {
                    let s = c.smallest();
                    victims[nv] = Some(c.drop_slot(s));
                    nv += 1;
                }
                let mut keep = true;
                if c.n == SLOTS {
                    let s = c.smallest();
                    if c.blocks[s].cap >= cap {
                        keep = false;
                        victims[nv] = Some(me);
                        nv += 1;
                    } else {
                        victims[nv] = Some(c.drop_slot(s));
                        nv += 1;
                    }
                }
                if keep {
                    let n = c.n;
                    c.blocks[n] = me;
                    c.n += 1;
                    c.cached += cap;
                }
            }
        }
        for v in victims.iter().take(nv).flatten() {
            // SAFETY: every victim is a block from allocate, freed once.
            unsafe { release_block(*v) };
        }
    }
}
impl Cache {
    fn drop_slot_keep(&mut self, i: usize) -> Block {
        self.drop_slot(i)
    }
}

/// Frees the blocks the library keeps for reuse (see the module
/// documentation). Never needed for correctness.
pub fn release_memory() {
    let mut held: [Option<Block>; SLOTS] = [None; SLOTS];
    let n;
    {
        let mut c = lock::lock();
        n = c.n;
        for (i, slot) in held.iter_mut().enumerate().take(n) {
            *slot = Some(c.blocks[i]);
        }
        c.n = 0;
        c.cached = 0;
    }
    for b in held.iter().take(n).flatten() {
        // SAFETY: the blocks were taken out of the cache under the lock.
        unsafe { release_block(*b) };
    }
}

/// Sets how many bytes of freed blocks the library keeps for the next sort
/// (32 MiB by default). 0 disables the cache; the blocks held now stay
/// until [`release_memory`] is called.
pub fn set_memory_cache_limit(bytes: usize) {
    LIMIT.store(bytes, Ordering::Relaxed);
}

//! The element contract, the array view, the instrumentation hooks, the
//! allocator interface and the scratch buffers the algorithm uses: the port
//! of `traits.hpp`.
//!
//! The algorithm is written against a view type `V: Arr` that carries the
//! element type, the hooks and the allocator. On the library's view the
//! hooks are `NoHooks` and every `if H::COUNTED` block is compiled out; the
//! benchmark harness supplies a counting hooks type, so the same code
//! counts every read, write, compare and table access.
use core::marker::PhantomData;
use core::ops::{BitAnd, BitOr, BitXor, Not, Shl, Shr};
use core::ptr::NonNull;

// ---- radix keys ----------------------------------------------------------------------------

/// An unsigned 32- or 64-bit radix key.
pub trait RadixKey:
    Copy + Ord + Default + core::fmt::Debug + BitAnd<Output = Self> + BitOr<Output = Self> + BitXor<Output = Self> + Not<Output = Self> + Shr<u32, Output = Self> + Shl<u32, Output = Self>
{
    /// Width in bits.
    const BITS: u32;
    /// All bits set.
    const MAX: Self;
    /// Zero.
    const ZERO: Self;
    /// One.
    const ONE: Self;
    /// Widened.
    fn to_u64(self) -> u64;
    /// Truncated.
    fn from_u64(v: u64) -> Self;
    /// Truncated to 32 bits (`static_cast<uint32_t>`).
    fn as_u32(self) -> u32;
    /// Wrapping subtraction.
    fn wsub(self, o: Self) -> Self;
    /// Wrapping addition.
    fn wadd(self, o: Self) -> Self;
    /// Number of set bits.
    fn count_ones(self) -> u32;
    /// Trailing zero bits.
    fn trailing_zeros(self) -> u32;
    /// Leading zero bits.
    fn leading_zeros(self) -> u32;
}
macro_rules! radix_key_impl {
    ($($t:ty),*) => {$(
        impl RadixKey for $t {
            const BITS: u32 = <$t>::BITS;
            const MAX: Self = <$t>::MAX;
            const ZERO: Self = 0;
            const ONE: Self = 1;
            #[inline(always)] fn to_u64(self) -> u64 { self as u64 }
            #[inline(always)] fn from_u64(v: u64) -> Self { v as $t }
            #[inline(always)] fn as_u32(self) -> u32 { self as u32 }
            #[inline(always)] fn wsub(self, o: Self) -> Self { self.wrapping_sub(o) }
            #[inline(always)] fn wadd(self, o: Self) -> Self { self.wrapping_add(o) }
            #[inline(always)] fn count_ones(self) -> u32 { <$t>::count_ones(self) }
            #[inline(always)] fn trailing_zeros(self) -> u32 { <$t>::trailing_zeros(self) }
            #[inline(always)] fn leading_zeros(self) -> u32 { <$t>::leading_zeros(self) }
        }
    )*};
}
radix_key_impl!(u32, u64);

/// Index of the top set bit (`k != 0`).
#[inline(always)]
pub fn highest_bit<K: RadixKey>(k: K) -> u32 {
    K::BITS - 1 - k.leading_zeros()
}

/// Portable PEXT: the bits of `k` that `mask` selects, compressed to the
/// bottom, one step per mask bit. The counted path uses it on every CPU so
/// the numbers it records do not depend on BMI2; the timed path uses the
/// instruction.
#[inline(always)]
pub fn pext_soft<K: RadixKey>(k: K, mask: K) -> K {
    let mut out = K::ZERO;
    let mut bit = K::ONE;
    let mut m = mask;
    while m != K::ZERO {
        let low = m & (!m).wadd(K::ONE); // lowest set bit of m
        if k & low != K::ZERO {
            out = out | bit;
        }
        m = m & m.wsub(K::ONE);
        bit = bit << 1;
    }
    out
}

// ---- elements ----------------------------------------------------------------------------

/// The layout promise of an element type for the vector paths.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SimdKind {
    /// No vector path.
    None,
    /// 8-byte elements with a signed 32-bit key in bytes 0-3 whose signed
    /// order is the element order.
    I32,
    /// 16-byte elements with a signed 64-bit key in bytes 0-7.
    I64,
    /// 16-byte elements with an IEEE double in bytes 0-7.
    F64,
}

/// An element the core algorithm sorts: a small plain value with an
/// order, a radix key per chunk, and for chunked (string) keys the chunk
/// bookkeeping.
pub trait Elem: Copy + Default {
    /// The radix key type: `u32` or `u64`.
    type Key: RadixKey;
    /// The key has several chunks (strings).
    const CHUNKED: bool;
    /// Layout promise for the vector paths.
    const SIMD: SimdKind;
    /// Strict order.
    fn less(a: Self, b: Self) -> bool;
    /// Three-way order.
    fn compare(a: Self, b: Self) -> i32;
    /// Three-way order from chunk `chunk` on (both agree on earlier chunks).
    fn compare_from(a: Self, b: Self, chunk: i32) -> i32;
    /// Order-preserving unsigned key of chunk `chunk`.
    fn radix_key(a: Self, chunk: i32) -> Self::Key;
    /// No chunk after this one.
    fn chunk_ends(a: Self, chunk: i32) -> bool;
    /// Counted path: the key bytes `radix_key(a, chunk)` loads, reported to
    /// `sink(address, bytes)`. Nothing for fixed keys.
    #[inline(always)]
    fn report_key_chunk(_a: &Self, _chunk: i32, _sink: &mut impl FnMut(*const u8, usize)) {}
    /// Counted path: the key bytes a compare from chunk `chunk` examines,
    /// reported to `sink` for each side. Nothing for fixed keys.
    #[inline(always)]
    fn report_key_compare(_a: &Self, _b: &Self, _chunk: i32, _sink: &mut impl FnMut(*const u8, usize)) {}
}

// ---- hooks -------------------------------------------------------------------------------

/// A counter of the algorithm's telemetry.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stat {
    /// Route 3 started and gave up.
    Giveups,
    /// Median splits.
    Splits,
    /// Splits whose first direction overflowed.
    SplitRetries,
    /// Partition sorts for 2-4 distinct keys.
    PartSorts,
    /// Dictionary radix attempts.
    DictTries,
    /// Dictionary radix successes.
    DictHits,
    /// Speculative plans that failed verification.
    PlanRetries,
    /// Scatter passes run.
    RadixPasses,
    /// Planned passes whose digit did not vary.
    PassesSkipped,
}

/// Instrumentation. Every call sits under `if H::COUNTED`, so with
/// [`NoHooks`] nothing here is ever compiled; the benchmark supplies a
/// hooks type that records the same calls.
pub trait Hooks {
    /// Whether the hooks record anything.
    const COUNTED: bool;
    /// An element load of `bytes` bytes at `p`.
    #[inline(always)]
    fn on_read(_p: *const u8, _bytes: usize) {}
    /// An element store.
    #[inline(always)]
    fn on_write(_p: *const u8, _bytes: usize) {}
    /// A table (histogram, index) entry read.
    #[inline(always)]
    fn on_table_read(_p: *const u8, _bytes: usize) {}
    /// A table entry write.
    #[inline(always)]
    fn on_table_write(_p: *const u8, _bytes: usize) {}
    /// A table entry read and write.
    #[inline(always)]
    fn on_table_rw(p: *const u8, bytes: usize) {
        Self::on_table_read(p, bytes);
        Self::on_table_write(p, bytes);
    }
    /// A whole table swept sequentially.
    #[inline(always)]
    fn on_table_sweep(_p: *const u8, _entries: usize, _entry_bytes: usize, _read: bool, _write: bool) {}
    /// Key bytes loaded by a compare or a chunk extraction.
    #[inline(always)]
    fn on_key(_p: *const u8, _bytes: usize) {}
    /// A comparison and its outcome.
    #[inline(always)]
    fn on_compare(_less: bool) {}
    /// Adds `n` to a telemetry counter.
    #[inline(always)]
    fn stat(_s: Stat, _n: u64) {}
    /// A route was taken for a range: 1 sorted, 2 reverse, 3 runs, 4
    /// displaced, 5 radix.
    #[inline(always)]
    fn note_route(_r: u8) {}
    /// Recursion depth.
    #[inline(always)]
    fn enter_depth() {}
    /// Recursion depth.
    #[inline(always)]
    fn leave_depth() {}
}

/// The hooks of the library: nothing is recorded.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoHooks;
impl Hooks for NoHooks {
    const COUNTED: bool = false;
}

/// Increments the recursion depth for its lifetime on a counted build.
pub struct DepthScope<H: Hooks>(PhantomData<H>);
impl<H: Hooks> DepthScope<H> {
    /// Enters.
    #[inline(always)]
    pub fn new() -> Self {
        if H::COUNTED {
            H::enter_depth();
        }
        DepthScope(PhantomData)
    }
}
impl<H: Hooks> Default for DepthScope<H> {
    fn default() -> Self {
        Self::new()
    }
}
impl<H: Hooks> Drop for DepthScope<H> {
    #[inline(always)]
    fn drop(&mut self) {
        if H::COUNTED {
            H::leave_depth();
        }
    }
}

// ---- allocation ----------------------------------------------------------------------------

/// An allocation failed. The sort completes anyway, by comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllocError;

/// The allocator of the sort's scratch memory. Blocks are 16-byte aligned.
pub trait Alloc {
    /// `bytes` bytes, 16-byte aligned; a request of 0 bytes is allowed.
    fn allocate(bytes: usize) -> Result<NonNull<u8>, AllocError>;
    /// Frees a block from [`Alloc::allocate`] with the same `bytes`.
    ///
    /// # Safety
    /// `p` came from `allocate(bytes)` on this allocator and is freed once.
    unsafe fn deallocate(p: NonNull<u8>, bytes: usize);
}

/// `n` objects of type `U`, uninitialised.
#[inline]
pub fn alloc_array<A: Alloc, U>(n: usize) -> Result<NonNull<U>, AllocError> {
    if core::mem::align_of::<U>() > 16 {
        return Err(AllocError); // blocks are 16-byte aligned: over-aligned elements take the in-place paths
    }
    let bytes = n.checked_mul(core::mem::size_of::<U>()).ok_or(AllocError)?;
    A::allocate(bytes).map(NonNull::cast)
}
/// Frees an array from [`alloc_array`] with the same `n`.
///
/// # Safety
/// `p` came from `alloc_array::<A, U>(n)` and is freed once.
#[inline]
pub unsafe fn free_array<A: Alloc, U>(p: NonNull<U>, n: usize) {
    // SAFETY: the caller's contract; the byte count is the one alloc_array computed.
    unsafe { A::deallocate(p.cast(), n * core::mem::size_of::<U>()) }
}

// ---- the array view ----------------------------------------------------------------------------

/// The array the algorithm works on: elements, hooks and allocator.
///
/// Views are small copyable handles over memory someone else owns; every
/// access is bounds-checked in debug builds and reported to the hooks on a
/// counted build.
pub trait Arr: Copy {
    /// The element type.
    type T: Elem;
    /// The instrumentation.
    type H: Hooks;
    /// The allocator.
    type A: Alloc;
    /// A view like this one over `n` elements at `p`.
    fn with_range(self, p: *mut Self::T, n: usize) -> Self;
    /// Number of elements.
    fn size(self) -> usize;
    /// The elements.
    fn data(self) -> *mut Self::T;
    /// Element `i`.
    fn get(self, i: usize) -> Self::T;
    /// Stores element `i`.
    fn set(self, i: usize, v: Self::T);
    /// Strict order.
    fn less(self, a: Self::T, b: Self::T) -> bool;
    /// Three-way order.
    fn compare(self, a: Self::T, b: Self::T) -> i32;
    /// Three-way order from a chunk on.
    fn compare_from(self, a: Self::T, b: Self::T, chunk: i32) -> i32;
    /// The radix key of a chunk.
    fn key(x: Self::T, chunk: i32) -> <Self::T as Elem>::Key;
    /// Swaps two elements.
    #[inline(always)]
    fn swap(self, i: usize, j: usize) {
        let t = self.get(i);
        self.set(i, self.get(j));
        self.set(j, t);
    }
    /// The sub-view `[off, off + len)`.
    #[inline(always)]
    fn sub(self, off: usize, len: usize) -> Self {
        debug_assert!(off + len <= self.size());
        // SAFETY: the sub-range lies inside the view.
        self.with_range(unsafe { self.data().add(off) }, len)
    }
    /// Scratch memory.
    #[inline(always)]
    fn alloc_array<U>(n: usize) -> Result<NonNull<U>, AllocError> {
        alloc_array::<Self::A, U>(n)
    }
    /// Frees scratch memory.
    ///
    /// # Safety
    /// As [`free_array`].
    #[inline(always)]
    unsafe fn free_array<U>(p: NonNull<U>, n: usize) {
        // SAFETY: the caller's contract.
        unsafe { free_array::<Self::A, U>(p, n) }
    }
}

/// The library's view: elements of type `T`, hooks `H`, allocator `A`.
pub struct View<T, H, A> {
    p: *mut T,
    n: usize,
    _m: PhantomData<fn() -> (H, A)>,
}
impl<T, H, A> Clone for View<T, H, A> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T, H, A> Copy for View<T, H, A> {}
impl<T, H, A> View<T, H, A> {
    /// A view over `n` elements at `p`.
    ///
    /// The memory must stay valid and unaliased for as long as the view is
    /// used; the view itself is a plain handle.
    #[inline(always)]
    pub fn new(p: *mut T, n: usize) -> Self {
        View { p, n, _m: PhantomData }
    }
}
impl<T: Elem, H: Hooks, A: Alloc> Arr for View<T, H, A> {
    type T = T;
    type H = H;
    type A = A;
    #[inline(always)]
    fn with_range(self, p: *mut T, n: usize) -> Self {
        View::new(p, n)
    }
    #[inline(always)]
    fn size(self) -> usize {
        self.n
    }
    #[inline(always)]
    fn data(self) -> *mut T {
        self.p
    }
    #[inline(always)]
    fn get(self, i: usize) -> T {
        debug_assert!(i < self.n);
        if H::COUNTED {
            H::on_read(self.p.wrapping_add(i) as *const u8, core::mem::size_of::<T>());
        }
        // SAFETY: i < n and the view's memory is valid (the constructor's contract).
        unsafe { *self.p.add(i) }
    }
    #[inline(always)]
    fn set(self, i: usize, v: T) {
        debug_assert!(i < self.n);
        if H::COUNTED {
            H::on_write(self.p.wrapping_add(i) as *const u8, core::mem::size_of::<T>());
        }
        // SAFETY: as get.
        unsafe { *self.p.add(i) = v }
    }
    #[inline(always)]
    fn less(self, a: T, b: T) -> bool {
        if H::COUNTED {
            let r = T::less(a, b);
            T::report_key_compare(&a, &b, 0, &mut |p, n| H::on_key(p, n));
            H::on_compare(r);
            r
        } else {
            T::less(a, b)
        }
    }
    #[inline(always)]
    fn compare(self, a: T, b: T) -> i32 {
        self.compare_from(a, b, 0)
    }
    #[inline(always)]
    fn compare_from(self, a: T, b: T, chunk: i32) -> i32 {
        if H::COUNTED {
            let c = T::compare_from(a, b, chunk);
            T::report_key_compare(&a, &b, chunk, &mut |p, n| H::on_key(p, n));
            H::on_compare(c < 0);
            c
        } else {
            T::compare_from(a, b, chunk)
        }
    }
    #[inline(always)]
    fn key(x: T, chunk: i32) -> T::Key {
        if H::COUNTED {
            T::report_key_chunk(&x, chunk, &mut |p, n| H::on_key(p, n));
        }
        T::radix_key(x, chunk)
    }
}

// ---- scratch buffers -----------------------------------------------------------------------

/// A buffer of elements whose accesses count like those of the main array.
pub struct AuxBuffer<V: Arr> {
    p: NonNull<V::T>,
    n: usize,
    like: V,
}
impl<V: Arr> AuxBuffer<V> {
    /// `n` elements; `like` provides the view type.
    pub fn new(like: V, n: usize) -> Result<Self, AllocError> {
        let p = V::alloc_array::<V::T>(n)?;
        Ok(AuxBuffer { p, n, like })
    }
    /// The buffer as a view.
    #[inline(always)]
    pub fn arr(&self) -> V {
        self.like.with_range(self.p.as_ptr(), self.n)
    }
    /// Capacity in elements.
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.n
    }
    /// The elements.
    #[inline(always)]
    pub fn data(&self) -> *mut V::T {
        self.p.as_ptr()
    }
    /// Drops the contents and reallocates with a new capacity. On failure
    /// the buffer is empty.
    pub fn resize_discard(&mut self, n: usize) -> Result<(), AllocError> {
        // SAFETY: p came from alloc_array with self.n.
        unsafe { V::free_array(self.p, self.n) };
        self.n = 0;
        self.p = NonNull::dangling();
        self.p = V::alloc_array::<V::T>(n)?;
        self.n = n;
        Ok(())
    }
}
impl<V: Arr> Drop for AuxBuffer<V> {
    fn drop(&mut self) {
        // SAFETY: p came from alloc_array with self.n (a dangling p has n == 0
        // and is freed as a zero-size block, which allocate/deallocate pair).
        if self.n > 0 || self.p != NonNull::dangling() {
            unsafe { V::free_array(self.p, self.n) };
        }
    }
}

/// Raw scratch (histograms, tables): tracked as memory only. The
/// algorithms report its accesses through the hooks on their counted path.
pub struct AuxRaw<U, V: Arr> {
    p: NonNull<U>,
    n: usize,
    _v: PhantomData<V>,
}
impl<U, V: Arr> AuxRaw<U, V> {
    /// `n` entries, uninitialised.
    pub fn new(n: usize) -> Result<Self, AllocError> {
        Ok(AuxRaw { p: V::alloc_array::<U>(n)?, n, _v: PhantomData })
    }
    /// The entries.
    #[inline(always)]
    pub fn data(&self) -> *mut U {
        self.p.as_ptr()
    }
    /// Number of entries.
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.n
    }
}
impl<U, V: Arr> Drop for AuxRaw<U, V> {
    fn drop(&mut self) {
        // SAFETY: p came from alloc_array with self.n.
        unsafe { V::free_array(self.p, self.n) };
    }
}

/// A view over `u32` indices whose accesses are counted like element
/// accesses, so index-based algorithms hide nothing (`AuxVec::View`).
pub struct IdxView<H: Hooks> {
    p: *mut u32,
    n: usize,
    _h: PhantomData<H>,
}
impl<H: Hooks> Clone for IdxView<H> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<H: Hooks> Copy for IdxView<H> {}
impl<H: Hooks> IdxView<H> {
    /// A view over `n` indices at `p`.
    #[inline(always)]
    pub fn new(p: *mut u32, n: usize) -> Self {
        IdxView { p, n, _h: PhantomData }
    }
    /// Index `i`.
    #[inline(always)]
    pub fn get(self, i: usize) -> u32 {
        debug_assert!(i < self.n);
        if H::COUNTED {
            H::on_read(self.p.wrapping_add(i) as *const u8, 4);
        }
        // SAFETY: i < n.
        unsafe { *self.p.add(i) }
    }
    /// Stores index `i`.
    #[inline(always)]
    pub fn set(self, i: usize, v: u32) {
        debug_assert!(i < self.n);
        if H::COUNTED {
            H::on_write(self.p.wrapping_add(i) as *const u8, 4);
        }
        // SAFETY: i < n.
        unsafe { *self.p.add(i) = v }
    }
}

// ---- small shared helpers ------------------------------------------------------------------

/// Insertion sort of `a[lo, hi)`.
#[inline]
pub fn insertion_sort<V: Arr>(a: V, lo: usize, hi: usize) {
    for i in lo + 1..hi {
        let v = a.get(i);
        let mut j = i;
        while j > lo {
            let p = a.get(j - 1);
            if !a.less(v, p) {
                break;
            }
            a.set(j, p);
            j -= 1;
        }
        a.set(j, v);
    }
}

/// Forward copy: safe for overlapping ranges when dst <= src.
#[inline]
pub fn copy_forward<V: Arr>(src: V, s: usize, dst: V, d: usize, len: usize) {
    for i in 0..len {
        dst.set(d + i, src.get(s + i));
    }
}
/// Reverses `a[lo, hi)`.
#[inline]
pub fn reverse_range<V: Arr>(a: V, mut lo: usize, mut hi: usize) {
    while lo + 1 < hi {
        hi -= 1;
        a.swap(lo, hi);
        lo += 1;
    }
}
/// floor(log2(n)); 0 for n < 2.
#[inline]
pub fn floor_log2(mut n: usize) -> usize {
    let mut r = 0;
    while n > 1 {
        n >>= 1;
        r += 1;
    }
    r
}

// ---- string chunk keys ---------------------------------------------------------------------
// A string key is consumed in 7-byte chunks. Chunk c is bytes [7c, 7c+7)
// big-endian in the top 56 bits of the key, and the number of valid bytes in
// this chunk (0..7) in the low byte. Lexicographic order of the chunk keys
// equals lexicographic order of the strings, and a shorter string sorts
// before a longer one with the same prefix. A string of length L has
// L / 7 + 1 chunks: the last one has fewer than 7 valid bytes.

/// Bytes per string chunk.
pub const STR_CHUNK_BYTES: u32 = 7;
/// The bytes of a chunk key, above the valid count.
pub const STR_KEY_MASK: u64 = !0xFFu64;

/// The chunk key of bytes `[ptr, ptr + len)` at chunk `chunk`.
///
/// # Safety
/// `len` bytes are readable at `ptr` (when `len > 0`).
#[inline(always)]
pub unsafe fn str_chunk_key(ptr: *const u8, len: u32, chunk: i32) -> u64 {
    let off = (chunk as u32).wrapping_mul(STR_CHUNK_BYTES);
    if off.wrapping_add(8) <= len && off < len {
        // fast path: one 8-byte load
        // SAFETY: off + 8 <= len bytes are readable.
        let w = unsafe { core::ptr::read_unaligned(ptr.add(off as usize) as *const u64) };
        return (u64::from_be(w) & STR_KEY_MASK) | STR_CHUNK_BYTES as u64;
    }
    if off >= len {
        return 0;
    }
    let valid = len - off; // 1..7 bytes left; this runs once per element per pass
    // SAFETY: valid bytes are readable at ptr + off.
    let q = unsafe { ptr.add(off as usize) };
    if valid >= 4 {
        // two overlapping 4-byte loads, branch-free assembly
        // SAFETY: q..q+valid is readable and valid >= 4.
        let hi = unsafe { core::ptr::read_unaligned(q as *const u32) };
        let lo = unsafe { core::ptr::read_unaligned(q.add(valid as usize - 4) as *const u32) };
        let w = ((u32::from_be(hi) as u64) << 32) | ((u32::from_be(lo) as u64) << (8 * (8 - valid)));
        return (w & STR_KEY_MASK) | valid as u64;
    }
    let mut w = 0u64;
    for i in 0..valid as usize {
        // SAFETY: i < valid.
        w = (w << 8) | unsafe { *q.add(i) } as u64;
    }
    w <<= 8 * (STR_CHUNK_BYTES - valid);
    (w << 8) | valid as u64
}
/// The same chunk for a key that sorts in descending order: the bytes are
/// inverted and the valid count is reversed, so a tie on the bytes puts the
/// longer string first.
///
/// # Safety
/// As [`str_chunk_key`].
#[inline(always)]
pub unsafe fn str_chunk_key_desc(ptr: *const u8, len: u32, chunk: i32) -> u64 {
    // SAFETY: the caller's contract.
    let k = unsafe { str_chunk_key(ptr, len, chunk) };
    (!k & STR_KEY_MASK) | (STR_CHUNK_BYTES as u64 - (k & 0xFF))
}
/// No chunk after `chunk` for a string of `len` bytes.
#[inline(always)]
pub fn str_chunk_ends(len: u32, chunk: i32) -> bool {
    (chunk as u32).wrapping_mul(STR_CHUNK_BYTES).wrapping_add(STR_CHUNK_BYTES) > len
}
/// Three-way compare starting at byte `off` (both strings are known to
/// agree on the first `off` bytes when off > 0). A plain byte compare on
/// purpose: inlined variants tripled the branch mispredictions on keys of
/// mixed length in the C++ measurements.
///
/// # Safety
/// `la0` bytes are readable at `pa`, `lb0` at `pb`.
#[inline(always)]
pub unsafe fn str_compare_from(pa: *const u8, la0: u32, pb: *const u8, lb0: u32, off: u32) -> i32 {
    let la = la0.saturating_sub(off);
    let lb = lb0.saturating_sub(off);
    let m = la.min(lb) as usize;
    // SAFETY: m bytes are readable at both offsets (off <= len on the shorter side or m == 0).
    let (sa, sb) = unsafe {
        (
            core::slice::from_raw_parts(if m == 0 { NonNull::dangling().as_ptr() } else { pa.add(off as usize) }, m),
            core::slice::from_raw_parts(if m == 0 { NonNull::dangling().as_ptr() } else { pb.add(off as usize) }, m),
        )
    };
    match sa.cmp(sb) {
        core::cmp::Ordering::Less => -1,
        core::cmp::Ordering::Greater => 1,
        core::cmp::Ordering::Equal => (la > lb) as i32 - (la < lb) as i32,
    }
}
/// The bytes of each string a compare from `off` looks at: up to and
/// including the first differing byte, or the whole common length. Counted
/// path only.
///
/// # Safety
/// As [`str_compare_from`].
pub unsafe fn str_examined(pa: *const u8, la0: u32, pb: *const u8, lb0: u32, off: u32) -> u32 {
    let la = la0.saturating_sub(off);
    let lb = lb0.saturating_sub(off);
    let m = la.min(lb);
    let mut i = 0u32;
    // SAFETY: i < m bytes are readable at both offsets.
    while i < m && unsafe { *pa.add((off + i) as usize) == *pb.add((off + i) as usize) } {
        i += 1;
    }
    if i < m { i + 1 } else { m }
}
/// Bytes `str_chunk_key` loads from a string of `len` bytes at `chunk`.
pub fn str_chunk_bytes(len: u32, chunk: i32) -> u32 {
    let off = (chunk as u32).wrapping_mul(STR_CHUNK_BYTES);
    if off.wrapping_add(8) <= len && off < len {
        return 8;
    }
    if off < len { len - off } else { 0 }
}

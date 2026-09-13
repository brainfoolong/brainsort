//! The public sorts: the port of `brainsort.hpp` minus the C++ overload
//! surface. A key projection is turned into records, the records are
//! sorted by the core algorithm, and the elements are permuted once;
//! sorted, reversed and nearly sorted input is recognised on the elements
//! before anything is built.
use crate::algorithm::displaced::sort_displaced;
use crate::algorithm::{Scratch, brainsort_impl};
use crate::key::{Key, Prescan};
use crate::record::{CompRec, Key32, Key64, KeyElem, Rec32, Rec64, Record, StrRec, radix64_of};
use crate::view::{Alloc, AllocError, Arr, Elem, NoHooks, SimdKind, View, alloc_array, free_array};
use core::cmp::Ordering;
use core::marker::PhantomData;
use core::ops::Deref;
use core::ptr::{self, NonNull};

/// Below this many elements the elements themselves are insertion sorted
/// by key: no allocation, and cheaper than building records.
pub const SMALL_SORT: usize = 32;
/// Elements up to this size are sorted in place on nearly sorted input;
/// larger ones through the records, where moving only the out-of-place
/// elements is cheaper than rewriting every element once.
pub const ELEMENT_ROUTE_MAX: usize = 16;
/// The comparator sort, which has no records to fall back on, sorts
/// nearly sorted elements up to this size in place (the C++ has no limit).
pub const COMPARATOR_ROUTE_MAX: usize = 64;
/// Below this many elements the key a comparator compares is not
/// inferred: the sample, the records and the check cost more than the
/// comparison sort of a small slice saves (level at about 5,000 elements).
pub const INFER_MIN: usize = 4096;
const PRESCAN_BLOCK: usize = 256;

// ---- projections --------------------------------------------------------------------------

/// A borrowed or owned key of an element.
pub struct Owned<K>(pub K);
impl<K> Deref for Owned<K> {
    type Target = K;
    #[inline(always)]
    fn deref(&self) -> &K {
        &self.0
    }
}

/// How the key of an element is obtained: the element itself, a borrow of
/// it, or a value computed from it.
pub trait Proj<T> {
    /// The key type.
    type K: Key + ?Sized;
    /// The held key: a reference or an owned value.
    type Held<'a>: Deref<Target = Self::K>
    where
        Self: 'a,
        T: 'a,
        Self::K: 'a;
    /// The owned key: the key type itself when keys are computed values,
    /// the element type when the elements are the keys.
    type Owned;
    /// The elements are the keys.
    const IDENTITY: bool;
    /// Keys are values that may own bytes: with a byte-string part they are
    /// kept alive for the whole sort.
    const OWNED: bool;
    /// The key of `t`.
    fn hold<'a>(&mut self, t: &'a T) -> Self::Held<'a>;
    /// The key of `t` as a value (only with `OWNED`).
    fn owned(&mut self, t: &T) -> Self::Owned;
    /// An owned key as a key (only with `OWNED`).
    fn as_key(o: &Self::Owned) -> &Self::K;
    /// The element of a radix value (only with `IDENTITY` and an exact key).
    fn from_radix(r: u64) -> Self::Owned;
}
/// The elements are the keys.
pub struct Identity;
impl<T: Key> Proj<T> for Identity {
    type K = T;
    type Held<'a>
        = &'a T
    where
        T: 'a;
    type Owned = T;
    const IDENTITY: bool = true;
    const OWNED: bool = false;
    #[inline(always)]
    fn hold<'a>(&mut self, t: &'a T) -> &'a T {
        t
    }
    fn owned(&mut self, _t: &T) -> T {
        unreachable!("the elements are the keys")
    }
    #[inline(always)]
    fn as_key(o: &T) -> &T {
        o
    }
    #[inline(always)]
    fn from_radix(r: u64) -> T {
        T::from_radix(r)
    }
}
/// A key borrowed from the element.
pub struct ByRef<F>(pub F);
impl<T, K: Key + ?Sized, F: for<'a> FnMut(&'a T) -> &'a K> Proj<T> for ByRef<F> {
    type K = K;
    type Held<'a>
        = &'a K
    where
        F: 'a,
        T: 'a,
        K: 'a;
    type Owned = ();
    const IDENTITY: bool = false;
    const OWNED: bool = false;
    #[inline(always)]
    fn hold<'a>(&mut self, t: &'a T) -> &'a K {
        (self.0)(t)
    }
    fn owned(&mut self, _t: &T) {
        unreachable!("borrowed keys are not materialised")
    }
    fn as_key(_o: &()) -> &K {
        unreachable!("borrowed keys are not materialised")
    }
    fn from_radix(_r: u64) {
        unreachable!("only the elements themselves are written back")
    }
}
/// A key computed from the element.
pub struct ByVal<F>(pub F);
impl<T, K: Key, F: FnMut(&T) -> K> Proj<T> for ByVal<F> {
    type K = K;
    type Held<'a>
        = Owned<K>
    where
        F: 'a,
        T: 'a,
        K: 'a;
    type Owned = K;
    const IDENTITY: bool = false;
    const OWNED: bool = true;
    #[inline(always)]
    fn hold(&mut self, t: &T) -> Owned<K> {
        Owned((self.0)(t))
    }
    #[inline(always)]
    fn owned(&mut self, t: &T) -> K {
        (self.0)(t)
    }
    #[inline(always)]
    fn as_key(o: &K) -> &K {
        o
    }
    fn from_radix(_r: u64) -> K {
        unreachable!("only the elements themselves are written back")
    }
}

// ---- raw buffers ----------------------------------------------------------------------------

/// An array of n objects of type U in memory from A: raw storage.
pub(crate) struct Buf<U, A: Alloc> {
    p: NonNull<U>,
    n: usize,
    _a: PhantomData<A>,
}
impl<U, A: Alloc> Buf<U, A> {
    pub(crate) fn new(n: usize) -> Result<Self, AllocError> {
        Ok(Buf { p: alloc_array::<A, U>(n)?, n, _a: PhantomData })
    }
    #[inline(always)]
    pub(crate) fn ptr(&self) -> *mut U {
        self.p.as_ptr()
    }
}
impl<U, A: Alloc> Drop for Buf<U, A> {
    fn drop(&mut self) {
        // SAFETY: allocated with n.
        unsafe { free_array::<A, U>(self.p, self.n) }
    }
}
/// An array of keys constructed one by one and dropped in the destructor.
struct KeyBuf<K, A: Alloc> {
    buf: Buf<K, A>,
    built: usize,
}
impl<K, A: Alloc> KeyBuf<K, A> {
    fn new(n: usize) -> Result<Self, AllocError> {
        Ok(KeyBuf { buf: Buf::new(n)?, built: 0 })
    }
    #[inline(always)]
    fn push(&mut self, k: K) {
        debug_assert!(self.built < self.buf.n);
        // SAFETY: built < n.
        unsafe { ptr::write(self.buf.ptr().add(self.built), k) };
        self.built += 1;
    }
    #[inline(always)]
    fn get(&self, i: usize) -> &K {
        debug_assert!(i < self.built);
        // SAFETY: i < built, so the slot holds a key.
        unsafe { &*self.buf.ptr().add(i) }
    }
}
impl<K, A: Alloc> Drop for KeyBuf<K, A> {
    fn drop(&mut self) {
        for i in (0..self.built).rev() {
            // SAFETY: the first `built` slots hold keys, dropped once here.
            unsafe { ptr::drop_in_place(self.buf.ptr().add(i)) };
        }
    }
}

// ---- the permutation --------------------------------------------------------------------------

/// An array of n element indices, read and written by position: the index
/// word of sorted records, or a plain index array.
pub(crate) trait Indices: Copy {
    /// The index at position i. Safety: i must be below the array's length.
    unsafe fn index(self, i: usize) -> u32;
    /// Sets the index at position i. Safety: as above.
    unsafe fn set_index(self, i: usize, v: u32);
}
impl<R: Record> Indices for *mut R {
    #[inline(always)]
    unsafe fn index(self, i: usize) -> u32 {
        // SAFETY: the caller's.
        unsafe { (*self.add(i)).index() }
    }
    #[inline(always)]
    unsafe fn set_index(self, i: usize, v: u32) {
        // SAFETY: the caller's.
        unsafe { (*self.add(i)).set_index(v) }
    }
}
impl Indices for *mut u32 {
    #[inline(always)]
    unsafe fn index(self, i: usize) -> u32 {
        // SAFETY: the caller's.
        unsafe { *self.add(i) }
    }
    #[inline(always)]
    unsafe fn set_index(self, i: usize, v: u32) {
        // SAFETY: the caller's.
        unsafe { *self.add(i) = v }
    }
}

/// Applies the permutation the sorted indices describe: out[i] = in[ix[i]].
/// In place, following cycles, with bitwise moves; works for any type.
fn permute_cycles<T, I: Indices>(v: &mut [T], ix: I, n: usize) {
    let first = v.as_mut_ptr();
    for i in 0..n {
        // SAFETY (whole loop): every index is below n; each element is read
        // once and written once per cycle, so no element is duplicated or
        // lost even though the moves are bitwise.
        unsafe {
            if ix.index(i) as usize == i {
                continue;
            }
            let tmp = ptr::read(first.add(i));
            let mut j = i;
            loop {
                let k = ix.index(j) as usize;
                ix.set_index(j, j as u32);
                if k == i {
                    ptr::write(first.add(j), tmp);
                    break;
                }
                ptr::copy_nonoverlapping(first.add(k), first.add(j), 1);
                j = k;
            }
        }
    }
}
/// The same through a gather buffer: sequential writes and independent
/// loads. Falls back to the cycle walk if the buffer cannot be allocated.
/// When the input was nearly sorted (`sparse`), most elements are already
/// in their final place and the cycle walk moves only the others, so it is
/// taken if at most an eighth are out of place.
fn permute<T, I: Indices, A: Alloc>(v: &mut [T], ix: I, n: usize, sparse: bool) {
    if sparse {
        let mut moved = 0usize;
        for i in 0..n {
            // SAFETY: i < n indices.
            moved += (unsafe { ix.index(i) } as usize != i) as usize;
        }
        if moved <= n / 8 {
            permute_cycles(v, ix, n);
            return;
        }
    }
    if core::mem::size_of::<T>() > 0 {
        if let Ok(tmp) = Buf::<T, A>::new(n) {
            let t = tmp.ptr();
            let first = v.as_mut_ptr();
            // SAFETY: t has n slots; every index is below n and the indices
            // are a permutation, so each element is copied out once and back
            // once.
            unsafe {
                for i in 0..n {
                    ptr::copy_nonoverlapping(first.add(ix.index(i) as usize), t.add(i), 1);
                }
                ptr::copy_nonoverlapping(t, first, n);
            }
            return;
        }
    }
    permute_cycles(v, ix, n);
}

// ---- the prescan --------------------------------------------------------------------------------

/// What one pass over the keys found, before any record is built or
/// memory allocated.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Shape {
    /// Nothing recognised.
    Unordered,
    /// No descents.
    Sorted,
    /// No ascents.
    Reversed,
    /// At most n/16 descents.
    NearlySorted,
}
/// The result of the prescan.
#[derive(Clone, Copy, Debug)]
pub struct PrescanResult {
    /// The shape.
    pub shape: Shape,
    /// Descents.
    pub descents: usize,
    /// Ascents seen by the pass: none are counted before the first descent.
    pub ascents: usize,
}
impl Default for PrescanResult {
    fn default() -> Self {
        PrescanResult { shape: Shape::Unordered, descents: 0, ascents: 0 }
    }
}
#[inline(always)]
fn classify(r: &mut PrescanResult, n: usize, desc: usize, asc: usize) {
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
}

/// Whether the pairs before `upto` hold an ascent.
fn prefix_ascent<T, P: Proj<T>>(v: &[T], upto: usize, proj: &mut P) -> bool {
    let mut prev = proj.hold(&v[0]);
    for i in 1..upto {
        let cur = proj.hold(&v[i]);
        if prev.cmp_key(&cur) == Ordering::Less {
            return true;
        }
        prev = cur;
    }
    false
}
/// One pass over the keys in three phases: the longest non-descending
/// prefix costs one compare and one branch per pair, so does the
/// non-ascending run after the first descent, and only from the first
/// ascent after that are the pairs counted. The prefix's ascents are looked
/// up only when the bail-out or the classification asks for them, so every
/// decision is the one a plain count would make. The pass stops as soon as
/// the counts prove the input unordered (a few hundred elements into random
/// input), so an expensive key function is called few times on such input.
fn prescan<T, P: Proj<T>>(v: &[T], proj: &mut P) -> PrescanResult {
    let n = v.len();
    #[cfg(all(target_arch = "x86_64", not(brainsort_no_simd)))]
    {
        if P::IDENTITY && <P::K as Key>::PRESCAN != Prescan::None && crate::cpu::have_avx2() {
            // SAFETY: AVX2 was detected; with IDENTITY the elements are the
            // keys, whose layout PRESCAN names.
            let (p, n) = (v.as_ptr() as *const u8, n);
            use crate::simd::x86::prescan_avx2;
            return unsafe {
                match <P::K as Key>::PRESCAN {
                    Prescan::I32 => prescan_avx2::<{ Prescan::I32.code() }>(p, n),
                    Prescan::U32 => prescan_avx2::<{ Prescan::U32.code() }>(p, n),
                    Prescan::I64 => prescan_avx2::<{ Prescan::I64.code() }>(p, n),
                    Prescan::U64 => prescan_avx2::<{ Prescan::U64.code() }>(p, n),
                    Prescan::F32 => prescan_avx2::<{ Prescan::F32.code() }>(p, n),
                    Prescan::F64 => prescan_avx2::<{ Prescan::F64.code() }>(p, n),
                    Prescan::None => unreachable!(),
                }
            };
        }
    }
    let mut r = PrescanResult::default();
    let mut prev = proj.hold(&v[0]);
    let mut i = 1;
    while i < n {
        let cur = proj.hold(&v[i]);
        let c = prev.cmp_key(&cur);
        prev = cur;
        if c == Ordering::Greater {
            break;
        }
        i += 1;
    }
    if i == n {
        r.shape = Shape::Sorted;
        return r;
    }
    let (mut desc, mut j) = (1usize, i + 1);
    let mut prefix: Option<bool> = None; // whether the prefix has an ascent, once asked
    while j < n {
        let cur = proj.hold(&v[j]);
        let c = prev.cmp_key(&cur);
        prev = cur;
        if c == Ordering::Less {
            break;
        }
        desc += (c == Ordering::Greater) as usize;
        // Unordered for certain: the displaced-element route gives up once
        // the displaced elements exceed scanned/8 + 64, and every descent
        // displaces at least one element.
        if j & (PRESCAN_BLOCK - 1) == 0 && desc > (j >> 3) + 64 && *prefix.get_or_insert_with(|| prefix_ascent(v, i, proj)) {
            return r;
        }
        j += 1;
    }
    if j == n {
        let asc = prefix.unwrap_or_else(|| prefix_ascent(v, i, proj)) as usize;
        classify(&mut r, n, desc, asc);
        return r;
    }
    let mut asc = 1usize;
    j += 1;
    while j < n {
        let cur = proj.hold(&v[j]);
        let c = prev.cmp_key(&cur);
        prev = cur;
        desc += (c == Ordering::Greater) as usize;
        asc += (c == Ordering::Less) as usize;
        if j & (PRESCAN_BLOCK - 1) == 0 && desc > (j >> 3) + 64 {
            return r;
        }
        j += 1;
    }
    classify(&mut r, n, desc, asc);
    r
}
/// Whether the pairs before `upto` hold an ascent, by the comparator.
fn prefix_ascent_cmp<T, F: FnMut(&T, &T) -> Ordering>(v: &[T], upto: usize, cmp: &mut F) -> bool {
    (1..upto).any(|i| cmp(&v[i - 1], &v[i]) == Ordering::Less)
}
/// The prescan of the comparator sort: the non-descending prefix and the
/// strictly descending run after the first descent cost one call and one
/// branch per pair; ties and the pairs after the first ascent are counted.
/// The bail-out a plain count would have taken inside the run is replayed
/// after it, so every decision is the count's.
pub(crate) fn prescan_cmp<T, F: FnMut(&T, &T) -> Ordering>(v: &[T], cmp: &mut F) -> PrescanResult {
    let n = v.len();
    let mut r = PrescanResult::default();
    let i = 1 + v.iter().zip(&v[1..]).take_while(|(p, c)| cmp(p, c) != Ordering::Greater).count();
    if i == n {
        r.shape = Shape::Sorted;
        return r;
    }
    let mut prefix: Option<bool> = None; // whether the prefix has an ascent, once asked
    // Asked the other way round, so that a comparator built from two `less`
    // calls answers a descent with one.
    let mut j = i + 1 + v[i..].iter().zip(&v[i + 1..]).take_while(|(p, c)| cmp(c, p) == Ordering::Less).count();
    let mut b = (i + PRESCAN_BLOCK) & !(PRESCAN_BLOCK - 1);
    while b < j {
        if b - i + 1 > (b >> 3) + 64 {
            if *prefix.get_or_insert_with(|| prefix_ascent_cmp(v, i, cmp)) {
                return r;
            }
            break;
        }
        b += PRESCAN_BLOCK;
    }
    let mut desc = j - i;
    if j == n {
        let asc = prefix.unwrap_or_else(|| prefix_ascent_cmp(v, i, cmp)) as usize;
        classify(&mut r, n, desc, asc);
        return r;
    }
    while j < n {
        let c = cmp(&v[j], &v[j - 1]);
        if c == Ordering::Greater {
            break;
        }
        desc += (c == Ordering::Less) as usize;
        if j & (PRESCAN_BLOCK - 1) == 0 && desc > (j >> 3) + 64 && *prefix.get_or_insert_with(|| prefix_ascent_cmp(v, i, cmp)) {
            return r;
        }
        j += 1;
    }
    if j == n {
        let asc = prefix.unwrap_or_else(|| prefix_ascent_cmp(v, i, cmp)) as usize;
        classify(&mut r, n, desc, asc);
        return r;
    }
    let mut asc = 1usize;
    j += 1;
    while j < n {
        let c = cmp(&v[j - 1], &v[j]);
        desc += (c == Ordering::Greater) as usize;
        asc += (c == Ordering::Less) as usize;
        if j & (PRESCAN_BLOCK - 1) == 0 && desc > (j >> 3) + 64 {
            return r;
        }
        j += 1;
    }
    classify(&mut r, n, desc, asc);
    r
}

/// Reverses a non-increasing range in place with each group of equal keys
/// put back into input order, so the result is stable. Strictly decreasing
/// input (every pair a descent) has no groups.
pub(crate) fn reverse_stable<T>(v: &mut [T], s: &PrescanResult, mut eq: impl FnMut(&T, &T) -> bool) {
    let n = v.len();
    v.reverse();
    if s.descents == n - 1 {
        return;
    }
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && eq(&v[i], &v[j]) {
            j += 1;
        }
        if j - i > 1 {
            v[i..j].reverse();
        }
        i = j;
    }
}

// ---- the small sort ------------------------------------------------------------------------------

/// Insertion sort of the elements by `less`; a panic in `less` leaves every
/// element in the slice (the element being inserted goes back into its hole).
pub(crate) fn small_sort<T>(v: &mut [T], mut less: impl FnMut(&T, &T) -> bool) {
    struct Hole<T> {
        src: *mut T,
        dst: *mut T,
    }
    impl<T> Drop for Hole<T> {
        fn drop(&mut self) {
            // SAFETY: src holds the element in flight, dst is the hole.
            unsafe { ptr::copy_nonoverlapping(self.src, self.dst, 1) };
        }
    }
    let n = v.len();
    let p = v.as_mut_ptr();
    for i in 1..n {
        // SAFETY: the element at i is moved into `tmp`; the hole travels
        // left while the predecessors shift right; the guard writes `tmp`
        // into the hole on every exit, panic included.
        unsafe {
            if !less(&*p.add(i), &*p.add(i - 1)) {
                continue;
            }
            let mut tmp = core::mem::ManuallyDrop::new(ptr::read(p.add(i)));
            let mut hole = Hole { src: &mut *tmp as *mut T, dst: p.add(i - 1) };
            ptr::copy_nonoverlapping(p.add(i - 1), p.add(i), 1);
            let mut j = i - 1;
            while j > 0 && less(&*hole.src, &*p.add(j - 1)) {
                ptr::copy_nonoverlapping(p.add(j - 1), p.add(j), 1);
                j -= 1;
                hole.dst = p.add(j);
            }
            drop(hole);
        }
    }
}

// ---- the record sort ------------------------------------------------------------------------------

/// Sorts `v` by the projection through records of type R. Returns
/// Ok(false), with the slice untouched, if the keys could not be
/// represented.
fn sort_records_as<T, P: Proj<T>, A: Alloc, R: Record>(v: &mut [T], proj: &mut P, sparse: bool) -> Result<bool, AllocError> {
    let n = v.len();
    let materialise = P::OWNED && <P::K as Key>::SHAPE.has_bytes(); // keys are temporaries that own their bytes
    let recs = Buf::<R, A>::new(n)?;
    let rec = recs.ptr();
    let mut scratch = Scratch::<View<R, NoHooks, A>>::new(); // the sorted records may end up in its buffer
    let mut keys: Option<KeyBuf<P::Owned, A>> = None; // lives until the elements are permuted
    if materialise {
        let mut kb = KeyBuf::<P::Owned, A>::new(n)?;
        for t in v.iter() {
            kb.push(proj.owned(t));
        }
        let kb = keys.insert(kb);
        for i in 0..n {
            match R::build(P::as_key(kb.get(i)), i as u32) {
                // SAFETY: i < n records.
                Some(r) => unsafe { ptr::write(rec.add(i), r) },
                None => return Ok(false),
            }
        }
    } else {
        for (i, t) in v.iter().enumerate() {
            let k = proj.hold(t);
            match R::build(&*k, i as u32) {
                // SAFETY: i < n records.
                Some(r) => unsafe { ptr::write(rec.add(i), r) },
                None => return Ok(false),
            }
        }
    }
    let in_src = brainsort_impl(View::<R, NoHooks, A>::new(rec, n), &mut scratch, true, 0)?;
    let rec: *mut R = if in_src { rec } else { scratch.buffer().expect("the result is in the scratch buffer") };
    permute::<T, *mut R, A>(v, rec, n, sparse);
    drop(keys);
    Ok(true)
}

/// The float element of a slice whose key type is `f32` or `f64` (by its
/// `PRESCAN`): whether it is a zero, and a negative zero.
///
/// # Safety
/// `PRESCAN` is `F32` or `F64` only for the key types `f32` and `f64`, and
/// `IDENTITY` makes `T` that type.
#[inline(always)]
unsafe fn float_zero<T, P: Proj<T>>(t: &T) -> (bool, bool) {
    // SAFETY: the caller's contract.
    unsafe {
        match <P::K as Key>::PRESCAN {
            Prescan::F64 => {
                let d = *(t as *const T as *const f64);
                (d == 0.0, d.to_bits() == 0x8000_0000_0000_0000)
            }
            Prescan::F32 => {
                let f = *(t as *const T as *const f32);
                (f == 0.0, f.to_bits() == 0x8000_0000)
            }
            _ => (false, false),
        }
    }
}

/// A slice whose elements are keys of up to 64 bits that invert from their
/// radix form (`i32`, `i64`, `u64`, `f64`, a pointer): the keys themselves
/// are sorted as `KE` (4 or 8 bytes per element, no index) and written back
/// from their radix form. The two zeros of an `f32` or `f64` share a key
/// and the sort keeps them in input order, so the negative ones are put
/// back by their rank among the zeros.
fn sort_keys_only<T, P: Proj<T>, A: Alloc, KE: KeyElem>(v: &mut [T], proj: &mut P) -> Result<(), AllocError> {
    let n = v.len();
    let flt = matches!(<P::K as Key>::PRESCAN, Prescan::F32 | Prescan::F64);
    let keys = Buf::<KE, A>::new(n)?;
    let k = keys.ptr();
    let mut scratch = Scratch::<View<KE, NoHooks, A>>::new(); // the sorted keys may end up in its buffer
    let mut neg_zeros = 0usize;
    for (i, t) in v.iter().enumerate() {
        let key = proj.hold(t);
        // SAFETY: i < n keys.
        unsafe { ptr::write(k.add(i), KE::from_radix(radix64_of(&*key))) };
        if flt {
            // SAFETY: PRESCAN is F32 or F64 only for the key types f32 and f64, and IDENTITY makes T that type.
            neg_zeros += unsafe { float_zero::<T, P>(t) }.1 as usize;
        }
    }
    let mut ranks: Option<Buf<u32, A>> = None; // of the negative zeros among the zeros, in input order
    if neg_zeros > 0 {
        let r = ranks.insert(Buf::<u32, A>::new(neg_zeros)?).ptr();
        let (mut z, mut j) = (0usize, 0usize);
        for t in v.iter() {
            // SAFETY: as above.
            let (zero, negative) = unsafe { float_zero::<T, P>(t) };
            if zero {
                if negative {
                    // SAFETY: j < neg_zeros, the number of negative zeros counted above.
                    unsafe { ptr::write(r.add(j), z as u32) };
                    j += 1;
                }
                z += 1;
            }
        }
    }
    let in_src = brainsort_impl(View::<KE, NoHooks, A>::new(k, n), &mut scratch, true, 0)?;
    let k: *const KE = if in_src { k } else { scratch.buffer().expect("the result is in the scratch buffer") };
    let first = v.as_mut_ptr() as *mut P::Owned;
    for i in 0..n {
        // SAFETY: IDENTITY means T is K is P::Owned, a plain fixed key without a destructor; i < n.
        unsafe { ptr::write(first.add(i), P::from_radix((*k.add(i)).radix())) };
    }
    if let Some(ranks) = ranks {
        // the zeros are the keys of signed value 0: the first is found by bisection
        let (mut lo, mut hi) = (0usize, n);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            // SAFETY: mid < n.
            if unsafe { (*k.add(mid)).is_negative() } {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let first = v.as_mut_ptr();
        for j in 0..neg_zeros {
            // SAFETY: rank j is below the number of zeros, which start at lo and end within n;
            // T is f32 or f64 by PRESCAN and IDENTITY.
            unsafe {
                let p = first.add(lo + *ranks.ptr().add(j) as usize);
                match <P::K as Key>::PRESCAN {
                    Prescan::F64 => ptr::write(p as *mut f64, -0.0),
                    _ => ptr::write(p as *mut f32, -0.0),
                }
            }
        }
    }
    Ok(())
}

fn sort_records<T, P: Proj<T>, A: Alloc>(v: &mut [T], proj: &mut P, sparse: bool) -> Result<bool, AllocError> {
    let s = <P::K as Key>::SHAPE;
    let keys_only = P::IDENTITY && (<P::K as Key>::EXACT_FIXED || matches!(<P::K as Key>::PRESCAN, Prescan::F32 | Prescan::F64));
    if s.all_fixed() && s.total_bits() <= 32 && keys_only {
        sort_keys_only::<T, P, A, Key32>(v, proj).map(|()| true)
    } else if s.all_fixed() && s.total_bits() <= 32 {
        sort_records_as::<T, P, A, Rec32>(v, proj, sparse)
    } else if s.all_fixed() && s.total_bits() <= 64 && keys_only {
        sort_keys_only::<T, P, A, Key64>(v, proj).map(|()| true)
    } else if s.all_fixed() && s.total_bits() <= 64 {
        sort_records_as::<T, P, A, Rec64>(v, proj, sparse)
    } else if s.n == 1 && s.part(0).bytes && !s.part(0).desc {
        sort_records_as::<T, P, A, StrRec<false>>(v, proj, sparse)
    } else if s.n == 1 && s.part(0).bytes && s.part(0).desc {
        sort_records_as::<T, P, A, StrRec<true>>(v, proj, sparse)
    } else {
        sort_records_as::<T, P, A, CompRec<P::K>>(v, proj, sparse)
    }
}

// ---- the element route --------------------------------------------------------------------------
// The elements themselves as the array the core's displaced-element route
// sorts, for nearly sorted input: no records, no permutation, the few
// displaced elements are pulled out, sorted and merged back in place. The
// elements are moved bitwise through a plain copyable stand-in of the same
// size and alignment.

/// A bitwise stand-in for an element: `W` words of `U`, possibly
/// uninitialised (an element's padding bytes are), so copying one is a
/// plain memory copy that never reads the bytes as values.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Bits<U: Copy, const W: usize>([core::mem::MaybeUninit<U>; W]);
impl<U: Copy, const W: usize> Default for Bits<U, W> {
    fn default() -> Self {
        Bits([core::mem::MaybeUninit::uninit(); W])
    }
}
impl<U: Copy, const W: usize> Elem for Bits<U, W> {
    type Key = u32;
    const CHUNKED: bool = false;
    const SIMD: SimdKind = SimdKind::None;
    fn less(_a: Self, _b: Self) -> bool {
        unreachable!("the element view compares through its order")
    }
    fn compare(_a: Self, _b: Self) -> i32 {
        unreachable!()
    }
    fn compare_from(_a: Self, _b: Self, _c: i32) -> i32 {
        unreachable!()
    }
    fn radix_key(_a: Self, _c: i32) -> u32 {
        unreachable!("the element route never takes the radix route")
    }
    fn chunk_ends(_a: Self, _c: i32) -> bool {
        true
    }
}

/// The view of the caller's elements for the displaced-element route.
pub struct ElemView<T, E: Elem, O, A> {
    p: *mut E,
    n: usize,
    order: *mut O,
    _m: PhantomData<fn() -> (T, A)>,
}
impl<T, E: Elem, O, A> Clone for ElemView<T, E, O, A> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T, E: Elem, O, A> Copy for ElemView<T, E, O, A> {}
impl<T, E: Elem, O: FnMut(&T, &T) -> i32, A: Alloc> Arr for ElemView<T, E, O, A> {
    type T = E;
    type H = NoHooks;
    type A = A;
    #[inline(always)]
    fn with_range(self, p: *mut E, n: usize) -> Self {
        ElemView { p, n, order: self.order, _m: PhantomData }
    }
    #[inline(always)]
    fn size(self) -> usize {
        self.n
    }
    #[inline(always)]
    fn data(self) -> *mut E {
        self.p
    }
    #[inline(always)]
    fn get(self, i: usize) -> E {
        debug_assert!(i < self.n);
        // SAFETY: i < n.
        unsafe { *self.p.add(i) }
    }
    #[inline(always)]
    fn set(self, i: usize, v: E) {
        debug_assert!(i < self.n);
        // SAFETY: i < n.
        unsafe { *self.p.add(i) = v }
    }
    #[inline(always)]
    fn less(self, a: E, b: E) -> bool {
        self.compare(a, b) < 0
    }
    #[inline(always)]
    fn compare(self, a: E, b: E) -> i32 {
        // SAFETY: E has the size and alignment of T and holds an element's
        // bytes; the order is called sequentially on one thread, so the
        // exclusive access to the closure is not aliased.
        unsafe { (*self.order)(&*(&a as *const E as *const T), &*(&b as *const E as *const T)) }
    }
    #[inline(always)]
    fn compare_from(self, a: E, b: E, _chunk: i32) -> i32 {
        self.compare(a, b)
    }
    fn key(_x: E, _chunk: i32) -> E::Key {
        unreachable!("the element route never takes the radix route")
    }
}

/// Runs the displaced-element route on elements of up to
/// `COMPARATOR_ROUTE_MAX` bytes. Ok(true) when the slice is sorted;
/// Ok(false), with the slice untouched, when the route gave up. A panic in
/// `order` leaves every element in the slice.
pub(crate) fn sort_displaced_elements<T, A: Alloc, O: FnMut(&T, &T) -> i32>(v: &mut [T], mut order: O) -> bool {
    macro_rules! run {
        ($u:ty, $w:expr) => {{
            let view = ElemView::<T, Bits<$u, $w>, O, A> { p: v.as_mut_ptr() as *mut Bits<$u, $w>, n: v.len(), order: &mut order as *mut O, _m: PhantomData };
            return sort_displaced(view, v.len()).unwrap_or(false); // nothing was moved: the route restores the range before it gives up
        }};
    }
    let (size, align) = (core::mem::size_of::<T>(), core::mem::align_of::<T>());
    if size == 0 || size > COMPARATOR_ROUTE_MAX || align > 8 {
        return false;
    }
    match (align, size) {
        (8, 8) => run!(u64, 1),
        (8, 16) => run!(u64, 2),
        (8, 24) => run!(u64, 3),
        (8, 32) => run!(u64, 4),
        (8, 40) => run!(u64, 5),
        (8, 48) => run!(u64, 6),
        (8, 56) => run!(u64, 7),
        (8, 64) => run!(u64, 8),
        (4, 4) => run!(u32, 1),
        (4, 8) => run!(u32, 2),
        (4, 12) => run!(u32, 3),
        (4, 16) => run!(u32, 4),
        (4, 20) => run!(u32, 5),
        (4, 24) => run!(u32, 6),
        (4, 28) => run!(u32, 7),
        (4, 32) => run!(u32, 8),
        (4, 36) => run!(u32, 9),
        (4, 40) => run!(u32, 10),
        (4, 44) => run!(u32, 11),
        (4, 48) => run!(u32, 12),
        (4, 52) => run!(u32, 13),
        (4, 56) => run!(u32, 14),
        (4, 60) => run!(u32, 15),
        (4, 64) => run!(u32, 16),
        (2, 2) => run!(u16, 1),
        (2, 4) => run!(u16, 2),
        (2, 6) => run!(u16, 3),
        (2, 8) => run!(u16, 4),
        (2, 10) => run!(u16, 5),
        (2, 12) => run!(u16, 6),
        (2, 14) => run!(u16, 7),
        (2, 16) => run!(u16, 8),
        (1, 1) => run!(u8, 1),
        (1, 2) => run!(u8, 2),
        (1, 3) => run!(u8, 3),
        (1, 4) => run!(u8, 4),
        (1, 5) => run!(u8, 5),
        (1, 6) => run!(u8, 6),
        (1, 7) => run!(u8, 7),
        (1, 8) => run!(u8, 8),
        (1, 9) => run!(u8, 9),
        (1, 10) => run!(u8, 10),
        (1, 11) => run!(u8, 11),
        (1, 12) => run!(u8, 12),
        (1, 13) => run!(u8, 13),
        (1, 14) => run!(u8, 14),
        (1, 15) => run!(u8, 15),
        (1, 16) => run!(u8, 16),
        _ => false,
    }
}

// ---- the entry points -----------------------------------------------------------------------

/// Sorts `v` by the projection. Generic over the allocator for the tests.
pub fn sort_by_key_impl<T, P: Proj<T>, A: Alloc>(v: &mut [T], mut proj: P) {
    const {
        assert!(<P::K as Key>::SHAPE.is_valid(), "brainsort: a key must have between 1 and 32 leaves");
    }
    let n = v.len();
    if n < 2 {
        return;
    }
    if n <= SMALL_SORT {
        small_sort(v, |a, b| proj.hold(a).cmp_key(&proj.hold(b)) == Ordering::Less);
        return;
    }
    let s = prescan(v, &mut proj);
    if s.shape == Shape::Sorted {
        return;
    }
    if s.shape == Shape::Reversed {
        reverse_stable(v, &s, |a, b| proj.hold(a).cmp_key(&proj.hold(b)) == Ordering::Equal);
        return;
    }
    let nearly = s.shape == Shape::NearlySorted;
    // Small elements are sorted in place by the displaced-element route;
    // large ones go through the records and a sparse permutation, which
    // moves only the elements that are out of place.
    if nearly && core::mem::size_of::<T>() <= ELEMENT_ROUTE_MAX && sort_displaced_elements::<T, A, _>(v, |a, b| ord3(proj.hold(a).cmp_key(&proj.hold(b)))) {
        return;
    }
    if n <= u32::MAX as usize {
        if let Ok(true) = sort_records::<T, P, A>(v, &mut proj, nearly) {
            return;
        }
        // the range is untouched: sort it by comparison instead
    }
    v.sort_by(|a, b| proj.hold(a).cmp_key(&proj.hold(b)));
}

#[inline(always)]
pub(crate) fn ord3(o: Ordering) -> i32 {
    match o {
        Ordering::Less => -1,
        Ordering::Equal => 0,
        Ordering::Greater => 1,
    }
}

/// Sorts `v` by a comparator: sorted, reversed and nearly sorted input is
/// handled on the elements; everything else is a comparison sort, the
/// standard library's stable sort, on the elements themselves up to 16
/// bytes and through an index array beyond that.
pub fn sort_by_impl<T, A: Alloc, F: FnMut(&T, &T) -> Ordering>(v: &mut [T], mut cmp: F) {
    let n = v.len();
    if n < 2 {
        return;
    }
    if n <= SMALL_SORT {
        small_sort(v, |a, b| cmp(a, b) == Ordering::Less);
        return;
    }
    let s = prescan_cmp(v, &mut cmp);
    if s.shape == Shape::Sorted {
        return;
    }
    if s.shape == Shape::Reversed {
        reverse_stable(v, &s, |a, b| cmp(a, b) == Ordering::Equal);
        return;
    }
    let nearly = s.shape == Shape::NearlySorted;
    if nearly && sort_displaced_elements::<T, A, _>(v, |a, b| ord3(cmp(a, b))) {
        return;
    }
    if core::mem::size_of::<T>() > ELEMENT_ROUTE_MAX && n <= u32::MAX as usize {
        // The element route above tried the displaced elements in place up
        // to COMPARATOR_ROUTE_MAX bytes and gave up; larger elements get
        // that route on the indices.
        let try_displaced = nearly && core::mem::size_of::<T>() > COMPARATOR_ROUTE_MAX;
        if sort_by_indices::<T, A, F>(v, &mut cmp, nearly, try_displaced, true) {
            return;
        }
    }
    v.sort_by(cmp);
}

/// The comparator sort of elements over 16 bytes: an array of the indices
/// 0..n is sorted by the order of the elements it points to (the
/// displaced-element route first when asked, the standard library's stable
/// sort otherwise) and the elements are permuted once. The passes move 4
/// bytes per element instead of the element, and a comparator that panics
/// leaves the slice untouched, because nothing moves before the last
/// comparison. False, with the slice untouched, if the index array cannot
/// be allocated, or, without `fallback`, if the displaced-element route
/// was the only one asked for and gave up.
pub(crate) fn sort_by_indices<T, A: Alloc, F: FnMut(&T, &T) -> Ordering>(v: &mut [T], cmp: &mut F, sparse: bool, try_displaced: bool, fallback: bool) -> bool {
    let n = v.len();
    let Ok(idx) = Buf::<u32, A>::new(n) else {
        return false;
    };
    let ix = idx.ptr();
    for i in 0..n {
        // SAFETY: n slots.
        unsafe { ptr::write(ix.add(i), i as u32) };
    }
    let base = v.as_ptr();
    // SAFETY: base points at n elements that are only read (through
    // indices below n) until the permutation, after the last comparison.
    let mut order = |i: &u32, j: &u32| unsafe { cmp(&*base.add(*i as usize), &*base.add(*j as usize)) };
    // SAFETY: ix holds n initialised indices, owned by `idx` for the whole
    // function.
    let indices = unsafe { core::slice::from_raw_parts_mut(ix, n) };
    if !(try_displaced && sort_displaced_elements::<u32, A, _>(indices, |i, j| ord3(order(i, j)))) {
        if !fallback {
            return false;
        }
        indices.sort_by(order);
    }
    permute::<T, *mut u32, A>(v, ix, n, sparse);
    true
}

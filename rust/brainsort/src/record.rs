//! The sort records: the port of `records.hpp`.
//!
//! The public API never sorts the caller's elements directly. It builds an
//! array of small records, each holding the radix form of one key and the
//! original index, sorts those with the core algorithm, and then permutes
//! the elements once. Four record types cover every key shape:
//!
//! - [`Rec32`]: fixed keys of up to 32 bits in total (8 bytes, AVX2 scout)
//! - [`Rec64`]: fixed keys of up to 64 bits in total (16 bytes, AVX2 scout)
//! - [`StrRec`]: a single byte string (16 bytes, chunked)
//! - [`CompRec`]: anything else: a sequence of fixed and byte-string parts
//!
//! Several fixed keys are packed into one radix value (a `(i32, i32)` is one
//! 64-bit key), so composite keys of integers stay on the fast paths.
use crate::key::{Key, Leaf, PartSink, SlotTree};
use crate::view::{Elem, STR_CHUNK_BYTES, SimdKind, str_chunk_bytes, str_chunk_ends, str_chunk_key, str_chunk_key_desc, str_compare_from, str_examined};

/// A sorted record: its original index can be read and rewritten.
pub trait Record: Elem {
    /// The record of `key` at index `idx`; `None` if the key cannot be
    /// represented (a string of 2^32 bytes or more).
    fn build<K: Key + ?Sized>(key: &K, idx: u32) -> Option<Self>;
    /// The original index.
    fn index(&self) -> u32;
    /// Rewrites the original index.
    fn set_index(&mut self, i: u32);
}

// ---- fixed-key records ------------------------------------------------------------
// The key is stored as the signed integer whose signed order equals the
// radix order (radix ^ sign bit), which is what the vector compares expect.

/// Fixed keys of up to 32 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Rec32 {
    /// The signed form of the radix key.
    pub key: i32,
    /// The original index.
    pub idx: u32,
}
/// Fixed keys of up to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Rec64 {
    /// The signed form of the radix key.
    pub key: i64,
    /// The original index.
    pub idx: u32,
    /// Spare (remembers a negative zero for `f64` write-back).
    pub pad: u32,
}
impl Elem for Rec32 {
    type Key = u32;
    const CHUNKED: bool = false;
    const SIMD: SimdKind = SimdKind::I32;
    #[inline(always)]
    fn less(a: Self, b: Self) -> bool {
        a.key < b.key
    }
    #[inline(always)]
    fn compare(a: Self, b: Self) -> i32 {
        (a.key > b.key) as i32 - (a.key < b.key) as i32
    }
    #[inline(always)]
    fn compare_from(a: Self, b: Self, _chunk: i32) -> i32 {
        Self::compare(a, b)
    }
    #[inline(always)]
    fn radix_key(a: Self, _chunk: i32) -> u32 {
        (a.key as u32) ^ 0x8000_0000
    }
    #[inline(always)]
    fn chunk_ends(_a: Self, _chunk: i32) -> bool {
        true
    }
}
impl Elem for Rec64 {
    type Key = u64;
    const CHUNKED: bool = false;
    const SIMD: SimdKind = SimdKind::I64;
    #[inline(always)]
    fn less(a: Self, b: Self) -> bool {
        a.key < b.key
    }
    #[inline(always)]
    fn compare(a: Self, b: Self) -> i32 {
        (a.key > b.key) as i32 - (a.key < b.key) as i32
    }
    #[inline(always)]
    fn compare_from(a: Self, b: Self, _chunk: i32) -> i32 {
        Self::compare(a, b)
    }
    #[inline(always)]
    fn radix_key(a: Self, _chunk: i32) -> u64 {
        (a.key as u64) ^ 0x8000_0000_0000_0000
    }
    #[inline(always)]
    fn chunk_ends(_a: Self, _chunk: i32) -> bool {
        true
    }
}

struct Build32 {
    acc: u32,
}
impl PartSink for Build32 {
    #[inline(always)]
    fn fixed(&mut self, radix: u64, bits: u8, desc: bool) {
        let mut r = radix as u32;
        if desc {
            r = (if bits >= 32 { u32::MAX } else { (1u32 << bits) - 1 }) - r;
        }
        self.acc = if bits >= 32 { r } else { (self.acc << bits) | r };
    }
    #[inline(always)]
    fn bytes(&mut self, _b: &[u8], _desc: bool) {}
}
struct Build64 {
    acc: u64,
}
impl PartSink for Build64 {
    #[inline(always)]
    fn fixed(&mut self, radix: u64, bits: u8, desc: bool) {
        let mut r = radix;
        if desc {
            r = (if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 }) - r;
        }
        self.acc = if bits >= 64 { r } else { (self.acc << bits) | r };
    }
    #[inline(always)]
    fn bytes(&mut self, _b: &[u8], _desc: bool) {}
}
impl Record for Rec32 {
    #[inline(always)]
    fn build<K: Key + ?Sized>(key: &K, idx: u32) -> Option<Self> {
        let mut b = Build32 { acc: 0 };
        key.write_parts(&mut b);
        Some(Rec32 { key: (b.acc ^ 0x8000_0000) as i32, idx })
    }
    #[inline(always)]
    fn index(&self) -> u32 {
        self.idx
    }
    #[inline(always)]
    fn set_index(&mut self, i: u32) {
        self.idx = i;
    }
}
impl Record for Rec64 {
    #[inline(always)]
    fn build<K: Key + ?Sized>(key: &K, idx: u32) -> Option<Self> {
        let mut b = Build64 { acc: 0 };
        key.write_parts(&mut b);
        Some(Rec64 { key: (b.acc ^ 0x8000_0000_0000_0000) as i64, idx, pad: 0 })
    }
    #[inline(always)]
    fn index(&self) -> u32 {
        self.idx
    }
    #[inline(always)]
    fn set_index(&mut self, i: u32) {
        self.idx = i;
    }
}
/// The radix key of a sorted 32-bit record, for the write-back of self-keyed elements.
#[inline(always)]
pub fn key_of32(r: &Rec32) -> u64 {
    ((r.key as u32) ^ 0x8000_0000) as u64
}
/// The radix key of a sorted 64-bit record.
#[inline(always)]
pub fn key_of64(r: &Rec64) -> u64 {
    (r.key as u64) ^ 0x8000_0000_0000_0000
}
/// A double is written back from its record as well: the radix transform
/// is a bijection on every bit pattern except that -0.0 and +0.0 share a
/// key, so the record's spare word remembers a negative zero.
#[inline(always)]
pub fn note_negative_zero(r: &mut Rec64, d: f64) {
    r.pad = (d == 0.0 && d.is_sign_negative()) as u32;
}
/// The double of a sorted record.
#[inline(always)]
pub fn double_of(r: &Rec64) -> f64 {
    let sign = 0x8000_0000_0000_0000u64;
    let u = (r.key as u64) ^ sign;
    let mut bits = if u & sign != 0 { u & !sign } else { sign | (sign - u) };
    if r.pad != 0 {
        bits = sign; // -0.0
    }
    f64::from_bits(bits)
}

// ---- string record ----------------------------------------------------------------

/// A single byte string; `DESC` reverses its order.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct StrRec<const DESC: bool> {
    /// The bytes.
    pub ptr: *const u8,
    /// Their number.
    pub len: u32,
    /// The original index.
    pub idx: u32,
}
impl<const DESC: bool> Default for StrRec<DESC> {
    fn default() -> Self {
        StrRec { ptr: core::ptr::NonNull::dangling().as_ptr(), len: 0, idx: 0 }
    }
}
impl<const DESC: bool> Elem for StrRec<DESC> {
    type Key = u64;
    const CHUNKED: bool = true;
    const SIMD: SimdKind = SimdKind::None;
    #[inline(always)]
    fn less(a: Self, b: Self) -> bool {
        Self::compare_from(a, b, 0) < 0
    }
    #[inline(always)]
    fn compare(a: Self, b: Self) -> i32 {
        Self::compare_from(a, b, 0)
    }
    #[inline(always)]
    fn compare_from(a: Self, b: Self, chunk: i32) -> i32 {
        // SAFETY: the record was built from a live byte slice.
        let c = unsafe { str_compare_from(a.ptr, a.len, b.ptr, b.len, (chunk as u32) * STR_CHUNK_BYTES) };
        if DESC { -c } else { c }
    }
    #[inline(always)]
    fn radix_key(a: Self, chunk: i32) -> u64 {
        // SAFETY: as above.
        unsafe { if DESC { str_chunk_key_desc(a.ptr, a.len, chunk) } else { str_chunk_key(a.ptr, a.len, chunk) } }
    }
    #[inline(always)]
    fn chunk_ends(a: Self, chunk: i32) -> bool {
        str_chunk_ends(a.len, chunk)
    }
    #[inline(always)]
    fn report_key_chunk(a: &Self, chunk: i32, sink: &mut impl FnMut(*const u8, usize)) {
        let off = (chunk as u32).wrapping_mul(STR_CHUNK_BYTES);
        sink(a.ptr.wrapping_add(off as usize), str_chunk_bytes(a.len, chunk) as usize);
    }
    #[inline(always)]
    fn report_key_compare(a: &Self, b: &Self, chunk: i32, sink: &mut impl FnMut(*const u8, usize)) {
        let off = (chunk as u32) * STR_CHUNK_BYTES;
        // SAFETY: the records were built from live byte slices.
        let ex = unsafe { str_examined(a.ptr, a.len, b.ptr, b.len, off) } as usize;
        sink(a.ptr.wrapping_add(off as usize), ex);
        sink(b.ptr.wrapping_add(off as usize), ex);
    }
}
struct BuildStr {
    ptr: *const u8,
    len: usize,
}
impl PartSink for BuildStr {
    #[inline(always)]
    fn fixed(&mut self, _radix: u64, _bits: u8, _desc: bool) {}
    #[inline(always)]
    fn bytes(&mut self, b: &[u8], _desc: bool) {
        self.ptr = b.as_ptr();
        self.len = b.len();
    }
}
impl<const DESC: bool> Record for StrRec<DESC> {
    #[inline(always)]
    fn build<K: Key + ?Sized>(key: &K, idx: u32) -> Option<Self> {
        let mut b = BuildStr { ptr: core::ptr::NonNull::dangling().as_ptr(), len: 0 };
        key.write_parts(&mut b);
        if b.len > u32::MAX as usize {
            return None;
        }
        Some(StrRec { ptr: b.ptr, len: b.len as u32, idx })
    }
    #[inline(always)]
    fn index(&self) -> u32 {
        self.idx
    }
    #[inline(always)]
    fn set_index(&mut self, i: u32) {
        self.idx = i;
    }
}

// ---- composite record -------------------------------------------------------------

/// A sequence of fixed and byte-string parts, stored inline in the
/// structure of the key type.
#[repr(C)]
pub struct CompRec<K: Key + ?Sized> {
    /// The leaves.
    pub slots: K::Slots,
    /// The original index.
    pub idx: u32,
    /// Unused.
    pub pad: u32,
}
impl<K: Key + ?Sized> Clone for CompRec<K> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<K: Key + ?Sized> Copy for CompRec<K> {}
impl<K: Key + ?Sized> Default for CompRec<K> {
    fn default() -> Self {
        CompRec { slots: K::Slots::zero(), idx: 0, pad: 0 }
    }
}
impl<K: Key + ?Sized> core::fmt::Debug for CompRec<K> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "CompRec(idx {})", self.idx)
    }
}
impl<K: Key + ?Sized> CompRec<K> {
    const N: usize = K::SHAPE.n;
    /// Chunks of part p of element a: one for a fixed part, len / 7 + 1 for a string.
    #[inline(always)]
    fn chunks_of(&self, p: usize) -> i32 {
        match self.slots.leaf(p) {
            Leaf::Fixed(_) => 1,
            Leaf::Bytes(_, len) => (len / STR_CHUNK_BYTES) as i32 + 1,
        }
    }
    /// The part that owns global chunk `chunk` of the element, and the
    /// chunk index within it; `N` past the end.
    #[inline(always)]
    fn locate(&self, mut chunk: i32) -> (usize, i32) {
        for p in 0..Self::N {
            let nc = self.chunks_of(p);
            if chunk < nc {
                return (p, chunk);
            }
            chunk -= nc;
        }
        (Self::N, 0)
    }
    #[inline(always)]
    fn part_chunk_key(&self, p: usize, local: i32) -> u64 {
        match self.slots.leaf(p) {
            Leaf::Fixed(v) => v,
            // SAFETY: the record was built from a live byte slice.
            Leaf::Bytes(ptr, len) => unsafe { if K::SHAPE.kind(p) == 2 { str_chunk_key_desc(ptr, len, local) } else { str_chunk_key(ptr, len, local) } },
        }
    }
    #[inline(always)]
    fn compare_part(&self, b: &Self, p: usize, off: u32) -> i32 {
        match (self.slots.leaf(p), b.slots.leaf(p)) {
            (Leaf::Fixed(x), Leaf::Fixed(y)) => (x > y) as i32 - (x < y) as i32,
            (Leaf::Bytes(pa, la), Leaf::Bytes(pb, lb)) => {
                // SAFETY: the records were built from live byte slices.
                let c = unsafe { str_compare_from(pa, la, pb, lb, off) };
                if K::SHAPE.kind(p) == 2 { -c } else { c }
            }
            _ => 0,
        }
    }
}
impl<K: Key + ?Sized> Elem for CompRec<K> {
    type Key = u64;
    const CHUNKED: bool = true;
    const SIMD: SimdKind = SimdKind::None;
    #[inline(always)]
    fn less(a: Self, b: Self) -> bool {
        Self::compare_from(a, b, 0) < 0
    }
    #[inline(always)]
    fn compare(a: Self, b: Self) -> i32 {
        Self::compare_from(a, b, 0)
    }
    /// Compare from chunk `chunk` on: a and b agree on every earlier chunk,
    /// so the chunk lies in the same part of both and the parts before it tie.
    #[inline(always)]
    fn compare_from(a: Self, b: Self, chunk: i32) -> i32 {
        let (mut p, local) = a.locate(chunk);
        if p >= Self::N {
            return 0;
        }
        let mut c = a.compare_part(&b, p, (local as u32) * STR_CHUNK_BYTES);
        p += 1;
        while c == 0 && p < Self::N {
            c = a.compare_part(&b, p, 0);
            p += 1;
        }
        c
    }
    #[inline(always)]
    fn radix_key(a: Self, chunk: i32) -> u64 {
        let (p, local) = a.locate(chunk);
        if p < Self::N { a.part_chunk_key(p, local) } else { 0 }
    }
    #[inline(always)]
    fn chunk_ends(a: Self, chunk: i32) -> bool {
        let (p, local) = a.locate(chunk);
        if p + 1 >= Self::N {
            if p >= Self::N {
                return true;
            }
            return K::SHAPE.kind(p) == 0 || local == a.chunks_of(p) - 1;
        }
        false
    }
    #[inline(always)]
    fn report_key_chunk(a: &Self, chunk: i32, sink: &mut impl FnMut(*const u8, usize)) {
        let (p, local) = a.locate(chunk);
        if p < Self::N {
            if let Leaf::Bytes(ptr, len) = a.slots.leaf(p) {
                let off = (local as u32) * STR_CHUNK_BYTES;
                sink(ptr.wrapping_add(off as usize), str_chunk_bytes(len, local) as usize);
            }
        }
    }
    #[inline(always)]
    fn report_key_compare(a: &Self, b: &Self, chunk: i32, sink: &mut impl FnMut(*const u8, usize)) {
        // Every string part a compare from `chunk` examines, up to the
        // first that decides.
        let (mut p, local) = a.locate(chunk);
        let mut off = (local as u32) * STR_CHUNK_BYTES;
        while p < Self::N {
            let c = a.compare_part(b, p, off);
            if let (Leaf::Bytes(pa, la), Leaf::Bytes(pb, lb)) = (a.slots.leaf(p), b.slots.leaf(p)) {
                // SAFETY: the records were built from live byte slices.
                let ex = unsafe { str_examined(pa, la, pb, lb, off) } as usize;
                sink(pa.wrapping_add(off as usize), ex);
                sink(pb.wrapping_add(off as usize), ex);
            }
            if c != 0 {
                break;
            }
            p += 1;
            off = 0;
        }
    }
}
struct BuildComp<K: Key + ?Sized> {
    rec: CompRec<K>,
    p: usize,
    ok: bool,
}
impl<K: Key + ?Sized> PartSink for BuildComp<K> {
    #[inline(always)]
    fn fixed(&mut self, radix: u64, bits: u8, desc: bool) {
        let mut r = radix;
        if desc {
            r = (if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 }) - r;
        }
        self.rec.slots.set_leaf(self.p, Leaf::Fixed(r));
        self.p += 1;
    }
    #[inline(always)]
    fn bytes(&mut self, b: &[u8], _desc: bool) {
        if b.len() > u32::MAX as usize {
            self.ok = false;
        }
        self.rec.slots.set_leaf(self.p, Leaf::Bytes(b.as_ptr(), b.len() as u32));
        self.p += 1;
    }
}
impl<K: Key + ?Sized> Record for CompRec<K> {
    #[inline(always)]
    fn build<K2: Key + ?Sized>(key: &K2, idx: u32) -> Option<Self> {
        let mut b = BuildComp::<K> { rec: CompRec::default(), p: 0, ok: true };
        key.write_parts(&mut b);
        b.rec.idx = idx;
        b.rec.pad = 0;
        if b.ok { Some(b.rec) } else { None }
    }
    #[inline(always)]
    fn index(&self) -> u32 {
        self.idx
    }
    #[inline(always)]
    fn set_index(&mut self, i: u32) {
        self.idx = i;
    }
}
// SAFETY: a record is plain data; the byte pointers are only read while
// the keys are alive, on the sorting thread.
unsafe impl<K: Key + ?Sized> Send for CompRec<K> {}
// SAFETY: as above.
unsafe impl<K: Key + ?Sized> Sync for CompRec<K> {}
// SAFETY: as above.
unsafe impl<const D: bool> Send for StrRec<D> {}
// SAFETY: as above.
unsafe impl<const D: bool> Sync for StrRec<D> {}

const _: () = assert!(core::mem::size_of::<Rec32>() == 8 && core::mem::size_of::<Rec64>() == 16);

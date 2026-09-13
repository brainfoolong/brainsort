//! The [`Key`] trait and its implementations for the built-in key types.
//!
//! A key is anything whose order can be expressed as a sequence of leaves,
//! each either an order-preserving unsigned integer of a fixed width or a
//! byte string. The sort reads the leaves once per element, packs them into
//! a small record and never calls back into the key type again; the
//! natural order [`Key::cmp_key`] is used only by the small-input sort, the
//! pass that recognises ordered input, and the fallbacks, so every path
//! orders alike (NaN and signed zeros included).
//!
//! This is the port of `keys.hpp`: `key_traits<K>` became the trait, the
//! compile-time `PartList` became [`Shape`], and the variadic composite
//! record storage became the nested [`SlotTree`] types.
pub use crate::shape::{MAX_PARTS, Part, Shape};
use core::cmp::Ordering;

/// Receives the leaves of a key, in order. Implemented by the record
/// builders; a user does not implement it.
pub trait PartSink {
    /// A fixed leaf: `radix` is an order-preserving unsigned value below
    /// 2^`bits`; `desc` asks for the reversed order.
    fn fixed(&mut self, radix: u64, bits: u8, desc: bool);
    /// A byte-string leaf, compared lexicographically as unsigned bytes with
    /// a shorter prefix first; `desc` asks for the reversed order.
    fn bytes(&mut self, b: &[u8], desc: bool);
}

/// Wraps a sink so that every leaf written through it is reversed: the
/// [`Desc`] keys write their inner key through it.
struct Flip<'a, S: PartSink>(&'a mut S);
impl<S: PartSink> PartSink for Flip<'_, S> {
    #[inline(always)]
    fn fixed(&mut self, radix: u64, bits: u8, desc: bool) {
        self.0.fixed(radix, bits, !desc);
    }
    #[inline(always)]
    fn bytes(&mut self, b: &[u8], desc: bool) {
        self.0.bytes(b, !desc);
    }
}

/// A leaf of a composite record.
#[derive(Clone, Copy, Debug)]
pub enum Leaf {
    /// A fixed part, as its 64-bit radix value.
    Fixed(u64),
    /// A byte string: pointer and length.
    Bytes(*const u8, u32),
}

/// One byte-string slot of a composite record.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BytesSlot {
    /// The bytes; never read when `len` is 0.
    pub ptr: *const u8,
    /// Their number.
    pub len: u32,
    /// Unused; keeps the slot at 16 bytes.
    pub pad: u32,
}
impl Default for BytesSlot {
    fn default() -> Self {
        BytesSlot { ptr: core::ptr::null(), len: 0, pad: 0 }
    }
}
// SAFETY: the pointer is a plain address that the sort dereferences only
// while the key it points into is alive and unchanged; records are never
// handed to another thread by the library.
unsafe impl Send for BytesSlot {}
// SAFETY: as above.
unsafe impl Sync for BytesSlot {}

/// The inline storage of a composite record: a fixed slot per fixed leaf
/// (8 bytes) and a byte-string slot per string leaf (16 bytes), nested in
/// the structure of the key type so the size is known at compile time.
pub trait SlotTree: Copy {
    /// Number of leaves.
    const N: usize;
    /// All slots empty.
    fn zero() -> Self;
    /// Leaf `p`, in key order.
    fn leaf(&self, p: usize) -> Leaf;
    /// Sets leaf `p`.
    fn set_leaf(&mut self, p: usize, v: Leaf);
}
impl SlotTree for u64 {
    const N: usize = 1;
    #[inline(always)]
    fn zero() -> Self {
        0
    }
    #[inline(always)]
    fn leaf(&self, _p: usize) -> Leaf {
        Leaf::Fixed(*self)
    }
    #[inline(always)]
    fn set_leaf(&mut self, _p: usize, v: Leaf) {
        if let Leaf::Fixed(x) = v {
            *self = x;
        }
    }
}
impl SlotTree for BytesSlot {
    const N: usize = 1;
    #[inline(always)]
    fn zero() -> Self {
        BytesSlot::default()
    }
    #[inline(always)]
    fn leaf(&self, _p: usize) -> Leaf {
        Leaf::Bytes(self.ptr, self.len)
    }
    #[inline(always)]
    fn set_leaf(&mut self, _p: usize, v: Leaf) {
        if let Leaf::Bytes(p, l) = v {
            self.ptr = p;
            self.len = l;
        }
    }
}
impl SlotTree for () {
    const N: usize = 0;
    #[inline(always)]
    fn zero() -> Self {}
    #[inline(always)]
    fn leaf(&self, _p: usize) -> Leaf {
        Leaf::Fixed(0)
    }
    #[inline(always)]
    fn set_leaf(&mut self, _p: usize, _v: Leaf) {}
}
impl<A: SlotTree, B: SlotTree> SlotTree for (A, B) {
    const N: usize = A::N + B::N;
    #[inline(always)]
    fn zero() -> Self {
        (A::zero(), B::zero())
    }
    #[inline(always)]
    fn leaf(&self, p: usize) -> Leaf {
        if p < A::N { self.0.leaf(p) } else { self.1.leaf(p - A::N) }
    }
    #[inline(always)]
    fn set_leaf(&mut self, p: usize, v: Leaf) {
        if p < A::N { self.0.set_leaf(p, v) } else { self.1.set_leaf(p - A::N, v) }
    }
}
impl<S: SlotTree, const N: usize> SlotTree for [S; N] {
    const N: usize = S::N * N;
    #[inline(always)]
    fn zero() -> Self {
        [S::zero(); N]
    }
    #[inline(always)]
    fn leaf(&self, p: usize) -> Leaf {
        self[p / S::N].leaf(p % S::N)
    }
    #[inline(always)]
    fn set_leaf(&mut self, p: usize, v: Leaf) {
        self[p / S::N].set_leaf(p % S::N, v)
    }
}

/// The layout of the elements of a plain slice, for the vectorised pass
/// that recognises ordered input. Not part of the public API.
#[doc(hidden)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Prescan {
    None,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
}

/// A sortable key.
///
/// Implemented for every integer, `bool`, `char`, `f32`, `f64`,
/// `Duration`, raw pointers, `Wrapping`, `NonZero`, `str`, `String`,
/// `[u8]`, `Vec<u8>`, `Box<str>`, `Rc<str>`, `Arc<str>`, `Cow<str>`,
/// `CStr`, `CString` (and `OsStr`, `OsString`, `Path`, `PathBuf` with the
/// `std` feature), references to keys, tuples of up to twelve keys, arrays
/// of keys, [`Desc`] and [`core::cmp::Reverse`].
///
/// Implement it for your own type through the macros
/// [`fixed_key!`](crate::fixed_key) and [`bytes_key!`](crate::bytes_key),
/// or by hand: give the [`Shape`], the [`Slots`](Key::Slots) type
/// (`u64` per fixed leaf, [`BytesSlot`] per string leaf, nested pairs for
/// a composite), write the leaves, and compare.
pub trait Key {
    /// The leaves of this key type.
    const SHAPE: Shape;
    /// The inline storage of a composite record of this key: `u64` for a
    /// fixed leaf, [`BytesSlot`] for a string leaf, nested pairs for
    /// composites.
    type Slots: SlotTree;
    /// Writes the leaves, in the order of [`Key::SHAPE`].
    fn write_parts<S: PartSink>(&self, sink: &mut S);
    /// The natural order: the order the leaves define.
    fn cmp_key(&self, other: &Self) -> Ordering;

    /// A single fixed leaf whose radix value inverts to the key: sorted
    /// keys can then be written back from their records. Not part of the
    /// public API.
    #[doc(hidden)]
    const EXACT_FIXED: bool = false;
    /// The inverse of the radix transform, when `EXACT_FIXED`. Not part
    /// of the public API.
    #[doc(hidden)]
    fn from_radix(_r: u64) -> Self
    where
        Self: Sized,
    {
        unreachable!("from_radix on a key without an exact radix")
    }
    /// The slice layout, for the vectorised prescan. Not part of the
    /// public API.
    #[doc(hidden)]
    const PRESCAN: Prescan = Prescan::None;
}

/// Reverses the order of a key: `sort_by_key(&mut v, |r| (r.group, Desc(r.score)))`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Desc<K>(pub K);

/// Wraps a key so that it sorts in descending order.
#[inline]
pub fn desc<K: Key>(k: K) -> Desc<K> {
    Desc(k)
}

impl<K: Key> Key for Desc<K> {
    const SHAPE: Shape = K::SHAPE.reversed();
    type Slots = K::Slots;
    #[inline(always)]
    fn write_parts<S: PartSink>(&self, sink: &mut S) {
        self.0.write_parts(&mut Flip(sink));
    }
    #[inline(always)]
    fn cmp_key(&self, other: &Self) -> Ordering {
        other.0.cmp_key(&self.0)
    }
}
impl<K: Key> Key for core::cmp::Reverse<K> {
    const SHAPE: Shape = K::SHAPE.reversed();
    type Slots = K::Slots;
    #[inline(always)]
    fn write_parts<S: PartSink>(&self, sink: &mut S) {
        self.0.write_parts(&mut Flip(sink));
    }
    #[inline(always)]
    fn cmp_key(&self, other: &Self) -> Ordering {
        other.0.cmp_key(&self.0)
    }
}

impl<K: Key + ?Sized> Key for &K {
    const SHAPE: Shape = K::SHAPE;
    type Slots = K::Slots;
    #[inline(always)]
    fn write_parts<S: PartSink>(&self, sink: &mut S) {
        (**self).write_parts(sink)
    }
    #[inline(always)]
    fn cmp_key(&self, other: &Self) -> Ordering {
        (**self).cmp_key(other)
    }
}
impl<K: Key + ?Sized> Key for &mut K {
    const SHAPE: Shape = K::SHAPE;
    type Slots = K::Slots;
    #[inline(always)]
    fn write_parts<S: PartSink>(&self, sink: &mut S) {
        (**self).write_parts(sink)
    }
    #[inline(always)]
    fn cmp_key(&self, other: &Self) -> Ordering {
        (**self).cmp_key(other)
    }
}

// ---- fixed keys -------------------------------------------------------------------------

/// Implements [`Key`] for a type that maps to a fixed-width unsigned value.
///
/// `$t` is the type, `$bits` the width (1..=64), and the closure-like
/// `|k| expr` turns a `&$t` into an order-preserving `u64` below 2^`$bits`.
///
/// ```
/// #[derive(Clone, Copy, PartialEq, Eq, Debug)]
/// struct UserId(u32);
/// brainsort::fixed_key!(UserId, 32, |k| k.0 as u64);
///
/// let mut v = vec![UserId(3), UserId(1), UserId(2)];
/// brainsort::sort(&mut v);
/// assert_eq!(v, [UserId(1), UserId(2), UserId(3)]);
/// ```
#[macro_export]
macro_rules! fixed_key {
    ($t:ty, $bits:expr, |$k:ident| $to:expr) => {
        impl $crate::Key for $t {
            const SHAPE: $crate::key::Shape = $crate::key::Shape::fixed($bits);
            type Slots = u64;
            #[inline(always)]
            fn write_parts<S: $crate::key::PartSink>(&self, sink: &mut S) {
                let $k = self;
                sink.fixed($to, $bits, false);
            }
            #[inline(always)]
            fn cmp_key(&self, other: &Self) -> ::core::cmp::Ordering {
                let a: u64 = {
                    let $k = self;
                    $to
                };
                let b: u64 = {
                    let $k = other;
                    $to
                };
                a.cmp(&b)
            }
        }
    };
}

/// Implements [`Key`] for a type ordered by a byte string it holds.
///
/// `$t` is the type and the closure-like `|k| expr` turns a `&$t` into a
/// `&[u8]`.
///
/// ```
/// struct Tag { name: String }
/// brainsort::bytes_key!(Tag, |k| k.name.as_bytes());
///
/// let mut v = vec![Tag { name: "b".into() }, Tag { name: "a".into() }];
/// brainsort::sort(&mut v);
/// assert_eq!(v[0].name, "a");
/// ```
#[macro_export]
macro_rules! bytes_key {
    ($t:ty, |$k:ident| $to:expr) => {
        impl $crate::Key for $t {
            const SHAPE: $crate::key::Shape = $crate::key::Shape::bytes();
            type Slots = $crate::key::BytesSlot;
            #[inline(always)]
            fn write_parts<S: $crate::key::PartSink>(&self, sink: &mut S) {
                let $k = self;
                sink.bytes($to, false);
            }
            #[inline(always)]
            fn cmp_key(&self, other: &Self) -> ::core::cmp::Ordering {
                let a: &[u8] = {
                    let $k = self;
                    $to
                };
                let b: &[u8] = {
                    let $k = other;
                    $to
                };
                a.cmp(b)
            }
        }
    };
}

// Integers: signed values are order-preserved by flipping the sign bit.
macro_rules! int_key {
    ($($t:ty => $u:ty, $bits:expr, $flip:expr, $prescan:expr;)*) => {$(
        impl Key for $t {
            const SHAPE: Shape = Shape::fixed($bits);
            type Slots = u64;
            const EXACT_FIXED: bool = true;
            const PRESCAN: Prescan = $prescan;
            #[inline(always)]
            fn write_parts<S: PartSink>(&self, sink: &mut S) {
                sink.fixed(((*self as $u) ^ ($flip as $u)) as u64, $bits, false);
            }
            #[inline(always)]
            fn cmp_key(&self, other: &Self) -> Ordering {
                self.cmp(other)
            }
            #[inline(always)]
            fn from_radix(r: u64) -> Self {
                ((r as $u) ^ ($flip as $u)) as $t
            }
        }
    )*};
}
int_key! {
    u8    => u8,  8,  0u8,  Prescan::None;
    u16   => u16, 16, 0u16, Prescan::None;
    u32   => u32, 32, 0u32, Prescan::U32;
    u64   => u64, 64, 0u64, Prescan::U64;
    i8    => u8,  8,  0x80u8,  Prescan::None;
    i16   => u16, 16, 0x8000u16, Prescan::None;
    i32   => u32, 32, 0x8000_0000u32, Prescan::I32;
    i64   => u64, 64, 0x8000_0000_0000_0000u64, Prescan::I64;
}
#[cfg(target_pointer_width = "64")]
int_key! {
    usize => u64, 64, 0u64, Prescan::U64;
    isize => u64, 64, 0x8000_0000_0000_0000u64, Prescan::I64;
}
#[cfg(target_pointer_width = "32")]
int_key! {
    usize => u32, 32, 0u32, Prescan::U32;
    isize => u32, 32, 0x8000_0000u32, Prescan::I32;
}

// 128-bit integers: two 64-bit leaves, high word first.
impl Key for u128 {
    const SHAPE: Shape = Shape::concat(&Shape::fixed(64), &Shape::fixed(64));
    type Slots = (u64, u64);
    #[inline(always)]
    fn write_parts<S: PartSink>(&self, sink: &mut S) {
        sink.fixed((*self >> 64) as u64, 64, false);
        sink.fixed(*self as u64, 64, false);
    }
    #[inline(always)]
    fn cmp_key(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }
}
impl Key for i128 {
    const SHAPE: Shape = Shape::concat(&Shape::fixed(64), &Shape::fixed(64));
    type Slots = (u64, u64);
    #[inline(always)]
    fn write_parts<S: PartSink>(&self, sink: &mut S) {
        let u = *self as u128;
        sink.fixed(((u >> 64) as u64) ^ (1u64 << 63), 64, false);
        sink.fixed(u as u64, 64, false);
    }
    #[inline(always)]
    fn cmp_key(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }
}

impl Key for bool {
    const SHAPE: Shape = Shape::fixed(1);
    type Slots = u64;
    const EXACT_FIXED: bool = true;
    #[inline(always)]
    fn write_parts<S: PartSink>(&self, sink: &mut S) {
        sink.fixed(*self as u64, 1, false);
    }
    #[inline(always)]
    fn cmp_key(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }
    #[inline(always)]
    fn from_radix(r: u64) -> Self {
        r != 0
    }
}
impl Key for char {
    const SHAPE: Shape = Shape::fixed(32);
    type Slots = u64;
    const EXACT_FIXED: bool = true;
    #[inline(always)]
    fn write_parts<S: PartSink>(&self, sink: &mut S) {
        sink.fixed(*self as u64, 32, false);
    }
    #[inline(always)]
    fn cmp_key(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }
    #[inline(always)]
    fn from_radix(r: u64) -> Self {
        // SAFETY: the value came from `*self as u64` of a char.
        unsafe { char::from_u32_unchecked(r as u32) }
    }
}

/// The order-preserving radix form of an `f32`. Not part of the public API.
#[doc(hidden)]
#[inline(always)]
pub const fn f32_radix(f: f32) -> u32 {
    let bits = f.to_bits();
    let sign = 0x8000_0000u32;
    if bits & sign != 0 { sign - (bits & !sign) } else { bits | sign }
}
/// The order-preserving radix form of an `f64`: a positive value maps to
/// `bits | 2^63`, a negative one to `2^63 - magnitude`, so `-0.0` and `+0.0`
/// both map to 2^63 and compare equal. A NaN with the sign bit clear sorts
/// after `+inf`, one with it set before `-inf`. Not part of the public API.
#[doc(hidden)]
#[inline(always)]
pub const fn f64_radix(f: f64) -> u64 {
    let bits = f.to_bits();
    let sign = 0x8000_0000_0000_0000u64;
    if bits & sign != 0 { sign - (bits & !sign) } else { bits | sign }
}
impl Key for f32 {
    const SHAPE: Shape = Shape::fixed(32);
    type Slots = u64;
    const PRESCAN: Prescan = Prescan::F32;
    #[inline(always)]
    fn write_parts<S: PartSink>(&self, sink: &mut S) {
        sink.fixed(f32_radix(*self) as u64, 32, false);
    }
    #[inline(always)]
    fn cmp_key(&self, other: &Self) -> Ordering {
        f32_radix(*self).cmp(&f32_radix(*other))
    }
}
impl Key for f64 {
    const SHAPE: Shape = Shape::fixed(64);
    type Slots = u64;
    const PRESCAN: Prescan = Prescan::F64;
    #[inline(always)]
    fn write_parts<S: PartSink>(&self, sink: &mut S) {
        sink.fixed(f64_radix(*self), 64, false);
    }
    #[inline(always)]
    fn cmp_key(&self, other: &Self) -> Ordering {
        f64_radix(*self).cmp(&f64_radix(*other))
    }
}

impl<T: Key + Copy> Key for core::num::Wrapping<T> {
    const SHAPE: Shape = T::SHAPE;
    type Slots = T::Slots;
    #[inline(always)]
    fn write_parts<S: PartSink>(&self, sink: &mut S) {
        self.0.write_parts(sink)
    }
    #[inline(always)]
    fn cmp_key(&self, other: &Self) -> Ordering {
        self.0.cmp_key(&other.0)
    }
}
macro_rules! nonzero_key {
    ($($nz:ty => $t:ty;)*) => {$(
        impl Key for $nz {
            const SHAPE: Shape = <$t as Key>::SHAPE;
            type Slots = u64;
            #[inline(always)]
            fn write_parts<S: PartSink>(&self, sink: &mut S) {
                self.get().write_parts(sink)
            }
            #[inline(always)]
            fn cmp_key(&self, other: &Self) -> Ordering {
                self.get().cmp(&other.get())
            }
        }
    )*};
}
nonzero_key! {
    core::num::NonZeroU8 => u8; core::num::NonZeroU16 => u16; core::num::NonZeroU32 => u32; core::num::NonZeroU64 => u64;
    core::num::NonZeroI8 => i8; core::num::NonZeroI16 => i16; core::num::NonZeroI32 => i32; core::num::NonZeroI64 => i64;
    core::num::NonZeroUsize => usize; core::num::NonZeroIsize => isize;
}

// Durations: seconds (64 bits) then nanoseconds (32 bits).
impl Key for core::time::Duration {
    const SHAPE: Shape = Shape::concat(&Shape::fixed(64), &Shape::fixed(32));
    type Slots = (u64, u64);
    #[inline(always)]
    fn write_parts<S: PartSink>(&self, sink: &mut S) {
        sink.fixed(self.as_secs(), 64, false);
        sink.fixed(self.subsec_nanos() as u64, 32, false);
    }
    #[inline(always)]
    fn cmp_key(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }
}

// Pointers: by address.
macro_rules! ptr_key {
    ($($p:ty),*) => {$(
        impl<T> Key for $p {
            const SHAPE: Shape = Shape::fixed(usize::BITS as u8);
            type Slots = u64;
            #[inline(always)]
            fn write_parts<S: PartSink>(&self, sink: &mut S) {
                sink.fixed(*self as *const () as usize as u64, usize::BITS as u8, false);
            }
            #[inline(always)]
            fn cmp_key(&self, other: &Self) -> Ordering {
                (*self as *const () as usize).cmp(&(*other as *const () as usize))
            }
        }
    )*};
}
ptr_key!(*const T, *mut T);

// ---- byte strings ------------------------------------------------------------------------

macro_rules! bytes_impl {
    ($($t:ty => |$k:ident| $to:expr;)*) => {$(
        impl Key for $t {
            const SHAPE: Shape = Shape::bytes();
            type Slots = BytesSlot;
            #[inline(always)]
            fn write_parts<S: PartSink>(&self, sink: &mut S) {
                let $k = self;
                sink.bytes($to, false);
            }
            #[inline(always)]
            fn cmp_key(&self, other: &Self) -> Ordering {
                let a: &[u8] = { let $k = self; $to };
                let b: &[u8] = { let $k = other; $to };
                a.cmp(b)
            }
        }
    )*};
}
bytes_impl! {
    str => |k| k.as_bytes();
    [u8] => |k| k;
    alloc::string::String => |k| k.as_bytes();
    alloc::vec::Vec<u8> => |k| k.as_slice();
    alloc::borrow::Cow<'_, str> => |k| k.as_bytes();
    alloc::borrow::Cow<'_, [u8]> => |k| k;
    core::ffi::CStr => |k| k.to_bytes();
    alloc::ffi::CString => |k| k.as_bytes();
}
#[cfg(feature = "std")]
bytes_impl! {
    std::ffi::OsStr => |k| k.as_encoded_bytes();
    std::ffi::OsString => |k| k.as_encoded_bytes();
    std::path::Path => |k| k.as_os_str().as_encoded_bytes();
    std::path::PathBuf => |k| k.as_os_str().as_encoded_bytes();
}

// ---- composites ------------------------------------------------------------------------------

macro_rules! tuple_key {
    ($(($t:ident, $a:ident, $b:ident))+) => {
        impl<$($t: Key),+> Key for ($($t,)+) {
            const SHAPE: Shape = tuple_key!(@shape $($t)+);
            type Slots = tuple_key!(@slots $($t)+);
            #[inline(always)]
            fn write_parts<S: PartSink>(&self, sink: &mut S) {
                let ($($a,)+) = self;
                $($a.write_parts(sink);)+
            }
            #[inline(always)]
            fn cmp_key(&self, other: &Self) -> Ordering {
                let ($($a,)+) = self;
                let ($($b,)+) = other;
                $(
                    let c = $a.cmp_key($b);
                    if c != Ordering::Equal {
                        return c;
                    }
                )+
                Ordering::Equal
            }
        }
    };
    (@shape $a:ident) => { <$a as Key>::SHAPE };
    (@shape $a:ident $($rest:ident)+) => { Shape::concat(&<$a as Key>::SHAPE, &tuple_key!(@shape $($rest)+)) };
    (@slots $a:ident) => { <$a as Key>::Slots };
    (@slots $a:ident $($rest:ident)+) => { (<$a as Key>::Slots, tuple_key!(@slots $($rest)+)) };
}
tuple_key!((A, a0, b0));
tuple_key!((A, a0, b0)(B, a1, b1));
tuple_key!((A, a0, b0)(B, a1, b1)(C, a2, b2));
tuple_key!((A, a0, b0)(B, a1, b1)(C, a2, b2)(D, a3, b3));
tuple_key!((A, a0, b0)(B, a1, b1)(C, a2, b2)(D, a3, b3)(E, a4, b4));
tuple_key!((A, a0, b0)(B, a1, b1)(C, a2, b2)(D, a3, b3)(E, a4, b4)(F, a5, b5));
tuple_key!((A, a0, b0)(B, a1, b1)(C, a2, b2)(D, a3, b3)(E, a4, b4)(F, a5, b5)(G, a6, b6));
tuple_key!((A, a0, b0)(B, a1, b1)(C, a2, b2)(D, a3, b3)(E, a4, b4)(F, a5, b5)(G, a6, b6)(H, a7, b7));
tuple_key!((A, a0, b0)(B, a1, b1)(C, a2, b2)(D, a3, b3)(E, a4, b4)(F, a5, b5)(G, a6, b6)(H, a7, b7)(I, a8, b8));
tuple_key!((A, a0, b0)(B, a1, b1)(C, a2, b2)(D, a3, b3)(E, a4, b4)(F, a5, b5)(G, a6, b6)(H, a7, b7)(I, a8, b8)(J, a9, b9));
tuple_key!((A, a0, b0)(B, a1, b1)(C, a2, b2)(D, a3, b3)(E, a4, b4)(F, a5, b5)(G, a6, b6)(H, a7, b7)(I, a8, b8)(J, a9, b9)(K2, a10, b10));
tuple_key!((A, a0, b0)(B, a1, b1)(C, a2, b2)(D, a3, b3)(E, a4, b4)(F, a5, b5)(G, a6, b6)(H, a7, b7)(I, a8, b8)(J, a9, b9)(K2, a10, b10)(L, a11, b11));

impl<K: Key, const N: usize> Key for [K; N] {
    const SHAPE: Shape = Shape::repeat(&K::SHAPE, N);
    type Slots = [K::Slots; N];
    #[inline(always)]
    fn write_parts<S: PartSink>(&self, sink: &mut S) {
        for k in self {
            k.write_parts(sink);
        }
    }
    #[inline(always)]
    fn cmp_key(&self, other: &Self) -> Ordering {
        for i in 0..N {
            let c = self[i].cmp_key(&other[i]);
            if c != Ordering::Equal {
                return c;
            }
        }
        Ordering::Equal
    }
}

impl<K: Key + ?Sized> Key for alloc::boxed::Box<K> {
    const SHAPE: Shape = K::SHAPE;
    type Slots = K::Slots;
    #[inline(always)]
    fn write_parts<S: PartSink>(&self, sink: &mut S) {
        (**self).write_parts(sink)
    }
    #[inline(always)]
    fn cmp_key(&self, other: &Self) -> Ordering {
        (**self).cmp_key(other)
    }
}
impl<K: Key + ?Sized> Key for alloc::rc::Rc<K> {
    const SHAPE: Shape = K::SHAPE;
    type Slots = K::Slots;
    #[inline(always)]
    fn write_parts<S: PartSink>(&self, sink: &mut S) {
        (**self).write_parts(sink)
    }
    #[inline(always)]
    fn cmp_key(&self, other: &Self) -> Ordering {
        (**self).cmp_key(other)
    }
}
impl<K: Key + ?Sized> Key for alloc::sync::Arc<K> {
    const SHAPE: Shape = K::SHAPE;
    type Slots = K::Slots;
    #[inline(always)]
    fn write_parts<S: PartSink>(&self, sink: &mut S) {
        (**self).write_parts(sink)
    }
    #[inline(always)]
    fn cmp_key(&self, other: &Self) -> Ordering {
        (**self).cmp_key(other)
    }
}

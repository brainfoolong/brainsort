//! brainsort: a stable sort for keys that map to an ordered integer:
//! numbers, strings, dates, ids. It avoids comparing keys wherever the key
//! type allows it and does less work when the input already has structure.
//!
//! ```
//! let mut v = vec![3, 1, 2];
//! brainsort::sort(&mut v);                                 // stable, by value
//! assert_eq!(v, [1, 2, 3]);
//!
//! struct Row { id: u64, name: String, score: f64 }
//! let mut rows = vec![Row { id: 2, name: "b".into(), score: 1.5 }, Row { id: 1, name: "a".into(), score: 2.5 }];
//! brainsort::sort_by_key(&mut rows, |r| r.id);              // by a key
//! brainsort::sort_by_key_ref(&mut rows, |r| r.name.as_str());   // by a borrowed key
//! brainsort::sort_by_key(&mut rows, |r| (r.id, brainsort::Desc(r.score)));
//! brainsort::sort_by(&mut rows, |a, b| a.name.cmp(&b.name));   // a comparator
//! assert_eq!(rows[0].name, "a");
//! ```
//!
//! Every sort is stable. Keys can be any integer, `bool`, `char`, `f32`,
//! `f64`, `Duration`, raw pointer, `str`, `String`, `[u8]`, `Vec<u8>`,
//! `CStr`, `OsStr`, `Path` and their owned forms, a tuple or array of those,
//! or [`Desc`] for a reversed order; your own key type implements
//! [`Key`] (see [`fixed_key!`] and [`bytes_key!`]). Elements can be
//! anything: they are permuted once, after the keys were sorted.
//!
//! # How it sorts
//!
//! One pass over the keys comes first, before anything is built: it counts
//! descents and ascents and stops as soon as they prove the input
//! unordered. Sorted input returns at once; reversed input is reversed in
//! place with equal keys kept in their order; nearly sorted input (at most
//! one descent in sixteen) is fixed in place when the elements are at most
//! 16 bytes, by pulling out the few displaced elements, sorting them and
//! merging them back. Plain slices of 32- and 64-bit numbers take this pass
//! with AVX2.
//!
//! Everything else goes through records: an array of small records, one
//! per element, holding the radix form of the key and the original index.
//! The records are sorted by the core algorithm (a scout pass that picks
//! between reversal, merging a few runs, the displaced-element route and
//! the radix route; the radix route samples the keys, splits at the
//! median, chooses a plan per part and verifies it while counting), then
//! the elements are permuted once. Keys of up to 32 bits become 8-byte
//! records, up to 64 bits 16-byte records, a string a 16-byte
//! pointer-and-length record, anything else a composite record sorted
//! chunk by chunk.
//!
//! # Floating point
//!
//! `-0.0` and `+0.0` compare equal (and stay in input order); a NaN with the
//! sign bit clear sorts after `+inf`, one with it set before `-inf`, ordered
//! by payload. This is a total order, so no input is unsafe; it is not
//! `f64::total_cmp`, which orders `-0.0` before `+0.0`.
//!
//! # Memory, failure, limits
//!
//! - **Memory.** n records of 8, 16 or more bytes, plus the algorithm's
//!   scratch of about n/2 records on full-entropy input, plus count tables
//!   of at most 64 KiB (about 400 KiB for a part that is scattered beyond
//!   the cache). Elements are permuted through a buffer of n elements, or
//!   in place through cycles if that buffer cannot be allocated. Sorted and
//!   reversed input, and nearly sorted small elements, need no memory
//!   beyond the displaced elements.
//! - **Memory is kept.** Freed blocks of 64 KiB and more are kept, up to
//!   32 MiB in total, for the next sort of the process; fresh pages cost
//!   more than a sort of a hundred thousand elements.
//!   [`set_memory_cache_limit`] changes the limit (0 disables it) and
//!   [`release_memory`] frees the blocks at any time. Without the `std`
//!   feature the cache is behind a spin lock.
//! - **Allocation fails.** The sort completes anyway through the standard
//!   library's stable sort with the same order.
//! - **The key function panics.** All keys are computed before anything
//!   moves, so the slice is unchanged if the panic happens during the
//!   first pass over the keys; a panic on a later call (the nearly sorted
//!   route compares elements again) leaves every element in the slice, in
//!   an unspecified order. A comparator that panics likewise.
//! - **Limits.** At most 2^32 - 1 elements per call and strings shorter
//!   than 2^32 bytes; beyond either the call goes to the standard
//!   library's stable sort. The block cache is the only shared state,
//!   under its own lock; concurrent sorts are fine.
//!
//! # Features
//!
//! - `std` (default): run-time CPU feature detection on x86-64, a mutex
//!   for the memory cache, `OsStr` and `Path` keys. Without it the crate is
//!   `no_std` + `alloc`, and the vector paths are used only when enabled
//!   at compile time (`-C target-feature=+avx2,+bmi2`).
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![allow(
    clippy::needless_range_loop,
    clippy::comparison_chain,
    clippy::collapsible_if,
    clippy::collapsible_else_if,
    clippy::too_many_arguments,
    clippy::manual_div_ceil,
    clippy::implicit_saturating_sub,
    clippy::int_plus_one,
    clippy::needless_late_init
)]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

mod algorithm;
mod api;
mod cpu;
pub mod key;
mod memory;
mod mergesort;
mod radix;
mod record;
mod shape;
mod simd;
mod view;

pub use key::{Desc, Key, desc};
pub use memory::{release_memory, set_memory_cache_limit};

use core::cmp::Ordering;

/// Sorts the slice by the elements themselves. Stable.
///
/// ```
/// let mut v = vec![2.5f64, -0.0, 0.0, -1.0];
/// brainsort::sort(&mut v);
/// assert_eq!(v.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), [-1.0f64, -0.0, 0.0, 2.5].map(f64::to_bits));
/// ```
pub fn sort<T: Key>(v: &mut [T]) {
    api::sort_by_key_impl::<T, api::Identity, memory::DefaultAlloc>(v, api::Identity)
}

/// Sorts the slice by a key computed from each element, once per element.
/// Stable.
///
/// ```
/// let mut v = vec!["bb", "a", "ccc"];
/// brainsort::sort_by_key(&mut v, |s| s.len());
/// assert_eq!(v, ["a", "bb", "ccc"]);
/// ```
pub fn sort_by_key<T, K: Key, F: FnMut(&T) -> K>(v: &mut [T], f: F) {
    api::sort_by_key_impl::<T, api::ByVal<F>, memory::DefaultAlloc>(v, api::ByVal(f))
}

/// Sorts the slice by a key borrowed from each element (a string field,
/// say). Stable.
///
/// ```
/// struct Row { name: String }
/// let mut v = vec![Row { name: "b".into() }, Row { name: "a".into() }];
/// brainsort::sort_by_key_ref(&mut v, |r| r.name.as_str());
/// assert_eq!(v[0].name, "a");
/// ```
pub fn sort_by_key_ref<T, K: Key + ?Sized, F: for<'a> FnMut(&'a T) -> &'a K>(v: &mut [T], f: F) {
    api::sort_by_key_impl::<T, api::ByRef<F>, memory::DefaultAlloc>(v, api::ByRef(f))
}

/// Sorts the slice by a comparator. Stable. Sorted, reversed and nearly
/// sorted input is handled on the elements as the key sorts do; everything
/// else is a comparison sort, the standard library's `slice::sort_by`.
///
/// ```
/// let mut v = vec![3, 1, 2];
/// brainsort::sort_by(&mut v, |a, b| b.cmp(a));
/// assert_eq!(v, [3, 2, 1]);
/// ```
pub fn sort_by<T, F: FnMut(&T, &T) -> Ordering>(v: &mut [T], cmp: F) {
    api::sort_by_impl::<T, memory::DefaultAlloc, F>(v, cmp)
}

/// The version of the crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The generic entry points, hooks and allocator interface, for the
/// benchmark harness. Not part of the public API: no semver guarantee.
#[cfg(feature = "__internals")]
#[doc(hidden)]
pub mod internals {
    pub use crate::algorithm::{Scratch, brainsort_impl};
    pub use crate::api::{ByRef, ByVal, Identity, Proj, sort_by_impl, sort_by_key_impl};
    pub use crate::cpu::{cache_sizes, have_avx2, have_bmi2};
    pub use crate::memory::DefaultAlloc;
    pub use crate::record::{CompRec, Rec32, Rec64, Record, StrRec};
    pub use crate::view::*;
}

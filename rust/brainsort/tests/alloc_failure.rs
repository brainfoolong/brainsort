//! Allocation failure at every allocation point: a counting allocator fails
//! call k, for every k up to the number of allocations a normal run makes,
//! and the result must still be sorted, stable and leak nothing.
#![cfg(feature = "__internals")]
mod common;
use brainsort::internals::{Alloc, AllocError, ByRef, Identity, sort_by_impl, sort_by_key_impl};
use common::*;
use std::cell::Cell;
use std::ptr::NonNull;

thread_local! {
    static COUNT: Cell<usize> = const { Cell::new(0) };
    static FAIL_AT: Cell<usize> = const { Cell::new(usize::MAX) };
    static LIVE: Cell<usize> = const { Cell::new(0) };
}
struct FailAlloc;
impl Alloc for FailAlloc {
    fn allocate(bytes: usize) -> Result<NonNull<u8>, AllocError> {
        let c = COUNT.with(|c| {
            let v = c.get();
            c.set(v + 1);
            v
        });
        if c == FAIL_AT.with(|f| f.get()) {
            return Err(AllocError);
        }
        let layout = std::alloc::Layout::from_size_align(bytes.max(1), 16).unwrap();
        let p = NonNull::new(unsafe { std::alloc::alloc(layout) }).ok_or(AllocError)?;
        LIVE.with(|l| l.set(l.get() + 1));
        Ok(p)
    }
    unsafe fn deallocate(p: NonNull<u8>, bytes: usize) {
        LIVE.with(|l| l.set(l.get() - 1));
        unsafe { std::alloc::dealloc(p.as_ptr(), std::alloc::Layout::from_size_align_unchecked(bytes.max(1), 16)) }
    }
}
fn reset(fail_at: usize) {
    COUNT.with(|c| c.set(0));
    FAIL_AT.with(|f| f.set(fail_at));
    LIVE.with(|l| l.set(0));
}

fn key_of<K>(t: &Tagged<K>) -> &K {
    &t.key
}

fn case<K: Gen + Clone + Natural + brainsort::Key + std::fmt::Debug>(name: &str, pattern: &str, n: usize) {
    // Miri interprets: the same allocation points at a fraction of the size (a
    // "runs" input needs several runs to allocate at all).
    let n = if !cfg!(miri) {
        n
    } else if pattern == "runs" {
        n / 2
    } else {
        n / 10
    };
    let mut pool = Pool;
    let input = make_input::<K>(pattern, n, 31 + n as u64, &mut pool);
    // How many allocations does a normal run make?
    reset(usize::MAX);
    let mut v = input.clone();
    sort_by_key_impl::<_, _, FailAlloc>(&mut v, ByRef(key_of::<K>));
    verify(&v, n, &format!("{name}/{pattern} baseline"));
    let total = COUNT.with(|c| c.get());
    assert_eq!(LIVE.with(|l| l.get()), 0, "{name}/{pattern}: scratch leaked");
    assert!(total > 0, "{name}/{pattern}: the baseline did not allocate");
    for k in 0..=total {
        reset(k);
        let mut v = input.clone();
        sort_by_key_impl::<_, _, FailAlloc>(&mut v, ByRef(key_of::<K>));
        verify(&v, n, &format!("{name}/{pattern} allocation {k} of {total} failed"));
        assert_eq!(LIVE.with(|l| l.get()), 0, "{name}/{pattern} allocation {k} failed: scratch leaked");
    }
}

/// The keys-only route of a plain slice of keys: the key array, the
/// scratch, and for `f32` and `f64` the ranks of the negative zeros; each failure
/// falls back to the standard library's sort and the result is the same,
/// bit for bit.
fn plain_case<K: Clone + Natural + brainsort::Key + std::fmt::Debug>(name: &str, input: &[K]) {
    let mut w = input.to_vec();
    w.sort_by(|x, y| x.natural_cmp(y));
    let check = |v: &[K], ctx: &str| {
        for i in 0..v.len() {
            assert!(v[i].identical(&w[i]), "{ctx}: differs at {i}: {:?} vs {:?}", v[i], w[i]);
        }
    };
    reset(usize::MAX);
    let mut v = input.to_vec();
    sort_by_key_impl::<_, Identity, FailAlloc>(&mut v, Identity);
    check(&v, &format!("{name} baseline"));
    let total = COUNT.with(|c| c.get());
    assert_eq!(LIVE.with(|l| l.get()), 0, "{name}: scratch leaked");
    assert!(total > 0, "{name}: the baseline did not allocate");
    for k in 0..=total {
        reset(k);
        let mut v = input.to_vec();
        sort_by_key_impl::<_, Identity, FailAlloc>(&mut v, Identity);
        check(&v, &format!("{name} allocation {k} of {total} failed"));
        assert_eq!(LIVE.with(|l| l.get()), 0, "{name} allocation {k} failed: scratch leaked");
    }
}

#[test]
fn plain_key_allocation_points() {
    let n = if cfg!(miri) { 500 } else { 5000 };
    let mut rng = Rng::new(11);
    let f: Vec<f64> = (0..n)
        .map(|i| match i % 5 {
            0 => -0.0,
            1 => 0.0,
            _ => (rng.next() % 1000) as f64 - 500.0,
        })
        .collect();
    plain_case("f64 with zeros", &f);
    let u: Vec<u64> = (0..n).map(|_| rng.next()).collect();
    plain_case("u64 random", &u);
    let f: Vec<f32> = (0..n)
        .map(|i| match i % 5 {
            0 => -0.0,
            1 => 0.0,
            _ => (rng.next() % 1000) as f32 - 500.0,
        })
        .collect();
    plain_case("f32 with zeros", &f);
    let i: Vec<i32> = (0..n).map(|_| rng.next() as i32).collect();
    plain_case("i32 random", &i);
}

/// The comparator sort of elements over 16 bytes: the index array, then
/// the permutation buffer; each failure falls back (to the standard
/// library's sort, to the cycle walk) and the result is the same.
fn comparator_case<const P: usize>(pattern: &str, n: usize) {
    #[derive(Clone, Copy)]
    struct Wide<const P: usize> {
        key: i32,
        id: u32,
        _payload: [u8; P],
    }
    let n = if cfg!(miri) { n / 10 } else { n };
    let mut pool = Pool;
    let input: Vec<Wide<P>> = make_input::<i32>(pattern, n, 37 + n as u64, &mut pool).iter().map(|t| Wide { key: t.key, id: t.id, _payload: [0; P] }).collect();
    let check = |v: &[Wide<P>], ctx: &str| {
        let tagged: Vec<Tagged<i32>> = v.iter().map(|w| Tagged { key: w.key, id: w.id }).collect();
        verify(&tagged, n, ctx);
    };
    reset(usize::MAX);
    let mut v = input.clone();
    sort_by_impl::<_, FailAlloc, _>(&mut v, |a, b| a.key.cmp(&b.key));
    check(&v, &format!("comparator {P}/{pattern} baseline"));
    let total = COUNT.with(|c| c.get());
    assert_eq!(LIVE.with(|l| l.get()), 0, "comparator {P}/{pattern}: scratch leaked");
    assert!(total > 0, "comparator {P}/{pattern}: the baseline did not allocate");
    for k in 0..=total {
        reset(k);
        let mut v = input.clone();
        sort_by_impl::<_, FailAlloc, _>(&mut v, |a, b| a.key.cmp(&b.key));
        check(&v, &format!("comparator {P}/{pattern} allocation {k} of {total} failed"));
        assert_eq!(LIVE.with(|l| l.get()), 0, "comparator {P}/{pattern} allocation {k} failed: scratch leaked");
    }
}

#[test]
fn comparator_allocation_points() {
    comparator_case::<56>("random", 5000);
    comparator_case::<56>("few_unique", 5000);
    comparator_case::<120>("nearly_sorted", 20000);
    comparator_case::<120>("random", 5000);
}

#[test]
fn every_allocation_point() {
    case::<i32>("i32", "random", 5000);
    case::<i32>("i32", "nearly_sorted", 20000);
    case::<i32>("i32", "few_unique", 5000);
    case::<i32>("i32", "organ_pipe", 5000);
    case::<i64>("i64", "random", 5000);
    case::<f64>("f64", "organ_pipe", 5000);
    case::<f64>("f64", "runs", 5000);
    case::<String>("String", "random", 3000);
    case::<String>("String", "few_unique", 3000);
    case::<(i64, i64)>("(i64, i64)", "random", 2000);
    case::<(i32, String)>("(i32, String)", "random", 2000);
}

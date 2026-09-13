//! Allocation failure at every allocation point: a counting allocator fails
//! call k, for every k up to the number of allocations a normal run makes,
//! and the result must still be sorted, stable and leak nothing.
#![cfg(feature = "__internals")]
mod common;
use brainsort::internals::{Alloc, AllocError, ByRef, sort_by_key_impl};
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

//! A key function or comparator that panics: the slice is unchanged when
//! the panic happens during the first pass over the keys, and every element
//! is still in the slice, exactly once, whenever it happens.
mod common;
use common::*;
use std::panic::{AssertUnwindSafe, catch_unwind};

fn ids_sorted<K>(v: &[Tagged<K>]) -> Vec<u32> {
    let mut ids: Vec<u32> = v.iter().map(|t| t.id).collect();
    ids.sort_unstable();
    ids
}

#[test]
fn key_function_panics() {
    for pattern in ["random", "nearly_sorted", "reverse", "sorted", "few_unique"] {
        let n = if cfg!(miri) { 2000 } else { 20000 };
        let mut pool = Pool;
        let input = make_input::<i32>(pattern, n, 12, &mut pool);
        let all_ids = ids_sorted(&input);
        for at in [0usize, 1, 5000, n - 1, n, n + 500, 2 * n, 2 * n + 1000, 4 * n] {
            let mut v = input.clone();
            let calls = std::cell::Cell::new(0usize);
            let r = catch_unwind(AssertUnwindSafe(|| {
                brainsort::sort_by_key(&mut v, |t| {
                    let c = calls.get();
                    calls.set(c + 1);
                    if c == at {
                        panic!("key function failed");
                    }
                    t.key
                });
            }));
            assert_eq!(ids_sorted(&v), all_ids, "{pattern}: key function panicking at call {at} keeps every element");
            if r.is_err() && at < n {
                // during the first pass: unchanged
                assert!(v.iter().zip(&input).all(|(a, b)| a.id == b.id), "{pattern}: a panic during the first pass over the keys leaves the slice unchanged (call {at})");
            }
            if r.is_ok() {
                verify(&v, n, &format!("{pattern}: no panic at call {at}"));
            }
        }
    }
    // Owned keys with heap storage: no leak and no double free (Miri and
    // the sanitizer check that), every element kept.
    let mut pool = Pool;
    let ns = if cfg!(miri) { 600 } else { 3000 };
    let input = make_input::<String>("random", ns, 3, &mut pool);
    let all_ids = ids_sorted(&input);
    for at in [0usize, 1, ns - 500, ns - 1, ns, ns + 500] {
        let mut v = input.clone();
        let calls = std::cell::Cell::new(0usize);
        let _ = catch_unwind(AssertUnwindSafe(|| {
            brainsort::sort_by_key(&mut v, |t| {
                let c = calls.get();
                calls.set(c + 1);
                if c == at {
                    panic!("key function failed");
                }
                format!("{}x", t.key)
            });
        }));
        assert_eq!(ids_sorted(&v), all_ids, "String keys: panic at call {at} keeps every element");
    }
}

#[test]
fn comparator_panics() {
    for pattern in ["random", "nearly_sorted", "reverse"] {
        let n = if cfg!(miri) { 1000 } else { 3000 };
        let mut pool = Pool;
        let input = make_input::<i32>(pattern, n, 11, &mut pool);
        let all_ids = ids_sorted(&input);
        for at in [0usize, 1, 100, 2999, 7000, 20000] {
            let mut v = input.clone();
            let calls = std::cell::Cell::new(0usize);
            let r = catch_unwind(AssertUnwindSafe(|| {
                brainsort::sort_by(&mut v, |a, b| {
                    let c = calls.get();
                    calls.set(c + 1);
                    if c == at {
                        panic!("comparator failed");
                    }
                    a.key.cmp(&b.key)
                });
            }));
            assert_eq!(ids_sorted(&v), all_ids, "{pattern}: comparator panicking at call {at} keeps every element");
            if r.is_ok() {
                verify(&v, n, &format!("{pattern}: comparator that did not panic at call {at}"));
            }
        }
    }
    // Elements with destructors: a panic must not drop anything twice.
    let mut pool = Pool;
    let input = make_input::<String>("nearly_sorted", if cfg!(miri) { 500 } else { 2000 }, 4, &mut pool);
    let rc: Vec<std::rc::Rc<Tagged<String>>> = input.iter().cloned().map(std::rc::Rc::new).collect();
    for at in [0usize, 10, 500, 3000] {
        let mut v = rc.clone();
        let calls = std::cell::Cell::new(0usize);
        let _ = catch_unwind(AssertUnwindSafe(|| {
            brainsort::sort_by(&mut v, |a, b| {
                let c = calls.get();
                calls.set(c + 1);
                if c == at {
                    panic!("comparator failed");
                }
                a.key.cmp(&b.key)
            });
        }));
        assert!(v.iter().all(|x| std::rc::Rc::strong_count(x) == 2), "no element was duplicated or lost at call {at}");
    }
}

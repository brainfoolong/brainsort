//! Tests of the public API: every built-in key type on ten input patterns
//! at many sizes, through `sort`, `sort_by_key` and `sort_by_key_ref`;
//! every result is checked for order (by an independently written natural
//! order), stability and being a permutation of the input. The port of
//! `tests/api_tests.cpp`.
mod common;
use common::*;

use brainsort::Desc;
use std::cmp::Ordering;

/// Sizes as in the C++ suite; `quick` (BRAINSORT_TEST_QUICK) caps them.
fn sizes(max_n: usize) -> Vec<usize> {
    let all = [0, 1, 2, 3, 5, 8, 16, 31, 32, 33, 64, 100, 255, 256, 257, 1000, 1023, 1024, 1025, 4096, 10000, 65535, 65536, 65537, 100000];
    all.iter().copied().filter(|&n| n <= max_n).collect()
}

fn test_key_type<K: Gen + Clone + Natural + brainsort::Key + std::fmt::Debug>(name: &str, max_n: usize) {
    for pattern in PATTERNS {
        for n in sizes(max_n) {
            for seed in [1u64, 2] {
                if seed == 2 && n > 10000 {
                    continue;
                }
                let mut pool = Pool;
                let input: Vec<Tagged<K>> = make_input::<K>(pattern, n, seed * 7919 + n as u64, &mut pool);
                let ctx = format!("{name}/{pattern} n={n} seed={seed}");
                if std::env::var_os("BRAINSORT_TEST_TRACE").is_some() {
                    eprintln!("{ctx}");
                }
                // by reference
                let mut a = input.clone();
                brainsort::sort_by_key_ref(&mut a, |t| &t.key);
                verify(&a, n, &format!("{ctx} [by ref]"));
                // by value
                let mut b = input.clone();
                brainsort::sort_by_key(&mut b, |t| t.key.clone());
                verify(&b, n, &format!("{ctx} [by value]"));
                assert!(a.iter().zip(&b).all(|(x, y)| x.id == y.id), "{ctx}: the two projection forms disagree");
            }
        }
    }
}

/// Sorting a vector of keys directly must agree with a stable sort under
/// the natural order.
fn test_self_keyed<K: Gen + Clone + Natural + brainsort::Key + std::fmt::Debug>(name: &str, max_n: usize) {
    for pattern in PATTERNS {
        for n in [0usize, 1, 7, 33, 1000, 4097, 100000] {
            if n > max_n {
                continue;
            }
            let mut pool = Pool;
            let input = make_input::<K>(pattern, n, n as u64 + 17, &mut pool);
            let mut v: Vec<K> = input.iter().map(|t| t.key.clone()).collect();
            let mut w = v.clone();
            brainsort::sort(&mut v);
            w.sort_by(|x, y| x.natural_cmp(y));
            for i in 0..n {
                assert!(v[i].identical(&w[i]), "{name}/{pattern} n={n}: brainsort::sort differs from a stable sort at {i}: {:?} vs {:?}", v[i], w[i]);
            }
        }
    }
}

#[test]
fn every_key_type() {
    let (big, mid) = if quick() { (4096, 2048) } else { (100_000, 10_000) };
    test_key_type::<i8>("i8", mid);
    test_key_type::<u8>("u8", mid);
    test_key_type::<i16>("i16", mid);
    test_key_type::<u16>("u16", mid);
    test_key_type::<i32>("i32", big);
    test_key_type::<u32>("u32", mid);
    test_key_type::<i64>("i64", big);
    test_key_type::<u64>("u64", mid);
    test_key_type::<i128>("i128", mid);
    test_key_type::<u128>("u128", mid);
    test_key_type::<usize>("usize", mid);
    test_key_type::<bool>("bool", mid);
    test_key_type::<char>("char", mid);
    test_key_type::<f32>("f32", mid);
    test_key_type::<f64>("f64", big);
    test_key_type::<Color>("enum Color", mid);
    test_key_type::<String>("String", big);
    test_key_type::<&'static str>("&str", mid);
    test_key_type::<Vec<u8>>("Vec<u8>", mid);
    test_key_type::<Box<str>>("Box<str>", mid);
    test_key_type::<std::sync::Arc<str>>("Arc<str>", mid);
    test_key_type::<std::borrow::Cow<'static, str>>("Cow<str>", mid);
    test_key_type::<std::time::Duration>("Duration", mid);
    test_key_type::<(i32, i32)>("(i32, i32) packed", mid);
    test_key_type::<(i8, u16, i32)>("(i8, u16, i32) packed", mid);
    test_key_type::<(u8, bool)>("(u8, bool) packed 32", mid);
    test_key_type::<(i64, i64)>("(i64, i64) composite", mid);
    test_key_type::<(i32, String)>("(i32, String)", mid);
    test_key_type::<(&'static str, i32)>("(&str, i32)", mid);
    test_key_type::<(String, String)>("(String, String)", mid);
    test_key_type::<[i32; 3]>("[i32; 3] composite", mid);
    test_key_type::<Desc<i32>>("Desc<i32>", mid);
    test_key_type::<Desc<f64>>("Desc<f64>", mid);
    test_key_type::<Desc<String>>("Desc<String>", mid);
    test_key_type::<(i32, Desc<f64>)>("(i32, Desc<f64>)", mid);
    test_key_type::<Desc<(i32, String)>>("Desc<(i32, String)>", mid);
    test_key_type::<(f64, String, Desc<u8>)>("(f64, String, Desc<u8>)", mid);
    test_key_type::<((i32, i32), (String,))>("nested tuple", mid);
    test_key_type::<std::cmp::Reverse<i16>>("Reverse<i16>", mid);
    test_key_type::<UserFixed>("user fixed key", mid);
    test_key_type::<UserBytes>("user bytes key", mid);
}

#[test]
fn self_keyed() {
    let (big, mid) = if quick() { (4096, 2048) } else { (100_000, 10_000) };
    test_self_keyed::<i32>("i32", big);
    test_self_keyed::<u64>("u64", mid);
    test_self_keyed::<i8>("i8", mid);
    test_self_keyed::<f32>("f32", mid);
    test_self_keyed::<f64>("f64", big);
    test_self_keyed::<String>("String", mid);
    test_self_keyed::<(i16, i16)>("(i16, i16)", mid);
    test_self_keyed::<u128>("u128", mid);
}

/// Plain slices of keys take the keys-only route: the sorted keys are
/// written back bit for bit, the two zeros of an `f32` or `f64` in input
/// order.
fn check_plain<K: Natural + Clone + std::fmt::Debug + brainsort::Key>(v: &[K], ctx: &str) {
    let mut a = v.to_vec();
    let mut b = v.to_vec();
    brainsort::sort(&mut a);
    b.sort_by(|x, y| x.natural_cmp(y));
    for i in 0..a.len() {
        assert!(a[i].identical(&b[i]), "{ctx} n={}: brainsort::sort differs from a stable sort at {i}: {:?} vs {:?}", v.len(), a[i], b[i]);
    }
}
#[test]
fn plain_keys() {
    let mut rng = Rng::new(5);
    let sizes: &[usize] = if quick() { &[33, 1000, 4097] } else { &[33, 1000, 4097, 100_000] };
    for &n in sizes {
        let pick = |rng: &mut Rng| -> f64 {
            match rng.next() % 16 {
                0..=2 => -0.0,
                3..=5 => 0.0,
                6 => f64::NAN,
                7 => -f64::NAN,
                8 => f64::INFINITY,
                9 => f64::NEG_INFINITY,
                10 => (rng.next() % 100) as f64 - 50.0,
                _ => f64::from_bits(rng.next()), // any bit pattern, NaNs with payloads included
            }
        };
        let f: Vec<f64> = (0..n).map(|_| pick(&mut rng)).collect();
        check_plain(&f, "f64 with zeros");
        let zeros: Vec<f64> = (0..n).map(|i| if i % 3 == 0 { -0.0 } else { 0.0 }).collect();
        check_plain(&zeros, "f64 all zeros");
        let mut sorted_zeros: Vec<f64> = (0..n).map(|i| (i / 4) as f64 - (n / 8) as f64).collect();
        for i in (0..n).step_by(7) {
            sorted_zeros[i] = -0.0;
        }
        check_plain(&sorted_zeros, "f64 nearly sorted with negative zeros");
        let u: Vec<u64> = (0..n).map(|_| rng.next()).collect();
        check_plain(&u, "u64 random");
        let i: Vec<i64> = (0..n).map(|_| (rng.next() % 7) as i64 - 3).collect();
        check_plain(&i, "i64 few unique");
        let s: Vec<usize> = (0..n).map(|i| n - i).collect();
        check_plain(&s, "usize reversed");
        let pick32 = |rng: &mut Rng| -> f32 {
            match rng.next() % 16 {
                0..=2 => -0.0,
                3..=5 => 0.0,
                6 => f32::NAN,
                7 => -f32::NAN,
                8 => f32::INFINITY,
                9 => f32::NEG_INFINITY,
                10 => (rng.next() % 100) as f32 - 50.0,
                _ => {
                    let x = f32::from_bits(rng.next() as u32); // any bit pattern; NaN payloads are covered above
                    if x.is_nan() { 1.5 } else { x }
                }
            }
        };
        let f: Vec<f32> = (0..n).map(|_| pick32(&mut rng)).collect();
        check_plain(&f, "f32 with zeros");
        let zeros: Vec<f32> = (0..n).map(|i| if i % 3 == 0 { -0.0 } else { 0.0 }).collect();
        check_plain(&zeros, "f32 all zeros");
        let few: Vec<i32> = (0..n).map(|_| (rng.next() % 7) as i32 - 3).collect();
        check_plain(&few, "i32 few unique");
        let u: Vec<u32> = (0..n).map(|_| rng.next() as u32).collect();
        check_plain(&u, "u32 random");
        let h: Vec<i16> = (0..n).map(|_| rng.next() as i16).collect();
        check_plain(&h, "i16 random");
        let b: Vec<u8> = (0..n).map(|_| rng.next() as u8).collect();
        check_plain(&b, "u8 random");
        let r: Vec<i8> = (0..n).map(|i| (((n - i) % 200) as i32 - 100) as i8).collect();
        check_plain(&r, "i8 reversed");
    }
    // raw pointers, by address
    let bytes = vec![0u8; 5000];
    let mut p: Vec<*const u8> = bytes.iter().map(|b| b as *const u8).collect();
    for i in (1..p.len()).rev() {
        let j = rng.next() as usize % (i + 1);
        p.swap(i, j);
    }
    let mut w = p.clone();
    brainsort::sort(&mut p);
    w.sort();
    assert_eq!(p, w, "pointers by address");
}

#[test]
fn float_order() {
    for n in [16usize, 40, 5000, 70000] {
        if quick() && n > 5000 {
            continue;
        }
        let mut rng = Rng::new(99);
        let quiet = 0x7FF8_0000_0000_0000u64;
        let sign = 0x8000_0000_0000_0000u64;
        let pool: Vec<f64> = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1e30,
            -1e30,
            f64::from_bits(quiet),
            f64::from_bits(quiet | sign),
            f64::from_bits(quiet | 1),
            f64::from_bits(quiet | sign | 1),
            f64::from_bits(1),
            -f64::from_bits(1),
            f64::MAX,
            f64::MIN,
        ];
        let mut v: Vec<Tagged<f64>> = (0..n).map(|i| Tagged { key: pool[rng.next() as usize % pool.len()], id: i as u32 }).collect();
        brainsort::sort_by_key(&mut v, |t| t.key);
        let cls = |x: f64| -> i32 {
            if x.is_nan() {
                return if x.is_sign_negative() { 0 } else { 6 };
            }
            if x.is_infinite() {
                return if x < 0.0 { 1 } else { 5 };
            }
            if x == 0.0 {
                return 3;
            }
            if x < 0.0 { 2 } else { 4 }
        };
        let mut seen = vec![false; n];
        for i in 0..n {
            assert!(!seen[v[i].id as usize], "permutation");
            seen[v[i].id as usize] = true;
            if i == 0 {
                continue;
            }
            let (a, b) = (v[i - 1].key, v[i].key);
            let (ca, cb) = (cls(a), cls(b));
            assert!(ca <= cb, "class order at {i}");
            if ca == cb && (ca == 2 || ca == 4) {
                assert!(b >= a, "value order at {i}");
            }
            if ca == cb && ca == 3 {
                assert!(v[i - 1].id < v[i].id, "zeros are equal and stable at {i}");
            }
            if ca == cb && (ca == 0 || ca == 6) {
                let (ma, mb) = (a.to_bits() & !sign, b.to_bits() & !sign);
                if ca == 6 {
                    assert!(ma <= mb, "positive NaN payload order at {i}");
                } else {
                    assert!(ma >= mb, "negative NaN payload order at {i}");
                }
                if ma == mb {
                    assert!(v[i - 1].id < v[i].id, "equal NaNs are stable at {i}");
                }
            }
        }
    }
    // f32 through the same rule
    let mut v: Vec<f32> = vec![f32::NAN, -0.0, 0.0, -f32::NAN, 1.0, -1.0, f32::INFINITY, f32::NEG_INFINITY];
    let mut w = v.clone();
    brainsort::sort(&mut v);
    w.sort_by(|a, b| a.natural_cmp(b));
    assert!(v.iter().zip(&w).all(|(a, b)| a.to_bits() == b.to_bits()), "{v:?} vs {w:?}");
}

#[test]
fn containers_and_element_kinds() {
    let mut rng = Rng::new(5);
    let n = if quick() { 4000 } else { 20000 };
    // rows: by an integer, by a string member by reference, by a temporary string, by a descending double
    #[derive(Clone, Debug, PartialEq)]
    struct Row {
        id: i64,
        name: String,
        score: f64,
    }
    let rows: Vec<Row> = (0..n).map(|_| Row { id: (rng.next() % 5000) as i64, name: word(&mut rng), score: (rng.next() % 1000) as f64 / 8.0 }).collect();
    let stable_ref = |less: &dyn Fn(&Row, &Row) -> Ordering| {
        let mut r = rows.clone();
        r.sort_by(less);
        r
    };
    let mut r1 = rows.clone();
    brainsort::sort_by_key(&mut r1, |r| r.id);
    assert_eq!(r1, stable_ref(&|a, b| a.id.cmp(&b.id)), "Row by id");
    let mut r2 = rows.clone();
    brainsort::sort_by_key_ref(&mut r2, |r| r.name.as_str());
    assert_eq!(r2, stable_ref(&|a, b| a.name.cmp(&b.name)), "Row by name (reference)");
    let mut r3 = rows.clone();
    brainsort::sort_by_key(&mut r3, |r| format!("{}!", r.name));
    assert_eq!(r3, stable_ref(&|a, b| format!("{}!", a.name).cmp(&format!("{}!", b.name))), "Row by temporary string");
    let mut r4 = rows.clone();
    brainsort::sort_by_key(&mut r4, |r| (r.id, Desc(r.score)));
    assert_eq!(r4, stable_ref(&|a, b| a.id.cmp(&b.id).then(b.score.partial_cmp(&a.score).unwrap())), "Row by (id, desc score)");
    let mut r5 = rows.clone();
    brainsort::sort_by_key(&mut r5, |r| (r.name.clone(), r.id));
    assert_eq!(r5, stable_ref(&|a, b| (&a.name, a.id).cmp(&(&b.name, b.id))), "Row by (name, id)");
    let mut r6 = rows.clone();
    brainsort::sort_by(&mut r6, |a, b| a.score.partial_cmp(&b.score).unwrap());
    assert_eq!(r6, stable_ref(&|a, b| a.score.partial_cmp(&b.score).unwrap()), "Row with comparator");
    let mut r7 = rows.clone();
    brainsort::sort_by(&mut r7, |a, b| b.id.cmp(&a.id));
    assert_eq!(r7, stable_ref(&|a, b| b.id.cmp(&a.id)), "Row with descending comparator");
    let mut r8 = rows.clone();
    brainsort::sort_by_key(&mut r8, |r| Desc(r.name.clone()));
    assert_eq!(r8, stable_ref(&|a, b| b.name.cmp(&a.name)), "Row by descending string");

    // vector<String> sorted directly, by &str, by bytes
    let s: Vec<String> = (0..n).map(|_| word(&mut rng)).collect();
    let mut reference = s.clone();
    reference.sort();
    let mut a = s.clone();
    brainsort::sort(&mut a);
    assert_eq!(a, reference, "Vec<String>");
    let mut b = s.clone();
    brainsort::sort_by_key_ref(&mut b, |x| x.as_bytes());
    assert_eq!(b, reference, "Vec<String> by bytes");
    let mut c: Vec<&str> = s.iter().map(|x| x.as_str()).collect();
    brainsort::sort(&mut c);
    assert!(c.iter().zip(&reference).all(|(x, y)| *x == y), "Vec<&str>");

    // boxed elements: permuted in place through bitwise moves
    let p: Vec<Box<i32>> = (0..n).map(|_| Box::new((rng.next() % 1000) as i32)).collect();
    let mut reference: Vec<i32> = p.iter().map(|x| **x).collect();
    reference.sort();
    let mut q = p;
    brainsort::sort_by_key(&mut q, |x| **x);
    assert!(q.iter().zip(&reference).all(|(x, y)| **x == *y), "Vec<Box<i32>>");
    // Rc elements with the reference count checked afterwards
    let rc: Vec<std::rc::Rc<u32>> = (0..n).map(|_| std::rc::Rc::new(rng.next() as u32 % 3000)).collect();
    let mut rc2 = rc.clone();
    brainsort::sort_by_key(&mut rc2, |x| **x);
    assert!(rc2.windows(2).all(|w| w[0] <= w[1]));
    assert!(rc2.iter().all(|x| std::rc::Rc::strong_count(x) == 2), "no element was duplicated or lost");

    // large and over-aligned elements
    #[derive(Clone, Copy)]
    struct Big {
        key: i64,
        payload: [u8; 120],
    }
    let mut big: Vec<Big> = (0..n).map(|i| Big { key: (rng.next() % 3000) as i64, payload: [i as u8; 120] }).collect();
    let mut bref = big.clone();
    bref.sort_by_key(|b| b.key);
    brainsort::sort_by_key(&mut big, |b| b.key);
    assert!(big.iter().zip(&bref).all(|(x, y)| x.key == y.key && x.payload == y.payload), "120-byte elements");
    #[derive(Clone, Copy)]
    #[repr(align(64))]
    struct Aligned {
        key: i32,
        pad: [i32; 15],
    }
    let mut al: Vec<Aligned> = (0..n).map(|i| Aligned { key: (rng.next() % 3000) as i32, pad: [i; 15] }).collect();
    let mut aref = al.clone();
    aref.sort_by_key(|a| a.key);
    brainsort::sort_by_key(&mut al, |a| a.key);
    assert!(al.iter().zip(&aref).all(|(x, y)| x.key == y.key && x.pad[0] == y.pad[0]), "over-aligned elements");
    // zero-sized elements
    let mut z = vec![(); 1000];
    brainsort::sort_by_key(&mut z, |_| 0u8);
    assert_eq!(z.len(), 1000);
    // tiny ranges
    let mut e: Vec<i32> = vec![];
    brainsort::sort(&mut e);
    let mut one = vec![7];
    brainsort::sort(&mut one);
    assert_eq!(one, [7]);
    let mut two = vec![9, 3];
    brainsort::sort(&mut two);
    assert_eq!(two, [3, 9]);
    // slices of a VecDeque
    let mut d: std::collections::VecDeque<u16> = (0..5000).map(|_| rng.next() as u16).collect();
    let mut dref: Vec<u16> = d.iter().copied().collect();
    dref.sort();
    brainsort::sort(d.make_contiguous());
    assert!(d.iter().zip(&dref).all(|(x, y)| x == y), "VecDeque::make_contiguous");
    // arrays, pointers by address, Option-free char keys
    let mut arr = [5u8, 3, 9, 1, 3];
    brainsort::sort(&mut arr);
    assert_eq!(arr, [1, 3, 3, 5, 9]);
    let words: Vec<String> = (0..500).map(|_| word(&mut rng)).collect();
    let mut ptrs: Vec<*const String> = words.iter().map(|w| w as *const String).collect();
    ptrs.reverse();
    brainsort::sort(&mut ptrs);
    assert!(ptrs.windows(2).all(|w| (w[0] as usize) <= (w[1] as usize)), "pointers by address");
    let mut chars: Vec<char> = "the quick brown fox".chars().collect();
    let mut cref = chars.clone();
    cref.sort();
    brainsort::sort(&mut chars);
    assert_eq!(chars, cref);
}

#[test]
fn comparator_paths() {
    let big = if quick() { 4096 } else { 100_000 };
    for pattern in PATTERNS {
        for n in [0usize, 1, 2, 16, 17, 32, 33, 100, 255, 1000, 4097, 10000, 100000] {
            if n > big {
                continue;
            }
            let mut pool = Pool;
            let input = make_input::<i32>(pattern, n, 5 * n as u64 + 3, &mut pool);
            let mut a = input.clone();
            brainsort::sort_by(&mut a, |x, y| x.key.cmp(&y.key));
            verify(&a, n, &format!("comparator 8-byte/{pattern} n={n}"));
            #[derive(Clone, Copy)]
            struct Wide {
                key: i32,
                id: u32,
                _payload: [u8; 56],
            }
            let mut w: Vec<Wide> = input.iter().map(|t| Wide { key: t.key, id: t.id, _payload: [0; 56] }).collect();
            brainsort::sort_by(&mut w, |x, y| x.key.cmp(&y.key));
            let tagged: Vec<Tagged<i32>> = w.iter().map(|t| Tagged { key: t.key, id: t.id }).collect();
            verify(&tagged, n, &format!("comparator 64-byte/{pattern} n={n}"));
        }
    }
    // non-trivial elements
    let mut pool = Pool;
    let mut v = make_input::<String>("random", 5000, 9, &mut pool);
    brainsort::sort_by(&mut v, |a, b| a.key.cmp(&b.key));
    verify(&v, 5000, "comparator on String elements");
}

/// `sort_by_inferred`: the same result as `sort_by` whether the comparator
/// is a window of the element, coarser than one, only agrees with one on
/// the sample, or is no window at all.
#[test]
fn inferred_comparator() {
    #[derive(Clone, Copy, Debug, PartialEq)]
    #[repr(C)]
    struct Mixed {
        score: f64,
        id: u32,
        group: i32,
    }
    // SAFETY: 8 + 4 + 4 bytes, no padding.
    unsafe impl brainsort::PlainBytes for Mixed {}
    let mut rng = Rng::new(23);
    let big = if quick() { 10_000 } else { 100_000 };
    for n in [40usize, 300, 5000, big] {
        let v: Vec<Mixed> = (0..n).map(|i| Mixed { score: (rng.next() % 100_000) as f64 / 7.0 - 5000.0, id: i as u32, group: (rng.next() % 50) as i32 }).collect();
        let same = |cmp: &dyn Fn(&Mixed, &Mixed) -> Ordering, ctx: &str| {
            let mut a = v.clone();
            let mut b = v.clone();
            brainsort::sort_by_inferred(&mut a, cmp);
            b.sort_by(cmp);
            assert_eq!(a, b, "{ctx} n={n}: sort_by_inferred differs from a stable sort");
        };
        same(&|a, b| a.score.total_cmp(&b.score), "f64 field");
        same(&|a, b| b.score.total_cmp(&a.score), "f64 field descending");
        same(&|a, b| a.group.cmp(&b.group), "i32 field with ties");
        same(&|a, b| (a.group / 10).cmp(&(b.group / 10)), "coarse comparator");
        same(&|a, b| a.group.cmp(&b.group).then(b.score.total_cmp(&a.score)), "two fields");
        same(&|a, b| a.score.abs().total_cmp(&b.score.abs()), "not a window");
        // agrees with the field on every pair but those with one rare value
        let k = |g: i32| if g == 42 { i32::MAX } else { g };
        same(&|a, b| k(a.group).cmp(&k(b.group)), "adversarial comparator");
    }
    let mut ints: Vec<i64> = (0..50_000).map(|_| rng.next() as i64 % 1000).collect();
    let mut w = ints.clone();
    brainsort::sort_by_inferred(&mut ints, |a, b| b.cmp(a));
    w.sort_by(|a, b| b.cmp(a));
    assert_eq!(ints, w, "i64 descending");
}

#[test]
fn random_shapes() {
    let mut rng = Rng::new(2026);
    let iterations = if quick() { 100 } else { 400 };
    for _ in 0..iterations {
        let n = (rng.next() % 3000) as usize;
        let pattern = PATTERNS[rng.next() as usize % PATTERNS.len()];
        let mut pool = Pool;
        let seed = rng.next();
        let mut a = make_input::<i32>(pattern, n, seed, &mut pool);
        brainsort::sort_by_key(&mut a, |t| t.key);
        verify(&a, n, &format!("random shape i32 {pattern} n={n}"));
        let mut b = make_input::<String>(pattern, n, seed, &mut pool);
        brainsort::sort_by_key_ref(&mut b, |t| t.key.as_str());
        verify(&b, n, &format!("random shape String {pattern} n={n}"));
        type C = (i16, &'static str, Desc<u32>);
        let mut c = make_input::<C>(pattern, n, seed, &mut pool);
        brainsort::sort_by_key_ref(&mut c, |t| &t.key);
        verify(&c, n, &format!("random shape composite {pattern} n={n}"));
    }
}

#[test]
fn memory_cache_can_be_released() {
    let mut pool = Pool;
    let mut v = make_input::<i64>("random", 200_000, 13, &mut pool);
    brainsort::sort_by_key(&mut v, |t| t.key);
    brainsort::release_memory();
    let mut w = make_input::<i64>("random", 200_000, 14, &mut pool);
    brainsort::set_memory_cache_limit(0);
    brainsort::sort_by_key(&mut w, |t| t.key);
    brainsort::set_memory_cache_limit(32 << 20);
    brainsort::release_memory();
    verify(&v, v.len(), "before release");
    verify(&w, w.len(), "with the cache disabled");
}

#[test]
fn threads() {
    let handles: Vec<_> = (0..8)
        .map(|t| {
            std::thread::spawn(move || {
                let mut pool = Pool;
                let n = if quick() { 20_000 } else { 200_000 };
                let mut a = make_input::<i32>("random", n, 10 + t, &mut pool);
                let mut b = make_input::<String>("random", n / 6, 20 + t, &mut pool);
                let mut c = make_input::<f64>("nearly_sorted", n / 2, 30 + t, &mut pool);
                brainsort::sort_by_key(&mut a, |x| x.key);
                brainsort::sort_by_key_ref(&mut b, |x| x.key.as_str());
                brainsort::sort_by_key(&mut c, |x| x.key);
                verify(&a, a.len(), "thread int");
                verify(&b, b.len(), "thread string");
                verify(&c, c.len(), "thread double");
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

#[test]
#[ignore = "large: about a minute; CI runs it once on Linux"]
fn large_inputs() {
    let mut rng = Rng::new(77);
    let mut v: Vec<i32> = (0..10_000_000).map(|_| rng.next() as i32).collect();
    let mut w = v.clone();
    brainsort::sort(&mut v);
    w.sort_unstable();
    assert_eq!(v, w, "10M i32");
    let mut pool = Pool;
    let mut a = make_input::<i64>("random", 2_000_000, 5, &mut pool);
    brainsort::sort_by_key(&mut a, |t| t.key);
    verify(&a, a.len(), "2M i64");
    let mut b = make_input::<f64>("sawtooth", 1_000_000, 6, &mut pool);
    brainsort::sort_by_key(&mut b, |t| t.key);
    verify(&b, b.len(), "1M f64");
    // 16-byte records beyond the cache: the MSD scatter on both halves
    let mut c = make_input::<f64>("random", 5_000_000, 8, &mut pool);
    brainsort::sort_by_key(&mut c, |t| t.key);
    verify(&c, c.len(), "5M f64");
    let mut d: Vec<f64> = c.iter().map(|t| t.key).collect();
    let mut rng = Rng::new(9);
    for i in (0..d.len()).rev() {
        let j = rng.next() as usize % (i + 1);
        d.swap(i, j);
    }
    for i in (0..d.len()).step_by(97) {
        d[i] = -0.0; // written back bit for bit
    }
    let mut e = d.clone();
    brainsort::sort(&mut d);
    e.sort_by(|a, b| a.natural_cmp(b));
    assert!(d.iter().zip(&e).all(|(x, y)| x.to_bits() == y.to_bits()), "5M f64 values, negative zeros kept");
    let mut s = make_input::<String>("random", 1_000_000, 7, &mut pool);
    brainsort::sort_by_key_ref(&mut s, |t| t.key.as_str());
    verify(&s, s.len(), "1M String");
}

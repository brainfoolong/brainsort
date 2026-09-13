//! The fuzz body shared by the cargo-fuzz target and the smoke test: the
//! input bytes choose a key type and the elements; the result is checked for
//! order, stability and permutation against an independent comparison.
//!
//! One mode hands the comparator overloads a comparator that is not an
//! order at all (random per call, or a hash of the pair) or an order the
//! key inference cannot express as a byte window: there only the
//! permutation is checked, and, for the orders, that the result is sorted
//! and stable by them.
//!
//! A leading size byte repeats the element bytes, so that a short fuzz
//! input reaches the routes that start at thousands of elements (the key
//! inference, the scatter beyond the cache, the index route) without the
//! fuzzer having to produce the bytes.
#![allow(dead_code)]
use brainsort::Desc;
use std::cmp::Ordering;

#[derive(Clone, Debug)]
pub struct Tagged<K> {
    pub key: K,
    pub id: u32,
}

/// A 16-byte element for the comparator overloads: every byte initialised.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Row {
    pub key: u64,
    pub id: u32,
    pub pad: u32,
}
// SAFETY: three fields of eight, four and four bytes, no padding.
unsafe impl brainsort::PlainBytes for Row {}

/// A 32-byte element: the comparator overloads sort it through indices.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Wide {
    pub key: u64,
    pub id: u32,
    pub pad: [u32; 5],
}
// SAFETY: eight, four and twenty bytes, no padding.
unsafe impl brainsort::PlainBytes for Wide {}

/// A comparator on elements, boxed: the flavours differ in what they capture.
pub type Cmp<E> = Box<dyn FnMut(&E, &E) -> Ordering>;

pub trait Elem: Copy + brainsort::PlainBytes {
    fn new(key: u64, id: u32) -> Self;
    fn key(&self) -> u64;
    fn id(&self) -> u32;
}
impl Elem for Row {
    fn new(key: u64, id: u32) -> Self {
        Row { key, id, pad: 0 }
    }
    fn key(&self) -> u64 {
        self.key
    }
    fn id(&self) -> u32 {
        self.id
    }
}
impl Elem for Wide {
    fn new(key: u64, id: u32) -> Self {
        Wide { key, id, pad: [0; 5] }
    }
    fn key(&self) -> u64 {
        self.key
    }
    fn id(&self) -> u32 {
        self.id
    }
}

fn check<K>(out: &[Tagged<K>], n: usize, cmp: impl Fn(&K, &K) -> Ordering, what: &str) {
    let mut seen = vec![false; n];
    for i in 0..n {
        assert!((out[i].id as usize) < n && !seen[out[i].id as usize], "{what}: not a permutation");
        seen[out[i].id as usize] = true;
        if i == 0 {
            continue;
        }
        match cmp(&out[i - 1].key, &out[i].key) {
            Ordering::Greater => panic!("{what}: not sorted at {i}"),
            Ordering::Equal => assert!(out[i - 1].id < out[i].id, "{what}: not stable at {i}"),
            Ordering::Less => {}
        }
    }
}
fn run<K: brainsort::Key + Ord + Clone>(mut v: Vec<Tagged<K>>, what: &str) {
    let n = v.len();
    brainsort::sort_by_key_ref(&mut v, |t| &t.key);
    check(&v, n, |a, b| a.cmp(b), what);
}

fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The comparators that break the contract, or the inference. Flavours 0
/// and 1 are not orders: only the permutation can be checked. Flavours 2
/// and 3 are total orders that no byte window of the element expresses,
/// so an inferred window that the sample agreed with has to fail the
/// verification pass.
fn adversary<E: Elem>(flavour: u8, seed: u64, threshold: u64) -> Cmp<E> {
    match flavour % 4 {
        // random per call: asymmetric, and a different answer next time
        0 => {
            let mut state = seed;
            Box::new(move |_, _| {
                state = mix(state);
                match state % 3 {
                    0 => Ordering::Less,
                    1 => Ordering::Equal,
                    _ => Ordering::Greater,
                }
            })
        }
        // a hash of the pair: antisymmetric and repeatable, not transitive
        1 => Box::new(move |a, b| {
            let (lo, hi) = (a.id().min(b.id()), a.id().max(b.id()));
            let o = match mix(seed ^ ((lo as u64) << 32 | hi as u64)) % 7 {
                0 => Ordering::Equal,
                1..=3 => Ordering::Less,
                _ => Ordering::Greater,
            };
            if a.id() > b.id() { o.reverse() } else { o }
        }),
        // ascending below the threshold, descending above it: the sample
        // of adjacent pairs mostly sees the ascending half
        2 => Box::new(move |a, b| if a.key() > threshold && b.key() > threshold { b.key().cmp(&a.key()) } else { a.key().cmp(&b.key()) }),
        // the order of the key times an odd constant
        _ => Box::new(move |a, b| a.key().wrapping_mul(0x9E37_79B9_7F4A_7C15).cmp(&b.key().wrapping_mul(0x9E37_79B9_7F4A_7C15))),
    }
}

/// The sort, and for a comparator that is not an order, the one panic
/// the standard library's sort documents for it, swallowed: the slice
/// has to be a permutation after it all the same. Any other panic (an
/// assertion inside brainsort) stays fatal. The hook is replaced for the
/// call so the swallowed panic prints nothing, and so the fuzz target's
/// abort-on-panic hook does not fire.
fn sort_tolerating_std_panic<E: Elem>(v: &mut [E], cmp: &mut Cmp<E>, inferred: bool, may_panic: bool) {
    let mut sort = |v: &mut [E]| {
        if inferred {
            brainsort::sort_by_inferred(v, &mut *cmp);
        } else {
            brainsort::sort_by(v, &mut *cmp);
        }
    };
    if !may_panic {
        sort(v);
        return;
    }
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sort(v)));
    std::panic::set_hook(prev);
    if let Err(e) = r {
        let msg = e.downcast_ref::<&str>().copied().or_else(|| e.downcast_ref::<String>().map(|s| s.as_str())).unwrap_or("");
        if !msg.contains("does not correctly implement a total order") {
            std::panic::resume_unwind(e);
        }
    }
}

/// Every element exactly once; for a comparator that is an order, sorted
/// and stable by it.
fn check_elems<E: Elem>(out: &[E], n: usize, order: Option<&mut Cmp<E>>, what: &str) {
    let mut seen = vec![false; n];
    for e in out {
        assert!((e.id() as usize) < n && !seen[e.id() as usize], "{what}: not a permutation");
        seen[e.id() as usize] = true;
    }
    if let Some(cmp) = order {
        for i in 1..n {
            match cmp(&out[i - 1], &out[i]) {
                Ordering::Greater => panic!("{what}: not sorted at {i}"),
                Ordering::Equal => assert!(out[i - 1].id() < out[i].id(), "{what}: not stable at {i}"),
                Ordering::Less => {}
            }
        }
    }
}

fn adversarial<E: Elem>(data: &[u8], flavour: u8, inferred: bool, repeat: usize) {
    let seed = mix(data.iter().take(8).fold(0u64, |s, &b| (s << 8) | b as u64));
    let threshold = data.get(8).copied().unwrap_or(0) as u64 * 256;
    let keys: Vec<u64> = data.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as u64).collect();
    let mut v: Vec<E> = (0..repeat).flat_map(|_| keys.iter().copied()).enumerate().map(|(i, k)| E::new(k, i as u32)).collect();
    let n = v.len();
    let mut cmp = adversary::<E>(flavour, seed, threshold);
    sort_tolerating_std_panic(&mut v, &mut cmp, inferred, flavour % 4 < 2);
    let what = format!("adversarial comparator, flavour {}, {} bytes, {}", flavour % 4, core::mem::size_of::<E>(), if inferred { "inferred" } else { "plain" });
    check_elems(&v, n, if flavour % 4 >= 2 { Some(&mut cmp) } else { None }, &what);
}

/// The element bytes are repeated this many times: 1 for most inputs, up
/// to 64 for one in eight, so an 8 KiB fuzz input reaches 100,000 elements.
/// Under Miri, which interprets, never.
fn repeats(b: u8) -> usize {
    if cfg!(miri) || b % 8 != 0 { 1 } else { 1 + (b as usize / 8) % 64 }
}

pub fn fuzz_one(data: &[u8]) {
    if data.len() < 2 {
        return;
    }
    let mode = data[0] % 8;
    let repeat = repeats(data[1]);
    let data = &data[2..];
    match mode {
        0 => {
            // int32 keys from 4-byte chunks
            let v: Vec<Tagged<i32>> = (0..repeat).flat_map(|_| data.chunks_exact(4)).enumerate().map(|(i, c)| Tagged { key: i32::from_le_bytes(c.try_into().unwrap()), id: i as u32 }).collect();
            run(v, "i32");
        }
        1 => {
            // 8-bit keys: many duplicates
            let v: Vec<Tagged<u8>> = (0..repeat).flat_map(|_| data.iter()).enumerate().map(|(i, &b)| Tagged { key: b, id: i as u32 }).collect();
            run(v, "u8");
        }
        2 => {
            // strings: split at zero bytes and at long runs
            let mut v: Vec<Tagged<Vec<u8>>> = Vec::new();
            let mut cur = Vec::new();
            for _ in 0..repeat {
                for &b in data {
                    if b == 0 || cur.len() >= 40 {
                        v.push(Tagged { key: std::mem::take(&mut cur), id: v.len() as u32 });
                    } else {
                        cur.push(b);
                    }
                }
            }
            v.push(Tagged { key: cur, id: v.len() as u32 });
            run(v, "bytes");
        }
        3 => {
            // doubles from 8-byte chunks, NaNs included: checked against the documented total order
            let mut v: Vec<Tagged<f64>> = (0..repeat).flat_map(|_| data.chunks_exact(8)).enumerate().map(|(i, c)| Tagged { key: f64::from_le_bytes(c.try_into().unwrap()), id: i as u32 }).collect();
            let n = v.len();
            brainsort::sort_by_key(&mut v, |t| t.key);
            check(&v, n, |a, b| brainsort::key::f64_radix(*a).cmp(&brainsort::key::f64_radix(*b)), "f64");
        }
        4 => {
            // composite: (i16, String, Desc<u32>)
            type K = (i16, String, Desc<u32>);
            let mut v: Vec<Tagged<K>> = Vec::new();
            for c in (0..repeat).flat_map(|_| data.chunks_exact(8)) {
                let a = i16::from_le_bytes([c[0], c[1]]);
                let u = u32::from_le_bytes([c[2], c[3], c[4], c[5]]);
                let s: String = c[6..6 + (c[6] % 3) as usize].iter().map(|&b| (b'a' + b % 26) as char).collect();
                v.push(Tagged { key: (a, s, Desc(u)), id: v.len() as u32 });
            }
            let n = v.len();
            brainsort::sort_by_key_ref(&mut v, |t| &t.key);
            check(&v, n, |x, y| (x.0, &x.1).cmp(&(y.0, &y.1)).then(y.2.0.cmp(&x.2.0)), "composite");
        }
        5 => {
            // int64 with structure: mostly sorted with a few edits
            let v: Vec<Tagged<i64>> =
                (0..repeat).flat_map(|_| data.iter()).enumerate().map(|(i, &b)| Tagged { key: i as i64 * 1000 + if b % 7 == 0 { -(b as i64) * 100000 } else { 0 }, id: i as u32 }).collect();
            run(v, "i64 structured");
        }
        6 => {
            // a comparator on 16-byte elements: the in-place routes and the comparison sort
            let mut v: Vec<Tagged<u64>> = (0..repeat).flat_map(|_| data.chunks_exact(2)).enumerate().map(|(i, c)| Tagged { key: u16::from_le_bytes([c[0], c[1]]) as u64 * 3, id: i as u32 }).collect();
            let n = v.len();
            brainsort::sort_by(&mut v, |a, b| a.key.cmp(&b.key));
            check(&v, n, |a, b| a.cmp(b), "comparator");
        }
        _ => {
            // a comparator that is not an order, or an order the inference
            // cannot express, on 16- and 32-byte elements, plain and inferred
            if data.is_empty() {
                return;
            }
            let (flavour, entry) = (data[0] % 4, data[0] / 4 % 4);
            let data = &data[1..];
            match entry {
                0 => adversarial::<Row>(data, flavour, false, repeat),
                1 => adversarial::<Wide>(data, flavour, false, repeat),
                2 => adversarial::<Row>(data, flavour, true, repeat),
                _ => adversarial::<Wide>(data, flavour, true, repeat),
            }
        }
    }
}

//! Shared by the integration tests: a natural order per key type written
//! independently of the library, key generators, the ten input patterns
//! and the verifier.
#![allow(dead_code, clippy::needless_range_loop)]
use brainsort::Desc;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::sync::Arc;
use std::time::Duration;

pub fn quick() -> bool {
    std::env::var_os("BRAINSORT_TEST_QUICK").is_some()
}

/// splitmix64: a small deterministic generator for the tests.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// The natural order of a key type, written independently of the library,
/// plus a bitwise identity for the self-keyed check.
pub trait Natural {
    fn natural_cmp(&self, other: &Self) -> Ordering;
    fn identical(&self, other: &Self) -> bool {
        self.natural_cmp(other) == Ordering::Equal
    }
}
macro_rules! natural_ord {
    ($($t:ty),*) => {$( impl Natural for $t { fn natural_cmp(&self, o: &Self) -> Ordering { self.cmp(o) } } )*};
}
natural_ord!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, usize, isize, bool, char, String, &'static str, Vec<u8>, Box<str>, Arc<str>, Cow<'static, str>, Duration);
fn float_rank(x: f64) -> (i32, u64) {
    // -NaN < -inf < negatives < zeros (equal) < positives < +inf < +NaN, NaNs by payload
    let sign = 0x8000_0000_0000_0000u64;
    if x.is_nan() {
        let payload = x.to_bits() & !sign;
        return if x.is_sign_negative() { (0, u64::MAX - payload) } else { (6, payload) };
    }
    if x.is_infinite() {
        return if x < 0.0 { (1, 0) } else { (5, 0) };
    }
    if x == 0.0 {
        return (3, 0);
    }
    if x < 0.0 { (2, u64::MAX - (x.to_bits() & !sign)) } else { (4, x.to_bits()) }
}
impl Natural for f64 {
    fn natural_cmp(&self, o: &Self) -> Ordering {
        float_rank(*self).cmp(&float_rank(*o))
    }
    fn identical(&self, o: &Self) -> bool {
        self.to_bits() == o.to_bits()
    }
}
impl Natural for f32 {
    fn natural_cmp(&self, o: &Self) -> Ordering {
        let widen = |f: f32| -> f64 {
            if f.is_nan() {
                let sign = 0x8000_0000u32;
                let bits = f.to_bits();
                // keep the payload order: map the 23-bit payload into the f64 payload space
                f64::from_bits(0x7FF8_0000_0000_0000 | ((bits & !sign & 0x3F_FFFF) as u64) | if bits & sign != 0 { 0x8000_0000_0000_0000 } else { 0 })
            } else {
                f as f64
            }
        };
        float_rank(widen(*self)).cmp(&float_rank(widen(*o)))
    }
    fn identical(&self, o: &Self) -> bool {
        self.to_bits() == o.to_bits()
    }
}
impl<K: Natural> Natural for Desc<K> {
    fn natural_cmp(&self, o: &Self) -> Ordering {
        o.0.natural_cmp(&self.0)
    }
}
impl<K: Natural> Natural for std::cmp::Reverse<K> {
    fn natural_cmp(&self, o: &Self) -> Ordering {
        o.0.natural_cmp(&self.0)
    }
}
impl<A: Natural, B: Natural> Natural for (A, B) {
    fn natural_cmp(&self, o: &Self) -> Ordering {
        self.0.natural_cmp(&o.0).then_with(|| self.1.natural_cmp(&o.1))
    }
}
impl<A: Natural> Natural for (A,) {
    fn natural_cmp(&self, o: &Self) -> Ordering {
        self.0.natural_cmp(&o.0)
    }
}
impl<A: Natural, B: Natural, C: Natural> Natural for (A, B, C) {
    fn natural_cmp(&self, o: &Self) -> Ordering {
        self.0.natural_cmp(&o.0).then_with(|| self.1.natural_cmp(&o.1)).then_with(|| self.2.natural_cmp(&o.2))
    }
}
impl<K: Natural, const N: usize> Natural for [K; N] {
    fn natural_cmp(&self, o: &Self) -> Ordering {
        for i in 0..N {
            let c = self[i].natural_cmp(&o[i]);
            if c != Ordering::Equal {
                return c;
            }
        }
        Ordering::Equal
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i8)]
pub enum Color {
    Red = -3,
    Green = 0,
    Blue = 5,
    Black = 100,
}
brainsort::fixed_key!(Color, 8, |k| ((*k as i8) as u8 ^ 0x80) as u64);
impl Natural for Color {
    fn natural_cmp(&self, o: &Self) -> Ordering {
        (*self as i8).cmp(&(*o as i8))
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct UserFixed {
    pub major: u16,
    pub minor: u16,
}
brainsort::fixed_key!(UserFixed, 32, |k| ((k.major as u64) << 16) | k.minor as u64);
impl Natural for UserFixed {
    fn natural_cmp(&self, o: &Self) -> Ordering {
        self.cmp(o)
    }
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct UserBytes {
    pub tag: String,
}
brainsort::bytes_key!(UserBytes, |k| k.tag.as_bytes());
impl Natural for UserBytes {
    fn natural_cmp(&self, o: &Self) -> Ordering {
        self.tag.cmp(&o.tag)
    }
}

/// Strings handed out as `&'static str` live here for the whole test.
#[derive(Default)]
pub struct Pool;
pub fn leak(s: String) -> &'static str {
    // Kept reachable, so that LeakSanitizer does not count them.
    static KEPT: std::sync::Mutex<Vec<&'static str>> = std::sync::Mutex::new(Vec::new());
    let s: &'static str = Box::leak(s.into_boxed_str());
    KEPT.lock().unwrap().push(s);
    s
}
pub fn padded(v: i64) -> String {
    format!("{:020}", v + (1i64 << 62))
}
pub fn word(rng: &mut Rng) -> String {
    let len = 3 + rng.next() as usize % 10;
    (0..len).map(|_| (b'a' + (rng.next() % 26) as u8) as char).collect()
}

/// make(v): a key that is monotone in v, so the integer patterns carry over
/// to every key type; random(): a full-entropy key.
pub trait Gen: Sized {
    fn make(v: i64, pool: &mut Pool, rng: &mut Rng) -> Self;
    fn random(pool: &mut Pool, rng: &mut Rng) -> Self;
}
macro_rules! gen_int {
    ($($t:ty),*) => {$(
        impl Gen for $t {
            fn make(v: i64, _: &mut Pool, _: &mut Rng) -> Self { v as $t }
            fn random(_: &mut Pool, rng: &mut Rng) -> Self { rng.next() as $t }
        }
    )*};
}
gen_int!(i8, u8, i16, u16, i32, u32, i64, u64, usize, isize);
impl Gen for i128 {
    fn make(v: i64, _: &mut Pool, _: &mut Rng) -> Self {
        v as i128
    }
    fn random(_: &mut Pool, rng: &mut Rng) -> Self {
        ((rng.next() as u128) << 64 | rng.next() as u128) as i128
    }
}
impl Gen for u128 {
    fn make(v: i64, _: &mut Pool, _: &mut Rng) -> Self {
        (v as u128).wrapping_add(1 << 62)
    }
    fn random(_: &mut Pool, rng: &mut Rng) -> Self {
        (rng.next() as u128) << 64 | rng.next() as u128
    }
}
impl Gen for bool {
    fn make(v: i64, _: &mut Pool, _: &mut Rng) -> Self {
        v & 1 != 0
    }
    fn random(_: &mut Pool, rng: &mut Rng) -> Self {
        rng.next() & 1 != 0
    }
}
impl Gen for char {
    fn make(v: i64, _: &mut Pool, _: &mut Rng) -> Self {
        char::from_u32((v.rem_euclid(0xD700)) as u32 + 32).unwrap()
    }
    fn random(_: &mut Pool, rng: &mut Rng) -> Self {
        char::from_u32((rng.next() % 0xD700) as u32 + 32).unwrap()
    }
}
impl Gen for f32 {
    fn make(v: i64, _: &mut Pool, _: &mut Rng) -> Self {
        v as f32 * 0.25
    }
    fn random(_: &mut Pool, rng: &mut Rng) -> Self {
        ((rng.next() >> 11) as f64 / (1u64 << 53) as f64 * 2e6 - 1e6) as f32
    }
}
impl Gen for f64 {
    fn make(v: i64, _: &mut Pool, _: &mut Rng) -> Self {
        v as f64 * 0.25
    }
    fn random(_: &mut Pool, rng: &mut Rng) -> Self {
        (rng.next() >> 11) as f64 / (1u64 << 53) as f64 * 2e6 - 1e6
    }
}
impl Gen for Color {
    fn make(v: i64, _: &mut Pool, _: &mut Rng) -> Self {
        [Color::Red, Color::Green, Color::Blue, Color::Black][v.rem_euclid(4) as usize]
    }
    fn random(p: &mut Pool, rng: &mut Rng) -> Self {
        let v = (rng.next() % 4) as i64;
        Self::make(v, p, rng)
    }
}
impl Gen for String {
    fn make(v: i64, _: &mut Pool, _: &mut Rng) -> Self {
        padded(v)
    }
    fn random(_: &mut Pool, rng: &mut Rng) -> Self {
        word(rng)
    }
}
impl Gen for &'static str {
    fn make(v: i64, _: &mut Pool, _: &mut Rng) -> Self {
        leak(padded(v))
    }
    fn random(_: &mut Pool, rng: &mut Rng) -> Self {
        leak(word(rng))
    }
}
impl Gen for Vec<u8> {
    fn make(v: i64, _: &mut Pool, _: &mut Rng) -> Self {
        padded(v).into_bytes()
    }
    fn random(_: &mut Pool, rng: &mut Rng) -> Self {
        word(rng).into_bytes()
    }
}
impl Gen for Box<str> {
    fn make(v: i64, _: &mut Pool, _: &mut Rng) -> Self {
        padded(v).into_boxed_str()
    }
    fn random(_: &mut Pool, rng: &mut Rng) -> Self {
        word(rng).into_boxed_str()
    }
}
impl Gen for Arc<str> {
    fn make(v: i64, _: &mut Pool, _: &mut Rng) -> Self {
        Arc::from(padded(v))
    }
    fn random(_: &mut Pool, rng: &mut Rng) -> Self {
        Arc::from(word(rng))
    }
}
impl Gen for Cow<'static, str> {
    fn make(v: i64, _: &mut Pool, _: &mut Rng) -> Self {
        Cow::Owned(padded(v))
    }
    fn random(_: &mut Pool, rng: &mut Rng) -> Self {
        Cow::Borrowed(leak(word(rng)))
    }
}
impl Gen for Duration {
    fn make(v: i64, _: &mut Pool, _: &mut Rng) -> Self {
        Duration::new((v + (1 << 40)) as u64 / 1000, ((v + (1 << 40)) % 1000) as u32 * 1_000_000)
    }
    fn random(_: &mut Pool, rng: &mut Rng) -> Self {
        Duration::new(rng.next() >> 20, (rng.next() % 1_000_000_000) as u32)
    }
}
impl<K: Gen> Gen for Desc<K> {
    fn make(v: i64, p: &mut Pool, r: &mut Rng) -> Self {
        Desc(K::make(-v, p, r))
    }
    fn random(p: &mut Pool, r: &mut Rng) -> Self {
        Desc(K::random(p, r))
    }
}
impl<K: Gen> Gen for std::cmp::Reverse<K> {
    fn make(v: i64, p: &mut Pool, r: &mut Rng) -> Self {
        std::cmp::Reverse(K::make(-v, p, r))
    }
    fn random(p: &mut Pool, r: &mut Rng) -> Self {
        std::cmp::Reverse(K::random(p, r))
    }
}
impl Gen for UserFixed {
    fn make(v: i64, _: &mut Pool, _: &mut Rng) -> Self {
        let u = (v + (1i64 << 40)) as u64;
        UserFixed { major: ((u / 1000) & 0xFFFF) as u16, minor: (u % 1000) as u16 }
    }
    fn random(_: &mut Pool, rng: &mut Rng) -> Self {
        UserFixed { major: rng.next() as u16, minor: rng.next() as u16 }
    }
}
impl Gen for UserBytes {
    fn make(v: i64, _: &mut Pool, _: &mut Rng) -> Self {
        UserBytes { tag: padded(v) }
    }
    fn random(_: &mut Pool, rng: &mut Rng) -> Self {
        UserBytes { tag: word(rng) }
    }
}
// Composite keys: v is split into digits, most significant first, so the
// lexicographic order of the parts follows v.
fn digits<const N: usize>(v: i64) -> [i64; N] {
    let u = v.abs(); // non-negative, so truncating division keeps the order
    let mut out = [0i64; N];
    let mut d = 1i64;
    for i in (0..N).rev() {
        out[i] = (u / d) % 1000;
        d *= 1000;
    }
    out
}
impl<A: Gen, B: Gen> Gen for (A, B) {
    fn make(v: i64, p: &mut Pool, r: &mut Rng) -> Self {
        let d = digits::<2>(v);
        (A::make(d[0], p, r), B::make(d[1], p, r))
    }
    fn random(p: &mut Pool, r: &mut Rng) -> Self {
        (A::random(p, r), B::random(p, r))
    }
}
impl<A: Gen> Gen for (A,) {
    fn make(v: i64, p: &mut Pool, r: &mut Rng) -> Self {
        (A::make(v, p, r),)
    }
    fn random(p: &mut Pool, r: &mut Rng) -> Self {
        (A::random(p, r),)
    }
}
impl<A: Gen, B: Gen, C: Gen> Gen for (A, B, C) {
    fn make(v: i64, p: &mut Pool, r: &mut Rng) -> Self {
        let d = digits::<3>(v);
        (A::make(d[0], p, r), B::make(d[1], p, r), C::make(d[2], p, r))
    }
    fn random(p: &mut Pool, r: &mut Rng) -> Self {
        (A::random(p, r), B::random(p, r), C::random(p, r))
    }
}
impl<K: Gen, const N: usize> Gen for [K; N] {
    fn make(v: i64, p: &mut Pool, r: &mut Rng) -> Self {
        let d = digits::<N>(v);
        core::array::from_fn(|i| K::make(d[i], p, r))
    }
    fn random(p: &mut Pool, r: &mut Rng) -> Self {
        core::array::from_fn(|_| K::random(p, r))
    }
}

#[derive(Clone, Debug)]
pub struct Tagged<K> {
    pub key: K,
    pub id: u32,
}

pub const PATTERNS: [&str; 10] = ["random", "sorted", "reverse", "nearly_sorted", "few_unique", "all_equal", "runs", "organ_pipe", "small_range", "sawtooth"];

pub fn make_input<K: Gen>(pattern: &str, n: usize, seed: u64, pool: &mut Pool) -> Vec<Tagged<K>> {
    let mut rng = Rng::new(seed);
    let mut v = vec![0i64; n];
    match pattern {
        "sorted" => {
            for i in 0..n {
                v[i] = i as i64;
            }
        }
        "reverse" => {
            for i in 0..n {
                v[i] = (n - 1 - i) as i64;
            }
        }
        "nearly_sorted" => {
            for i in 0..n {
                v[i] = i as i64;
            }
            if n > 1 {
                for _ in 0..n / 100 + 1 {
                    let (a, b) = (rng.next() as usize % n, rng.next() as usize % n);
                    v.swap(a, b);
                }
            }
        }
        "few_unique" => {
            for x in v.iter_mut() {
                *x = (rng.next() % 100) as i64;
            }
        }
        "all_equal" => {
            for x in v.iter_mut() {
                *x = 42;
            }
        }
        "runs" => {
            let mut i = 0;
            while i < n {
                let len = 16 + rng.next() as usize % 2000;
                let base = (rng.next() % 100000) as i64;
                let mut k = 0;
                while k < len && i < n {
                    v[i] = base + k as i64;
                    k += 1;
                    i += 1;
                }
            }
        }
        "organ_pipe" => {
            for i in 0..n {
                v[i] = if i < n / 2 { i } else { n - i } as i64;
            }
        }
        "small_range" => {
            for x in v.iter_mut() {
                *x = (rng.next() % 4) as i64;
            }
        }
        "sawtooth" => {
            for i in 0..n {
                v[i] = (i % 1000) as i64;
            }
        }
        _ => {}
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let key = if pattern == "random" { K::random(pool, &mut rng) } else { K::make(v[i], pool, &mut rng) };
        out.push(Tagged { key, id: i as u32 });
    }
    out
}

/// Order by the natural order, stability, permutation.
pub fn verify<K: Natural>(out: &[Tagged<K>], n: usize, ctx: &str) {
    assert_eq!(out.len(), n, "{ctx}: size changed");
    let mut seen = vec![false; n];
    for i in 0..n {
        assert!((out[i].id as usize) < n && !seen[out[i].id as usize], "{ctx}: not a permutation at {i}");
        seen[out[i].id as usize] = true;
        if i > 0 {
            let c = out[i - 1].key.natural_cmp(&out[i].key);
            assert!(c != Ordering::Greater, "{ctx}: not sorted at {i}");
            assert!(!(c == Ordering::Equal && out[i - 1].id > out[i].id), "{ctx}: not stable at {i}");
        }
    }
}

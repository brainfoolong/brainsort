//! The input distributions, generated exactly as the C++ `datasets.hpp`
//! does: a port of `std::mt19937_64` (the standard fixes every constant of
//! the engine, so the same seed gives the same stream on every platform)
//! and of the two integer and real draws the C++ writes out itself, so no
//! implementation-defined standard distribution is involved.
use crate::{DblItem, I64Item, Item, Item2, StrItem};

/// `std::mt19937_64`.
pub struct Mt19937_64 {
    mt: [u64; 312],
    mti: usize,
}
impl Mt19937_64 {
    pub fn new(seed: u64) -> Self {
        let mut mt = [0u64; 312];
        mt[0] = seed;
        for i in 1..312 {
            mt[i] = 6364136223846793005u64.wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 62)).wrapping_add(i as u64);
        }
        Mt19937_64 { mt, mti: 312 }
    }
    pub fn next_u64(&mut self) -> u64 {
        const NN: usize = 312;
        const MM: usize = 156;
        const MATRIX_A: u64 = 0xB502_6F5A_A966_19E9;
        const UM: u64 = 0xFFFF_FFFF_8000_0000;
        const LM: u64 = 0x7FFF_FFFF;
        if self.mti >= NN {
            let mt = &mut self.mt;
            for i in 0..NN - MM {
                let x = (mt[i] & UM) | (mt[i + 1] & LM);
                mt[i] = mt[i + MM] ^ (x >> 1) ^ if x & 1 != 0 { MATRIX_A } else { 0 };
            }
            for i in NN - MM..NN - 1 {
                let x = (mt[i] & UM) | (mt[i + 1] & LM);
                mt[i] = mt[i + MM - NN] ^ (x >> 1) ^ if x & 1 != 0 { MATRIX_A } else { 0 };
            }
            let x = (mt[NN - 1] & UM) | (mt[0] & LM);
            mt[NN - 1] = mt[MM - 1] ^ (x >> 1) ^ if x & 1 != 0 { MATRIX_A } else { 0 };
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= (y >> 29) & 0x5555_5555_5555_5555;
        y ^= (y << 17) & 0x71D6_7FFF_EDA6_0000;
        y ^= (y << 37) & 0xFFF7_EEE0_0000_0000;
        y ^= y >> 43;
        y
    }
}

/// A uniform integer in [lo, hi], by rejection on the masked low bits of
/// the 64-bit draw.
pub fn uniform_u64(rng: &mut Mt19937_64, lo: u64, hi: u64) -> u64 {
    let range = hi.wrapping_sub(lo);
    if range == u64::MAX {
        return rng.next_u64();
    }
    let mut mask = range;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    mask |= mask >> 32;
    loop {
        let r = rng.next_u64() & mask;
        if r <= range {
            return lo.wrapping_add(r);
        }
    }
}
pub fn uniform_i64(rng: &mut Mt19937_64, lo: i64, hi: i64) -> i64 {
    uniform_u64(rng, 0, (hi as u64).wrapping_sub(lo as u64)).wrapping_add(lo as u64) as i64
}
/// A uniform double in [lo, hi): the top 53 bits of the draw scaled to [0, 1).
pub fn uniform_real(rng: &mut Mt19937_64, lo: f64, hi: f64) -> f64 {
    let u = (rng.next_u64() >> 11) as f64 * (1.0f64 / (1u64 << 53) as f64);
    lo + (hi - lo) * u
}

pub const DATASETS: [&str; 12] = ["random", "sorted", "reverse", "nearly_sorted", "few_unique", "all_equal", "runs", "organ_pipe", "small_range", "sawtooth", "prefixed", "sparse_bits"];
pub const TYPES: [&str; 4] = ["int32", "double", "int64", "string"];
pub fn dataset_applies(name: &str, ty: &str) -> bool {
    name != "sparse_bits" || ty == "int32"
}

/// Per-type key generation: a random key, a key from a small integer
/// (monotone, distinct for distinct integers), and a "prefixed" key.
pub trait KeyGen {
    type Key: Clone + PartialOrd;
    fn random(rng: &mut Mt19937_64) -> Self::Key;
    fn from_int(v: i64) -> Self::Key;
    fn prefixed(rng: &mut Mt19937_64) -> Self::Key;
    fn sparse_bits(rng: &mut Mt19937_64) -> Self::Key;
}
fn sparse_bits_i32(rng: &mut Mt19937_64) -> i32 {
    let r = (rng.next_u64() & 0xFF) as u32;
    (((r >> 4) << 28) | (r & 0xF)) as i32
}
impl KeyGen for Item {
    type Key = i32;
    fn random(rng: &mut Mt19937_64) -> i32 {
        rng.next_u64() as u32 as i32
    }
    fn from_int(v: i64) -> i32 {
        v as i32
    }
    fn prefixed(rng: &mut Mt19937_64) -> i32 {
        (0x4000_0000u64 | (rng.next_u64() & 0xF_FFFF)) as i32
    }
    fn sparse_bits(rng: &mut Mt19937_64) -> i32 {
        sparse_bits_i32(rng)
    }
}
impl KeyGen for DblItem {
    type Key = f64;
    fn random(rng: &mut Mt19937_64) -> f64 {
        uniform_real(rng, -1e6, 1e6)
    }
    fn from_int(v: i64) -> f64 {
        v as f64 * 0.25
    }
    fn prefixed(rng: &mut Mt19937_64) -> f64 {
        1e9 + uniform_real(rng, 0.0, 1e6)
    }
    fn sparse_bits(rng: &mut Mt19937_64) -> f64 {
        Self::from_int(sparse_bits_i32(rng) as i64)
    }
}
impl KeyGen for I64Item {
    type Key = i64;
    fn random(rng: &mut Mt19937_64) -> i64 {
        rng.next_u64() as i64
    }
    fn from_int(v: i64) -> i64 {
        v
    }
    fn prefixed(rng: &mut Mt19937_64) -> i64 {
        (0x5A5A_0000_0000_0000u64 | (rng.next_u64() & 0xF_FFFF)) as i64
    }
    fn sparse_bits(rng: &mut Mt19937_64) -> i64 {
        sparse_bits_i32(rng) as i64
    }
}
impl KeyGen for StrItem {
    type Key = Vec<u8>;
    fn random(rng: &mut Mt19937_64) -> Vec<u8> {
        // a lowercase word, 3..12 letters
        let len = 3 + (rng.next_u64() % 10) as usize;
        (0..len).map(|_| b'a' + (rng.next_u64() % 26) as u8).collect()
    }
    fn from_int(v: i64) -> Vec<u8> {
        // zero-padded decimal, monotone in v
        format!("{:012}", (v + (1i64 << 40)) as u64).into_bytes()
    }
    fn prefixed(rng: &mut Mt19937_64) -> Vec<u8> {
        format!("customer-{:08}", (rng.next_u64() % 100_000_000) as u32).into_bytes()
    }
    fn sparse_bits(rng: &mut Mt19937_64) -> Vec<u8> {
        Self::from_int(sparse_bits_i32(rng) as i64)
    }
}

/// A generated input. For string elements the characters live in `pool`.
pub struct Dataset<T> {
    pub items: Vec<T>,
    pub pool: Option<Vec<u8>>,
}

pub fn generate_keys<G: KeyGen>(name: &str, n: usize, seed: u64) -> Vec<G::Key> {
    let mut rng = Mt19937_64::new(seed);
    let mut v: Vec<G::Key> = Vec::with_capacity(n);
    let random_all = |rng: &mut Mt19937_64, v: &mut Vec<G::Key>| {
        for _ in 0..n {
            v.push(G::random(rng));
        }
    };
    let sort = |v: &mut Vec<G::Key>| v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    match name {
        "random" => random_all(&mut rng, &mut v),
        "sorted" => {
            random_all(&mut rng, &mut v);
            sort(&mut v);
        }
        "reverse" => {
            random_all(&mut rng, &mut v);
            sort(&mut v);
            v.reverse();
        }
        "nearly_sorted" => {
            random_all(&mut rng, &mut v);
            sort(&mut v);
            if n > 1 {
                let swaps = (n / 100).max(1);
                for _ in 0..swaps {
                    let i = uniform_u64(&mut rng, 0, n as u64 - 1) as usize;
                    let j = uniform_u64(&mut rng, 0, n as u64 - 1) as usize;
                    v.swap(i, j);
                }
            }
        }
        "few_unique" => {
            let pool: Vec<G::Key> = (0..100).map(|_| G::random(&mut rng)).collect();
            for _ in 0..n {
                v.push(pool[(rng.next_u64() % 100) as usize].clone());
            }
        }
        "all_equal" => {
            for _ in 0..n {
                v.push(G::from_int(42));
            }
        }
        "runs" => {
            v.resize(n, G::from_int(0));
            let mut i = 0;
            while i < n {
                let l = (uniform_u64(&mut rng, 16, 2000) as usize).min(n - i);
                let mut key = uniform_i64(&mut rng, -100000, 100000);
                for k in 0..l {
                    v[i + k] = G::from_int(key);
                    key += uniform_u64(&mut rng, 0, 10) as i64;
                }
                i += l;
            }
        }
        "organ_pipe" => {
            let half = n / 2;
            for i in 0..n {
                v.push(G::from_int(if i < half { i } else { n - i } as i64));
            }
        }
        "small_range" => {
            for _ in 0..n {
                v.push(G::from_int((rng.next_u64() % 4) as i64));
            }
        }
        "sawtooth" => {
            for i in 0..n {
                v.push(G::from_int((i % 1000) as i64));
            }
        }
        "prefixed" => {
            for _ in 0..n {
                v.push(G::prefixed(&mut rng));
            }
        }
        "sparse_bits" => {
            for _ in 0..n {
                v.push(G::sparse_bits(&mut rng));
            }
        }
        other => panic!("unknown dataset: {other}"),
    }
    v
}

pub trait Generate<T> {
    fn build(keys: Vec<<Self as GenKeys>::Key>) -> Dataset<T>
    where
        Self: GenKeys;
}
pub trait GenKeys {
    type Key;
}
impl<T> GenKeys for Dataset<T>
where
    T: KeyGen,
{
    type Key = T::Key;
}
impl Generate<Item> for Dataset<Item> {
    fn build(keys: Vec<i32>) -> Dataset<Item> {
        Dataset { items: keys.into_iter().enumerate().map(|(i, k)| Item { key: k, id: i as u32 }).collect(), pool: None }
    }
}
impl Generate<DblItem> for Dataset<DblItem> {
    fn build(keys: Vec<f64>) -> Dataset<DblItem> {
        Dataset { items: keys.into_iter().enumerate().map(|(i, k)| DblItem { key: k, id: i as u32, pad: 0 }).collect(), pool: None }
    }
}
impl Generate<I64Item> for Dataset<I64Item> {
    fn build(keys: Vec<i64>) -> Dataset<I64Item> {
        Dataset { items: keys.into_iter().enumerate().map(|(i, k)| I64Item { key: k, id: i as u32, pad: 0 }).collect(), pool: None }
    }
}
impl Generate<StrItem> for Dataset<StrItem> {
    fn build(keys: Vec<Vec<u8>>) -> Dataset<StrItem> {
        let total: usize = keys.iter().map(|k| k.len()).sum();
        let mut pool = Vec::with_capacity(total + 1);
        let mut offs = Vec::with_capacity(keys.len());
        for k in &keys {
            offs.push((pool.len(), k.len()));
            pool.extend_from_slice(k);
        }
        pool.push(0);
        let base = pool.as_ptr();
        let items = offs.iter().enumerate().map(|(i, &(o, l))| StrItem { ptr: base.wrapping_add(o), len: l as u32, id: i as u32 }).collect();
        Dataset { items, pool: Some(pool) }
    }
}

/// The dataset `name` of `n` elements from `seed`, for element type `T`.
pub fn generate<T: Item2 + KeyGen>(name: &str, n: usize, seed: u64) -> Dataset<T>
where
    Dataset<T>: Generate<T>,
{
    Dataset::<T>::build(generate_keys::<T>(name, n, seed))
}

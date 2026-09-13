//! The fuzz body shared by the cargo-fuzz target and the smoke test: the
//! input bytes choose a key type and the elements; the result is checked for
//! order, stability and permutation against an independent comparison.
#![allow(dead_code)]
use brainsort::Desc;
use std::cmp::Ordering;

#[derive(Clone, Debug)]
pub struct Tagged<K> {
    pub key: K,
    pub id: u32,
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

pub fn fuzz_one(data: &[u8]) {
    if data.is_empty() {
        return;
    }
    let mode = data[0] % 7;
    let data = &data[1..];
    match mode {
        0 => {
            // int32 keys from 4-byte chunks
            let v: Vec<Tagged<i32>> = data.chunks_exact(4).enumerate().map(|(i, c)| Tagged { key: i32::from_le_bytes(c.try_into().unwrap()), id: i as u32 }).collect();
            run(v, "i32");
        }
        1 => {
            // 8-bit keys: many duplicates
            let v: Vec<Tagged<u8>> = data.iter().enumerate().map(|(i, &b)| Tagged { key: b, id: i as u32 }).collect();
            run(v, "u8");
        }
        2 => {
            // strings: split at zero bytes and at long runs
            let mut v: Vec<Tagged<Vec<u8>>> = Vec::new();
            let mut cur = Vec::new();
            for &b in data {
                if b == 0 || cur.len() >= 40 {
                    v.push(Tagged { key: std::mem::take(&mut cur), id: v.len() as u32 });
                } else {
                    cur.push(b);
                }
            }
            v.push(Tagged { key: cur, id: v.len() as u32 });
            run(v, "bytes");
        }
        3 => {
            // doubles from 8-byte chunks, NaNs included: checked against the documented total order
            let mut v: Vec<Tagged<f64>> = data.chunks_exact(8).enumerate().map(|(i, c)| Tagged { key: f64::from_le_bytes(c.try_into().unwrap()), id: i as u32 }).collect();
            let n = v.len();
            brainsort::sort_by_key(&mut v, |t| t.key);
            check(&v, n, |a, b| brainsort::key::f64_radix(*a).cmp(&brainsort::key::f64_radix(*b)), "f64");
        }
        4 => {
            // composite: (i16, String, Desc<u32>)
            type K = (i16, String, Desc<u32>);
            let mut v: Vec<Tagged<K>> = Vec::new();
            for c in data.chunks_exact(8) {
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
            let v: Vec<Tagged<i64>> = data.iter().enumerate().map(|(i, &b)| Tagged { key: i as i64 * 1000 + if b % 7 == 0 { -(b as i64) * 100000 } else { 0 }, id: i as u32 }).collect();
            run(v, "i64 structured");
        }
        _ => {
            // a comparator on 16-byte elements: the in-place routes and the comparison sort
            let mut v: Vec<Tagged<u64>> = data.chunks_exact(2).enumerate().map(|(i, c)| Tagged { key: u16::from_le_bytes([c[0], c[1]]) as u64 * 3, id: i as u32 }).collect();
            let n = v.len();
            brainsort::sort_by(&mut v, |a, b| a.key.cmp(&b.key));
            check(&v, n, |a, b| a.cmp(b), "comparator");
        }
    }
}

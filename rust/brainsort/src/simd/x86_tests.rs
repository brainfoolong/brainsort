//! Every vector kernel against its scalar twin on the same inputs, in
//! buffers of exactly the size the kernel is told about: an overread or
//! overwrite past a tail is then outside the allocation, which the
//! AddressSanitizer job sees. Miri cannot interpret the intrinsics, so
//! this is the check the vector code gets that the scalar code gets from
//! Miri. Every tail length is covered (n runs over every remainder modulo
//! the vector width), and the results are compared byte for byte where
//! the contract makes them equal.
//!
//! The scout and the prescan may stop early on a rule that is rigorous
//! (the input is unordered for certain) but that the kernel tests at
//! other positions than the scalar loop: there the small sizes, below the
//! first check, are compared exactly, and a large input's early stop is
//! checked against the rule on a prefix.
use super::*;
use crate::algorithm::partition::{partition2_scalar, split2_scalar};
use crate::algorithm::radix_route::split_forward_scalar;
use crate::algorithm::scout::scout_scalar;
use crate::cpu::{have_avx2, have_bmi2};
use crate::memory::DefaultAlloc;
use crate::radix::{PextKeySoft, prefix_scalar};
use crate::record::{Key32, Key64, Rec32, Rec64};
use crate::view::{NoHooks, View};
use alloc::vec::Vec;

type V<T> = View<T, NoHooks, DefaultAlloc>;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// The bytes of a slice, for the byte-for-byte comparisons.
fn bytes<T>(v: &[T]) -> &[u8] {
    // SAFETY: the slice's own memory; every element type here is plain data.
    unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, core::mem::size_of_val(v)) }
}

/// The element types with a vector layout, from a random key: a few
/// distinct values (`small`) or the full range.
trait Gen: Elem {
    fn make(rng: &mut Rng, small: bool) -> Self;
}
fn key_bits(rng: &mut Rng, small: bool) -> u64 {
    if small { rng.below(4) * 1_000_003 } else { rng.next() }
}
impl Gen for Rec32 {
    fn make(rng: &mut Rng, small: bool) -> Self {
        Rec32 { key: key_bits(rng, small) as i32, idx: rng.next() as u32 }
    }
}
impl Gen for Rec64 {
    fn make(rng: &mut Rng, small: bool) -> Self {
        Rec64 { key: key_bits(rng, small) as i64, idx: rng.next() as u32, pad: 0 }
    }
}
impl Gen for Key64 {
    fn make(rng: &mut Rng, small: bool) -> Self {
        Key64 { key: key_bits(rng, small) as i64 }
    }
}
impl Gen for Key32 {
    fn make(rng: &mut Rng, small: bool) -> Self {
        Key32 { key: key_bits(rng, small) as i32 }
    }
}
fn elems<T: Gen>(rng: &mut Rng, n: usize, small: bool) -> Vec<T> {
    (0..n).map(|_| T::make(rng, small)).collect()
}
/// n up to 200 (every tail), then a few sizes past the commit block.
fn sizes() -> Vec<usize> {
    let mut s: Vec<usize> = (1..=200).collect();
    s.extend([1023, 1024, 1025, 1500, 2049, 4096, 4111]);
    s
}

// ---- reverse ------------------------------------------------------------------------------------

fn check_reverse<T: Copy + PartialEq + core::fmt::Debug>(make: impl Fn(u64) -> T) {
    let mut rng = Rng(1);
    for n in (0..=100).chain([1000, 1001, 4097]) {
        let v: Vec<T> = (0..n).map(|_| make(rng.next())).collect();
        let mut a = v.clone();
        let mut b = v.clone();
        // SAFETY: AVX2 was checked by the caller; a holds n elements of 4, 8 or 16 bytes.
        unsafe { reverse_avx2::<T>(a.as_mut_ptr(), n) };
        b.reverse();
        assert_eq!(bytes(&a), bytes(&b), "reverse of {n} elements of {} bytes", core::mem::size_of::<T>());
    }
}
#[test]
fn reverse_matches_scalar() {
    if !have_avx2() {
        return;
    }
    check_reverse(|x| x as u32);
    check_reverse(|x| x);
    check_reverse(|x| (x as u128) << 64 | x.rotate_left(17) as u128);
}

// ---- prefix sums --------------------------------------------------------------------------------

#[test]
fn prefix_sums_match_scalar() {
    if !have_avx2() {
        return;
    }
    let mut rng = Rng(2);
    for n in [16usize, 32, 256, 1024, 8192] {
        let v: Vec<u32> = (0..n).map(|_| rng.next() as u32).collect();
        let mut a = v.clone();
        let mut b = v.clone();
        // SAFETY: AVX2 was checked; a has n entries, n a power of two >= 16.
        unsafe { prefix_avx2_u32(a.as_mut_ptr(), n) };
        prefix_scalar(b.as_mut_ptr(), n);
        assert_eq!(a, b, "prefix sums of {n}");
    }
}

// ---- PEXT ---------------------------------------------------------------------------------------

#[test]
fn pext_matches_soft() {
    if !have_bmi2() {
        return;
    }
    let mut rng = Rng(3);
    for _ in 0..10_000 {
        let (k, m) = (rng.next(), rng.next() & rng.next());
        let hard = <PextKey<u64> as KeyFn<V<Key64>>>::raw(&PextKey { chunk: 0, mask: m }, k);
        let soft = <PextKeySoft<u64> as KeyFn<V<Key64>>>::raw(&PextKeySoft { chunk: 0, mask: m }, k);
        assert_eq!(hard, soft, "pext64 of {k:#x} by {m:#x}");
        let (k, m) = (k as u32, m as u32);
        let hard = <PextKey<u32> as KeyFn<V<Key32>>>::raw(&PextKey { chunk: 0, mask: m }, k);
        let soft = <PextKeySoft<u32> as KeyFn<V<Key32>>>::raw(&PextKeySoft { chunk: 0, mask: m }, k);
        assert_eq!(hard, soft, "pext32 of {k:#x} by {m:#x}");
    }
}

// ---- the forward split --------------------------------------------------------------------------

/// What a split leaves on overflow: some prefix of the input stably
/// partitioned (the kept elements, then the buffered ones, each side in
/// input order) and the rest untouched. The kernel stops at the first
/// element that does not fit, the scalar loop compacts on past it, so the
/// prefix length differs; the state is checked at the shortest prefix the
/// untouched suffix allows.
fn partial_partition<T: Gen>(input: &[T], out: &[T], ge: impl Fn(&T) -> bool) -> bool {
    let n = input.len();
    let mut l = 0;
    while l < n && bytes(&out[n - 1 - l..n - l]) == bytes(&input[n - 1 - l..n - l]) {
        l += 1;
    }
    let s = n - l;
    let mut expect: Vec<T> = input[..s].iter().copied().filter(|e| !ge(e)).collect();
    expect.extend(input[..s].iter().copied().filter(|e| ge(e)));
    bytes(&out[..s]) == bytes(&expect)
}

fn check_split_forward<T: Gen>() {
    let mut rng = Rng(4);
    for n in sizes() {
        for small in [false, true] {
            for cap in [n / 4, n / 2 + 1, n] {
                let v: Vec<T> = elems(&mut rng, n, small);
                let pivot = v[rng.below(n as u64) as usize];
                let mut a = v.clone();
                let mut b = v.clone();
                let mut ba: Vec<T> = alloc::vec![T::default(); cap];
                let mut bb: Vec<T> = alloc::vec![T::default(); cap];
                let (mut ga, mut gb) = (0usize, 0usize);
                let (mut ma, mut mb) = (0u64, 0u64);
                // SAFETY: AVX2 was checked; the views hold n and cap elements.
                let ra = unsafe { split_forward_avx2::<V<T>>(V::new(a.as_mut_ptr(), n), V::new(ba.as_mut_ptr(), cap), n, cap, pivot, &mut ga, Some(&mut ma)) };
                let rb = split_forward_scalar::<V<T>>(V::new(b.as_mut_ptr(), n), V::new(bb.as_mut_ptr(), cap), n, cap, T::radix_key(pivot, 0).to_u64(), 0, &mut gb, Some(&mut mb));
                let what = alloc::format!("split_forward of {n} elements of {} bytes, cap {cap}, small {small}", core::mem::size_of::<T>());
                assert_eq!(ra, rb, "{what}: result");
                if ra {
                    // on overflow the mask is partial (the scalar loop has
                    // seen the element that overflowed, the kernel not) and
                    // only ever ORed under the retry's full mask
                    assert_eq!(ma, mb, "{what}: mask");
                    assert_eq!(ga, gb, "{what}: n_ge");
                    // the kept side and the buffered side; the rest is dead space
                    assert_eq!(bytes(&a[..n - ga]), bytes(&b[..n - gb]), "{what}: kept side");
                    assert_eq!(bytes(&ba[..ga]), bytes(&bb[..gb]), "{what}: buffered side");
                } else {
                    let pk = T::radix_key(pivot, 0);
                    assert!(partial_partition(&v, &a, |e| T::radix_key(*e, 0) >= pk), "{what}: the kernel's partial partition");
                    assert!(partial_partition(&v, &b, |e| T::radix_key(*e, 0) >= pk), "{what}: the scalar partial partition");
                }
            }
        }
    }
}
#[test]
fn split_forward_matches_scalar() {
    if !have_avx2() {
        return;
    }
    check_split_forward::<Rec32>();
    check_split_forward::<Rec64>();
    check_split_forward::<Key64>();
    check_split_forward::<Key32>();
}

// ---- the partition sort's two passes ------------------------------------------------------------

/// Four distinct sampled keys in order, and the thresholds the partition
/// sort derives from them.
fn sampled<T: Gen>(v: &[T]) -> Option<[T::Key; 4]> {
    let mut keys: Vec<T::Key> = v.iter().map(|&e| T::radix_key(e, 0)).collect();
    keys.sort_unstable_by_key(|k| k.to_u64());
    keys.dedup();
    if keys.len() < 4 {
        return None;
    }
    Some([keys[0], keys[1], keys[2], keys[3]])
}
fn check_split2<T: Gen>() {
    let mut rng = Rng(5);
    for n in sizes() {
        for unknown in [false, true] {
            for cap in [n / 4, n / 2 + 1, n] {
                let mut v: Vec<T> = elems(&mut rng, n, true);
                if unknown {
                    let i = rng.below(n as u64) as usize;
                    v[i] = T::make(&mut rng, false);
                }
                let Some(vk) = sampled(&v) else { continue };
                let (t1, t2, t3) = (vk[1], vk[2], vk[3]);
                let mut a = v.clone();
                let mut b = v.clone();
                let mut ba: Vec<T> = alloc::vec![T::default(); cap];
                let mut bb: Vec<T> = alloc::vec![T::default(); cap];
                let (mut xa, mut xb) = (0u64, 0u64);
                let (mut la, mut lb, mut ha, mut hb, mut ga, mut gb) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
                let (mut ka, mut kb) = (true, true);
                // SAFETY: AVX2 was checked; the views hold n and cap elements.
                let ra = unsafe { split2_avx2::<V<T>>(V::new(a.as_mut_ptr(), n), V::new(ba.as_mut_ptr(), cap), n, cap, &vk, t1, t2, t3, &mut xa, &mut la, &mut ha, &mut ga, &mut ka) };
                let rb = split2_scalar::<V<T>>(V::new(b.as_mut_ptr(), n), V::new(bb.as_mut_ptr(), cap), n, cap, 0, &vk, t1, t2, t3, &mut xb, &mut lb, &mut hb, &mut gb, &mut kb);
                let what = alloc::format!("split2 of {n} elements of {} bytes, cap {cap}, unknown {unknown}", core::mem::size_of::<T>());
                assert_eq!(ra, rb, "{what}: result");
                if ra {
                    // on overflow the mask and `known` are partial, as in
                    // the forward split, and unused
                    assert_eq!((xa, ka), (xb, kb), "{what}: mask and known");
                    assert_eq!((la, ha, ga), (lb, hb, gb), "{what}: counts");
                    assert_eq!(bytes(&a[..n - ga]), bytes(&b[..n - gb]), "{what}: kept side");
                    assert_eq!(bytes(&ba[..ga]), bytes(&bb[..gb]), "{what}: buffered side");
                } else {
                    assert!(partial_partition(&v, &a, |e| T::radix_key(*e, 0) >= t2), "{what}: the kernel's partial partition");
                    assert!(partial_partition(&v, &b, |e| T::radix_key(*e, 0) >= t2), "{what}: the scalar partial partition");
                }
            }
        }
    }
}
fn check_partition2<T: Gen>() {
    let mut rng = Rng(6);
    for m in sizes() {
        let v: Vec<T> = elems(&mut rng, m, true);
        let Some(vk) = sampled(&v) else { continue };
        for t in [vk[1], vk[2]] {
            let lower = v.iter().filter(|&&e| T::radix_key(e, 0) < t).count();
            let o0 = rng.below(8) as usize;
            let (o1, end1) = (o0 + lower, o0 + m);
            let mut da: Vec<T> = alloc::vec![T::default(); end1];
            let mut db: Vec<T> = alloc::vec![T::default(); end1];
            let src = v.clone();
            // SAFETY: AVX2 was checked; src holds m elements, dst end1.
            unsafe { partition2_avx2::<T>(src.as_ptr(), m, da.as_mut_ptr(), o0, o1, end1, t) };
            partition2_scalar::<V<T>>(V::new(src.as_ptr() as *mut T, m), m, V::new(db.as_mut_ptr(), end1), o0, o1, end1, 0, t);
            assert_eq!(bytes(&da[o0..]), bytes(&db[o0..]), "partition2 of {m} elements of {} bytes at {o0}", core::mem::size_of::<T>());
        }
    }
}
#[test]
fn split2_matches_scalar() {
    if !have_avx2() {
        return;
    }
    check_split2::<Rec32>();
    check_split2::<Key64>();
    check_split2::<Key32>();
}
#[test]
fn partition2_matches_scalar() {
    if !have_avx2() {
        return;
    }
    check_partition2::<Rec32>();
    check_partition2::<Key64>();
    check_partition2::<Key32>();
}

// ---- the scout ----------------------------------------------------------------------------------

fn same_scout(a: &ScoutResult, b: &ScoutResult, what: &str) {
    assert_eq!((a.mask, a.has_mask, a.committed), (b.mask, b.has_mask, b.committed), "{what}: mask and flags");
    assert_eq!((a.descents, a.ascents), (b.descents, b.ascents), "{what}: counts");
    assert_eq!((a.runs, a.tracking, a.cur_dir), (b.runs, b.tracking, b.cur_dir), "{what}: runs");
    if a.tracking {
        assert_eq!(a.bound[..=a.runs], b.bound[..=b.runs], "{what}: run bounds");
        assert_eq!(a.dir[..a.runs], b.dir[..b.runs], "{what}: run directions");
    }
}
/// Whether the commit rule held on a prefix the kernels test it on (one
/// past a whole number of vectors after the first element, so 1 modulo
/// 8, from the first commit block on): the only way a kernel may stop
/// early.
fn could_commit<T: Gen>(v: &[T]) -> bool {
    (COMMIT_BLOCK + 1..=v.len()).step_by(8).any(|i| {
        let r = scout_scalar::<V<T>>(V::new(v.as_ptr() as *mut T, i), i, 0);
        r.committed || commit_now(&r, i)
    })
}
fn check_scout<T: Gen>() {
    let mut rng = Rng(7);
    for n in sizes() {
        for shape in 0..4 {
            let mut v: Vec<T> = elems(&mut rng, n, shape == 1);
            match shape {
                2 => {
                    // nearly sorted: a sorted array with a few elements moved
                    v.sort_by_key(|&e| T::radix_key(e, 0).to_u64());
                    for _ in 0..n / 64 + 1 {
                        let (i, j) = (rng.below(n as u64) as usize, rng.below(n as u64) as usize);
                        v.swap(i, j);
                    }
                }
                3 => {
                    // a few monotone runs
                    v.sort_by_key(|&e| T::radix_key(e, 0).to_u64());
                    let cut = rng.below(n as u64) as usize;
                    v[cut..].reverse();
                }
                _ => {}
            }
            let what = alloc::format!("scout of {n} elements of {} bytes, shape {shape}", core::mem::size_of::<T>());
            // SAFETY: AVX2 was checked; v holds n >= 1 elements.
            let mut a = unsafe { scout_avx2::<T>(v.as_ptr(), n) };
            a.has_mask = !a.committed; // as the caller sets it
            let b = scout_scalar::<V<T>>(V::new(v.as_mut_ptr(), n), n, 0);
            if !a.committed && !b.committed {
                same_scout(&a, &b, &what);
            } else if a.committed {
                assert!(could_commit(&v), "{what}: the kernel committed where the rule never held");
            }
        }
    }
}
#[test]
fn scout_matches_scalar() {
    if !have_avx2() {
        return;
    }
    check_scout::<Rec32>();
    check_scout::<Rec64>();
    check_scout::<Key64>();
    check_scout::<Key32>();
}

// ---- the prescan --------------------------------------------------------------------------------

/// The natural order of a prescan layout, and whether a value is NaN.
trait Nat: Copy {
    const KIND: Prescan;
    fn cmp_nat(a: Self, b: Self) -> core::cmp::Ordering;
    fn nan(self) -> bool;
    fn from_bits(x: u64) -> Self;
}
macro_rules! nat {
    ($($t:ty => $k:expr, $nan:expr;)*) => {$(
        impl Nat for $t {
            const KIND: Prescan = $k;
            fn cmp_nat(a: Self, b: Self) -> core::cmp::Ordering { a.partial_cmp(&b).unwrap_or(core::cmp::Ordering::Equal) }
            fn nan(self) -> bool { $nan(self) }
            fn from_bits(x: u64) -> Self {
                // SAFETY: eight bytes reinterpreted as one or two plain values.
                unsafe { core::mem::transmute::<u64, [Self; 8 / core::mem::size_of::<Self>()]>(x)[0] }
            }
        }
    )*};
}
nat! {
    i32 => Prescan::I32, |_| false;
    u32 => Prescan::U32, |_| false;
    i64 => Prescan::I64, |_| false;
    u64 => Prescan::U64, |_| false;
    f32 => Prescan::F32, f32::is_nan;
    f64 => Prescan::F64, f64::is_nan;
}
fn prescan_of<T: Nat>(v: &[T]) -> PrescanResult {
    // SAFETY: AVX2 was checked; v holds n >= 2 elements of the layout KIND names.
    unsafe {
        let (p, n) = (v.as_ptr() as *const u8, v.len());
        match T::KIND {
            Prescan::I32 => prescan_avx2::<{ Prescan::I32.code() }>(p, n),
            Prescan::U32 => prescan_avx2::<{ Prescan::U32.code() }>(p, n),
            Prescan::I64 => prescan_avx2::<{ Prescan::I64.code() }>(p, n),
            Prescan::U64 => prescan_avx2::<{ Prescan::U64.code() }>(p, n),
            Prescan::F32 => prescan_avx2::<{ Prescan::F32.code() }>(p, n),
            _ => prescan_avx2::<{ Prescan::F64.code() }>(p, n),
        }
    }
}
fn check_prescan<T: Nat>() {
    let mut rng = Rng(8);
    for n in sizes().into_iter().filter(|&n| n >= 2) {
        for shape in 0..4 {
            let mut v: Vec<T> = (0..n).map(|_| T::from_bits(if shape == 1 { rng.below(4) * 0x0010_0010_0010_0010 } else { rng.next() })).collect();
            if shape >= 2 {
                // a NaN would not sort; the random shapes keep theirs
                for x in v.iter_mut() {
                    if x.nan() {
                        *x = T::from_bits(0);
                    }
                }
                v.sort_by(|&a, &b| T::cmp_nat(a, b));
                if shape == 2 {
                    for _ in 0..n / 64 + 1 {
                        let (i, j) = (rng.below(n as u64) as usize, rng.below(n as u64) as usize);
                        v.swap(i, j);
                    }
                } else {
                    v.reverse();
                }
            }
            let what = alloc::format!("prescan of {n} elements of {} bytes, shape {shape}", core::mem::size_of::<T>());
            let r = prescan_of(&v);
            if v.iter().any(|&x| x.nan()) {
                assert_eq!(r.shape, Shape::Unordered, "{what}: NaN");
                continue;
            }
            // the reference: full counts, and whether the early-stop rule
            // held at any of the kernel's checkpoints
            let (mut desc, mut asc, mut could_stop) = (0usize, 0usize, false);
            for i in 1..n {
                match T::cmp_nat(v[i - 1], v[i]) {
                    core::cmp::Ordering::Greater => desc += 1,
                    core::cmp::Ordering::Less => asc += 1,
                    _ => {}
                }
                could_stop |= i + 1 >= 256 && asc > 0 && desc > ((i + 1) >> 3) + 64;
            }
            match r.shape {
                Shape::Sorted => assert_eq!(desc, 0, "{what}: sorted"),
                Shape::Reversed => assert!(desc > 0 && asc == 0, "{what}: reversed"),
                Shape::NearlySorted => assert!(desc > 0 && asc > 0 && desc <= n / 16 && (r.descents, r.ascents) == (desc, asc), "{what}: nearly sorted, {} {} vs {desc} {asc}", r.descents, r.ascents),
                Shape::Unordered => assert!(desc > n / 16 || could_stop, "{what}: unordered with {desc} descents and {asc} ascents"),
            }
            if r.shape != Shape::Unordered {
                assert_eq!((r.descents, r.ascents), (desc, asc), "{what}: counts");
            }
        }
    }
}
#[test]
fn prescan_matches_reference() {
    if !have_avx2() {
        return;
    }
    check_prescan::<i32>();
    check_prescan::<u32>();
    check_prescan::<i64>();
    check_prescan::<u64>();
    check_prescan::<f32>();
    check_prescan::<f64>();
}

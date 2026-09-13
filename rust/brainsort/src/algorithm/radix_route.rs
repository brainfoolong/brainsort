//! Route 4: the radix route. A strided sample gives the median (split
//! pivot), the sampled varying-bit mask and key range, and whether keys
//! repeat. When the plan needs three passes or more and the data is
//! unordered, the range is split by the pivot into two parts through a
//! buffer of about half the array, and each part is radix sorted with the
//! cheapest plan: the raw key, the contiguous varying-bit range, only the
//! varying bits (PEXT), or key minus a base. A plan made from the sample is
//! verified inside the histogram pass and replaced by the exact plan if a
//! key contradicts it. Few distinct keys take a dictionary radix. Parts
//! beyond the cache are scattered once by their top digit into cache-sized
//! buckets first.
use super::partition::partition_sort_few;
use super::scout::compute_mask;
use super::{INSERTION_MAX, SAMPLE, SPLIT_MIN, Scratch};
use crate::radix::{FullKey, KeyFn, MAX_PASSES, PextKeySoft, ShiftKey, SubKey, live_passes, make_plan, prefix_sums, radix_passes};
#[cfg(all(target_arch = "x86_64", not(brainsort_no_simd)))]
use crate::view::SimdKind;
use crate::view::{AllocError, Arr, Elem, Hooks, RadixKey, Stat, copy_forward, floor_log2, highest_bit, insertion_sort};

/// What a strided sample of the range says: the median (the split pivot),
/// the sampled varying-bit mask and key range, and whether the sampled keys
/// repeat enough for the dictionary radix to be worth a try.
#[derive(Clone, Copy, Debug, Default)]
pub struct Pivot {
    /// Sample size.
    pub n_sample: usize,
    /// Sample keys >= the pivot key.
    pub n_ge: usize,
    /// OR of (sample key XOR key0): bits that vary in the sample.
    pub sample_mask: u64,
    /// Smallest sampled key.
    pub smin: u64,
    /// Largest sampled key.
    pub smax: u64,
    /// Every sampled key is the same.
    pub all_equal: bool,
    /// Few distinct sampled keys and a collision-free bucket hash for them.
    pub dict: bool,
    /// The multiplier of that hash.
    pub dict_mul: u64,
    /// Distinct sampled keys, when counted (0: not counted); `few` holds up
    /// to four, ascending.
    pub n_distinct: usize,
    /// The first four distinct keys.
    pub few: [u64; 4],
    /// How often each of them was sampled.
    pub few_cnt: [usize; 4],
}

// Dictionary radix: for a range with few distinct keys, one hashed bucket
// per key, the buckets laid out in key order. The count and key tables
// together are sized to the count tables a plain radix on the same part
// would use, so the route costs no extra memory: 2048 buckets for 32-bit
// keys (16 KiB), 4096 for 64-bit keys (48 KiB, inside the 64 KiB arena of
// 13-bit digits). Chunked (string) keys stay at 2048 buckets too.
#[inline(always)]
const fn dict_bits<T: Elem>() -> u32 {
    if !T::CHUNKED && <T::Key as RadixKey>::BITS == 64 { 12 } else { 11 }
}
const DICT_MAX_DISTINCT: usize = 128; // dictionary only up to this many distinct sampled keys
#[inline(always)]
const fn dict_buckets<T: Elem>() -> usize {
    1usize << dict_bits::<T>()
}
/// u32 entries of the dictionary tables: count + representative key.
#[inline(always)]
pub const fn dict_entries<T: Elem>() -> usize {
    dict_buckets::<T>() * (1 + <T::Key as RadixKey>::BITS as usize / 32)
}
#[inline(always)]
fn dict_bucket<T: Elem>(k: T::Key, mul: u64) -> u32 {
    (k.to_u64().wrapping_mul(mul) >> (64 - dict_bits::<T>())) as u32
}

/// Deterministic odd multipliers to try (splitmix64 outputs).
const DICT_MULS: [u64; 32] = {
    let mut m = [0u64; 32];
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut i = 0;
    while i < 32 {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        m[i] = (z ^ (z >> 31)) | 1;
        i += 1;
    }
    m
};

/// A multiplier whose bucket hash is injective on the d distinct sampled keys.
fn dict_pick<T: Elem>(keys: &[u64], d: usize) -> Option<u64> {
    for &m in DICT_MULS.iter() {
        let mut seen = [0u64; 64]; // dict_buckets / 64 <= 64
        let mut ok = true;
        for &k in keys.iter().take(d) {
            let h = dict_bucket::<T>(T::Key::from_u64(k), m);
            let w = &mut seen[(h >> 6) as usize];
            let b = 1u64 << (h & 63);
            if *w & b != 0 {
                ok = false;
                break;
            }
            *w |= b;
        }
        if ok {
            return Some(m);
        }
    }
    None
}

/// Pivot for the split: the median radix key of a strided sample.
/// Deterministic (no RNG), O(sample). For chunked keys the sample is taken
/// on the current chunk. If the median key repeats in the sample, the
/// distinct sampled keys are counted and, if there are few, a dictionary
/// hash is prepared for them.
pub fn pick_pivot<V: Arr>(a: V, n: usize, chunk: i32, info: &mut Pivot) -> V::T {
    let mut k = n >> 5;
    if k > SAMPLE {
        k = SAMPLE;
    }
    let mut smp = [V::T::default(); SAMPLE];
    let mut keys = [0u64; SAMPLE];
    let stride = n / k;
    let k0 = V::key(a.get(stride / 2), chunk).to_u64();
    let (mut m, mut lo, mut hi) = (0u64, k0, k0);
    for i in 0..k {
        smp[i] = a.get(i * stride + stride / 2);
        let ki = V::key(smp[i], chunk).to_u64();
        keys[i] = ki;
        m |= ki ^ k0;
        lo = lo.min(ki);
        hi = hi.max(ki);
    }
    info.sample_mask = m;
    info.all_equal = m == 0;
    info.smin = lo;
    info.smax = hi;
    // The median is taken on the keys extracted above, not on the elements
    // with a key-extracting comparator, so the counted path tallies no
    // extra extractions. The pivot is the first sample element with the
    // median key.
    let mut med = [0u64; SAMPLE];
    med[..k].copy_from_slice(&keys[..k]);
    let (_, &mut pk, _) = med[..k].select_nth_unstable(k / 2);
    let mut pi = 0;
    while keys[pi] != pk {
        pi += 1;
    }
    let pivot = smp[pi];
    info.n_sample = k;
    info.n_ge = 0;
    let mut n_eq = 0;
    for &key in keys.iter().take(k) {
        info.n_ge += (key >= pk) as usize;
        n_eq += (key == pk) as usize;
    }
    info.dict = false;
    if n_eq >= 2 && m != 0 {
        // the median repeats: few distinct keys are likely
        keys[..k].sort_unstable();
        let mut d = 0;
        let mut i = 0;
        while i < k {
            let mut j = i + 1;
            while j < k && keys[j] == keys[i] {
                j += 1;
            }
            if d < 4 {
                info.few[d] = keys[i];
                info.few_cnt[d] = j - i;
            }
            keys[d] = keys[i];
            d += 1;
            i = j;
        }
        info.n_distinct = d;
        if d <= DICT_MAX_DISTINCT {
            if let Some(mul) = dict_pick::<V::T>(&keys, d) {
                info.dict = true;
                info.dict_mul = mul;
            }
        }
    }
    pivot
}

/// Stable two-way split of a[0,n) by key >= pivot key. One side is
/// compacted in place, the other goes to buf[0,cap) (forward: the >= side,
/// from the front; backward: the < side, from the back). Both sides are
/// written every iteration and only the matching cursor advances, which
/// keeps the loop free of unpredictable branches; the junk write always
/// lands on the element's own slot or on dead space. If the buffer
/// overflows (the sample misjudged which side is smaller), the buffered
/// elements are copied into the gap they came from - a stable partial
/// partition - and false is returned; the caller then runs the opposite
/// direction, which is guaranteed to fit. On success the buffered side stays
/// in buf and n_ge is set. When `mask` is given, the OR of (key XOR key0)
/// over all elements is accumulated into it.
pub fn split_forward_scalar<V: Arr>(a: V, buf: V, n: usize, cap: usize, pk: u64, chunk: i32, n_ge: &mut usize, mask: Option<&mut u64>) -> bool {
    let pk = <V::T as Elem>::Key::from_u64(pk);
    let k0 = V::key(a.get(0), chunk);
    let mut m = <V::T as Elem>::Key::ZERO;
    let (mut w, mut b, mut i) = (0usize, 0usize, 0usize);
    if V::H::COUNTED {
        while i < n {
            let e = a.get(i);
            let k = V::key(e, chunk);
            m = m | (k ^ k0);
            if k >= pk {
                if b == cap {
                    break;
                }
                buf.set(b, e);
                b += 1;
            } else {
                a.set(w, e);
                w += 1;
            }
            i += 1;
        }
    } else {
        let pa = a.data();
        let pb = buf.data();
        let src = a.data() as *const V::T;
        while i < n {
            // A stretch with enough buffer room for every element of it.
            let end = n.min(i + (cap - b));
            if end == i {
                break;
            }
            while i < end {
                // SAFETY: i < n; w <= i (own slot or dead space); b < cap.
                unsafe {
                    let e = *src.add(i);
                    let k = <V::T as Elem>::radix_key(e, chunk);
                    let g = (k >= pk) as usize;
                    m = m | (k ^ k0);
                    *pa.add(w) = e;
                    *pb.add(b) = e;
                    w += 1 - g;
                    b += g;
                }
                i += 1;
            }
        }
    }
    if let Some(mask) = mask {
        *mask |= m.to_u64();
    }
    if i < n {
        // overflow: fold the buffer back into the gap a[w, i)
        copy_forward(buf, 0, a, w, b);
        return false;
    }
    *n_ge = b; // the >= side lives in buf[0, n_ge)
    true
}
/// The backward split: the < side goes to the buffer, from its back.
pub fn split_backward_scalar<V: Arr>(a: V, buf: V, n: usize, cap: usize, pk: u64, chunk: i32, n_ge: &mut usize, mask: Option<&mut u64>) -> bool {
    let pk = <V::T as Elem>::Key::from_u64(pk);
    let k0 = V::key(a.get(0), chunk);
    let mut m = <V::T as Elem>::Key::ZERO;
    let (mut w, mut b, mut i) = (n, cap, n);
    if V::H::COUNTED {
        while i > 0 {
            let e = a.get(i - 1);
            let k = V::key(e, chunk);
            m = m | (k ^ k0);
            if k >= pk {
                w -= 1;
                a.set(w, e);
            } else {
                if b == 0 {
                    break;
                }
                b -= 1;
                buf.set(b, e);
            }
            i -= 1;
        }
    } else {
        let pa = a.data();
        let pb = buf.data();
        let src = a.data() as *const V::T;
        while i > 0 {
            let begin = if i > b { i - b } else { 0 };
            if begin == i {
                break;
            }
            while i > begin {
                i -= 1;
                // SAFETY: i < n; w-1 >= i (own slot or dead space); b >= 1 throughout the stretch.
                unsafe {
                    let e = *src.add(i);
                    let k = <V::T as Elem>::radix_key(e, chunk);
                    let g = (k >= pk) as usize;
                    m = m | (k ^ k0);
                    *pa.add(w - 1) = e;
                    *pb.add(b - 1) = e;
                    w -= g;
                    b -= 1 - g;
                }
            }
        }
    }
    if let Some(mask) = mask {
        *mask |= m.to_u64();
    }
    if i > 0 {
        // overflow: fold the buffer back into the gap a[i, w)
        copy_forward(buf, b, a, i, cap - b);
        return false;
    }
    *n_ge = n - w; // the < side lives in buf[cap - n_lt, cap)
    true
}
/// The forward split, vectorised where the element type allows it.
#[inline]
pub fn split_forward<V: Arr>(a: V, buf: V, n: usize, cap: usize, pivot: V::T, chunk: i32, n_ge: &mut usize, mask: Option<&mut u64>) -> bool {
    #[cfg(all(target_arch = "x86_64", not(brainsort_no_simd)))]
    {
        if !<V::H as Hooks>::COUNTED && <V::T as Elem>::SIMD != SimdKind::None && chunk == 0 && crate::cpu::have_avx2() {
            // SAFETY: AVX2 was detected; the views hold n and cap elements.
            return unsafe { crate::simd::x86::split_forward_avx2::<V>(a, buf, n, cap, pivot, n_ge, mask) };
        }
    }
    split_forward_scalar(a, buf, n, cap, V::key(pivot, chunk).to_u64(), chunk, n_ge, mask)
}

/// Digit width for `bits` key bits over m elements. The width is capped
/// (13 bits on the split path: two 13-bit count tables are 64 KiB, which
/// stays under the allocator's mmap threshold, so the histogram arena is
/// served from the heap and reused across sorts). Within the cap, the width
/// balances the passes (32 bits -> 3x11, not 13+13+6).
#[inline]
pub fn digit_width(bits: u32, m: usize, max_digit: u32) -> u32 {
    let cap = (floor_log2(m) as u32).clamp(8, max_digit);
    if bits <= cap {
        return bits;
    }
    let passes = (bits + cap - 1) / cap;
    (bits + passes - 1) / passes
}

/// The whole-array (no-split) radix uses up to 16-bit digits.
pub const RADIX_MAX_DIGIT: u32 = 16;
/// The split path caps at 13 bits (64 KiB tables).
pub const SPLIT_MAX_DIGIT: u32 = 13;
/// The split is taken only from this many passes up.
pub const SPLIT_MIN_PASSES: u32 = 3;

/// The kind of key function of a radix plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanKind {
    /// The raw radix key.
    Full,
    /// The contiguous varying-bit range, shifted down.
    Shift,
    /// Only the varying bits, compressed (PEXT).
    Pext,
    /// Key minus a base below the sampled minimum.
    Sub,
}
/// A radix plan: the key function, the number of bits it yields, the digit
/// width, and whether the plan is exact (derived from the true varying-bit
/// mask) or a guess from the sample that the histogram pass must verify
/// against every key.
#[derive(Clone, Copy, Debug)]
pub struct RadixPlan {
    /// The key function.
    pub kind: PlanKind,
    /// Key bits in the key function's domain; 0: nothing varies.
    pub bits: u32,
    /// Digit width.
    pub width: u32,
    /// Shift: bits below `low` are constant.
    pub low: u32,
    /// Pext: the varying bits.
    pub pmask: u64,
    /// Sub: key - base.
    pub base: u64,
    /// A valid plan needs the true XOR mask within this.
    pub assumed: u64,
    /// No verification needed.
    pub exact: bool,
}
impl Default for RadixPlan {
    fn default() -> Self {
        RadixPlan { kind: PlanKind::Full, bits: 0, width: 1, low: 0, pmask: 0, base: 0, assumed: !0, exact: true }
    }
}
impl RadixPlan {
    /// Number of passes.
    #[inline(always)]
    pub fn passes(&self) -> u32 {
        if self.bits > 0 { (self.bits + self.width - 1) / self.width } else { 0 }
    }
}

/// The cheapest plan for a range of m elements whose varying bits are `est`
/// (exact if mask_exact, else the best estimate so far), optionally with a
/// sampled key range. Fewest passes first, then markedly smaller tables,
/// then exact over speculative. The sampled range is widened by itself on
/// both sides, so an unsampled key outside it is rare; if one turns up
/// anyway the histogram pass catches it and the exact plan takes over.
/// `allow_pext`: the CPU has BMI2, or the run is counted.
#[allow(clippy::too_many_arguments)]
pub fn choose_plan<K: RadixKey>(est64: u64, mask_exact: bool, have_range: bool, rmin: u64, rmax: u64, m: usize, max_digit: u32, force_width: u32, allow_pext: bool) -> RadixPlan {
    let kb = K::BITS;
    let est = K::from_u64(est64);
    let mut best = RadixPlan::default();
    if mask_exact && est == K::ZERO {
        return best; // nothing varies
    }
    let mut have = false;
    let mut consider = |mut p: RadixPlan| {
        p.width = if force_width > 0 { force_width } else { digit_width(p.bits, m, max_digit) };
        if !have {
            best = p;
            have = true;
            return;
        }
        if p.passes() != best.passes() {
            if p.passes() < best.passes() {
                best = p;
            }
            return;
        }
        // Smaller tables win a tie only when they are at least 8x smaller;
        // between two cache-resident sizes the extra shift or pext per
        // element costs more than they save.
        if p.width + 3 <= best.width {
            best = p;
            return;
        }
        if best.width + 3 <= p.width {
            return;
        }
        if p.exact != best.exact {
            if p.exact {
                best = p;
            }
        }
    };
    {
        // full: always exact, whatever the mask
        let p = RadixPlan { kind: PlanKind::Full, bits: kb, exact: true, ..RadixPlan::default() };
        consider(p);
    }
    if est != K::ZERO {
        let low = est.trailing_zeros();
        let high = highest_bit(est);
        let bits = high - low + 1;
        let p = RadixPlan { kind: PlanKind::Shift, bits, low, exact: mask_exact, assumed: if bits >= kb { !0u64 } else { ((1u64 << bits) - 1) << low }, ..RadixPlan::default() };
        consider(p);
        if allow_pext {
            let q = RadixPlan { kind: PlanKind::Pext, bits: est.count_ones(), pmask: est.to_u64(), assumed: est.to_u64(), exact: mask_exact, ..RadixPlan::default() };
            consider(q);
        }
    }
    if have_range && rmax > rmin {
        let lo = K::from_u64(rmin);
        let hi = K::from_u64(rmax);
        let r = hi.wsub(lo);
        let base = if lo < r { K::ZERO } else { lo.wsub(r) };
        let top = if hi > K::MAX.wsub(r) { K::MAX } else { hi.wadd(r) };
        let p = RadixPlan { kind: PlanKind::Sub, bits: highest_bit(top.wsub(base)) + 1, base: base.to_u64(), exact: false, ..RadixPlan::default() };
        consider(p);
    }
    best
}

/// Histogram storage handed to the radix: reused across every radix call of
/// one sort so the count tables are allocated once.
#[derive(Clone, Copy)]
pub struct HistStore {
    /// The entries.
    pub p: *mut u32,
}

// ---- MSD levels --------------------------------------------------------------
// A part too large for the cache is scattered once by its top key bits into
// buckets of about the size of the first-level cache, through
// write-combining buffers: one cache line per bucket, written to the array
// whole and aligned with non-temporal stores. Each bucket is then sorted on
// the remaining bits while it sits in cache, by the fused LSD passes (or by
// one more such level if it is still too large).
const WC_LINE: usize = 64; // bytes per bucket buffer: one cache line
const MAX_LEVELS: usize = 10; // 64-bit keys in 8-bit digits, plus one
const MSD_MAX_BITS: u32 = SPLIT_MAX_DIGIT; // widest top digit that is scattered this way

#[inline]
fn msd_min_bytes() -> usize {
    let third = crate::cpu::cache_sizes().l3 / 3;
    third.max(4 << 20)
}
#[inline]
fn bucket_bytes() -> usize {
    crate::cpu::cache_sizes().l1d.clamp(16 << 10, 64 << 10)
}
/// The top digit for a level of `bytes` bytes whose keys vary in `bits` bits.
#[inline]
fn msd_width(bytes: usize, bits: u32) -> u32 {
    let w = (floor_log2(bytes / bucket_bytes()) as u32).clamp(8, MSD_MAX_BITS);
    w.min(bits)
}
/// Element types the write-combining buffers can hold: a whole number per line.
#[inline(always)]
const fn wc_ok<T>() -> bool {
    WC_LINE % core::mem::size_of::<T>() == 0 && core::mem::size_of::<T>() <= WC_LINE / 2
}

/// The count tables of the levels: one per level (the bucket ends are read
/// while the buckets are sorted), two for a bucket's LSD passes.
#[derive(Clone, Copy)]
struct LevelTables {
    base: *mut u32,
    width: usize, // entries per table
}
impl LevelTables {
    #[inline(always)]
    fn at(&self, level: usize) -> *mut u32 {
        self.base.wrapping_add(level * self.width)
    }
    #[inline(always)]
    fn need(width: usize) -> usize {
        (MAX_LEVELS + 2) * width
    }
}

/// Write-combined scatter of s[0,n) into d by digit (key(e) >> shift) & dmask.
/// pos[b] is the start of bucket b on entry and its end on return. A
/// bucket's first flush is cut at the next line boundary, so every later
/// flush writes one whole, aligned line.
///
/// # Safety
/// `s` holds n elements, `d` room for them at the bucket positions; `wc` is
/// a 64-byte aligned buffer of B lines; `fill` and `cap` have B entries.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn wc_scatter<T: Elem, F: Fn(T) -> u64>(s: *const T, d: *mut T, n: usize, key: F, shift: u32, dmask: u32, pos: *mut u32, b_count: usize, wc: *mut u8, fill: *mut u8, cap: *mut u8) {
    let l = WC_LINE / core::mem::size_of::<T>();
    // SAFETY: all indices are below the counts the caller guarantees.
    unsafe {
        for b in 0..b_count {
            *fill.add(b) = 0;
            let off = (d.add(*pos.add(b) as usize) as usize) & (WC_LINE - 1);
            let head = ((WC_LINE - off) & (WC_LINE - 1)) / core::mem::size_of::<T>();
            *cap.add(b) = if head == 0 { l } else { head } as u8;
        }
        for i in 0..n {
            let e = *s.add(i);
            let b = ((key(e) >> shift) as u32 & dmask) as usize;
            let line = wc.add(b * WC_LINE) as *mut T;
            let mut f = *fill.add(b) as usize;
            *line.add(f) = e;
            f += 1;
            if f == *cap.add(b) as usize {
                let out = d.add(*pos.add(b) as usize);
                if f == l {
                    #[cfg(all(target_arch = "x86_64", not(brainsort_no_simd)))]
                    {
                        use core::arch::x86_64::{__m128i, _mm_load_si128, _mm_stream_si128};
                        for k in 0..WC_LINE / 16 {
                            _mm_stream_si128((out as *mut __m128i).add(k), _mm_load_si128((line as *const __m128i).add(k)));
                        }
                    }
                    #[cfg(not(all(target_arch = "x86_64", not(brainsort_no_simd))))]
                    core::ptr::copy_nonoverlapping(line as *const u8, out as *mut u8, WC_LINE);
                } else {
                    core::ptr::copy_nonoverlapping(line as *const u8, out as *mut u8, f * core::mem::size_of::<T>());
                }
                *pos.add(b) += f as u32;
                *cap.add(b) = l as u8;
                f = 0;
            }
            *fill.add(b) = f as u8;
        }
        for b in 0..b_count {
            let f = *fill.add(b) as usize;
            if f != 0 {
                core::ptr::copy_nonoverlapping(wc.add(b * WC_LINE), d.add(*pos.add(b) as usize) as *mut u8, f * core::mem::size_of::<T>());
                *pos.add(b) += f as u32;
            }
        }
        #[cfg(all(target_arch = "x86_64", not(brainsort_no_simd)))]
        core::arch::x86_64::_mm_sfence();
    }
}

/// Scatter s[0,m) into d by one digit, with pos[] the bucket starts (ends
/// on return): write-combined when the level is out of cache, plain
/// otherwise.
///
/// # Safety
/// `pos` has `b_count` entries whose values are valid positions in `d`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn msd_scatter<V: Arr, F: KeyFn<V>>(s: V, d: V, m: usize, key: F, shift: u32, dmask: u32, pos: *mut u32, b_count: usize, scratch: &mut Scratch<V>) -> Result<(), AllocError> {
    if V::H::COUNTED {
        V::H::stat(Stat::RadixPasses, 1);
    }
    if !V::H::COUNTED && wc_ok::<V::T>() && m * core::mem::size_of::<V::T>() >= msd_min_bytes() {
        let raw = scratch.wc(b_count * WC_LINE + WC_LINE + 2 * b_count)?;
        // SAFETY: the arena has room for the aligned lines plus the two byte arrays.
        unsafe {
            let buf = raw.add((WC_LINE - (raw as usize & (WC_LINE - 1))) & (WC_LINE - 1));
            let fill = buf.add(b_count * WC_LINE);
            let cap = fill.add(b_count);
            wc_scatter::<V::T, _>(s.data() as *const V::T, d.data(), m, |e| key.key(e).to_u64(), shift, dmask, pos, b_count, buf, fill, cap);
        }
        return Ok(());
    }
    for i in 0..m {
        let e = s.get(i);
        let b = (key.key(e) >> shift).as_u32() & dmask;
        if V::H::COUNTED {
            V::H::on_table_rw(pos.wrapping_add(b as usize) as *const u8, 4);
        }
        // SAFETY: b < b_count.
        unsafe {
            let slot = pos.add(b as usize);
            d.set(*slot as usize, e);
            *slot += 1;
        }
    }
    Ok(())
}

/// The levels below the top one: sorts one bucket, s[0,m), whose elements
/// agree on every key bit from `bits` up, with d[0,m) as the other buffer
/// and the result in s or d as asked. `mask` is the parent's varying-bit
/// mask below `bits`: exact for the part as a whole, a superset for this
/// bucket. The bucket's histogram pass refines it, so digits that are
/// constant within the bucket are skipped.
///
/// # Safety
/// `tabs` holds `MAX_LEVELS + 2` tables of `tabs.width` entries.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn level_run<V: Arr, F: KeyFn<V>>(s: V, d: V, m: usize, mask: u64, key: F, result_in_s: bool, tabs: LevelTables, level: usize, scratch: &mut Scratch<V>) -> Result<(), AllocError> {
    type K<V> = <<V as Arr>::T as Elem>::Key;
    if m < 2 || mask == 0 {
        if !result_in_s {
            copy_forward(s, 0, d, 0, m);
        }
        return Ok(());
    }
    if m <= INSERTION_MAX {
        insertion_sort(s, 0, m);
        if !result_in_s {
            copy_forward(s, 0, d, 0, m);
        }
        return Ok(());
    }
    let bits = highest_bit(mask) + 1;
    let tab = [tabs.at(level), tabs.at(level + 1)];
    let bytes = m * core::mem::size_of::<V::T>();
    if bytes >= msd_min_bytes() && bits > 8 && level + 1 < MAX_LEVELS {
        // still out of cache: one more level
        let w = msd_width(bytes, bits);
        let shift = bits - w;
        let dmask = (1u32 << w) - 1;
        let b_count = 1usize << w;
        // SAFETY: the tables have tabs.width >= B entries.
        unsafe { core::ptr::write_bytes(tab[0], 0, b_count) };
        if V::H::COUNTED {
            V::H::on_table_sweep(tab[0] as *const u8, b_count, 4, false, true);
        }
        let k0 = key.key(s.get(0));
        let mut xm = K::<V>::ZERO;
        for i in 0..m {
            let u = key.key(s.get(i));
            let dg = (u >> shift).as_u32() & dmask;
            xm = xm | (u ^ k0);
            if V::H::COUNTED {
                V::H::on_table_rw(tab[0].wrapping_add(dg as usize) as *const u8, 4);
            }
            // SAFETY: dg < B.
            unsafe { *tab[0].add(dg as usize) += 1 };
        }
        if V::H::COUNTED {
            V::H::on_table_sweep(tab[0] as *const u8, b_count, 4, true, true);
        }
        prefix_sums(tab[0], b_count);
        // SAFETY: the prefix sums are valid positions in d.
        unsafe { msd_scatter(s, d, m, key, shift, dmask, tab[0], b_count, scratch)? };
        let low = xm.to_u64() & ((1u64 << shift) - 1);
        let mut lo = 0usize;
        for b in 0..b_count {
            if V::H::COUNTED {
                V::H::on_table_read(tab[0].wrapping_add(b) as *const u8, 4);
            }
            // SAFETY: b < B.
            let hi = unsafe { *tab[0].add(b) } as usize;
            if hi > lo {
                // SAFETY: the same tables, one level deeper.
                unsafe { level_run(d.sub(lo, hi - lo), s.sub(lo, hi - lo), hi - lo, low, key, !result_in_s, tabs, level + 1, scratch)? };
            }
            lo = hi;
        }
        return Ok(());
    }
    // In cache: the fused LSD passes. The histogram of the lowest live digit
    // also measures the bucket's exact mask; if that digit turns out
    // constant, the histogram of the next live one is built instead.
    let plan = make_plan(bits, digit_width(bits, m, SPLIT_MAX_DIGIT));
    let mut order = [0usize; MAX_PASSES];
    let mut live = live_passes(&plan, mask, &mut order);
    if live == 0 {
        if !result_in_s {
            copy_forward(s, 0, d, 0, m);
        }
        return Ok(());
    }
    let w = plan.width();
    let mut p0 = order[0];
    let mut shift = plan.shift[p0];
    let mut dmask = (1u32 << plan.bits[p0]) - 1;
    // SAFETY: the tables have tabs.width >= W entries.
    unsafe { core::ptr::write_bytes(tab[0], 0, w) };
    if V::H::COUNTED {
        V::H::on_table_sweep(tab[0] as *const u8, w, 4, false, true);
    }
    let k0 = key.key(s.get(0));
    let mut xm = K::<V>::ZERO;
    for i in 0..m {
        let u = key.key(s.get(i));
        let dg = (u >> shift).as_u32() & dmask;
        xm = xm | (u ^ k0);
        if V::H::COUNTED {
            V::H::on_table_rw(tab[0].wrapping_add(dg as usize) as *const u8, 4);
        }
        // SAFETY: dg < W.
        unsafe { *tab[0].add(dg as usize) += 1 };
    }
    if xm.to_u64() != mask {
        live = live_passes(&plan, xm.to_u64(), &mut order);
        if live == 0 {
            if !result_in_s {
                copy_forward(s, 0, d, 0, m);
            }
            return Ok(());
        }
        if order[0] != p0 {
            p0 = order[0];
            shift = plan.shift[p0];
            dmask = (1u32 << plan.bits[p0]) - 1;
            // SAFETY: as above.
            unsafe { core::ptr::write_bytes(tab[0], 0, w) };
            if V::H::COUNTED {
                V::H::on_table_sweep(tab[0] as *const u8, w, 4, false, true);
            }
            for i in 0..m {
                let dg = (key.key(s.get(i)) >> shift).as_u32() & dmask;
                if V::H::COUNTED {
                    V::H::on_table_rw(tab[0].wrapping_add(dg as usize) as *const u8, 4);
                }
                // SAFETY: dg < W.
                unsafe { *tab[0].add(dg as usize) += 1 };
            }
        }
    }
    if V::H::COUNTED {
        V::H::stat(Stat::PassesSkipped, (plan.passes - live) as u64);
    }
    // SAFETY: two tables of W entries.
    unsafe { radix_passes(s, d, m, &plan, key, &order, live, 0, tab, result_in_s, false) };
    Ok(())
}

/// Radix of src[0,n) with dst[0,n) as the other buffer under plan P, with
/// the result in src or dst as requested. Builds the histogram of the
/// first pass here (the top digit when the part takes an MSD level, the
/// lowest live digit otherwise), verifying a speculative plan on the way:
/// the exact XOR mask of the keys and any key bits above the plan's width
/// are accumulated, and if either shows the sample misjudged the keys the
/// function returns Ok(false) without moving anything (xm_out then holds
/// the exact mask for the retry). With `free` the result may stay in
/// either buffer and `ended_in_src` says which.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn radix_exec<V: Arr, F: KeyFn<V>>(
    src: V,
    dst: V,
    n: usize,
    chunk: i32,
    plan_p: &RadixPlan,
    key: F,
    domain_mask: u64,
    result_in_src: bool,
    scratch: &mut Scratch<V>,
    xm_out: &mut u64,
    free: bool,
    ended_in_src: &mut bool,
) -> Result<bool, AllocError> {
    type K<V> = <<V as Arr>::T as Elem>::Key;
    let kb = K::<V>::BITS;
    let plan = make_plan(plan_p.bits, plan_p.width);
    let mut order = [0usize; MAX_PASSES];
    let live = live_passes(&plan, domain_mask, &mut order);
    if live == 0 {
        if !free && !result_in_src {
            copy_forward(src, 0, dst, 0, n);
        }
        *ended_in_src = free || result_in_src;
        return Ok(true);
    }
    let width = plan.width();
    // The MSD level: the part is out of cache and takes two passes or more.
    // Its digit is the top of the bits that vary (the domain mask is exact
    // for an exact plan; a speculative plan assumes every bit varies).
    let dm = domain_mask & if plan_p.bits >= 64 { !0u64 } else { (1u64 << plan_p.bits) - 1 };
    let bits_eff = if dm != 0 { highest_bit(dm) + 1 } else { plan_p.bits };
    let msd = live >= 2 && n * core::mem::size_of::<V::T>() >= msd_min_bytes() && bits_eff > 8;
    let mut tabs = LevelTables { base: core::ptr::null_mut(), width: 0 };
    let tab: [*mut u32; 2];
    let shift: u32;
    let dmask: u32;
    let mut b_count = 0usize;
    let mut w = width;
    if msd {
        let wd = msd_width(n * core::mem::size_of::<V::T>(), bits_eff);
        shift = bits_eff - wd;
        dmask = (1u32 << wd) - 1;
        b_count = 1usize << wd;
        w = b_count;
        let hs = scratch.hist(LevelTables::need(1usize << MSD_MAX_BITS))?;
        tabs = LevelTables { base: hs.p, width: 1usize << MSD_MAX_BITS };
        tab = [tabs.at(0), tabs.at(1)];
    } else {
        shift = plan.shift[order[0]];
        dmask = (1u32 << plan.bits[order[0]]) - 1;
        let hs = scratch.hist(if live > 1 { 2 * width } else { width })?;
        tab = [hs.p, if live > 1 { hs.p.wrapping_add(width) } else { hs.p }];
    }
    // SAFETY: the arena has W entries.
    unsafe { core::ptr::write_bytes(tab[0], 0, w) };
    if V::H::COUNTED {
        if !msd {
            V::H::stat(Stat::PassesSkipped, (plan.passes - live) as u64);
        }
        V::H::on_table_sweep(tab[0] as *const u8, w, 4, false, true);
    }
    let mut domain_xm = K::<V>::ZERO; // the exact mask in the plan's domain, measured when the buckets need it
    if plan_p.exact && !msd {
        for i in 0..n {
            let dg = (key.key(src.get(i)) >> shift).as_u32() & dmask;
            if V::H::COUNTED {
                V::H::on_table_rw(tab[0].wrapping_add(dg as usize) as *const u8, 4);
            }
            // SAFETY: dg < W.
            unsafe { *tab[0].add(dg as usize) += 1 };
        }
    } else if plan_p.exact {
        let u0 = key.key(src.get(0));
        for i in 0..n {
            let u = key.key(src.get(i));
            let dg = (u >> shift).as_u32() & dmask;
            domain_xm = domain_xm | (u ^ u0);
            if V::H::COUNTED {
                V::H::on_table_rw(tab[0].wrapping_add(dg as usize) as *const u8, 4);
            }
            // SAFETY: dg < W.
            unsafe { *tab[0].add(dg as usize) += 1 };
        }
    } else {
        // The XOR mask check covers shift and pext (bits outside the plan
        // must be constant); sub needs every key within [base, base + 2^bits).
        let k0 = V::key(src.get(0), chunk);
        let u0 = key.raw(k0);
        let ovmask = if plan_p.kind == PlanKind::Sub && plan_p.bits < kb { K::<V>::MAX << plan_p.bits } else { K::<V>::ZERO };
        let (mut xm, mut ov) = (K::<V>::ZERO, K::<V>::ZERO);
        for i in 0..n {
            let e = src.get(i);
            let kr = V::key(e, chunk);
            let u = key.raw(kr);
            xm = xm | (kr ^ k0);
            ov = ov | (u & ovmask);
            domain_xm = domain_xm | (u ^ u0);
            let dg = (u >> shift).as_u32() & dmask;
            if V::H::COUNTED {
                V::H::on_table_rw(tab[0].wrapping_add(dg as usize) as *const u8, 4);
            }
            // SAFETY: dg < W.
            unsafe { *tab[0].add(dg as usize) += 1 };
        }
        *xm_out = xm.to_u64();
        if (xm & !K::<V>::from_u64(plan_p.assumed)) != K::<V>::ZERO || ov != K::<V>::ZERO {
            return Ok(false);
        }
    }
    if !msd {
        // SAFETY: two tables of W entries.
        *ended_in_src = unsafe { radix_passes(src, dst, n, &plan, key, &order, live, 0, tab, result_in_src, free) };
        return Ok(true);
    }
    let low = domain_xm.to_u64() & ((1u64 << shift) - 1);
    if V::H::COUNTED {
        V::H::on_table_sweep(tab[0] as *const u8, b_count, 4, true, true);
    }
    prefix_sums(tab[0], b_count);
    // SAFETY: the prefix sums are valid positions in dst.
    unsafe { msd_scatter(src, dst, n, key, shift, dmask, tab[0], b_count, scratch)? };
    if shift == 0 {
        // the digit covered every varying bit: the scatter sorted the part
        if !free && result_in_src {
            copy_forward(dst, 0, src, 0, n);
        }
        *ended_in_src = !free && result_in_src;
        return Ok(true);
    }
    // The buckets all end in the same buffer: src, whose copies are in cache.
    let rs = free || result_in_src;
    *ended_in_src = rs;
    let mut lo = 0usize;
    for b in 0..b_count {
        if V::H::COUNTED {
            V::H::on_table_read(tab[0].wrapping_add(b) as *const u8, 4);
        }
        // SAFETY: b < B.
        let hi = unsafe { *tab[0].add(b) } as usize;
        if hi > lo {
            // SAFETY: the level tables were allocated with MAX_LEVELS + 2 tables.
            unsafe { level_run(dst.sub(lo, hi - lo), src.sub(lo, hi - lo), hi - lo, low, key, !rs, tabs, 1, scratch)? };
        }
        lo = hi;
    }
    Ok(true)
}

/// Dispatch on the plan's key function. `est` is the varying-bit mask the
/// plan was made from; for an exact plan it tells which passes are trivial.
#[allow(clippy::too_many_arguments)]
pub fn radix_run<V: Arr>(
    src: V,
    dst: V,
    n: usize,
    chunk: i32,
    p: &RadixPlan,
    est: u64,
    result_in_src: bool,
    scratch: &mut Scratch<V>,
    xm_out: &mut u64,
    free: bool,
    ended_in_src: &mut bool,
) -> Result<bool, AllocError> {
    type K<V> = <<V as Arr>::T as Elem>::Key;
    let kb = K::<V>::BITS;
    let all = !0u64;
    match p.kind {
        PlanKind::Shift => radix_exec(src, dst, n, chunk, p, ShiftKey { chunk, low: p.low }, if p.exact { est >> p.low } else { all }, result_in_src, scratch, xm_out, free, ended_in_src),
        PlanKind::Pext => {
            let dm = if p.exact && p.bits < kb { (1u64 << p.bits) - 1 } else { all };
            // The counted path runs the portable PEXT on every CPU so its
            // numbers do not depend on BMI2; the timed path only gets here
            // when the CPU has it.
            #[cfg(all(target_arch = "x86_64", not(brainsort_no_simd)))]
            {
                if !<V::H as Hooks>::COUNTED {
                    // SAFETY: BMI2 was detected before the plan allowed pext.
                    return unsafe { crate::simd::x86::radix_exec_pext(src, dst, n, chunk, p, dm, result_in_src, scratch, xm_out, free, ended_in_src) };
                }
            }
            radix_exec(src, dst, n, chunk, p, PextKeySoft { chunk, mask: K::<V>::from_u64(p.pmask) }, dm, result_in_src, scratch, xm_out, free, ended_in_src)
        }
        PlanKind::Sub => radix_exec(src, dst, n, chunk, p, SubKey { chunk, base: K::<V>::from_u64(p.base) }, all, result_in_src, scratch, xm_out, free, ended_in_src),
        PlanKind::Full => radix_exec(src, dst, n, chunk, p, FullKey { chunk }, if p.exact { est } else { all }, result_in_src, scratch, xm_out, free, ended_in_src),
    }
}

/// Dictionary radix of src[0,n): one hashed bucket per distinct key,
/// verified as it is counted (every element of a bucket must carry the
/// bucket's key), then the occupied buckets are laid out in key order and
/// one scatter pass sorts. Returns Ok(false), with nothing moved, if two
/// distinct keys shared a bucket; xm_out then holds the exact XOR mask for
/// the fallback.
#[allow(clippy::too_many_arguments)]
pub fn dict_sort<V: Arr>(src: V, dst: V, n: usize, chunk: i32, mul: u64, result_in_src: bool, hs: HistStore, xm_out: &mut u64, free: bool, ended_in_src: &mut bool) -> bool {
    type K<V> = <<V as Arr>::T as Elem>::Key;
    let b_count = dict_buckets::<V::T>();
    let cnt = hs.p;
    let rep = hs.p.wrapping_add(b_count) as *mut K<V>;
    // SAFETY: hs has dict_entries entries: B counts then B keys.
    unsafe { core::ptr::write_bytes(cnt, 0, b_count) };
    if V::H::COUNTED {
        V::H::on_table_sweep(cnt as *const u8, b_count, 4, false, true);
    }
    let k0 = V::key(src.get(0), chunk);
    let (mut xm, mut bad) = (K::<V>::ZERO, K::<V>::ZERO);
    for i in 0..n {
        let e = src.get(i);
        let k = V::key(e, chunk);
        let h = dict_bucket::<V::T>(k, mul) as usize;
        if V::H::COUNTED {
            V::H::on_table_rw(cnt.wrapping_add(h) as *const u8, 4);
            V::H::on_table_rw(rep.wrapping_add(h) as *const u8, core::mem::size_of::<K<V>>());
        }
        // SAFETY: h < B.
        unsafe {
            let c = *cnt.add(h);
            xm = xm | (k ^ k0);
            if c == 0 {
                *rep.add(h) = k; // first sight of this bucket: rare, predictable
            } else {
                bad = bad | (*rep.add(h) ^ k);
            }
            *cnt.add(h) = c + 1;
        }
    }
    *xm_out = xm.to_u64();
    if bad != K::<V>::ZERO {
        return false;
    }
    let mut occ = [0u32; 4096];
    let mut nocc = 0usize;
    if V::H::COUNTED {
        V::H::on_table_sweep(cnt as *const u8, b_count, 4, true, false);
    }
    for h in 0..b_count {
        // SAFETY: h < B.
        if unsafe { *cnt.add(h) } != 0 {
            occ[nocc] = h as u32;
            nocc += 1;
        }
    }
    // SAFETY: every occupied bucket has its representative key set.
    occ[..nocc].sort_unstable_by_key(|&x| unsafe { *rep.add(x as usize) });
    let mut sum = 0u32;
    for &h in occ.iter().take(nocc) {
        if V::H::COUNTED {
            V::H::on_table_rw(cnt.wrapping_add(h as usize) as *const u8, 4);
        }
        // SAFETY: h < B.
        unsafe {
            let c = *cnt.add(h as usize);
            *cnt.add(h as usize) = sum;
            sum += c;
        }
    }
    if V::H::COUNTED {
        V::H::stat(Stat::RadixPasses, 1);
    }
    for i in 0..n {
        let e = src.get(i);
        let h = dict_bucket::<V::T>(V::key(e, chunk), mul) as usize;
        if V::H::COUNTED {
            V::H::on_table_rw(cnt.wrapping_add(h) as *const u8, 4);
        }
        // SAFETY: h < B.
        unsafe {
            let slot = cnt.add(h);
            dst.set(*slot as usize, e);
            *slot += 1;
        }
    }
    if !free && result_in_src {
        copy_forward(dst, 0, src, 0, n);
    }
    *ended_in_src = !free && result_in_src;
    true
}

/// Radix sort of one part, src[0,n) with dst[0,n) as the other buffer, the
/// result in src or dst as requested. `mask` is the part's varying-bit
/// mask, exact or estimated; [rmin, rmax] the sampled key range if
/// have_range. The dictionary is tried first when the sample called for it
/// and the plan would otherwise need two passes or more; a speculative plan
/// that fails its verification is replaced by the exact plan (the failed
/// pass measured the exact mask), so no pass is ever wasted on a wrong
/// assumption twice.
#[allow(clippy::too_many_arguments)]
pub fn radix_part<V: Arr>(
    src: V,
    dst: V,
    n: usize,
    chunk: i32,
    scratch: &mut Scratch<V>,
    mut mask: u64,
    mut mask_exact: bool,
    info: &Pivot,
    have_range: bool,
    rmin: u64,
    rmax: u64,
    max_digit: u32,
    result_in_src: bool,
    free: bool,
    digit_bits: u32,
) -> Result<bool, AllocError> {
    type K<V> = <<V as Arr>::T as Elem>::Key;
    let mut ended = true;
    let done = |ended: &mut bool| {
        if !free && !result_in_src {
            copy_forward(src, 0, dst, 0, n);
        }
        *ended = free || result_in_src;
    };
    if n < 2 {
        done(&mut ended);
        return Ok(ended);
    }
    let allow_pext = V::H::COUNTED || crate::cpu::have_bmi2();
    let mut p = choose_plan::<K<V>>(mask, mask_exact, have_range, rmin, rmax, n, max_digit, digit_bits, allow_pext);
    if p.bits == 0 {
        done(&mut ended);
        return Ok(ended);
    }
    let mut xm = 0u64;
    if info.dict && p.passes() >= 2 {
        if V::H::COUNTED {
            V::H::stat(Stat::DictTries, 1);
        }
        let hs = scratch.hist(dict_entries::<V::T>())?;
        if dict_sort(src, dst, n, chunk, info.dict_mul, result_in_src, hs, &mut xm, free, &mut ended) {
            if V::H::COUNTED {
                V::H::stat(Stat::DictHits, 1);
            }
            return Ok(ended);
        }
        mask = xm;
        mask_exact = true;
        p = choose_plan::<K<V>>(mask, true, false, 0, 0, n, max_digit, digit_bits, allow_pext);
        if p.bits == 0 {
            done(&mut ended);
            return Ok(ended);
        }
    }
    let _ = mask_exact;
    if radix_run(src, dst, n, chunk, &p, mask, result_in_src, scratch, &mut xm, free, &mut ended)? {
        return Ok(ended);
    }
    if V::H::COUNTED {
        V::H::stat(Stat::PlanRetries, 1);
    }
    p = choose_plan::<K<V>>(xm, true, false, 0, 0, n, max_digit, digit_bits, allow_pext); // exact: cannot fail
    if p.bits == 0 {
        done(&mut ended);
        return Ok(ended);
    }
    radix_run(src, dst, n, chunk, &p, xm, result_in_src, scratch, &mut xm, free, &mut ended)?;
    Ok(ended)
}

/// Sorts the two sides a split left behind: the buffered side straight out
/// of the buffer, the array side through the buffer. Shared by the median
/// split and the partition sort's fallback.
#[allow(clippy::too_many_arguments)]
pub fn sort_split_parts<V: Arr>(
    a: V,
    buf: V,
    cap: usize,
    n: usize,
    n_ge: usize,
    buf_holds_ge: bool,
    chunk: i32,
    scratch: &mut Scratch<V>,
    mask: u64,
    info: &Pivot,
    have_range: bool,
    lo: u64,
    pk: u64,
    hi: u64,
    digit_bits: u32,
) -> Result<(), AllocError> {
    let n_lt = n - n_ge;
    let lt_max = if pk > 0 { pk - 1 } else { 0 }; // the < side lies in [lo, pk), the >= side in [pk, hi]
    if buf_holds_ge {
        radix_part(buf.sub(0, n_ge), a.sub(n_lt, n_ge), n_ge, chunk, scratch, mask, true, info, have_range, pk, hi, SPLIT_MAX_DIGIT, false, false, digit_bits)?;
        let tmp = scratch.ensure(a, n_lt)?; // grows only if the sample misjudged the sizes
        radix_part(a.sub(0, n_lt), tmp.sub(0, n_lt), n_lt, chunk, scratch, mask, true, info, have_range, lo, lt_max, SPLIT_MAX_DIGIT, true, false, digit_bits)?;
    } else {
        radix_part(buf.sub(cap - n_lt, n_lt), a.sub(0, n_lt), n_lt, chunk, scratch, mask, true, info, have_range, lo, lt_max, SPLIT_MAX_DIGIT, false, false, digit_bits)?;
        let tmp = scratch.ensure(a, n_ge)?;
        radix_part(a.sub(n_lt, n_ge), tmp.sub(0, n_ge), n_ge, chunk, scratch, mask, true, info, have_range, pk, hi, SPLIT_MAX_DIGIT, true, false, digit_bits)?;
    }
    Ok(())
}

/// Route 4 on a[0,n). If the scratch buffer is not yet large enough for
/// the whole range (the top level), the range is split by a sampled pivot
/// with a buffer of about half the size; each part is then radix sorted,
/// the part in the buffer straight out of it. `mask` is the varying-bit
/// mask if `mask_known`, otherwise the bits the scout saw before it stopped
/// early. For chunked keys, chunks shared by every element are skipped
/// first (chunk advances).
#[allow(clippy::too_many_arguments)]
pub fn radix_route<V: Arr>(
    a: V,
    scratch: &mut Scratch<V>,
    n: usize,
    chunk: &mut i32,
    mut mask: u64,
    mut mask_known: bool,
    unordered: bool,
    free: bool,
    ended_in_src: &mut bool,
    digit_bits: u32,
) -> Result<(), AllocError> {
    type K<V> = <<V as Arr>::T as Elem>::Key;
    *ended_in_src = true;
    let mut info = Pivot::default();
    let mut pivot = V::T::default();
    let mut have_pivot = false;
    if n >= SPLIT_MIN {
        pivot = pick_pivot(a, n, *chunk, &mut info);
        have_pivot = true;
    }
    if <V::T as Elem>::CHUNKED {
        // A chunk shared by every element carries no information: move on
        // to the next chunk without scouting again (same elements, same
        // order). The sample tells cheaply whether that is likely; the full
        // pass confirms.
        if !mask_known && (!have_pivot || info.all_equal) {
            mask = compute_mask(a, n, *chunk);
            mask_known = true;
        }
        if mask_known {
            let chunk0 = *chunk;
            while mask == 0 {
                let e0 = a.get(0);
                let _k0 = V::key(e0, *chunk); // counted path: the chunk bytes this test loads
                if <V::T as Elem>::chunk_ends(e0, *chunk) {
                    break;
                }
                *chunk += 1;
                mask = compute_mask(a, n, *chunk);
            }
            if mask == 0 {
                return Ok(());
            }
            if have_pivot && *chunk != chunk0 {
                pivot = pick_pivot(a, n, *chunk, &mut info); // re-sample on the new chunk
            }
        }
    }
    let est = if mask_known { mask } else { mask | info.sample_mask };
    let have_range = have_pivot && !info.all_equal;
    // Two to four distinct sampled keys on 8-byte elements: the partition
    // sort, no counters at all.
    if !<V::T as Elem>::CHUNKED
        && core::mem::size_of::<V::T>() == 8
        && have_pivot
        && info.n_distinct >= 2
        && info.n_distinct <= 4
        && partition_sort_few(a, scratch, n, *chunk, &info, have_range, digit_bits)?
    {
        if V::H::COUNTED {
            V::H::stat(Stat::PartSorts, 1);
        }
        return Ok(());
    }
    // Split only when it pays: the buffer is not already large enough; the
    // radix will run enough passes (planned from the estimate) to amortise
    // the split's extra streaming pass; and the data is substantially
    // unordered (a split interleaves the two halves and so destroys any
    // key locality the input had).
    let want_split = have_pivot
        && scratch.capacity() < n
        && unordered
        && choose_plan::<K<V>>(est, mask_known, have_range, info.smin, info.smax, n, SPLIT_MAX_DIGIT, digit_bits, V::H::COUNTED || crate::cpu::have_bmi2()).passes() >= SPLIT_MIN_PASSES;
    if want_split {
        let cap = n / 2 + n / 32 + 8; // sample error margin plus vector slack; the retry is rare
        let buf = scratch.ensure(a, cap)?;
        let pk = V::key(pivot, *chunk).to_u64();
        let mut n_ge = 0usize;
        let buf_holds_ge;
        if V::H::COUNTED {
            V::H::stat(Stat::Splits, 1);
        }
        let mut local_mask = mask;
        fn mp(mask_known: bool, m: &mut u64) -> Option<&mut u64> {
            if mask_known { None } else { Some(m) }
        }
        {
            if info.n_ge * 16 <= info.n_sample * 9 {
                // >= side not clearly larger: forward (vectorised)
                buf_holds_ge = split_forward(a, buf, n, cap, pivot, *chunk, &mut n_ge, mp(mask_known, &mut local_mask));
                if !buf_holds_ge {
                    if V::H::COUNTED {
                        V::H::stat(Stat::SplitRetries, 1);
                    }
                    split_backward_scalar(a, buf, n, cap, pk, *chunk, &mut n_ge, mp(mask_known, &mut local_mask));
                }
            } else {
                buf_holds_ge = !split_backward_scalar(a, buf, n, cap, pk, *chunk, &mut n_ge, mp(mask_known, &mut local_mask));
                if buf_holds_ge {
                    if V::H::COUNTED {
                        V::H::stat(Stat::SplitRetries, 1);
                    }
                    split_forward(a, buf, n, cap, pivot, *chunk, &mut n_ge, mp(mask_known, &mut local_mask));
                }
            }
        }
        // mask is now exact (the split accumulated it when it was not known)
        sort_split_parts(a, buf, cap, n, n_ge, buf_holds_ge, *chunk, scratch, local_mask, &info, have_range, info.smin, pk, info.smax, digit_bits)?;
        return Ok(());
    }
    let tmp = scratch.ensure(a, n)?;
    *ended_in_src = radix_part(a, tmp, n, *chunk, scratch, est, mask_known, &info, have_range, info.smin, info.smax, RADIX_MAX_DIGIT, true, free, digit_bits)?;
    Ok(())
}

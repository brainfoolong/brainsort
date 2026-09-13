//! The scout pass: one linear pass over the keys that computes, at the same
//! time, the OR of (key XOR key[0]), the number of descents and ascents, and
//! the monotone runs (timsort's definition) with their exact bounds while
//! there are at most four of them. It decides the route.
use super::{COMMIT_BLOCK, MAX_RUNS};
use crate::view::{Arr, Elem, RadixKey};

/// What the scout pass found.
#[derive(Clone, Copy, Debug)]
pub struct ScoutResult {
    /// OR of (key XOR key[0]) over the elements seen.
    pub mask: u64,
    /// `mask` covers every element (chunked keys compute it lazily).
    pub has_mask: bool,
    /// Stopped early: routes 1-3 are provably out, go straight to route 4.
    pub committed: bool,
    /// Descents (a[i] < a[i-1]).
    pub descents: usize,
    /// Ascents (a[i] > a[i-1]).
    pub ascents: usize,
    /// Monotone runs by timsort's definition (non-decreasing, or strictly
    /// decreasing), tracked exactly while there are at most MAX_RUNS of
    /// them: run j is a[bound[j], bound[j+1]) with direction dir[j] (+1
    /// non-decreasing, -1 strictly decreasing). Once a (MAX_RUNS+1)-th run
    /// starts, tracking stops and `runs` stays at MAX_RUNS + 1.
    pub runs: usize,
    /// Run bounds.
    pub bound: [usize; MAX_RUNS + 1],
    /// Run directions.
    pub dir: [i8; MAX_RUNS + 1],
    /// Runs are still being tracked.
    pub tracking: bool,
    /// Direction of the run in progress; 0 while it has one element.
    pub cur_dir: i8,
}
impl Default for ScoutResult {
    fn default() -> Self {
        ScoutResult { mask: 0, has_mask: false, committed: false, descents: 0, ascents: 0, runs: 1, bound: [0; MAX_RUNS + 1], dir: [0; MAX_RUNS + 1], tracking: true, cur_dir: 0 }
    }
}

/// Early commit to route 4 after `scanned` elements. Rigorous, not a guess:
/// routes 1 and 2 need one of the counts to be zero; route 2b needs at most
/// MAX_RUNS runs; route 3 gives up once the displaced elements exceed
/// scanned/8 + 64, and every descent displaces at least one element.
#[inline(always)]
pub fn commit_now(r: &ScoutResult, scanned: usize) -> bool {
    r.ascents > 0 && !r.tracking && r.descents > (scanned >> 3) + 64
}

/// Feeds the compare of element i against element i-1 (c < 0: descent,
/// c > 0: ascent, 0: equal) into the run tracking.
#[inline(always)]
pub fn track_pair(r: &mut ScoutResult, c: i32, i: usize) {
    if r.cur_dir == 0 {
        r.cur_dir = if c < 0 { -1 } else { 1 };
        return;
    }
    if (c < 0) != (r.cur_dir < 0) {
        // the run ends: element i starts the next one
        if r.runs >= MAX_RUNS {
            r.tracking = false;
            r.runs = MAX_RUNS + 1;
            return;
        }
        r.dir[r.runs - 1] = r.cur_dir;
        r.bound[r.runs] = i;
        r.runs += 1;
        r.cur_dir = 0;
    }
}
/// Closes the last run.
#[inline(always)]
pub fn finish_runs(r: &mut ScoutResult, n: usize) {
    if !r.tracking {
        return;
    }
    r.dir[r.runs - 1] = if r.cur_dir < 0 { -1 } else { 1 };
    r.bound[r.runs] = n;
}

/// Scalar scout pass (also the counted path: every read and compare
/// tallied). One three-way compare per element, starting at the
/// shared-prefix offset.
pub fn scout_scalar<V: Arr>(a: V, n: usize, chunk: i32) -> ScoutResult {
    let mut r = ScoutResult::default();
    let mut prev = a.get(0);
    let k0 = V::key(prev, chunk).to_u64();
    // The counts live in locals (the tracker takes r by reference, which
    // would otherwise keep every increment in memory) and are written back
    // at each commit check and at the end.
    let (mut desc, mut asc) = (0usize, 0usize);
    let mut mask = 0u64;
    for i in 1..n {
        let cur = a.get(i);
        if !<V::T as Elem>::CHUNKED {
            mask |= V::key(cur, chunk).to_u64() ^ k0; // cheap for fixed keys
        }
        let c = a.compare_from(cur, prev, chunk);
        // Only a pair that could end the current run (or start one) goes to
        // the tracker: on monotone data this test is all the tracking costs.
        // Equal pairs take one branch and nothing else, as they always did.
        if c != 0 {
            desc += (c < 0) as usize;
            asc += (c > 0) as usize;
            if r.tracking && (r.cur_dir == 0 || (c < 0) != (r.cur_dir < 0)) {
                track_pair(&mut r, c, i);
            }
        } else if r.tracking && r.cur_dir <= 0 {
            track_pair(&mut r, 0, i);
        }
        prev = cur;
        if i & (COMMIT_BLOCK - 1) == 0 {
            r.descents = desc;
            r.ascents = asc;
            if commit_now(&r, i) {
                r.mask = mask;
                r.committed = true;
                return r;
            }
        }
    }
    r.descents = desc;
    r.ascents = asc;
    r.mask = mask;
    r.has_mask = !<V::T as Elem>::CHUNKED;
    finish_runs(&mut r, n);
    r
}

/// The varying-bit mask alone, for chunked keys that reach the radix route.
pub fn compute_mask<V: Arr>(a: V, n: usize, chunk: i32) -> u64 {
    let k0 = V::key(a.get(0), chunk).to_u64();
    let mut mask = 0u64;
    for i in 1..n {
        mask |= V::key(a.get(i), chunk).to_u64() ^ k0;
    }
    mask
}

/// Scalar tail shared by the vector scouts.
#[inline]
pub fn scout_tail<T: Elem>(r: &mut ScoutResult, p: *const T, mut i: usize, n: usize) {
    // SAFETY: the caller passes an array of n elements.
    let k0 = T::radix_key(unsafe { *p }, 0).to_u64();
    while i < n {
        // SAFETY: i < n.
        let (cur, prev) = unsafe { (*p.add(i), *p.add(i - 1)) };
        r.mask |= T::radix_key(cur, 0).to_u64() ^ k0;
        let desc = T::less(cur, prev);
        let asc = T::less(prev, cur);
        r.descents += desc as usize;
        r.ascents += asc as usize;
        if r.tracking && (r.cur_dir == 0 || desc != (r.cur_dir < 0)) {
            track_pair(
                r,
                if desc {
                    -1
                } else if asc {
                    1
                } else {
                    0
                },
                i,
            );
        }
        i += 1;
    }
}

/// The scout pass with the vector kernel where the element type allows it.
#[inline]
pub fn scout<V: Arr>(a: V, n: usize, chunk: i32) -> ScoutResult {
    #[cfg(all(target_arch = "x86_64", not(brainsort_no_simd)))]
    {
        use crate::view::{Hooks, SimdKind};
        if !<V::H as Hooks>::COUNTED && <V::T as Elem>::SIMD != SimdKind::None && chunk == 0 && crate::cpu::have_avx2() {
            // SAFETY: AVX2 was detected; the view holds n elements.
            let mut r = unsafe { crate::simd::x86::scout_avx2::<V::T>(a.data() as *const V::T, n) };
            r.has_mask = !r.committed;
            return r;
        }
    }
    scout_scalar(a, n, chunk)
}

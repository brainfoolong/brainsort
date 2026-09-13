//! Routes 1 and 2: sorted input, and non-increasing input reversed in
//! place with equal keys kept in their original order.
use super::scout::ScoutResult;
use crate::view::{Arr, reverse_range};

/// Reverses `a[0, n)` in place.
#[inline]
pub fn reverse_all<V: Arr>(a: V, n: usize) {
    #[cfg(all(target_arch = "x86_64", not(brainsort_no_simd)))]
    {
        use crate::view::Hooks;
        let sz = core::mem::size_of::<V::T>();
        if !<V::H as Hooks>::COUNTED && (sz == 4 || sz == 8 || sz == 16) && crate::cpu::have_avx2() {
            // SAFETY: AVX2 was detected; the view holds n elements of 4, 8 or 16 bytes.
            unsafe { crate::simd::x86::reverse_avx2::<V::T>(a.data(), n) };
            return;
        }
    }
    reverse_range(a, 0, n);
}

/// Route 2: reverses a non-increasing range while keeping equal keys in
/// their original order. With the scout's run bounds the tie groups are
/// known without another pass: in a non-increasing sequence a strictly
/// decreasing run can only end at a tie, and a non-decreasing run consists
/// of equal keys. Each group is reversed back after the whole range was
/// reversed. Without the bounds (more than MAX_RUNS runs) the groups are
/// found by a scan.
pub fn stable_reverse<V: Arr>(a: V, n: usize, s: &ScoutResult) {
    reverse_all(a, n);
    if s.descents == n - 1 {
        return; // strictly decreasing: no ties
    }
    if s.tracking {
        for j in 0..s.runs {
            let (lo, hi);
            if s.dir[j] > 0 {
                // equal keys, plus the tie that ended a decreasing run before them
                lo = if j > 0 && s.dir[j - 1] < 0 { s.bound[j] - 1 } else { s.bound[j] };
                hi = s.bound[j + 1];
            } else if j + 1 < s.runs && s.dir[j + 1] < 0 {
                // two decreasing runs meet at a tie pair
                lo = s.bound[j + 1] - 1;
                hi = lo + 2;
            } else {
                continue;
            }
            if hi - lo > 1 {
                reverse_range(a, n - hi, n - lo); // the group's image after the reversal
            }
        }
        return;
    }
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        let first = a.get(i);
        while j < n {
            let cur = a.get(j);
            if a.less(first, cur) || a.less(cur, first) {
                break;
            }
            j += 1;
        }
        if j - i > 1 {
            reverse_range(a, i, j);
        }
        i = j;
    }
}

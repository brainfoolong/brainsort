//! Route 2b: a handful of monotone runs, merged pairwise from the shorter
//! side with a buffer of at most n/2.
use super::Scratch;
use super::scout::ScoutResult;
use crate::view::{AllocError, Arr, copy_forward, reverse_range};

/// Stable in-place merge of a[lo,mid) and a[mid,hi) with a buffer of at
/// least min(mid-lo, hi-mid) elements: the shorter run is copied out and the
/// merge runs towards the free space, so the output never overtakes unread
/// input. The merge branches on the compare on purpose: the runs this route
/// sees are structured (an organ pipe, a few concatenated runs), so the
/// branch predicts well and the loop runs ahead of the compare latency; a
/// branch-free select measured 2.5x slower here.
pub fn merge_with_buffer<V: Arr>(a: V, buf: V, lo: usize, mid: usize, hi: usize) {
    let (len1, len2) = (mid - lo, hi - mid);
    if len1 == 0 || len2 == 0 {
        return;
    }
    if len1 <= len2 {
        // left run out, merge forward
        copy_forward(a, lo, buf, 0, len1);
        let (mut i, mut j, mut o) = (0usize, mid, lo);
        let (mut x, mut y) = (buf.get(0), a.get(j));
        loop {
            if a.less(y, x) {
                a.set(o, y);
                o += 1;
                j += 1;
                if j == hi {
                    break;
                }
                y = a.get(j);
            } else {
                a.set(o, x);
                o += 1;
                i += 1;
                if i == len1 {
                    break;
                }
                x = buf.get(i);
            }
        }
        while i < len1 {
            a.set(o, buf.get(i));
            o += 1;
            i += 1;
        }
    } else {
        // right run out, merge backward (ties: right stays right)
        copy_forward(a, mid, buf, 0, len2);
        let (mut i, mut j, mut o) = (mid, len2, hi);
        let (mut x, mut y) = (a.get(i - 1), buf.get(j - 1));
        loop {
            if a.less(y, x) {
                o -= 1;
                a.set(o, x);
                i -= 1;
                if i == lo {
                    break;
                }
                x = a.get(i - 1);
            } else {
                o -= 1;
                a.set(o, y);
                j -= 1;
                if j == 0 {
                    break;
                }
                y = buf.get(j - 1);
            }
        }
        while j > 0 {
            o -= 1;
            j -= 1;
            a.set(o, buf.get(j));
        }
    }
}

/// Route 2b: a handful of monotone runs, with the bounds the scout recorded
/// (timsort's run definition: ascending = non-decreasing, descending =
/// strictly decreasing, so reversing a descending run in place keeps
/// stability). No second detection pass. The buffer (n/2, the most any
/// merge needs) is allocated first. Merges go pairwise, from the shorter
/// side.
pub fn sort_few_runs<V: Arr>(a: V, scratch: &mut Scratch<V>, n: usize, s: &ScoutResult) -> Result<(), AllocError> {
    let buf = scratch.ensure(a, n / 2 + 1)?;
    let bounds = &s.bound;
    let runs = s.runs;
    for j in 0..runs {
        if s.dir[j] < 0 {
            reverse_range(a, bounds[j], bounds[j + 1]);
        }
    }
    // Merge plan: (0,1), (2,3), then the two halves.
    if runs >= 2 {
        merge_with_buffer(a, buf, bounds[0], bounds[1], bounds[2]);
    }
    if runs == 3 {
        merge_with_buffer(a, buf, bounds[0], bounds[2], bounds[3]);
    }
    if runs == 4 {
        merge_with_buffer(a, buf, bounds[2], bounds[3], bounds[4]);
        merge_with_buffer(a, buf, bounds[0], bounds[2], bounds[4]);
    }
    Ok(())
}

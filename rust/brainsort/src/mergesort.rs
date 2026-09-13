//! Classic top-down merge sort with a full-size scratch buffer: the fallback
//! for ranges of 2^32 elements or more (positions inside the main algorithm
//! are 32-bit). The input is copied to the scratch buffer once and the
//! recursion then alternates between the two buffers so no per-level
//! copy-back is needed. Leaves of <= 16 elements are insertion sorted.
//! O(n log n) always, stable, n extra elements of memory.
use crate::view::{AllocError, Arr, AuxBuffer, DepthScope, copy_forward, insertion_sort};

const LEAF: usize = 16;

/// Merges sorted src[lo,mid) and src[mid,hi) into dst[lo,hi). Stable: on
/// ties the left element wins.
fn merge<V: Arr>(src: V, dst: V, lo: usize, mid: usize, hi: usize) {
    let (mut i, mut j, mut k) = (lo, mid, lo);
    if i < mid && j < hi {
        let (mut x, mut y) = (src.get(i), src.get(j));
        loop {
            if src.less(y, x) {
                dst.set(k, y);
                k += 1;
                j += 1;
                if j == hi {
                    break;
                }
                y = src.get(j);
            } else {
                dst.set(k, x);
                k += 1;
                i += 1;
                if i == mid {
                    break;
                }
                x = src.get(i);
            }
        }
    }
    while i < mid {
        dst.set(k, src.get(i));
        k += 1;
        i += 1;
    }
    while j < hi {
        dst.set(k, src.get(j));
        k += 1;
        j += 1;
    }
}

/// Precondition: src[lo,hi) and dst[lo,hi) hold identical contents.
/// Postcondition: dst[lo,hi) is sorted.
fn split_merge<V: Arr>(src: V, dst: V, lo: usize, hi: usize) {
    let _depth = DepthScope::<V::H>::new();
    if hi - lo <= LEAF {
        insertion_sort(dst, lo, hi);
        return;
    }
    let mid = lo + (hi - lo) / 2;
    split_merge(dst, src, lo, mid); // sorted halves end up in src
    split_merge(dst, src, mid, hi);
    merge(src, dst, lo, mid, hi);
}

/// Stable merge sort of the whole view.
pub fn merge_sort<V: Arr>(a: V) -> Result<(), AllocError> {
    let n = a.size();
    if n < 2 {
        return Ok(());
    }
    let buf = AuxBuffer::new(a, n)?;
    let b = buf.arr();
    copy_forward(a, 0, b, 0, n);
    split_merge(b, a, 0, n);
    Ok(())
}

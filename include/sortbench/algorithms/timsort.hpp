// TimSort (Tim Peters, 2002). The default sort of CPython's list.sort(),
// Java's Arrays.sort(Object[]) / Collections.sort, Android, V8's
// Array.prototype.sort and (in a modified form) Rust's stable slice::sort.
//
// This is a faithful port of the CPython / Java (java.util.TimSort) design:
//   * natural runs are detected; strictly descending runs are reversed,
//   * runs shorter than minrun (32..64) are extended with binary insertion sort,
//   * runs are pushed on a stack and merged whenever the length invariants
//     (with the 2015 de Gouw et al. fix) are violated, keeping the stack
//     logarithmic and merges balanced,
//   * merges first gallop to trim the parts of the runs that are already in
//     place, then merge with an adaptive galloping mode (MIN_GALLOP = 7).
// Stable, O(n log n) worst case, O(n) on presorted data, at most n/2 extra
// elements of memory.
#pragma once
#include "sortbench/core.hpp"

#include <algorithm>
#include <cstddef>

namespace sb {

namespace tim_detail {

using idx = std::ptrdiff_t;

constexpr idx MIN_MERGE      = 32;
constexpr idx MIN_GALLOP     = 7;
constexpr idx INITIAL_TMP    = 256;
constexpr idx MAX_PENDING    = 85;   // enough for any array that fits in memory

// minrun: choose so that n / minrun is a power of two or slightly below it.
inline idx compute_minrun(idx n) {
    idx r = 0;
    while (n >= MIN_MERGE) {
        r |= (n & 1);
        n >>= 1;
    }
    return n + r;
}

// Sort a[lo,hi) by binary insertion, where a[lo,start) is already sorted.
template <class A>
inline void binary_insertion_sort(A a, idx lo, idx hi, idx start) {
    if (start == lo) ++start;
    for (; start < hi; ++start) {
        const typename A::value_type pivot = a.get(start);
        idx left  = lo;
        idx right = start;
        while (left < right) {
            const idx mid = left + ((right - left) >> 1);
            if (a.less(pivot, a.get(mid))) right = mid;
            else                           left = mid + 1;
        }
        for (idx k = start; k > left; --k) a.set(k, a.get(k - 1));
        a.set(left, pivot);
    }
}

// Returns the length of the run starting at lo, and reverses it in place if it
// was strictly descending (strictness preserves stability).
template <class A>
inline idx count_run_and_make_ascending(A a, idx lo, idx hi) {
    idx run_hi = lo + 1;
    if (run_hi == hi) return 1;
    if (a.less(a.get(run_hi), a.get(lo))) {
        while (run_hi < hi && a.less(a.get(run_hi), a.get(run_hi - 1))) ++run_hi;
        reverse_range(a, static_cast<size_t>(lo), static_cast<size_t>(run_hi));
    } else {
        while (run_hi < hi && !a.less(a.get(run_hi), a.get(run_hi - 1))) ++run_hi;
    }
    return run_hi - lo;
}

// Locate the leftmost insertion point of key in sorted a[base, base+len):
// returns k with a[base+k-1] < key <= a[base+k]. hint is where to start
// searching; the result is found in O(log |k - hint|) comparisons.
template <class A>
inline idx gallop_left(A a, typename A::value_type key, idx base, idx len, idx hint) {
    idx last_ofs = 0;
    idx ofs      = 1;
    if (a.less(a.get(base + hint), key)) {
        // a[hint] < key: gallop right until a[hint+last_ofs] < key <= a[hint+ofs]
        const idx max_ofs = len - hint;
        while (ofs < max_ofs) {
            if (a.less(a.get(base + hint + ofs), key)) {
                last_ofs = ofs;
                ofs = (ofs << 1) + 1;
                if (ofs <= 0) ofs = max_ofs;  // overflow guard
            } else {
                break;
            }
        }
        if (ofs > max_ofs) ofs = max_ofs;
        last_ofs += hint;
        ofs += hint;
    } else {
        // key <= a[hint]: gallop left until a[hint-ofs] < key <= a[hint-last_ofs]
        const idx max_ofs = hint + 1;
        while (ofs < max_ofs) {
            if (a.less(a.get(base + hint - ofs), key)) break;
            last_ofs = ofs;
            ofs = (ofs << 1) + 1;
            if (ofs <= 0) ofs = max_ofs;
        }
        if (ofs > max_ofs) ofs = max_ofs;
        const idx tmp = last_ofs;
        last_ofs = hint - ofs;
        ofs      = hint - tmp;
    }
    // Now a[base+last_ofs] < key <= a[base+ofs]; binary search the rest.
    ++last_ofs;
    while (last_ofs < ofs) {
        const idx m = last_ofs + ((ofs - last_ofs) >> 1);
        if (a.less(a.get(base + m), key)) last_ofs = m + 1;
        else                              ofs = m;
    }
    return ofs;
}

// Like gallop_left but returns the rightmost insertion point:
// k with a[base+k-1] <= key < a[base+k].
template <class A>
inline idx gallop_right(A a, typename A::value_type key, idx base, idx len, idx hint) {
    idx ofs      = 1;
    idx last_ofs = 0;
    if (a.less(key, a.get(base + hint))) {
        // key < a[hint]: gallop left until a[hint-ofs] <= key < a[hint-last_ofs]
        const idx max_ofs = hint + 1;
        while (ofs < max_ofs) {
            if (a.less(key, a.get(base + hint - ofs))) {
                last_ofs = ofs;
                ofs = (ofs << 1) + 1;
                if (ofs <= 0) ofs = max_ofs;
            } else {
                break;
            }
        }
        if (ofs > max_ofs) ofs = max_ofs;
        const idx tmp = last_ofs;
        last_ofs = hint - ofs;
        ofs      = hint - tmp;
    } else {
        // a[hint] <= key: gallop right until a[hint+last_ofs] <= key < a[hint+ofs]
        const idx max_ofs = len - hint;
        while (ofs < max_ofs) {
            if (a.less(key, a.get(base + hint + ofs))) break;
            last_ofs = ofs;
            ofs = (ofs << 1) + 1;
            if (ofs <= 0) ofs = max_ofs;
        }
        if (ofs > max_ofs) ofs = max_ofs;
        last_ofs += hint;
        ofs += hint;
    }
    ++last_ofs;
    while (last_ofs < ofs) {
        const idx m = last_ofs + ((ofs - last_ofs) >> 1);
        if (a.less(key, a.get(base + m))) ofs = m;
        else                              last_ofs = m + 1;
    }
    return ofs;
}

template <class A>
class Sorter {
public:
    explicit Sorter(A a)
        : a_(a),
          tmp_(static_cast<size_t>(initial_tmp_len(static_cast<idx>(a.size())))) {}

    void sort() {
        const idx n = static_cast<idx>(a_.size());
        if (n < 2) return;

        if (n < MIN_MERGE) {
            const idx run = count_run_and_make_ascending(a_, 0, n);
            binary_insertion_sort(a_, 0, n, run);
            return;
        }

        const idx minrun   = compute_minrun(n);
        idx       lo        = 0;
        idx       remaining = n;
        do {
            idx run_len = count_run_and_make_ascending(a_, lo, n);
            if (run_len < minrun) {
                const idx force = std::min(minrun, remaining);
                binary_insertion_sort(a_, lo, lo + force, lo + run_len);
                run_len = force;
            }
            push_run(lo, run_len);
            merge_collapse();
            lo += run_len;
            remaining -= run_len;
        } while (remaining != 0);

        merge_force_collapse();
    }

private:
    struct Run { idx base, len; };

    A                      a_;
    AuxBuffer<typename A::value_type, A::counted> tmp_;
    idx                    min_gallop_ = MIN_GALLOP;
    Run                    runs_[MAX_PENDING];
    idx                    n_runs_ = 0;

    static idx initial_tmp_len(idx n) {
        // Same policy as java.util.TimSort.
        return n < 2 * INITIAL_TMP ? n >> 1 : INITIAL_TMP;
    }

    A ensure_tmp(idx need) {
        if (static_cast<idx>(tmp_.size()) < need) {
            idx new_size = need;
            new_size |= new_size >> 1;
            new_size |= new_size >> 2;
            new_size |= new_size >> 4;
            new_size |= new_size >> 8;
            new_size |= new_size >> 16;
            if constexpr (sizeof(idx) > 4) new_size |= new_size >> 32;
            ++new_size;
            new_size = std::min(new_size, static_cast<idx>(a_.size()) >> 1);
            tmp_.resize_discard(static_cast<size_t>(new_size));
        }
        return tmp_.arr();
    }

    void push_run(idx base, idx len) {
        assert(n_runs_ < MAX_PENDING);
        runs_[n_runs_].base = base;
        runs_[n_runs_].len  = len;
        ++n_runs_;
        if constexpr (A::counted) { if (static_cast<uint32_t>(n_runs_) > g_stats.max_depth) g_stats.max_depth = static_cast<uint32_t>(n_runs_); }
    }

    // Restore the stack invariants:  len[i-2] > len[i-1] + len[i]  and
    // len[i-1] > len[i], for all i (checked for the top runs, including the
    // 2015 fix that also inspects one run deeper).
    void merge_collapse() {
        while (n_runs_ > 1) {
            idx n = n_runs_ - 2;
            if ((n > 0 && runs_[n - 1].len <= runs_[n].len + runs_[n + 1].len) ||
                (n > 1 && runs_[n - 2].len <= runs_[n - 1].len + runs_[n].len)) {
                if (runs_[n - 1].len < runs_[n + 1].len) --n;
                merge_at(n);
            } else if (runs_[n].len <= runs_[n + 1].len) {
                merge_at(n);
            } else {
                break;
            }
        }
    }

    void merge_force_collapse() {
        while (n_runs_ > 1) {
            idx n = n_runs_ - 2;
            if (n > 0 && runs_[n - 1].len < runs_[n + 1].len) --n;
            merge_at(n);
        }
    }

    void merge_at(idx i) {
        idx base1 = runs_[i].base;
        idx len1  = runs_[i].len;
        idx base2 = runs_[i + 1].base;
        idx len2  = runs_[i + 1].len;

        runs_[i].len = len1 + len2;
        if (i == n_runs_ - 3) runs_[i + 1] = runs_[i + 2];
        --n_runs_;

        // Where does the first element of run2 go in run1? Elements of run1
        // before that are already in place.
        const idx k = gallop_right(a_, a_.get(base2), base1, len1, 0);
        base1 += k;
        len1  -= k;
        if (len1 == 0) return;

        // Where does the last element of run1 go in run2? Elements of run2
        // after that are already in place.
        len2 = gallop_left(a_, a_.get(base1 + len1 - 1), base2, len2, len2 - 1);
        if (len2 == 0) return;

        if (len1 <= len2) merge_lo(base1, len1, base2, len2);
        else              merge_hi(base1, len1, base2, len2);
    }

    // Merge adjacent runs with len1 <= len2, copying run1 to tmp and merging
    // from the front.
    void merge_lo(idx base1, idx len1, idx base2, idx len2) {
        A a   = a_;
        A tmp = ensure_tmp(len1);
        copy_forward(a, static_cast<size_t>(base1), tmp, 0, static_cast<size_t>(len1));

        idx cursor1 = 0;       // in tmp
        idx cursor2 = base2;   // in a
        idx dest    = base1;   // in a

        a.set(dest++, a.get(cursor2++));
        if (--len2 == 0) {
            copy_forward(tmp, static_cast<size_t>(cursor1), a, static_cast<size_t>(dest), static_cast<size_t>(len1));
            return;
        }
        if (len1 == 1) {
            copy_forward(a, static_cast<size_t>(cursor2), a, static_cast<size_t>(dest), static_cast<size_t>(len2));
            a.set(dest + len2, tmp.get(cursor1));
            return;
        }

        idx min_gallop = min_gallop_;
        enum { kContinue, kSucceed, kCopyB } state = kContinue;

        for (;;) {
            idx count1 = 0;   // consecutive wins of run1
            idx count2 = 0;   // consecutive wins of run2

            // Straightforward one-at-a-time merge until one run is winning
            // consistently.
            do {
                if (a.less(a.get(cursor2), tmp.get(cursor1))) {
                    a.set(dest++, a.get(cursor2++));
                    ++count2;
                    count1 = 0;
                    if (--len2 == 0) { state = kSucceed; break; }
                } else {
                    a.set(dest++, tmp.get(cursor1++));
                    ++count1;
                    count2 = 0;
                    if (--len1 == 1) { state = kCopyB; break; }
                }
            } while ((count1 | count2) < min_gallop);
            if (state != kContinue) break;

            // One run is winning consistently: gallop.
            ++min_gallop;
            do {
                min_gallop -= min_gallop > 1;
                min_gallop_ = min_gallop;

                count1 = gallop_right(tmp, a.get(cursor2), cursor1, len1, 0);
                if (count1 != 0) {
                    copy_forward(tmp, static_cast<size_t>(cursor1), a, static_cast<size_t>(dest), static_cast<size_t>(count1));
                    dest    += count1;
                    cursor1 += count1;
                    len1    -= count1;
                    if (len1 == 1) { state = kCopyB; break; }
                    if (len1 == 0) { state = kSucceed; break; }   // only with an inconsistent comparator
                }
                a.set(dest++, a.get(cursor2++));
                if (--len2 == 0) { state = kSucceed; break; }

                count2 = gallop_left(a, tmp.get(cursor1), cursor2, len2, 0);
                if (count2 != 0) {
                    copy_forward(a, static_cast<size_t>(cursor2), a, static_cast<size_t>(dest), static_cast<size_t>(count2));
                    dest    += count2;
                    cursor2 += count2;
                    len2    -= count2;
                    if (len2 == 0) { state = kSucceed; break; }
                }
                a.set(dest++, tmp.get(cursor1++));
                if (--len1 == 1) { state = kCopyB; break; }
            } while (count1 >= MIN_GALLOP || count2 >= MIN_GALLOP);
            if (state != kContinue) break;

            ++min_gallop;   // penalise leaving gallop mode
            min_gallop_ = min_gallop;
        }

        min_gallop_ = min_gallop < 1 ? 1 : min_gallop;
        if (state == kSucceed) {
            if (len1 != 0) copy_forward(tmp, static_cast<size_t>(cursor1), a, static_cast<size_t>(dest), static_cast<size_t>(len1));
        } else {  // kCopyB: one element of run1 left; it goes after the rest of run2
            copy_forward(a, static_cast<size_t>(cursor2), a, static_cast<size_t>(dest), static_cast<size_t>(len2));
            a.set(dest + len2, tmp.get(cursor1));
        }
    }

    // Merge adjacent runs with len1 >= len2, copying run2 to tmp and merging
    // from the back.
    void merge_hi(idx base1, idx len1, idx base2, idx len2) {
        A a   = a_;
        A tmp = ensure_tmp(len2);
        copy_forward(a, static_cast<size_t>(base2), tmp, 0, static_cast<size_t>(len2));

        idx cursor1 = base1 + len1 - 1;   // in a
        idx cursor2 = len2 - 1;           // in tmp
        idx dest    = base2 + len2 - 1;   // in a

        a.set(dest--, a.get(cursor1--));
        if (--len1 == 0) {
            copy_forward(tmp, 0, a, static_cast<size_t>(dest - (len2 - 1)), static_cast<size_t>(len2));
            return;
        }
        if (len2 == 1) {
            dest    -= len1;
            cursor1 -= len1;
            copy_backward(a, static_cast<size_t>(cursor1 + 1), a, static_cast<size_t>(dest + 1), static_cast<size_t>(len1));
            a.set(dest, tmp.get(cursor2));
            return;
        }

        idx min_gallop = min_gallop_;
        enum { kContinue, kSucceed, kCopyA } state = kContinue;

        for (;;) {
            idx count1 = 0;
            idx count2 = 0;

            do {
                if (a.less(tmp.get(cursor2), a.get(cursor1))) {
                    a.set(dest--, a.get(cursor1--));
                    ++count1;
                    count2 = 0;
                    if (--len1 == 0) { state = kSucceed; break; }
                } else {
                    a.set(dest--, tmp.get(cursor2--));
                    ++count2;
                    count1 = 0;
                    if (--len2 == 1) { state = kCopyA; break; }
                }
            } while ((count1 | count2) < min_gallop);
            if (state != kContinue) break;

            ++min_gallop;
            do {
                min_gallop -= min_gallop > 1;
                min_gallop_ = min_gallop;

                count1 = len1 - gallop_right(a, tmp.get(cursor2), base1, len1, len1 - 1);
                if (count1 != 0) {
                    dest    -= count1;
                    cursor1 -= count1;
                    len1    -= count1;
                    copy_backward(a, static_cast<size_t>(cursor1 + 1), a, static_cast<size_t>(dest + 1), static_cast<size_t>(count1));
                    if (len1 == 0) { state = kSucceed; break; }
                }
                a.set(dest--, tmp.get(cursor2--));
                if (--len2 == 1) { state = kCopyA; break; }

                count2 = len2 - gallop_left(tmp, a.get(cursor1), 0, len2, len2 - 1);
                if (count2 != 0) {
                    dest    -= count2;
                    cursor2 -= count2;
                    len2    -= count2;
                    copy_forward(tmp, static_cast<size_t>(cursor2 + 1), a, static_cast<size_t>(dest + 1), static_cast<size_t>(count2));
                    if (len2 == 1) { state = kCopyA; break; }
                    if (len2 == 0) { state = kSucceed; break; }   // only with an inconsistent comparator
                }
                a.set(dest--, a.get(cursor1--));
                if (--len1 == 0) { state = kSucceed; break; }
            } while (count1 >= MIN_GALLOP || count2 >= MIN_GALLOP);
            if (state != kContinue) break;

            ++min_gallop;
            min_gallop_ = min_gallop;
        }

        min_gallop_ = min_gallop < 1 ? 1 : min_gallop;
        if (state == kSucceed) {
            if (len2 != 0) copy_forward(tmp, 0, a, static_cast<size_t>(dest - (len2 - 1)), static_cast<size_t>(len2));
        } else {  // kCopyA: one element of run2 left; it goes before the rest of run1
            dest    -= len1;
            cursor1 -= len1;
            copy_backward(a, static_cast<size_t>(cursor1 + 1), a, static_cast<size_t>(dest + 1), static_cast<size_t>(len1));
            a.set(dest, tmp.get(cursor2));
        }
    }
};

}  // namespace tim_detail

template <class A>
inline void timsort(A a) {
    if (a.size() < 2) return;
    tim_detail::Sorter<A> s(a);
    s.sort();
}

}  // namespace sb

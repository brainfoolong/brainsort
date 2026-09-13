//! The Rust sorts' deterministic numbers: `results/rust-counts.csv` is
//! recomputed and every column must come out the same, and brainsort's
//! rows must agree with the C++ golden file. Up to 10,000 elements in a
//! debug build, 100,000 in release (`brainsort-bench --rust-golden` does
//! the same from the command line).
use brainsort_bench::counts;
use std::path::Path;

#[test]
fn rust_counts_are_reproduced() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let max_n = if cfg!(debug_assertions) { 10_000 } else { 100_000 };
    let report = counts::check_file(&root.join("results/rust-counts.csv"), max_n, Some(&root.join("results/counts.csv")), |_| {}).expect("read the counts files");
    assert!(report.checked > 0, "no rows checked");
    assert!(report.failures.is_empty(), "{} of {} rows differ:\n{}", report.failures.len(), report.checked, report.failures.join("\n"));
}

#[test]
fn every_rust_sort_is_counted_on_one_cell() {
    // The counters see something for every algorithm: comparisons for every
    // sort but brainsort (whose scout pass compares once per element), key
    // reads for the cached-key sort; every sort produces the one stable order.
    let rows = counts::run_cell::<brainsort_bench::Item>("random", 1000);
    assert_eq!(rows.len(), counts::ALGORITHMS.len());
    for r in &rows {
        assert!(r.ok, "{}: {}", r.algorithm, r.error);
        if r.algorithm == "slice::sort_by_cached_key" {
            assert_eq!(r.key_reads, Some(1000), "{}: key reads {:?}", r.algorithm, r.key_reads);
        } else {
            assert_eq!(r.key_reads, None, "{}: key reads {:?}", r.algorithm, r.key_reads);
        }
        if r.algorithm != "brainsort" {
            assert!(r.compares >= 1000, "{}: {} compares", r.algorithm, r.compares);
        }
    }
    let stable: Vec<&counts::CountRow> = rows.iter().filter(|r| r.stable).collect();
    assert!(stable.windows(2).all(|w| w[0].order_hash == w[1].order_hash), "the stable sorts must produce the same order");
}

//! The deterministic numbers of the Rust sorts as shipped: everything that
//! can be counted of `slice::sort`, `slice::sort_by_cached_key` and
//! glidesort without touching their code. A comparison sort shows its
//! comparisons through the comparator it is given (the cached-key sort
//! through the `Ord` of the key it caches), the cached-key sort its key
//! reads through the key function, and every sort its scratch memory
//! through the global allocator. What cannot be observed is an element
//! move: a Rust move is a plain copy with no hook, so bytes moved, reads,
//! writes and the cache model exist only for brainsort, in
//! `results/counts.csv`, and the golden test holds the crate to them.
//!
//! The opponents are the Rust sorts that can do what brainsort does, as
//! shipped: stable, every key type, an arbitrary comparator
//! (docs/decisions/0017.md).
//!
//! One row per (key type, input, algorithm, size), the same cells as the
//! C++ counts, into `results/rust-counts.csv`. The number of allocations is
//! not recorded: for a crate built without hooks it is a property of the
//! optimiser, which elides dead allocations in a release build, not of the
//! algorithm; the peak is the same in every build. brainsort's row comes from
//! the counted view (the same run the golden test makes) with its scratch
//! memory taken through the same allocator hook as everyone else's; the
//! check compares it with `counts.csv`, so the two files cannot drift
//! apart.
use crate::datasets::{self, Dataset, Generate, KeyGen, dataset_applies};
use crate::{DblItem, FNV_OFFSET, FNV_PRIME, I64Item, Item, Item2, StrItem, alloc_arm, alloc_disarm, counted_run, make_reference, verify_sorted};
use std::cell::Cell;
use std::cmp::Ordering;

// ---- the counters ---------------------------------------------------------------
// Comparisons and their flips are seen in the comparator; key reads in the
// key function the cached-key sort calls. Thread-local: every sort here
// runs on the calling thread.

thread_local! {
    static COMPARES: Cell<u64> = const { Cell::new(0) };
    static FLIPS: Cell<u64> = const { Cell::new(0) };
    static LAST: Cell<u8> = const { Cell::new(2) };   // 2: no comparison yet
    static KEY_READS: Cell<u64> = const { Cell::new(0) };
}
fn reset_counters() {
    COMPARES.set(0);
    FLIPS.set(0);
    LAST.set(2);
    KEY_READS.set(0);
}
/// One comparison with outcome `less`, as the trace counts brainsort's:
/// a flip is an outcome different from the one before.
#[inline]
fn note_compare(less: bool) {
    COMPARES.set(COMPARES.get() + 1);
    let last = LAST.get();
    if last != 2 && (last == 1) != less {
        FLIPS.set(FLIPS.get() + 1);
    }
    LAST.set(less as u8);
}
#[inline]
fn note_key() {
    KEY_READS.set(KEY_READS.get() + 1);
}
/// The three-way comparator on the key, counted: one call is one comparison.
#[inline]
fn counted_cmp<T: Item2>(a: &T, b: &T) -> Ordering {
    let c = T::compare(*a, *b);
    note_compare(c < 0);
    c.cmp(&0)
}

// ---- the key as slice::sort_by_cached_key sees it ---------------------------
// The cached-key sort takes a key per element, keeps the keys in its own
// buffer with the indices, sorts the pairs and permutes the elements. The
// key is a counted wrapper: its `Ord` is the harness's comparison on the
// key, so every comparison the sort makes is seen; the key function counts
// the reads. Doubles compare by `total_cmp`, what a Rust program has to
// write to give f64 an `Ord`; the inputs hold no NaN and no negative zero,
// so that is the natural order.

pub trait CachedKey: Copy {
    fn cmp_key(a: Self, b: Self) -> Ordering;
}
impl CachedKey for i32 {
    #[inline]
    fn cmp_key(a: Self, b: Self) -> Ordering {
        a.cmp(&b)
    }
}
impl CachedKey for i64 {
    #[inline]
    fn cmp_key(a: Self, b: Self) -> Ordering {
        a.cmp(&b)
    }
}
impl CachedKey for f64 {
    #[inline]
    fn cmp_key(a: Self, b: Self) -> Ordering {
        a.total_cmp(&b)
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Counted<K: CachedKey>(pub K);
impl<K: CachedKey> PartialEq for Counted<K> {
    #[inline]
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl<K: CachedKey> Eq for Counted<K> {}
impl<K: CachedKey> PartialOrd for Counted<K> {
    #[inline]
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl<K: CachedKey> Ord for Counted<K> {
    #[inline]
    fn cmp(&self, o: &Self) -> Ordering {
        let c = K::cmp_key(self.0, o.0);
        note_compare(c == Ordering::Less);
        c
    }
}

/// Whether a key type has a cached-key opponent, and how to call it. A
/// string's key would be the string itself, copied once per element: not
/// what a program would write, so strings have none.
pub trait RustOpponents: Item2 + KeyGen
where
    Dataset<Self>: Generate<Self>,
{
    const CACHED_KEY: bool;
    fn sort_by_cached_key(_v: &mut Vec<Self>) {}
}
macro_rules! scalar_opponents {
    ($item:ty) => {
        impl RustOpponents for $item {
            const CACHED_KEY: bool = true;
            fn sort_by_cached_key(v: &mut Vec<Self>) {
                v.sort_by_cached_key(|x| {
                    note_key();
                    Counted(x.key)
                });
            }
        }
    };
}
scalar_opponents!(Item);
scalar_opponents!(I64Item);
scalar_opponents!(DblItem);
impl RustOpponents for StrItem {
    const CACHED_KEY: bool = false;
}

// ---- the rows -------------------------------------------------------------------

/// The algorithms of the file, in order, with their stability (every one
/// is stable: that is the entry condition of the pool).
pub const ALGORITHMS: [(&str, bool); 4] = [("brainsort", true), ("slice::sort", true), ("slice::sort_by_cached_key", true), ("glidesort", true)];
/// The input patterns, the same twelve as `results/counts.csv`.
pub const DATASETS: [&str; 12] = ["random", "sorted", "reverse", "nearly_sorted", "few_unique", "all_equal", "runs", "organ_pipe", "small_range", "sawtooth", "prefixed", "sparse_bits"];
pub const TYPES: [&str; 4] = ["int32", "double", "int64", "string"];
pub const SEED: u64 = 20260912;
pub const COLUMNS: [&str; 13] = ["type", "dataset", "algorithm", "stable", "n", "seed", "ok", "error", "compares", "cmp_flips", "key_reads", "aux_peak_bytes", "order_hash"];

/// One row of `rust-counts.csv`.
#[derive(Clone, Debug, PartialEq)]
pub struct CountRow {
    pub ty: String,
    pub dataset: String,
    pub algorithm: String,
    pub stable: bool,
    pub n: usize,
    pub seed: u64,
    pub ok: bool,
    pub error: String,
    pub compares: u64,
    pub cmp_flips: u64,
    /// Calls of the key function: `None` for a comparison sort, and for
    /// brainsort, whose key work is counted as table accesses in counts.csv;
    /// one per element for the cached-key sort.
    pub key_reads: Option<u64>,
    pub aux_peak_bytes: usize,
    pub order_hash: u64,
}
impl CountRow {
    pub fn csv(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{:016x}",
            self.ty,
            self.dataset,
            self.algorithm,
            self.stable as u8,
            self.n,
            self.seed,
            self.ok as u8,
            self.error.replace(',', ";"),
            self.compares,
            self.cmp_flips,
            self.key_reads.map(|k| k.to_string()).unwrap_or_default(),
            self.aux_peak_bytes,
            self.order_hash
        )
    }
    /// The columns a check compares, as (name, value) strings.
    pub fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("ok", (self.ok as u8).to_string()),
            ("compares", self.compares.to_string()),
            ("cmp_flips", self.cmp_flips.to_string()),
            ("key_reads", self.key_reads.map(|k| k.to_string()).unwrap_or_default()),
            ("aux_peak_bytes", self.aux_peak_bytes.to_string()),
            ("order_hash", format!("{:016x}", self.order_hash)),
        ]
    }
}
pub fn csv_header() -> String {
    COLUMNS.join(",")
}

fn order_hash<T: Item2>(v: &[T]) -> u64 {
    let mut h = FNV_OFFSET;
    for x in v {
        h = (h ^ x.id() as u64).wrapping_mul(FNV_PRIME);
    }
    h
}
/// Sorted by key, and the same elements as the input (for an unstable sort,
/// which may order equal keys as it likes).
fn verify_permutation<T: Item2>(out: &[T], reference: &[T]) -> Result<(), String> {
    if out.len() != reference.len() {
        return Err("size changed".into());
    }
    for i in 1..out.len() {
        if T::less(out[i], out[i - 1]) {
            return Err(format!("not sorted at index {i}"));
        }
    }
    let mut ids: Vec<u32> = out.iter().map(|x| x.id()).collect();
    ids.sort_unstable();
    if ids.iter().enumerate().any(|(i, &id)| id != i as u32) {
        return Err("not a permutation of the input".into());
    }
    Ok(())
}

/// One opponent on one input: counters reset, the allocator armed around the
/// call, the result checked.
fn run_one<T: Item2>(items: &[T], reference: &[T], name: &str, stable: bool, sort: impl FnOnce(&mut Vec<T>)) -> CountRow {
    let mut work = items.to_vec(); // before the allocator is armed: not scratch
    reset_counters();
    alloc_arm();
    sort(&mut work);
    let a = alloc_disarm();
    let check = if stable { verify_sorted(&work, reference) } else { verify_permutation(&work, reference) };
    let mut error = check.err().unwrap_or_default();
    if error.is_empty() && a.leak != 0 {
        error = format!("aux_memory_leak:{}", a.leak);
    }
    CountRow {
        ty: T::NAME.into(),
        dataset: String::new(),
        algorithm: name.into(),
        stable,
        n: items.len(),
        seed: SEED,
        ok: error.is_empty(),
        error,
        compares: COMPARES.get(),
        cmp_flips: FLIPS.get(),
        key_reads: if KEY_READS.get() > 0 { Some(KEY_READS.get()) } else { None },
        aux_peak_bytes: a.peak,
        order_hash: order_hash(&work),
    }
}

/// One-time state, taken out of the counted windows: brainsort's CPU and
/// cache detection (a temporary buffer on first use) and whatever a crate
/// sets up on its first call. Scratch is what a call allocates and frees;
/// this is neither.
pub fn warm_up() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _ = (brainsort::internals::cache_sizes(), brainsort::internals::have_avx2(), brainsort::internals::have_bmi2());
        for ty in TYPES {
            let _ = cell_rows(ty, "random", 64); // its rows carry the one-time state; discarded
        }
    });
}

/// Every algorithm on one cell.
pub fn run_cell<T: RustOpponents>(dataset: &str, n: usize) -> Vec<CountRow>
where
    Dataset<T>: Generate<T>,
{
    warm_up();
    run_cell_cold::<T>(dataset, n)
}
/// `run_cell` without the warm-up: what the warm-up itself runs.
fn run_cell_cold<T: RustOpponents>(dataset: &str, n: usize) -> Vec<CountRow>
where
    Dataset<T>: Generate<T>,
{
    let data = datasets::generate::<T>(dataset, n, SEED);
    let reference = make_reference(&data.items);
    let mut rows = Vec::new();
    // brainsort: the counted view, its scratch through the same allocator hook.
    let (det, out) = counted_run(&data.items, data.pool.as_deref(), &reference);
    let get = |c: &str| det.values.iter().find(|(k, _)| *k == c).map(|(_, v)| v.parse::<u64>().unwrap_or(0)).unwrap_or(0);
    let mut error = det.error.clone();
    if error.is_empty() && (det.alloc_peak != get("aux_peak_bytes") as usize || det.alloc_allocs != get("aux_allocs")) {
        let sizes: Vec<String> = det.alloc_sizes.iter().take(det.alloc_allocs.min(16) as usize).map(|s| s.to_string()).collect();
        error = format!("allocator_mismatch:{}/{}_vs_{}/{}_sizes_{}", det.alloc_peak, det.alloc_allocs, get("aux_peak_bytes"), get("aux_allocs"), sizes.join("+"));
    }
    rows.push(CountRow {
        ty: T::NAME.into(),
        dataset: dataset.into(),
        algorithm: "brainsort".into(),
        stable: true,
        n,
        seed: SEED,
        ok: error.is_empty(),
        error,
        compares: get("compares"),
        cmp_flips: get("cmp_flips"),
        key_reads: None,
        aux_peak_bytes: det.alloc_peak,
        order_hash: order_hash(&out),
    });
    let items = &data.items;
    rows.push(run_one(items, &reference, "slice::sort", true, |v| v.sort_by(counted_cmp)));
    if T::CACHED_KEY {
        rows.push(run_one(items, &reference, "slice::sort_by_cached_key", true, T::sort_by_cached_key));
    }
    rows.push(run_one(items, &reference, "glidesort", true, |v| glidesort::sort_by(v, counted_cmp)));
    for r in &mut rows {
        r.dataset = dataset.into();
    }
    rows
}

fn cell_rows(ty: &str, dataset: &str, n: usize) -> Vec<CountRow> {
    match ty {
        "int32" => run_cell_cold::<Item>(dataset, n),
        "double" => run_cell_cold::<DblItem>(dataset, n),
        "int64" => run_cell_cold::<I64Item>(dataset, n),
        "string" => run_cell_cold::<StrItem>(dataset, n),
        _ => Vec::new(),
    }
}

/// The whole matrix: every size, type, input and algorithm; `log` sees
/// every row as it is made.
pub fn run_matrix(sizes: &[usize], mut log: impl FnMut(&CountRow)) -> Vec<CountRow> {
    warm_up();
    let mut out = Vec::new();
    for &n in sizes {
        for ty in TYPES {
            for ds in DATASETS {
                if !dataset_applies(ds, ty) {
                    continue;
                }
                for r in cell_rows(ty, ds, n) {
                    log(&r);
                    out.push(r);
                }
            }
        }
    }
    out
}
pub fn to_csv(rows: &[CountRow]) -> String {
    let mut s = csv_header();
    s.push('\n');
    for r in rows {
        s.push_str(&r.csv());
        s.push('\n');
    }
    s
}

/// A row of the file as written: the key of the cell and its checked columns.
#[derive(Clone, Debug)]
pub struct FileRow {
    pub ty: String,
    pub dataset: String,
    pub algorithm: String,
    pub n: usize,
    pub fields: Vec<(String, String)>,
}
pub fn load_file(path: &std::path::Path) -> Result<Vec<FileRow>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().ok_or("empty file")?.split(',').collect();
    let col = |name: &str| header.iter().position(|h| *h == name).ok_or_else(|| format!("no column {name}"));
    let (ct, cd, ca, cn) = (col("type")?, col("dataset")?, col("algorithm")?, col("n")?);
    let mut rows = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        let fields = ["ok", "compares", "cmp_flips", "key_reads", "aux_peak_bytes", "order_hash"]
            .iter()
            .filter_map(|c| header.iter().position(|h| h == c).map(|i| (c.to_string(), f.get(i).copied().unwrap_or("").to_string())))
            .collect();
        rows.push(FileRow { ty: f[ct].into(), dataset: f[cd].into(), algorithm: f[ca].into(), n: f[cn].parse().map_err(|_| "bad n")?, fields });
    }
    Ok(rows)
}

/// What a check found.
#[derive(Default, Debug)]
pub struct Report {
    pub checked: usize,
    pub skipped: usize,
    pub failures: Vec<String>,
}

/// Recomputes every row of the file with n <= max_n and reports every
/// column that differs; then, if the C++ golden file is given, holds the
/// brainsort rows of this file to it on the columns the two share.
pub fn check_file(path: &std::path::Path, max_n: usize, cpp_golden: Option<&std::path::Path>, mut log: impl FnMut(&str)) -> Result<Report, String> {
    let file = load_file(path)?;
    warm_up();
    let mut report = Report::default();
    let mut cells: Vec<(String, String, usize)> = Vec::new();
    for r in &file {
        let key = (r.ty.clone(), r.dataset.clone(), r.n);
        if !cells.contains(&key) {
            cells.push(key);
        }
    }
    for (ty, ds, n) in cells {
        if n > max_n {
            report.skipped += file.iter().filter(|r| r.ty == ty && r.dataset == ds && r.n == n).count();
            continue;
        }
        let got = cell_rows(&ty, &ds, n);
        for want in file.iter().filter(|r| r.ty == ty && r.dataset == ds && r.n == n) {
            report.checked += 1;
            let Some(g) = got.iter().find(|g| g.algorithm == want.algorithm) else {
                report.failures.push(format!("{ty} {ds} n={n} {}: not produced", want.algorithm));
                continue;
            };
            let gf = g.fields();
            let diff: Vec<String> = want.fields.iter().filter_map(|(c, w)| gf.iter().find(|(k, _)| k == c).filter(|(_, v)| v != w).map(|(_, v)| format!("{c} {w} -> {v}"))).collect();
            if diff.is_empty() {
                log(&format!("ok    {ty:<7} {ds:<14} n={n:<8} {}", want.algorithm));
            } else {
                let line = format!("FAIL  {ty:<7} {ds:<14} n={n:<8} {}: {}", want.algorithm, diff.join(", "));
                log(&line);
                report.failures.push(line);
            }
        }
    }
    if let Some(cpp) = cpp_golden {
        let golden = crate::load_golden(cpp)?;
        for want in file.iter().filter(|r| r.algorithm == "brainsort" && r.n <= max_n) {
            let Some(g) = golden.iter().find(|g| g.ty == want.ty && g.dataset == want.dataset && g.n == want.n) else {
                report.failures.push(format!("{} {} n={}: brainsort row not in {}", want.ty, want.dataset, want.n, cpp.display()));
                continue;
            };
            for c in ["compares", "cmp_flips", "aux_peak_bytes"] {
                let a = want.fields.iter().find(|(k, _)| k == c).map(|(_, v)| v.as_str()).unwrap_or("");
                let b = g.values.iter().find(|(k, _)| k == c).map(|(_, v)| v.as_str()).unwrap_or("");
                if a != b {
                    let line = format!("FAIL  {:<7} {:<14} n={:<8} brainsort: {c} {a} here, {b} in {}", want.ty, want.dataset, want.n, cpp.display());
                    log(&line);
                    report.failures.push(line);
                }
            }
        }
    }
    Ok(report)
}

//! The public API against the Rust ecosystem on plain vectors: what a user
//! of `brainsort::sort` gets, including the cost of building the records
//! and permuting the elements. The twin of the C++ `bench/api_bench.cpp`:
//! the same key types, input patterns, sizes and repetition scheme, so the
//! two languages can be compared cell by cell on one machine.
//!
//! Opponents, as shipped: `slice::sort` (the standard library's stable
//! sort, driftsort), `slice::sort_unstable` (ipnsort), and the radix crates
//! `radsort` (stable), `voracious_radix_sort` (stable and unstable) and
//! `rdst` (unstable, single-threaded here) where they support the key type.
//! Every result is checked.
use crate::datasets::Mt19937_64;
use std::time::Instant;

/// A 64-byte record sorted by its first field.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Row {
    pub key: i64,
    pub payload: [u8; 56],
}
impl Default for Row {
    fn default() -> Self {
        Row { key: 0, payload: [0; 56] }
    }
}
// SAFETY: an i64 and 56 bytes, no padding.
unsafe impl brainsort::PlainBytes for Row {}
impl rdst::RadixKey for Row {
    const LEVELS: usize = 8;
    #[inline]
    fn get_level(&self, level: usize) -> u8 {
        self.key.get_level(level)
    }
}

/// The input generators of the C++ API benchmark: `random()` a full-entropy
/// key, `from(v)` a key monotone in v. Doubles use the harness's own
/// uniform draw (the C++ uses `std::uniform_real_distribution`, whose
/// mapping is implementation-defined): the same distribution, not the same
/// values.
pub trait BenchKey: Clone + Send + Sync + 'static {
    const NAME: &'static str;
    fn random(rng: &mut Mt19937_64) -> Self;
    fn from_int(v: i64) -> Self;
    fn less(a: &Self, b: &Self) -> bool;
}
impl BenchKey for i32 {
    const NAME: &'static str = "i32";
    fn random(r: &mut Mt19937_64) -> Self {
        r.next_u64() as u32 as i32
    }
    fn from_int(v: i64) -> Self {
        v as i32
    }
    fn less(a: &Self, b: &Self) -> bool {
        a < b
    }
}
impl BenchKey for i64 {
    const NAME: &'static str = "i64";
    fn random(r: &mut Mt19937_64) -> Self {
        r.next_u64() as i64
    }
    fn from_int(v: i64) -> Self {
        v
    }
    fn less(a: &Self, b: &Self) -> bool {
        a < b
    }
}
impl BenchKey for f64 {
    const NAME: &'static str = "f64";
    fn random(r: &mut Mt19937_64) -> Self {
        crate::datasets::uniform_real(r, -1e6, 1e6)
    }
    fn from_int(v: i64) -> Self {
        v as f64 * 0.25
    }
    fn less(a: &Self, b: &Self) -> bool {
        a < b
    }
}
impl BenchKey for String {
    const NAME: &'static str = "String";
    fn random(r: &mut Mt19937_64) -> Self {
        let len = 3 + (r.next_u64() % 10) as usize;
        (0..len).map(|_| (b'a' + (r.next_u64() % 26) as u8) as char).collect()
    }
    fn from_int(v: i64) -> Self {
        format!("{:012}", v + (1i64 << 40))
    }
    fn less(a: &Self, b: &Self) -> bool {
        a < b
    }
}
impl BenchKey for Row {
    const NAME: &'static str = "64-byte struct by i64";
    fn random(r: &mut Mt19937_64) -> Self {
        Row { key: r.next_u64() as i64, payload: [0; 56] }
    }
    fn from_int(v: i64) -> Self {
        Row { key: v, payload: [0; 56] }
    }
    fn less(a: &Self, b: &Self) -> bool {
        a.key < b.key
    }
}

pub const DATASETS: [&str; 5] = ["random", "sorted", "reverse", "nearly_sorted", "few_unique"];

pub fn make<T: BenchKey>(ds: &str, n: usize, seed: u64) -> Vec<T> {
    let mut rng = Mt19937_64::new(seed);
    let mut v: Vec<T> = Vec::with_capacity(n);
    match ds {
        "random" => (0..n).for_each(|_| v.push(T::random(&mut rng))),
        "sorted" => (0..n).for_each(|i| v.push(T::from_int(i as i64))),
        "reverse" => (0..n).for_each(|i| v.push(T::from_int((n - i) as i64))),
        "few_unique" => (0..n).for_each(|_| v.push(T::from_int((rng.next_u64() % 100) as i64))),
        "nearly_sorted" => {
            (0..n).for_each(|i| v.push(T::from_int(i as i64)));
            for _ in 0..n / 100 {
                let (a, b) = (rng.next_u64() as usize % n, rng.next_u64() as usize % n);
                v.swap(a, b);
            }
        }
        _ => {}
    }
    v
}

/// Median wall time in ms of `reps` runs of `sorter` on a copy of `input`,
/// each result checked for order.
/// The size the repetition scheme is defined at: below it, `batch_for(n)`
/// independent copies are sorted back to back in one timed region and the
/// time per sort is reported, so that ten elements are not timed as one
/// call of a few nanoseconds.
pub const REFERENCE_N: usize = 100_000;
pub fn batch_for(n: usize) -> usize {
    if n == 0 || n >= REFERENCE_N { 1 } else { REFERENCE_N.div_ceil(n) }
}

/// The median wall time of one sort in ms over `reps` repetitions; every
/// result is checked.
pub fn median_ms<T: BenchKey>(input: &[T], reps: usize, mut sorter: impl FnMut(&mut Vec<T>)) -> f64 {
    let batch = batch_for(input.len());
    let mut times = Vec::with_capacity(reps);
    let mut work: Vec<Vec<T>> = (0..batch).map(|_| Vec::with_capacity(input.len())).collect();
    for _ in 0..reps {
        for w in &mut work {
            w.clear();
            w.extend_from_slice(input);
        }
        let t0 = Instant::now();
        for w in &mut work {
            sorter(w);
        }
        let dt = t0.elapsed();
        for w in &work {
            assert!(w.windows(2).all(|p| !T::less(&p[1], &p[0])), "not sorted!");
        }
        times.push(dt.as_secs_f64() * 1e3 / batch as f64);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

/// One measured cell: the times of brainsort and of every opponent that
/// supports the type (`None` where it does not).
pub struct Cell {
    pub ty: &'static str,
    pub n: usize,
    pub ds: &'static str,
    pub brainsort: f64,
    pub others: Vec<Option<f64>>,
}
pub const OPPONENTS: [&str; 6] = ["std sort (stable)", "std sort_unstable", "radsort", "voracious stable", "voracious unstable", "rdst"];

/// A type's opponents: which of the six apply and how to call them.
pub trait Opponents: BenchKey {
    fn brainsort(v: &mut Vec<Self>);
    fn std_stable(v: &mut Vec<Self>);
    fn std_unstable(v: &mut Vec<Self>);
    fn radsort(_v: &mut Vec<Self>) -> bool {
        false
    }
    fn voracious_stable(_v: &mut Vec<Self>) -> bool {
        false
    }
    fn voracious_unstable(_v: &mut Vec<Self>) -> bool {
        false
    }
    fn rdst(_v: &mut Vec<Self>) -> bool {
        false
    }
}
macro_rules! scalar_opponents {
    ($($t:ty),*) => {$(
        impl Opponents for $t {
            fn brainsort(v: &mut Vec<Self>) { brainsort::sort(v) }
            fn std_stable(v: &mut Vec<Self>) { v.sort_by(|a, b| a.partial_cmp(b).unwrap()) }
            fn std_unstable(v: &mut Vec<Self>) { v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap()) }
            fn radsort(v: &mut Vec<Self>) -> bool { radsort::sort(v); true }
            fn voracious_stable(v: &mut Vec<Self>) -> bool { use voracious_radix_sort::RadixSort; v.voracious_stable_sort(); true }
            fn voracious_unstable(v: &mut Vec<Self>) -> bool { use voracious_radix_sort::RadixSort; v.voracious_sort(); true }
            fn rdst(v: &mut Vec<Self>) -> bool { use rdst::RadixSort; v.radix_sort_builder().with_single_threaded_tuner().with_parallel(false).sort(); true }
        }
    )*};
}
scalar_opponents!(i32, i64, f64);
impl Opponents for String {
    fn brainsort(v: &mut Vec<Self>) {
        brainsort::sort(v)
    }
    fn std_stable(v: &mut Vec<Self>) {
        v.sort()
    }
    fn std_unstable(v: &mut Vec<Self>) {
        v.sort_unstable()
    }
}
impl Opponents for Row {
    fn brainsort(v: &mut Vec<Self>) {
        brainsort::sort_by_key(v, |r| r.key)
    }
    fn std_stable(v: &mut Vec<Self>) {
        v.sort_by_key(|r| r.key)
    }
    fn std_unstable(v: &mut Vec<Self>) {
        v.sort_unstable_by_key(|r| r.key)
    }
    fn radsort(v: &mut Vec<Self>) -> bool {
        radsort::sort_by_key(v, |r| r.key);
        true
    }
    fn rdst(v: &mut Vec<Self>) -> bool {
        use rdst::RadixSort;
        v.radix_sort_builder().with_single_threaded_tuner().with_parallel(false).sort();
        true
    }
}

fn bench_type<T: Opponents>(sizes: &[usize], reps: usize, out: &mut Vec<Cell>, mut log: impl FnMut(&Cell)) {
    for &n in sizes {
        if (T::NAME == "String" || T::NAME.starts_with("64-byte")) && n > 1_000_000 {
            continue; // 10M strings or rows: memory
        }
        for ds in DATASETS {
            let input = make::<T>(ds, n, 20260912);
            let bs = median_ms(&input, reps, T::brainsort);
            let mut others = vec![Some(median_ms(&input, reps, T::std_stable)), Some(median_ms(&input, reps, T::std_unstable)), None, None, None, None];
            let mut probe = |i: usize, f: fn(&mut Vec<T>) -> bool| {
                let mut w = input.clone();
                if f(&mut w) {
                    others[i] = Some(median_ms(&input, reps, |v| {
                        f(v);
                    }));
                }
            };
            probe(2, T::radsort);
            probe(3, T::voracious_stable);
            probe(4, T::voracious_unstable);
            probe(5, T::rdst);
            let cell = Cell { ty: T::NAME, n, ds, brainsort: bs, others };
            log(&cell);
            out.push(cell);
        }
    }
}

/// A comparator on the elements: brainsort's `sort_by`, or with `infer`
/// its `sort_by_inferred`, against the standard library's two comparator
/// sorts.
fn bench_comparator<T: BenchKey + brainsort::PlainBytes>(name: &'static str, infer: bool, sizes: &[usize], reps: usize, out: &mut Vec<Cell>, mut log: impl FnMut(&Cell)) {
    for &n in sizes {
        if name.starts_with("64-byte") && n > 1_000_000 {
            continue;
        }
        for ds in DATASETS {
            let input = make::<T>(ds, n, 20260912);
            let cmp = |a: &T, b: &T| {
                if T::less(a, b) {
                    std::cmp::Ordering::Less
                } else if T::less(b, a) {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Equal
                }
            };
            let bs = if infer { median_ms(&input, reps, |v| brainsort::sort_by_inferred(v, cmp)) } else { median_ms(&input, reps, |v| brainsort::sort_by(v, cmp)) };
            let ss = median_ms(&input, reps, |v| v.sort_by(cmp));
            let su = median_ms(&input, reps, |v| v.sort_unstable_by(cmp));
            let cell = Cell { ty: name, n, ds, brainsort: bs, others: vec![Some(ss), Some(su), None, None, None, None] };
            log(&cell);
            out.push(cell);
        }
    }
}

/// Runs the whole benchmark; `log` sees every cell as it is measured.
pub fn run_all(sizes: &[usize], reps: usize, mut log: impl FnMut(&Cell)) -> Vec<Cell> {
    let mut out = Vec::new();
    bench_type::<i32>(sizes, reps, &mut out, &mut log);
    bench_type::<i64>(sizes, reps, &mut out, &mut log);
    bench_type::<f64>(sizes, reps, &mut out, &mut log);
    bench_type::<String>(sizes, reps, &mut out, &mut log);
    bench_type::<Row>(sizes, reps, &mut out, &mut log);
    bench_comparator::<i32>("i32 by comparator", false, sizes, reps, &mut out, &mut log);
    bench_comparator::<Row>("64-byte struct by i64 by comparator", false, sizes, reps, &mut out, &mut log);
    bench_comparator::<i32>("i32 by comparator, inferred", true, sizes, reps, &mut out, &mut log);
    bench_comparator::<Row>("64-byte struct by i64 by comparator, inferred", true, sizes, reps, &mut out, &mut log);
    out
}

/// Milliseconds with three decimals, and more below one ms so that a sort
/// of ten elements keeps three significant digits.
pub fn fmt_ms(v: f64) -> String {
    if v >= 1.0 {
        format!("{v:.3}")
    } else if v >= 0.01 {
        format!("{v:.4}")
    } else {
        format!("{v:.6}")
    }
}

/// One line of the Markdown table.
pub fn cell_line(c: &Cell) -> String {
    let best = c.others.iter().flatten().cloned().fold(f64::INFINITY, f64::min);
    let mut s = format!("| {} | {} | {} | {} |", c.ty, c.n, c.ds, fmt_ms(c.brainsort));
    for o in &c.others {
        match o {
            Some(v) => s.push_str(&format!(" {} |", fmt_ms(*v))),
            None => s.push_str(" n/a |"),
        }
    }
    s.push_str(&format!(" {:.2}x |", best / c.brainsort));
    s
}
pub fn table_header() -> String {
    let mut h = String::from("| type | n | dataset | brainsort |");
    for o in OPPONENTS {
        h.push_str(&format!(" {o} |"));
    }
    h.push_str(" vs best |\n|---|---:|---|---:|");
    for _ in OPPONENTS {
        h.push_str("---:|");
    }
    h.push_str("---:|");
    h
}

// ---- the run stamp ---------------------------------------------------------------------

fn cmd(program: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(program).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}
pub fn cpu_name() -> String {
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in s.lines() {
                if let Some(v) = line.strip_prefix("model name") {
                    return v.trim_start_matches([' ', '\t', ':']).trim().to_string();
                }
            }
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Some(s) = cmd("sysctl", &["-n", "machdep.cpu.brand_string"]) {
            return s;
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Some(s) = cmd("reg", &["query", r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0", "/v", "ProcessorNameString"])
            && let Some(line) = s.lines().find(|l| l.contains("ProcessorNameString"))
            && let Some(v) = line.split("REG_SZ").nth(1)
        {
            return v.trim().to_string();
        }
    }
    String::from("unknown CPU")
}
pub fn os_name() -> &'static str {
    match std::env::consts::OS {
        "linux" => "Linux",
        "windows" => "Windows",
        "macos" => "macOS",
        o => o,
    }
}
pub fn arch_name() -> &'static str {
    match std::env::consts::ARCH {
        "x86_64" => "x86-64",
        "aarch64" => "ARM64",
        "x86" => "x86",
        a => a,
    }
}
pub fn compiler() -> String {
    let v = cmd("rustc", &["-V"]).unwrap_or_else(|| "rustc".into());
    format!("{v} release")
}
pub fn commit() -> String {
    cmd("git", &["rev-parse", "--short=12", "HEAD"]).unwrap_or_default()
}
/// UTC now as `YYYY-MM-DDTHH:MM:SSZ`.
pub fn now_utc() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    let days = secs.div_euclid(86400);
    let rem = secs.rem_euclid(86400);
    // civil from days (Howard Hinnant)
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, (rem % 3600) / 60, rem % 60)
}
/// The paths of the Rust sources a measurement depends on, relative to
/// the repository root.
pub const MEASURED_PATHS: [&str; 3] = ["rust/brainsort/src", "rust/brainsort/Cargo.toml", "rust/brainsort-bench/src"];
/// FNV-1a over the sorted relative paths and contents (CR stripped) of the
/// measured files: the same definition as the C++ stamp and
/// `scripts/website.py`, which recomputes it to detect stale results.
pub fn code_fingerprint(root: &std::path::Path) -> String {
    fn walk(root: &std::path::Path, p: &std::path::Path, files: &mut Vec<(String, std::path::PathBuf)>) {
        if p.is_file() {
            let rel = p.strip_prefix(root).unwrap_or(p).to_string_lossy().replace('\\', "/");
            files.push((rel, p.to_path_buf()));
        } else if p.is_dir()
            && let Ok(rd) = std::fs::read_dir(p)
        {
            for e in rd.flatten() {
                walk(root, &e.path(), files);
            }
        }
    }
    let mut files = Vec::new();
    for p in MEASURED_PATHS {
        walk(root, &root.join(p), &mut files);
    }
    if files.is_empty() {
        return String::new();
    }
    files.sort();
    let mut h = 0xcbf29ce484222325u64;
    let mut mix = |c: u8| h = (h ^ c as u64).wrapping_mul(0x100000001b3);
    for (rel, path) in files {
        rel.bytes().for_each(&mut mix);
        mix(0);
        if let Ok(data) = std::fs::read(&path) {
            data.iter().filter(|&&c| c != b'\r').for_each(|&c| mix(c));
        }
        mix(0);
    }
    format!("{h:016x}")
}
pub fn json_str(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

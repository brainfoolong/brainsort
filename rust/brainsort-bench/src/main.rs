//! The bench crate's command line.
//!
//!   brainsort-bench --golden [--max-n N] [--golden-file PATH]
//!       recompute the brainsort rows of results/counts.csv through the Rust
//!       port's counted view and report every column that differs (the
//!       golden equivalence test; exit status 1 on any mismatch)
//!   brainsort-bench --counts --type T --dataset D [--n N] [--seed S]
//!       print the deterministic columns of one cell
//!   brainsort-bench --rust-counts [--sizes 1000,10000,...] [--out FILE]
//!       the deterministic numbers of every Rust sort as shipped (comparisons,
//!       compare flips, key reads, scratch memory) on the cells of counts.csv,
//!       into results/rust-counts.csv (1,000 to 1,000,000 elements by default)
//!   brainsort-bench --rust-golden [--max-n N] [--rust-file PATH] [--golden-file PATH]
//!       recompute every row of results/rust-counts.csv and report every
//!       column that differs, then hold its brainsort rows to counts.csv
//!       (exit status 1 on any mismatch)
//!   brainsort-bench --bench [--max-n N] [--reps R] [--host TEXT] [--out FILE]
//!       time brainsort against the Rust ecosystem on plain vectors and write
//!       the Markdown table with its run stamp (the twin of brainsort_api_bench):
//!       100,000 elements and 3 repetitions by default; --max-n 1000000 or
//!       10000000 adds the larger sizes
use brainsort_bench::*;
use std::path::PathBuf;

fn usage() -> ! {
    eprintln!(
        "usage: brainsort-bench --golden [--max-n N] [--golden-file PATH]\n       brainsort-bench --counts --type T --dataset D [--n N] [--seed S]\n       brainsort-bench --rust-counts [--sizes 1000,10000,...] [--out FILE]\n       brainsort-bench --rust-golden [--max-n N] [--rust-file PATH] [--golden-file PATH]\n       brainsort-bench --bench [--max-n N] [--reps R] [--host TEXT] [--out FILE]"
    );
    std::process::exit(2)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    let (mut golden, mut counts, mut bench, mut rust_counts, mut rust_golden) = (false, false, false, false, false);
    let mut sizes: Vec<usize> = vec![1_000, 10_000, 100_000, 1_000_000];
    let mut rust_file = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../results/rust-counts.csv"));
    let mut max_n = 100_000usize;
    let mut bench_max_n = 100_000usize;
    let mut reps = 3usize;
    let (mut host, mut out_path) = (String::new(), String::new());
    let mut golden_file = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../results/counts.csv"));
    let (mut ty, mut ds) = (String::from("int32"), String::from("random"));
    let (mut n, mut seed) = (100_000usize, 20260912u64);
    while i < args.len() {
        let need = |i: &mut usize| -> String {
            *i += 1;
            args.get(*i).cloned().unwrap_or_else(|| usage())
        };
        match args[i].as_str() {
            "--golden" => golden = true,
            "--counts" => counts = true,
            "--bench" => bench = true,
            "--rust-counts" => rust_counts = true,
            "--rust-golden" => rust_golden = true,
            "--sizes" => sizes = need(&mut i).split(',').map(|v| v.trim().parse().unwrap_or_else(|_| usage())).collect(),
            "--rust-file" => rust_file = PathBuf::from(need(&mut i)),
            "--reps" => reps = need(&mut i).parse().unwrap_or_else(|_| usage()),
            "--host" => host = need(&mut i),
            "--out" => out_path = need(&mut i),
            "--max-n" => {
                let v: usize = need(&mut i).parse().unwrap_or_else(|_| usage());
                max_n = v;
                bench_max_n = v;
            }
            "--golden-file" => golden_file = PathBuf::from(need(&mut i)),
            "--type" => ty = need(&mut i),
            "--dataset" => ds = need(&mut i),
            "--n" => n = need(&mut i).parse().unwrap_or_else(|_| usage()),
            "--seed" => seed = need(&mut i).parse().unwrap_or_else(|_| usage()),
            _ => usage(),
        }
        i += 1;
    }
    if bench {
        run_bench(reps, bench_max_n, &host, &out_path);
        return;
    }
    if rust_counts {
        let out = if out_path.is_empty() { rust_file.clone() } else { PathBuf::from(&out_path) };
        let rows = counts::run_matrix(&sizes, |r| eprintln!("{}", r.csv()));
        std::fs::write(&out, counts::to_csv(&rows)).expect("write the counts file");
        // A row that failed its check (a sort that did not sort, or did not keep
        // equal keys in order) is a finding, recorded with ok = 0, not an error.
        let failed: Vec<String> = rows.iter().filter(|r| !r.ok).map(|r| format!("{} {} n={} {}: {}", r.ty, r.dataset, r.n, r.algorithm, r.error)).collect();
        eprintln!("wrote {}: {} rows, {} failed their check (recorded with ok = 0)", out.display(), rows.len(), failed.len());
        for f in &failed {
            eprintln!("  {f}");
        }
    }
    if rust_golden {
        let report = match counts::check_file(&rust_file, max_n, Some(&golden_file), |line| println!("{line}")) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(2);
            }
        };
        println!(
            "rust-golden: {} rows of {} recomputed, {} mismatches, {} rows above n = {max_n} skipped; brainsort rows held to {}",
            report.checked,
            rust_file.display(),
            report.failures.len(),
            report.skipped,
            golden_file.display()
        );
        std::process::exit(if report.failures.is_empty() { 0 } else { 1 });
    }
    if golden {
        let rows = match load_golden(&golden_file) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(2);
            }
        };
        let (mut checked, mut mismatched, mut skipped) = (0, 0, 0);
        for row in &rows {
            if row.n > max_n {
                skipped += 1;
                continue;
            }
            checked += 1;
            match check_row(row) {
                Ok(diff) if diff.is_empty() => println!("ok    {:<7} {:<14} n={}", row.ty, row.dataset, row.n),
                Ok(diff) => {
                    mismatched += 1;
                    println!("FAIL  {:<7} {:<14} n={}: {}", row.ty, row.dataset, row.n, diff.join(", "));
                }
                Err(e) => {
                    mismatched += 1;
                    println!("FAIL  {:<7} {:<14} n={}: {e}", row.ty, row.dataset, row.n);
                }
            }
        }
        println!("golden: {checked} brainsort rows of {} recomputed, {mismatched} mismatches, {skipped} rows above n = {max_n} skipped", golden_file.display());
        std::process::exit(if mismatched == 0 { 0 } else { 1 });
    }
    if counts {
        let row = GoldenRow { ty: ty.clone(), dataset: ds.clone(), n, seed, values: vec![] };
        let det = match ty.as_str() {
            "int32" => one::<Item>(&row),
            "double" => one::<DblItem>(&row),
            "int64" => one::<I64Item>(&row),
            "string" => one::<StrItem>(&row),
            _ => usage(),
        };
        if !det.ok {
            println!("error: {}", det.error);
            std::process::exit(1);
        }
        for (c, v) in &det.values {
            println!("{c}={v}");
        }
        return;
    }
    usage();
}

fn run_bench(reps: usize, max_n: usize, host: &str, out_path: &str) {
    use timing::*;
    let sizes: Vec<usize> = [100_000usize, 1_000_000, 10_000_000].into_iter().filter(|&n| n <= max_n).collect();
    let reps = reps.max(1);
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let (cpu, os, arch, comp, when, commit, code) = (cpu_name(), os_name(), arch_name(), compiler(), now_utc(), commit(), code_fingerprint(&root));
    let host_part = if host.is_empty() { String::new() } else { format!(", {host}") };
    let head = format!(
        "{os} {arch}, {comp}, {cpu}, code {code}, commit {commit}, {when}{host_part}. brainsort::sort on a plain Vec against the standard library's stable and unstable sorts and the radix crates radsort, voracious_radix_sort and rdst; median of {reps} runs, wall ms. Generated by brainsort-bench --bench."
    );
    let fields = [
        ("host", json_str(host)),
        ("os", json_str(os)),
        ("arch", json_str(arch)),
        ("compiler", json_str(&comp)),
        ("cpu", json_str(&cpu)),
        ("measured", json_str(&when)),
        ("commit", json_str(&commit)),
        ("code", json_str(&code)),
        ("measured_paths", json_str(&MEASURED_PATHS.join(" "))),
        ("reps", reps.to_string()),
        ("lang", json_str("rust")),
    ];
    let mut stamp = String::from(
        "<!-- stamp {
",
    );
    for (i, (k, v)) in fields.iter().enumerate() {
        stamp.push_str(&format!(
            "  \"{k}\": {v}{}
",
            if i + 1 < fields.len() { "," } else { "" }
        ));
    }
    stamp.push_str("} -->");
    let mut text = format!(
        "{head}
{stamp}

{}
",
        table_header()
    );
    eprintln!("{head}");
    let cells = run_all(&sizes, reps, |c| eprintln!("{}", cell_line(c)));
    for c in &cells {
        text.push_str(&cell_line(c));
        text.push('\n');
    }
    if out_path.is_empty() {
        print!("{text}");
    } else {
        std::fs::write(out_path, text).expect("write the benchmark file");
        eprintln!("wrote {out_path}");
    }
}

fn one<T: Item2 + datasets::KeyGen>(row: &GoldenRow) -> DetRecord
where
    datasets::Dataset<T>: datasets::Generate<T>,
{
    let data = datasets::generate::<T>(&row.dataset, row.n, row.seed);
    let reference = make_reference(&data.items);
    counted_run(&data.items, data.pool.as_deref(), &reference).0
}

// Benchmark driver.
//
// Parent mode (default): for every (size, type, dataset, algorithm) cell
// spawn a fresh child process of this same executable, so each measurement
// starts from a clean heap and its peak-RSS figure is independent of what ran
// before. Collects one result line per child, prints tables and writes CSV /
// Markdown, plus a .meta.json stamp next to the CSV (sortbench/stamp.hpp):
// the code fingerprint, time, CPU, compiler and settings of the run.
//
// Child mode (--child): generate the input, compute the reference, run the
// counted variant once (the deterministic numbers + verification), then run
// the uncounted variant `reps` times under the hardware/time meter and report
// medians. With --counts-only only the counted run happens, which yields
// results/counts.csv: the same on every machine. With --timing-only the
// counted run is skipped and only the timed passes are made (the site takes
// the deterministic columns from the counts file).
//
// Sizes: --n is repeatable. Below 100,000 elements the timed region sorts
// ceil(100,000 / n) independent copies back to back and reports the time per
// sort, so the meter's fixed overhead stays small against the measurement.
// Above 100,000 the repetitions are scaled down in proportion (never below
// 3), so a cell costs about the same time at every size.
#include "sortbench/core.hpp"
#include "sortbench/counted_run.hpp"
#include "sortbench/datasets.hpp"
#include "sortbench/metrics.hpp"
#include "sortbench/registry.hpp"
#include "sortbench/stamp.hpp"
#include "sortbench/trace_alloc.hpp"
#include "sortbench/verify.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX   // windows.h's min/max macros break std::min / std::max
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

using namespace sb;

namespace {

constexpr size_t kReferenceN = 100000;   // the size the repetition count and the batching are defined at

int reps_for(size_t n, int reps) {
    if (n <= kReferenceN || reps <= 0) return reps;
    const double scaled = static_cast<double>(reps) * static_cast<double>(kReferenceN) / static_cast<double>(n);
    return std::max(3, static_cast<int>(scaled + 0.5));
}

size_t batch_for(size_t n) {
    return n == 0 || n >= kReferenceN ? 1 : (kReferenceN + n - 1) / n;
}

// ---------------------------------------------------------------------------
// Child
// ---------------------------------------------------------------------------

template <class T>
T median(std::vector<T> v) {
    if (v.empty()) return T{};
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : static_cast<T>((v[n / 2 - 1] + v[n / 2]) / 2);
}

template <class T>
std::string run_child_t(const std::string& algo_name, const std::string& ds_name, size_t n, uint64_t seed,
                        int reps, int pin_cpu, bool counted) {
    const AlgoInfo<T>* algo = find_algorithm<T>(algo_name);
    if (!algo) return "ok=0 error=unknown_algorithm";
    if (!dataset_exists(ds_name)) return "ok=0 error=unknown_dataset";

    const int pinned = pin_to_cpu(pin_cpu);

    const Dataset<T>      data  = generate_dataset<T>(ds_name, n, seed);
    const std::vector<T>& input = data.items;
    const std::vector<T>  ref   = make_reference(input);
    const size_t          batch = reps > 0 ? batch_for(n) : 1;
    std::vector<std::vector<T>> work(batch, std::vector<T>(n));
    std::string           err;

    // Everything the harness itself needs is now allocated and touched
    // (the cache model's tables included, so they do not show up in the
    // RSS delta).
    g_trace.cache.reset();
    const uint64_t rss_before = peak_rss_bytes();

    // 1. Counted pass: access counts, table and key traffic, the cache
    //    model, telemetry, fingerprints, and verification of the
    //    instrumented code path.
    DetRecord det;
    if (counted) {
        det = counted_run(*algo, data, ref, true, work[0]);
        if (!det.ok) return "ok=0 error=" + det.error;
    }

    // 2. Timed passes on the uncounted variant (none with --counts-only).
    Meter meter;
    std::vector<double>   wall, cpu;
    std::vector<uint64_t> instr, cycles;
    bool has_instr = false, has_cycles = false, has_cpu = false;

    // Calibrate the meter's own overhead (counter ioctls, clock reads) on an
    // empty region; it is subtracted from every sample.
    Measurement overhead;
    if (reps > 0) {
        std::vector<double>   ow, oc;
        std::vector<uint64_t> oi, oy;
        for (int r = 0; r < 101; ++r) {
            meter.start();
            const Measurement m = meter.stop();
            ow.push_back(m.wall_ns);
            oc.push_back(m.cpu_ns);
            oi.push_back(m.instructions);
            oy.push_back(m.cycles);
        }
        overhead.wall_ns = median(ow);
        overhead.cpu_ns = median(oc);
        overhead.instructions = median(oi);
        overhead.cycles = median(oy);
    }

    // One untimed warm-up run so code pages and the aux allocator are hot.
    if (reps > 0) {
        std::copy(input.begin(), input.end(), work[0].begin());
        algo->run_raw(work[0].data(), n);
    }

    const double per = 1.0 / static_cast<double>(batch);
    for (int r = 0; r < reps; ++r) {
        for (auto& w : work) std::copy(input.begin(), input.end(), w.begin());
        meter.start();
        for (auto& w : work) algo->run_raw(w.data(), n);
        const Measurement m = meter.stop();
        for (auto& w : work)
            if (!verify_sorted(w, ref, algo->stable, err))
                return "ok=0 error=timed_pass_failed_rep" + std::to_string(r) + ":" + err;
        wall.push_back(std::max(0.0, m.wall_ns - overhead.wall_ns) * per);
        cpu.push_back(std::max(0.0, m.cpu_ns - overhead.cpu_ns) * per);
        has_cpu |= m.has_cpu;
        if (m.has_instructions) { instr.push_back(static_cast<uint64_t>((m.instructions > overhead.instructions ? m.instructions - overhead.instructions : 0) * per)); has_instr = true; }
        if (m.has_cycles)       { cycles.push_back(static_cast<uint64_t>((m.cycles > overhead.cycles ? m.cycles - overhead.cycles : 0) * per)); has_cycles = true; }
    }
    const uint64_t rss_after = peak_rss_bytes();

    char buf[1024];
    std::snprintf(buf, sizeof buf,
                "ok=1 type=%s algo=%s dataset=%s n=%zu seed=%llu reps=%d batch=%zu pinned=%d hw=%s meter_overhead_ns=%.0f "
                "wall_ns_med=%.0f wall_ns_min=%.0f has_cpu=%d cpu_ns_med=%.0f "
                "has_instr=%d instr_med=%llu has_cycles=%d cycles_med=%llu "
                "counts_accesses=%d rss_delta_bytes=%llu verified=1",
                KeyTraits<T>::name, algo->name, ds_name.c_str(), n, static_cast<unsigned long long>(seed), reps, batch, pinned,
                meter.backend(), overhead.wall_ns,
                median(wall), wall.empty() ? 0.0 : *std::min_element(wall.begin(), wall.end()), has_cpu ? 1 : 0, median(cpu),
                has_instr ? 1 : 0, static_cast<unsigned long long>(has_instr ? median(instr) : 0),
                has_cycles ? 1 : 0, static_cast<unsigned long long>(has_cycles ? median(cycles) : 0),
                counted && algo->counts_accesses ? 1 : 0,
                static_cast<unsigned long long>(rss_after - rss_before));
    std::string line = buf;
    if (counted) {
        const std::vector<std::string> dv = det_values(det);
        for (size_t i = 0; i < kDetColumnCount; ++i) line += std::string(" ") + kDetColumns[i] + "=" + dv[i];
    }
    return line;
}

std::string run_child(const std::string& type, const std::string& algo, const std::string& ds, size_t n,
                      uint64_t seed, int reps, int pin, bool counted) {
    std::string out = "ok=0 error=unknown_type";
    with_type(type, [&](auto tag) {
        using T = typename decltype(tag)::type;
        out = run_child_t<T>(algo, ds, n, seed, reps, pin, counted);
    });
    return out;
}

// ---------------------------------------------------------------------------
// Parent
// ---------------------------------------------------------------------------

std::string self_path(const char* argv0) {
#if defined(_WIN32)
    char buf[MAX_PATH];
    DWORD len = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (len > 0 && len < MAX_PATH) return std::string(buf, len);
#elif defined(__linux__)
    char buf[4096];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof buf - 1);
    if (len > 0) return std::string(buf, static_cast<size_t>(len));
#endif
    return argv0;
}

std::string run_process(const std::string& cmd) {
    std::string out;
#if defined(_WIN32)
    SECURITY_ATTRIBUTES sa{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE rd = nullptr, wr = nullptr;
    if (!CreatePipe(&rd, &wr, &sa, 0)) return "ok=0 error=CreatePipe";
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOA si{};
    si.cb         = sizeof si;
    si.dwFlags    = STARTF_USESTDHANDLES;
    si.hStdOutput = wr;
    si.hStdError  = GetStdHandle(STD_ERROR_HANDLE);
    si.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi{};
    std::string mutable_cmd = cmd;
    if (!CreateProcessA(nullptr, mutable_cmd.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi)) {
        CloseHandle(rd); CloseHandle(wr);
        return "ok=0 error=CreateProcess";
    }
    CloseHandle(wr);
    char  buf[4096];
    DWORD got = 0;
    while (ReadFile(rd, buf, sizeof buf, &got, nullptr) && got > 0) out.append(buf, got);
    WaitForSingleObject(pi.hProcess, INFINITE);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    CloseHandle(rd);
#else
    FILE* f = popen(cmd.c_str(), "r");
    if (!f) return "ok=0 error=popen";
    char buf[4096];
    size_t got;
    while ((got = fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, got);
    pclose(f);
#endif
    return out;
}

using Fields = std::map<std::string, std::string>;

Fields parse_fields(const std::string& line) {
    Fields f;
    std::istringstream is(line);
    std::string tok;
    while (is >> tok) {
        const size_t eq = tok.find('=');
        if (eq != std::string::npos) f[tok.substr(0, eq)] = tok.substr(eq + 1);
    }
    return f;
}

std::string fmt_num(double v, int prec = 2) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(prec) << v;
    return os.str();
}
std::string fmt_int(unsigned long long v) {
    std::string s = std::to_string(v), out;
    int c = 0;
    for (size_t i = s.size(); i-- > 0;) {
        out.insert(out.begin(), s[i]);
        if (++c % 3 == 0 && i > 0) out.insert(out.begin(), ',');
    }
    return out;
}
std::string fmt_bytes(unsigned long long b) {
    if (b == 0) return "0";
    if (b < 1024) return std::to_string(b) + " B";
    if (b < 1024 * 1024) return fmt_num(b / 1024.0, 1) + " KiB";
    return fmt_num(b / (1024.0 * 1024.0), 2) + " MiB";
}

struct Row {
    size_t      n = 0;
    std::string type, dataset, algo;
    Fields f;
    bool ok = false;
    std::string error;
    unsigned long long u(const char* k) const { auto it = f.find(k); return it == f.end() ? 0 : std::strtoull(it->second.c_str(), nullptr, 10); }
    double d(const char* k) const { auto it = f.find(k); return it == f.end() ? 0 : std::strtod(it->second.c_str(), nullptr); }
    std::string s(const char* k) const { auto it = f.find(k); return it == f.end() ? std::string() : it->second; }
};

struct TypeInfo {
    std::string name;
    size_t      elem_bytes = 0;
    std::vector<std::string> algo_names;
};

TypeInfo describe_type(const std::string& type) {
    TypeInfo t;
    t.name = type;
    with_type(type, [&](auto tag) {
        using T = typename decltype(tag)::type;
        t.elem_bytes = sizeof(T);
        for (const auto& a : algorithms<T>()) t.algo_names.push_back(a.name);
    });
    return t;
}

bool algo_is_stable(const std::string& type, const std::string& algo) {
    bool stable = false;
    with_type(type, [&](auto tag) {
        using T = typename decltype(tag)::type;
        const AlgoInfo<T>* a = find_algorithm<T>(algo);
        stable = a && a->stable;
    });
    return stable;
}

std::string meta_path_for(const std::string& csv_path) {
    return (csv_path.size() > 4 && csv_path.compare(csv_path.size() - 4, 4, ".csv") == 0
                ? csv_path.substr(0, csv_path.size() - 4) : csv_path) + ".meta.json";
}

void print_usage() {
    std::puts(
        "usage: sortbench [options]\n"
        "  --n N            number of elements (repeatable; default 100000). Below 100,000 the timed\n"
        "                   region sorts ceil(100000/N) copies and reports the time per sort; above,\n"
        "                   the repetitions are scaled down in proportion (never below 3)\n"
        "  --reps R         timed repetitions per cell at n <= 100000, medians reported (default 21)\n"
        "  --rounds K       run the whole matrix K times, keep the lowest median per cell (default 2)\n"
        "  --seed S         RNG seed (default 20260912)\n"
        "  --type NAME      element type: int32 (default), double, int64, string (repeatable)\n"
        "  --all-types      benchmark every element type\n"
        "  --algo NAME      restrict to one algorithm (repeatable)\n"
        "  --dataset NAME   restrict to one dataset (repeatable; default: the 5 primary ones)\n"
        "  --all-datasets   use every dataset\n"
        "  --pin CPU        pin the measuring thread to CPU (default: auto, -1 disables)\n"
        "  --counts-only    no timing: one counted run per cell, only the deterministic columns\n"
        "                   are filled (results/counts.csv is made this way)\n"
        "  --timing-only    no counted run: only the timed columns are filled\n"
        "  --id ID          file id written into the stamp (default: <os>-<arch>-<compiler>)\n"
        "  --host TEXT      description of the machine written into the stamp\n"
        "  --print-id       print the default id and exit\n"
        "  --print-code     print the code fingerprint of the source tree and exit\n"
        "  --csv FILE       write results as CSV (with it, <name>.meta.json is written too)\n"
        "  --md FILE        write results as Markdown\n"
        "  --no-fork        run in-process instead of one child per cell\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);

    // ---- child mode ----
    if (args.size() >= 9 && args[0] == "--child") {
        const std::string line = run_child(args[1], args[2], args[3], std::strtoull(args[4].c_str(), nullptr, 10),
                                           std::strtoull(args[5].c_str(), nullptr, 10), std::atoi(args[6].c_str()),
                                           std::atoi(args[7].c_str()), args[8] == "1");
        std::puts(line.c_str());
        return line.rfind("ok=1", 0) == 0 ? 0 : 1;
    }

    std::vector<size_t> sizes;
    int      reps    = 21;
    int      rounds  = 2;
    uint64_t seed    = 20260912;
    int      pin     = default_pin_cpu();
    bool     no_fork = false;
    bool     all_datasets = false;
    bool     all_types = false;
    bool     counts_only = false;
    bool     timing_only = false;
    std::string csv_path, md_path, id, host;
    std::vector<std::string> algos, datasets, types;

    for (size_t i = 0; i < args.size(); ++i) {
        auto need = [&](const char* opt) -> std::string {
            if (i + 1 >= args.size()) { std::fprintf(stderr, "%s needs a value\n", opt); std::exit(2); }
            return args[++i];
        };
        const std::string& a = args[i];
        if      (a == "--n")        sizes.push_back(std::strtoull(need("--n").c_str(), nullptr, 10));
        else if (a == "--reps")     reps = std::atoi(need("--reps").c_str());
        else if (a == "--rounds")   rounds = std::atoi(need("--rounds").c_str());
        else if (a == "--seed")     seed = std::strtoull(need("--seed").c_str(), nullptr, 10);
        else if (a == "--type")     types.push_back(need("--type"));
        else if (a == "--algo")     algos.push_back(need("--algo"));
        else if (a == "--dataset")  datasets.push_back(need("--dataset"));
        else if (a == "--pin")      pin = std::atoi(need("--pin").c_str());
        else if (a == "--csv")      csv_path = need("--csv");
        else if (a == "--md")       md_path = need("--md");
        else if (a == "--id")       id = need("--id");
        else if (a == "--host")     host = need("--host");
        else if (a == "--no-fork")  no_fork = true;
        else if (a == "--all-datasets") all_datasets = true;
        else if (a == "--all-types") all_types = true;
        else if (a == "--counts-only") counts_only = true;
        else if (a == "--timing-only") timing_only = true;
        else if (a == "--print-id") { std::puts(RunStamp::now().default_id().c_str()); return 0; }
        else if (a == "--print-code") { std::puts(code_fingerprint().c_str()); return 0; }
        else if (a == "--help" || a == "-h") { print_usage(); return 0; }
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); print_usage(); return 2; }
    }
    if (counts_only && timing_only) { std::fprintf(stderr, "--counts-only and --timing-only exclude each other\n"); return 2; }
    if (sizes.empty()) sizes.push_back(kReferenceN);
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
    if (reps < 1) reps = 1;
    if (rounds < 1) rounds = 1;
    if (counts_only) { reps = 0; rounds = 1; }
    const bool counted = !timing_only;
    if (types.empty()) {
        if (all_types) for (const char* t : kTypeNames) types.push_back(t);
        else types.push_back("int32");
    }
    for (const auto& t : types) if (!type_exists(t)) { std::fprintf(stderr, "unknown type %s\n", t.c_str()); return 2; }
    if (datasets.empty()) {
        const size_t count = all_datasets ? std::size(kDatasets) : kDefaultDatasetCount;
        for (size_t i = 0; i < count; ++i) datasets.push_back(kDatasets[i].name);
    }
    for (const auto& d : datasets) if (!dataset_exists(d)) { std::fprintf(stderr, "unknown dataset %s\n", d.c_str()); return 2; }

    const std::string exe = self_path(argv[0]);
    const RunStamp stamp = RunStamp::now();
    if (id.empty()) id = stamp.default_id();
    std::vector<Row> rows;

    std::string sizes_txt;
    for (size_t n : sizes) sizes_txt += (sizes_txt.empty() ? "" : ", ") + fmt_int(n);
    std::printf("sortbench: n = %s, reps=%d rounds=%d seed=%llu, %zu type(s) x %zu datasets%s\n", sizes_txt.c_str(), reps, rounds,
                static_cast<unsigned long long>(seed), types.size(), datasets.size(),
                counts_only ? " (counts only)" : timing_only ? " (timing only)" : "");
    std::printf("stamp: %s, id %s%s\n", stamp.summary().c_str(), id.c_str(), host.empty() ? "" : (", " + host).c_str());
    std::fflush(stdout);

    std::map<std::string, std::vector<std::string>> algos_by_type;
    for (const auto& type : types) {
        const TypeInfo ti = describe_type(type);
        std::vector<std::string> selected;
        if (!algos.empty()) {
            for (const auto& a : algos)
                if (std::find(ti.algo_names.begin(), ti.algo_names.end(), a) != ti.algo_names.end()) selected.push_back(a);
                else std::fprintf(stderr, "note: algorithm %s not available for type %s\n", a.c_str(), type.c_str());
        } else {
            selected = ti.algo_names;
        }
        algos_by_type[type] = selected;
    }

    // Bursty interference (another process, the hypervisor) can only inflate a
    // median, so the matrix is run `rounds` times and the lowest median per
    // cell is kept.
    for (int round = 0; round < rounds; ++round) {
        for (size_t n : sizes) {
            const int reps_n = reps_for(n, reps);
            for (const auto& type : types) {
                const std::vector<std::string>& selected = algos_by_type[type];
                for (const auto& ds : datasets) {
                    if (!dataset_applies(ds, type)) continue;
                    for (const auto& al : selected) {
                        Row row;
                        row.n       = n;
                        row.type    = type;
                        row.dataset = ds;
                        row.algo    = al;
                        std::string line;
                        if (no_fork) {
                            line = run_child(type, al, ds, n, seed, reps_n, pin, counted);
                        } else {
                            std::ostringstream cmd;
                            cmd << '"' << exe << "\" --child " << type << ' ' << al << ' ' << ds << ' ' << n << ' ' << seed
                                << ' ' << reps_n << ' ' << pin << ' ' << (counted ? 1 : 0);
                            line = run_process(cmd.str());
                        }
                        row.f  = parse_fields(line);
                        row.ok = row.f["ok"] == "1";
                        if (!row.ok) row.error = row.f.count("error") ? row.f["error"] : ("no output: " + line);
                        auto existing = std::find_if(rows.begin(), rows.end(), [&](const Row& r) {
                            return r.n == n && r.type == type && r.dataset == ds && r.algo == al;
                        });
                        const bool better = existing == rows.end() ||
                                            (row.ok && (!existing->ok || row.d("wall_ns_med") < existing->d("wall_ns_med")));
                        if (existing == rows.end()) rows.push_back(row);
                        else if (better) *existing = row;
                        std::printf("  r%d n=%-8zu %-7s %-13s %-16s %s%s\n", round + 1, n, type.c_str(), ds.c_str(), al.c_str(),
                                    !row.ok       ? ("FAILED: " + row.error).c_str()
                                    : counts_only ? ("ok  " + fmt_bytes(row.u("traffic_bytes")) + " traffic, " + fmt_int(row.u("compares")) + " compares").c_str()
                                                  : ("ok  " + fmt_num(row.d("wall_ns_med") / 1e6, 3) + " ms").c_str(),
                                    round > 0 && better ? "  (kept)" : "");
                        std::fflush(stdout);
                    }
                }
            }
        }
    }

    // ---- console tables + markdown ----
    bool any_instr = false, any_cycles = false;
    for (const auto& r : rows) { any_instr |= r.u("has_instr") == 1; any_cycles |= r.u("has_cycles") == 1; }
    const std::string backend = rows.empty() || !rows[0].f.count("hw") ? "" : rows[0].f.at("hw");
    if (!rows.empty() && rows[0].ok)
        std::printf("\nmeasurement backend: %s, pinned cpu: %s, meter overhead subtracted: %.1f us\n",
                    backend.c_str(), rows[0].f.at("pinned").c_str(), rows[0].d("meter_overhead_ns") / 1e3);

    std::ostringstream md;
    md << "# Sorting benchmark results\n\n"
       << "- " << stamp.summary() << (host.empty() ? "" : ", " + host) << "\n"
       << "- n = " << sizes_txt << " elements; every element carries its original index for verification\n"
       << "- " << reps << " timed repetitions per cell at n = 100,000 (scaled down above, copies batched below), medians reported, lowest median of "
       << rounds << " round(s) kept; every run is verified against std::stable_sort\n"
       << "- one fresh process per (size, type, dataset, algorithm) cell" << (no_fork ? " (disabled: --no-fork)" : "") << "\n"
       << "- backend: " << backend << " (meter overhead of " << (rows.empty() ? 0.0 : rows[0].d("meter_overhead_ns") / 1e3)
       << " us subtracted from every sample)\n"
       << "- reads/writes = element loads/stores on the main array and any scratch buffer (counted on a separate instrumented run; the upstream code through an element wrapper)\n"
       << "- aux peak = exact scratch heap memory requested by the algorithm; RSS delta = growth of the process peak resident set during the sort\n"
       << "- the second table per dataset holds the deterministic numbers: identical on every machine and platform. traffic = element bytes read and written + histogram/index table bytes + string key bytes; "
          "L1/L2 = misses of a fixed cache model (64-byte lines, LRU, 32 KiB 8-way / 1 MiB 16-way) on allocation-relative addresses; depth = deepest recursion (timsort: run stack); route = brainsort's route for the whole array\n\n";

    for (size_t n : sizes) {
        for (const auto& type : types) {
            const TypeInfo ti = describe_type(type);
            std::printf("\n#### n = %s, type: %s (%zu-byte elements) ####\n", fmt_int(n).c_str(), type.c_str(), ti.elem_bytes);
            md << "# n = " << fmt_int(n) << ", type: " << type << " (" << ti.elem_bytes << "-byte elements)\n\n";
            for (const auto& ds : datasets) {
                if (!dataset_applies(ds, type)) continue;
                std::printf("\n== %s / %s ==\n", type.c_str(), ds.c_str());
                std::printf("%-16s %10s %10s", "algorithm", "wall ms", "cpu ms");
                if (any_instr)  std::printf(" %12s", "instr (M)");
                if (any_cycles) std::printf(" %12s", "cycles (M)");
                std::printf(" %14s %14s %14s %11s %11s\n", "reads", "writes", "compares", "aux peak", "rss delta");

                md << "## " << type << " / " << ds << "\n\n";
                md << "| algorithm | stable | wall ms | cpu ms |";
                if (any_instr)  md << " instructions (M) |";
                if (any_cycles) md << " cycles (M) |";
                md << " array reads | array writes | compares | aux peak | RSS delta |\n";
                md << "|---|---|---:|---:|";
                if (any_instr)  md << "---:|";
                if (any_cycles) md << "---:|";
                md << "---:|---:|---:|---:|---:|\n";

                for (const auto& r : rows) {
                    if (r.n != n || r.type != type || r.dataset != ds) continue;
                    if (!r.ok) {
                        std::printf("%-16s FAILED: %s\n", r.algo.c_str(), r.error.c_str());
                        md << "| " << r.algo << " | | FAILED: " << r.error << " |\n";
                        continue;
                    }
                    const bool ca = r.u("counts_accesses") == 1;
                    const std::string reads  = ca ? fmt_int(r.u("reads"))  : "n/a";
                    const std::string writes = ca ? fmt_int(r.u("writes")) : "n/a";
                    const std::string aux    = ca ? fmt_bytes(r.u("aux_peak_bytes")) : "n/a";
                    const std::string comps  = counted ? fmt_int(r.u("compares")) : "n/a";
                    const std::string cpu_ms = r.u("has_cpu") ? fmt_num(r.d("cpu_ns_med") / 1e6, 3) : "n/a";
                    std::printf("%-16s %10s %10s", r.algo.c_str(), fmt_num(r.d("wall_ns_med") / 1e6, 3).c_str(), cpu_ms.c_str());
                    if (any_instr)  std::printf(" %12s", r.u("has_instr") ? fmt_num(r.d("instr_med") / 1e6, 2).c_str() : "n/a");
                    if (any_cycles) std::printf(" %12s", r.u("has_cycles") ? fmt_num(r.d("cycles_med") / 1e6, 2).c_str() : "n/a");
                    std::printf(" %14s %14s %14s %11s %11s\n", reads.c_str(), writes.c_str(),
                                comps.c_str(), aux.c_str(), fmt_bytes(r.u("rss_delta_bytes")).c_str());

                    md << "| " << r.algo << " | " << (algo_is_stable(type, r.algo) ? "yes" : "no") << " | "
                       << fmt_num(r.d("wall_ns_med") / 1e6, 3) << " | " << cpu_ms << " |";
                    if (any_instr)  md << ' ' << (r.u("has_instr") ? fmt_num(r.d("instr_med") / 1e6, 2) : "n/a") << " |";
                    if (any_cycles) md << ' ' << (r.u("has_cycles") ? fmt_num(r.d("cycles_med") / 1e6, 2) : "n/a") << " |";
                    md << ' ' << reads << " | " << writes << " | " << comps << " | " << aux << " | "
                       << fmt_bytes(r.u("rss_delta_bytes")) << " |\n";
                }
                if (counted) {
                    md << "\n| algorithm | traffic | element reads | element writes | table reads | table writes | key bytes | compares | compare flips | L1 misses | L2 misses | depth | route |\n"
                       << "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n";
                    for (const auto& r : rows) {
                        if (r.n != n || r.type != type || r.dataset != ds || !r.ok) continue;
                        md << "| " << r.algo << " | " << fmt_bytes(r.u("traffic_bytes")) << " | " << fmt_int(r.u("reads")) << " | " << fmt_int(r.u("writes")) << " | "
                           << fmt_int(r.u("table_reads")) << " | " << fmt_int(r.u("table_writes")) << " | " << fmt_bytes(r.u("key_bytes")) << " | "
                           << fmt_int(r.u("compares")) << " | " << fmt_int(r.u("cmp_flips")) << " | " << fmt_int(r.u("l1_misses")) << " | " << fmt_int(r.u("l2_misses")) << " | "
                           << r.u("max_depth") << " | " << (r.s("route") == "-" ? "" : r.s("route")) << " |\n";
                    }
                }
                md << "\n";
            }
        }
    }

    if (!csv_path.empty()) {
        std::ofstream csv(csv_path);
        // Timing columns first (empty with --counts-only), then every
        // deterministic column from counted_run.hpp (empty with --timing-only).
        csv << "type,dataset,algorithm,stable,n,seed,reps,batch,wall_ns_med,wall_ns_min,cpu_ns_med,instr_med,cycles_med,"
               "counts_accesses,rss_delta_bytes,backend,ok,error";
        for (const char* c : kDetColumns) csv << ',' << c;
        csv << '\n';
        for (const auto& r : rows) {
            const bool timed = r.u("reps") > 0;
            csv << r.type << ',' << r.dataset << ',' << r.algo << ',' << (algo_is_stable(r.type, r.algo) ? 1 : 0) << ','
                << r.n << ',' << seed << ',' << r.u("reps") << ',' << std::max<unsigned long long>(1, r.u("batch")) << ','
                << (timed ? std::to_string(r.u("wall_ns_med")) : "") << ',' << (timed ? std::to_string(r.u("wall_ns_min")) : "") << ','
                << (r.u("has_cpu") ? std::to_string(r.u("cpu_ns_med")) : "") << ','
                << (r.u("has_instr") ? std::to_string(r.u("instr_med")) : "") << ','
                << (r.u("has_cycles") ? std::to_string(r.u("cycles_med")) : "") << ','
                << r.u("counts_accesses") << ',' << (timed ? std::to_string(r.u("rss_delta_bytes")) : "") << ','
                << (r.f.count("hw") ? r.f.at("hw") : "") << ',' << (r.ok ? 1 : 0) << ',' << r.error;
            for (const char* c : kDetColumns) csv << ',' << r.s(c);
            csv << '\n';
        }
        std::printf("\nwrote %s\n", csv_path.c_str());
        if (!counts_only) {
            // The stamp of this timing run. The deterministic file needs none:
            // it is the same for every tree that does not change the numbers.
            const std::string mp = meta_path_for(csv_path);
            std::ofstream meta(mp);
            std::string sizes_json;
            for (size_t n : sizes) sizes_json += (sizes_json.empty() ? "" : ", ") + std::to_string(n);
            meta << "{\n"
                 << "  \"id\": " << json_str(id) << ",\n"
                 << "  \"host\": " << json_str(host) << ",\n"
                 << stamp.json_fields() << ",\n"
                 << "  \"backend\": " << json_str(backend) << ",\n"
                 << "  \"sizes\": [" << sizes_json << "],\n"
                 << "  \"reps\": " << reps << ",\n"
                 << "  \"rounds\": " << rounds << ",\n"
                 << "  \"seed\": " << seed << ",\n"
                 << "  \"pinned\": " << (rows.empty() || !rows[0].f.count("pinned") ? -1 : std::atoi(rows[0].f.at("pinned").c_str())) << ",\n"
                 << "  \"cells\": " << rows.size() << ",\n"
                 << "  \"timing_only\": " << (timing_only ? "true" : "false") << "\n"
                 << "}\n";
            std::printf("wrote %s\n", mp.c_str());
        }
    }
    if (!md_path.empty()) {
        std::ofstream out(md_path);
        out << md.str();
        std::printf("wrote %s\n", md_path.c_str());
    }

    bool all_ok = true;
    for (const auto& r : rows) all_ok &= r.ok;
    return all_ok ? 0 : 1;
}

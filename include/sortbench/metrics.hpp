// Platform measurement: wall time, thread CPU time, retired instructions,
// CPU cycles and peak resident memory.
//
//   Linux   : perf_event_open (PERF_COUNT_HW_INSTRUCTIONS / CPU_CYCLES, user
//             space only), clock_gettime, getrusage(ru_maxrss).
//   Windows : QueryPerformanceCounter, GetThreadTimes (coarse, ~1 ms),
//             QueryThreadCycleTime for cycles; no instruction counter is
//             available from user mode, so instructions are reported as n/a.
#pragma once
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>

#if defined(__linux__)
#include <linux/perf_event.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#endif

namespace sb {

struct Measurement {
    double   wall_ns         = 0;
    double   cpu_ns          = 0;
    uint64_t instructions    = 0;
    uint64_t cycles          = 0;
    bool     has_cpu          = false;   // false where the OS clock is too coarse (Windows: ~16 ms ticks)
    bool     has_instructions = false;
    bool     has_cycles       = false;
};

class Meter {
public:
    Meter() {
#if defined(__linux__)
        fd_instr_  = open_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
        fd_cycles_ = open_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
#endif
    }
    ~Meter() {
#if defined(__linux__)
        if (fd_instr_ >= 0)  close(fd_instr_);
        if (fd_cycles_ >= 0) close(fd_cycles_);
#endif
    }
    Meter(const Meter&) = delete;
    Meter& operator=(const Meter&) = delete;

    const char* backend() const {
#if defined(__linux__)
        return fd_instr_ >= 0 ? "linux-perf" : "linux-clock-only";
#elif defined(_WIN32)
        return "windows-qpc-cycletime";
#else
        return "portable-clock-only";
#endif
    }

    void start() {
        cpu_start_  = thread_cpu_ns();
        wall_start_ = std::chrono::steady_clock::now();
#if defined(__linux__)
        if (fd_instr_ >= 0)  { ioctl(fd_instr_, PERF_EVENT_IOC_RESET, 0);  ioctl(fd_instr_, PERF_EVENT_IOC_ENABLE, 0); }
        if (fd_cycles_ >= 0) { ioctl(fd_cycles_, PERF_EVENT_IOC_RESET, 0); ioctl(fd_cycles_, PERF_EVENT_IOC_ENABLE, 0); }
#elif defined(_WIN32)
        QueryThreadCycleTime(GetCurrentThread(), &cycles_start_);
#endif
    }

    Measurement stop() {
        Measurement m;
#if defined(__linux__)
        if (fd_instr_ >= 0)  { ioctl(fd_instr_, PERF_EVENT_IOC_DISABLE, 0);  m.has_instructions = read_counter(fd_instr_, m.instructions); }
        if (fd_cycles_ >= 0) { ioctl(fd_cycles_, PERF_EVENT_IOC_DISABLE, 0); m.has_cycles = read_counter(fd_cycles_, m.cycles); }
#elif defined(_WIN32)
        ULONG64 c = 0;
        QueryThreadCycleTime(GetCurrentThread(), &c);
        m.cycles = c - cycles_start_;
        m.has_cycles = true;
#endif
        const auto wall_end = std::chrono::steady_clock::now();
        m.wall_ns = std::chrono::duration<double, std::nano>(wall_end - wall_start_).count();
        m.cpu_ns  = thread_cpu_ns() - cpu_start_;
#if defined(__linux__)
        m.has_cpu = true;
#endif
        return m;
    }

private:
    std::chrono::steady_clock::time_point wall_start_;
    double cpu_start_ = 0;
#if defined(__linux__)
    int fd_instr_  = -1;
    int fd_cycles_ = -1;

    static int open_counter(uint32_t type, uint64_t config) {
        perf_event_attr attr;
        std::memset(&attr, 0, sizeof attr);
        attr.type           = type;
        attr.size           = sizeof attr;
        attr.config         = config;
        attr.disabled       = 1;
        attr.exclude_kernel = 1;
        attr.exclude_hv     = 1;
        return static_cast<int>(syscall(__NR_perf_event_open, &attr, 0, -1, -1, 0));
    }
    static bool read_counter(int fd, uint64_t& out) {
        uint64_t v = 0;
        return read(fd, &v, sizeof v) == static_cast<ssize_t>(sizeof v) && (out = v, true);
    }
#elif defined(_WIN32)
    ULONG64 cycles_start_ = 0;
#endif

    static double thread_cpu_ns() {
#if defined(__linux__)
        timespec ts;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
        return ts.tv_sec * 1e9 + ts.tv_nsec;
#elif defined(_WIN32)
        FILETIME c, e, k, u;
        GetThreadTimes(GetCurrentThread(), &c, &e, &k, &u);
        auto to64 = [](FILETIME f) { return (static_cast<uint64_t>(f.dwHighDateTime) << 32) | f.dwLowDateTime; };
        return static_cast<double>(to64(k) + to64(u)) * 100.0;  // 100 ns units
#else
        return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count());
#endif
    }
};

// Peak resident set size of this process so far (monotonic).
inline uint64_t peak_rss_bytes() {
#if defined(__linux__)
    rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return static_cast<uint64_t>(ru.ru_maxrss) * 1024;
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    pmc.cb = sizeof pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) return pmc.PeakWorkingSetSize;
    return 0;
#else
    return 0;
#endif
}

// Pin the calling thread to one CPU to reduce migration noise. Returns the
// CPU chosen or -1 if pinning was not possible.
inline int pin_to_cpu(int cpu) {
    if (cpu < 0) return -1;
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return sched_setaffinity(0, sizeof set, &set) == 0 ? cpu : -1;
#elif defined(_WIN32)
    if (cpu >= 64) return -1;
    return SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << cpu) ? cpu : -1;
#else
    return -1;
#endif
}

inline int default_pin_cpu() {
    unsigned n = std::thread::hardware_concurrency();
    return n >= 4 ? static_cast<int>(n - 2) : -1;  // a core the OS is unlikely to favour
}

}  // namespace sb

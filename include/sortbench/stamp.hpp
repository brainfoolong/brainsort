// The run stamp: which code was measured, when, on what.
//
// Every timing file the benchmark writes carries this stamp so a number can
// be held against the code it describes. The decisive field is the code
// fingerprint: a hash over every file under the measured paths (include/,
// src/, third_party/, CMakeLists.txt) of the tree the binary was built from.
// scripts/website.py recomputes it over the current tree and treats a timing
// file whose fingerprint differs as stale. Unlike a commit hash this works
// for uncommitted trees, shallow clones and rewritten history.
//
// The fingerprint is FNV-1a (64-bit) over, for every regular file under the
// measured paths in byte order of its slash-separated relative path: the
// path, a zero byte, the file's bytes with every '\r' removed, a zero byte.
// scripts/website.py implements the same function; the two must agree.
#pragma once
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace sb {

inline const char* const kMeasuredPaths[] = {"include", "src", "third_party", "CMakeLists.txt"};

inline std::string trim(std::string v) {
    while (!v.empty() && (v.back() == ' ' || v.back() == '\n' || v.back() == '\r' || v.back() == '\t')) v.pop_back();
    size_t i = 0;
    while (i < v.size() && (v[i] == ' ' || v[i] == '\t')) ++i;
    return v.substr(i);
}

// Runs a command and returns its trimmed stdout, or "" when it fails.
inline std::string shell_line(const std::string& cmd) {
#if defined(_WIN32)
    FILE* p = _popen((cmd + " 2>nul").c_str(), "r");
#else
    FILE* p = popen((cmd + " 2>/dev/null").c_str(), "r");
#endif
    if (!p) return "";
    std::string out;
    char buf[512];
    while (std::fgets(buf, sizeof buf, p)) out += buf;
#if defined(_WIN32)
    const int rc = _pclose(p);
#else
    const int rc = pclose(p);
#endif
    return rc == 0 ? trim(out) : "";
}

inline std::string os_name() {
#if defined(_WIN32)
    return "Windows";
#elif defined(__APPLE__)
    return "macOS";
#elif defined(__linux__)
    std::ifstream v("/proc/version");
    std::string line;
    if (std::getline(v, line) && (line.find("microsoft") != std::string::npos || line.find("WSL") != std::string::npos))
        return "Linux (WSL2)";
    return "Linux";
#else
    return "unknown";
#endif
}

inline std::string arch_name() {
#if defined(__x86_64__) || defined(_M_X64)
    return "x86-64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "ARM64";
#elif defined(__i386__) || defined(_M_IX86)
    return "x86 (32-bit)";
#elif defined(__arm__) || defined(_M_ARM)
    return "ARM (32-bit)";
#else
    return "unknown";
#endif
}

inline std::string compiler_name() {
#if defined(__apple_build_version__)
    return std::string("Apple Clang ") + __clang_version__;
#elif defined(__clang__)
    return std::string("Clang ") + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__) + "." + std::to_string(__clang_patchlevel__);
#elif defined(_MSC_VER)
    return std::string("MSVC ") + std::to_string(_MSC_VER / 100) + "." + std::to_string(_MSC_VER % 100);
#elif defined(__GNUC__)
    return std::string("GCC ") + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." + std::to_string(__GNUC_PATCHLEVEL__);
#else
    return "unknown compiler";
#endif
}

inline std::string cpu_name() {
#if defined(_WIN32)
    char  buf[256];
    DWORD size = sizeof buf;
    if (RegGetValueA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", "ProcessorNameString",
                     RRF_RT_REG_SZ, nullptr, buf, &size) == ERROR_SUCCESS)
        return trim(buf);
    return "";
#elif defined(__APPLE__)
    return shell_line("sysctl -n machdep.cpu.brand_string");
#else
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    while (std::getline(in, line))
        if (line.rfind("model name", 0) == 0) {
            const size_t c = line.find(':');
            return c == std::string::npos ? "" : trim(line.substr(c + 1));
        }
    // ARM kernels do not print a model name; lscpu knows one.
    const std::string l = shell_line("lscpu | grep -m1 'Model name'");
    const size_t c = l.find(':');
    return c == std::string::npos ? "" : trim(l.substr(c + 1));
#endif
}

inline std::string utc_now() {
    char buf[32];
    std::time_t t = std::time(nullptr);
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
    return buf;
}

// The source root: what the build was configured from (SB_SOURCE_DIR), else
// the working directory.
inline std::filesystem::path source_root() {
#ifdef SB_SOURCE_DIR
    return std::filesystem::path(SB_SOURCE_DIR);
#else
    return std::filesystem::current_path();
#endif
}

inline std::string code_fingerprint(const std::filesystem::path& root = source_root()) {
    namespace fs = std::filesystem;
    std::vector<std::pair<std::string, fs::path>> files;
    std::error_code ec;
    for (const char* p : kMeasuredPaths) {
        const fs::path base = root / p;
        if (fs::is_regular_file(base, ec)) files.emplace_back(p, base);
        else if (fs::is_directory(base, ec))
            for (fs::recursive_directory_iterator it(base, ec), end; it != end && !ec; it.increment(ec))
                if (it->is_regular_file(ec))
                    files.emplace_back(fs::relative(it->path(), root, ec).generic_string(), it->path());
    }
    if (files.empty()) return "";
    std::sort(files.begin(), files.end());
    uint64_t h = 0xcbf29ce484222325ull;
    auto mix = [&](unsigned char c) { h ^= c; h *= 0x100000001b3ull; };
    for (const auto& [rel, path] : files) {
        for (unsigned char c : rel) mix(c);
        mix(0);
        std::ifstream in(path, std::ios::binary);
        char buf[65536];
        while (in.read(buf, sizeof buf) || in.gcount() > 0) {
            for (std::streamsize i = 0; i < in.gcount(); ++i)
                if (buf[i] != '\r') mix(static_cast<unsigned char>(buf[i]));
        }
        mix(0);
    }
    char out[17];
    std::snprintf(out, sizeof out, "%016llx", static_cast<unsigned long long>(h));
    return out;
}

inline std::string json_str(const std::string& v) {
    std::string o = "\"";
    for (char c : v) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (static_cast<unsigned char>(c) < 0x20) o += ' ';
        else o += c;
    }
    return o + "\"";
}

struct RunStamp {
    std::string os, arch, compiler, cpu, commit, measured, code;

    static RunStamp now() {
        RunStamp st;
        st.os       = os_name();
        st.arch     = arch_name();
        st.compiler = compiler_name() + " -O2";
        st.cpu      = cpu_name();
        st.commit   = shell_line("git rev-parse --short=12 HEAD");
        st.measured = utc_now();
        st.code     = code_fingerprint();
        return st;
    }

    // A default file id such as linux-x86-64-gcc13: lower case, one token per field.
    std::string default_id() const {
        std::string os_tok = os.substr(0, os.find(' '));
        std::string cc = compiler.substr(0, compiler.find(' '));
        std::string major;
        for (size_t i = compiler.find(' ') + 1; i < compiler.size() && compiler[i] != '.' && compiler[i] != ' '; ++i) major += compiler[i];
        if (cc == "Apple") { cc = "appleclang"; major.clear(); }
        std::string id = os_tok + "-" + arch + "-" + cc + major;
        for (char& c : id) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (c == ' ' || c == '(' || c == ')') c = '-';
        }
        std::string out;
        for (char c : id) if (!(c == '-' && !out.empty() && out.back() == '-')) out += c;
        while (!out.empty() && out.back() == '-') out.pop_back();
        return out;
    }

    // One line for humans, the same fields the JSON carries.
    std::string summary() const {
        return os + " " + arch + ", " + compiler + ", " + (cpu.empty() ? "unknown CPU" : cpu) + ", code " +
               (code.empty() ? "unknown" : code) + (commit.empty() ? "" : ", commit " + commit) + ", " + measured;
    }

    std::string json_fields() const {
        return "  \"os\": " + json_str(os) + ",\n  \"arch\": " + json_str(arch) + ",\n  \"compiler\": " + json_str(compiler) +
               ",\n  \"cpu\": " + json_str(cpu) + ",\n  \"measured\": " + json_str(measured) + ",\n  \"commit\": " + json_str(commit) +
               ",\n  \"code\": " + json_str(code) + ",\n  \"measured_paths\": \"include src third_party CMakeLists.txt\"";
    }
};

}  // namespace sb

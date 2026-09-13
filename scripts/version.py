#!/usr/bin/env python3
"""The one version of both libraries, set or checked in every place at once.

    python3 scripts/version.py --check           every location agrees
    python3 scripts/version.py --check --tag vX  ... and the tag matches
    python3 scripts/version.py 0.5.0             set 0.5.0 everywhere
    python3 scripts/version.py 0.5.0 --force     even if not above the current

The locations: the C++ header (the string and the three numeric macros),
the crate, the two crates that depend on it by version (the benchmark and
the fuzz target), the workspace lock file, the README's version sentence,
the release example in setup.md, and the changelog, whose "Unreleased"
section becomes the version's. Setting a version regenerates the single
header and refreshes the lock file. The check is a CTest and the first
step of the release workflow; it exits 1 and names every location that
disagrees.
"""
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG = ROOT / "include/brainsort/detail/config.hpp"
SINGLE = ROOT / "single_include/brainsort.hpp"
CRATE = ROOT / "rust/brainsort/Cargo.toml"
LOCK = ROOT / "rust/Cargo.lock"
DEPENDENTS = [ROOT / "rust/brainsort-bench/Cargo.toml", ROOT / "rust/brainsort/fuzz/Cargo.toml"]
README = ROOT / "README.md"
SETUP = ROOT / "setup.md"
CHANGELOG = ROOT / "rust/brainsort/CHANGELOG.md"

SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")

# Each location: file, a regex whose group 1 is the version, how many matches
# there must be, and what a replacement writes (the match with the version
# swapped). A location that must appear once but matches more or less is an
# inconsistency in itself.
LOCATIONS = [
    ("C++ header, BRAINSORT_VERSION", CONFIG, re.compile(r'#define BRAINSORT_VERSION "(\d+\.\d+\.\d+)"'), 1),
    ("crate, [package] version", CRATE, re.compile(r'^version = "(\d+\.\d+\.\d+)"$', re.M), 1),
    ("benchmark crate, brainsort dependency", DEPENDENTS[0], re.compile(r'^brainsort = \{ path = "[^"]*", version = "(\d+\.\d+\.\d+)"', re.M), 1),
    ("fuzz crate, brainsort dependency", DEPENDENTS[1], re.compile(r'^brainsort = \{ path = "[^"]*", version = "(\d+\.\d+\.\d+)"', re.M), 1),
    ("workspace lock file, brainsort entry", LOCK, re.compile(r'^name = "brainsort"\nversion = "(\d+\.\d+\.\d+)"$', re.M), 1),
    ("README, the version sentence", README, re.compile(r"MIT licensed, version\s+(\d+\.\d+\.\d+)\."), 1),
    ("setup.md, the tag example", SETUP, re.compile(r"git tag v(\d+\.\d+\.\d+) && git push origin v\d+\.\d+\.\d+"), 1),
    ("single header, the banner", SINGLE, re.compile(r"^// brainsort (\d+\.\d+\.\d+) - single-header distribution\.$", re.M), 1),
    ("single header, BRAINSORT_VERSION", SINGLE, re.compile(r'#define BRAINSORT_VERSION "(\d+\.\d+\.\d+)"'), 1),
]
MACROS = re.compile(r"#define BRAINSORT_VERSION_MAJOR (\d+)\n#define BRAINSORT_VERSION_MINOR (\d+)\n#define BRAINSORT_VERSION_PATCH (\d+)\n")


def read(p):
    return p.read_text(encoding="utf-8")


def write(p, text):
    p.write_text(text, encoding="utf-8", newline="\n")


def found():
    """Every location's version, as (name, version or None, count)."""
    out = []
    for name, path, rx, count in LOCATIONS:
        ms = rx.findall(read(path))
        out.append((name, ms[0] if len(ms) == count else None, len(ms)))
    for path in (CONFIG, SINGLE):
        m = MACROS.search(read(path))
        which = "C++ header" if path is CONFIG else "single header"
        out.append((f"{which}, the numeric macros", ".".join(m.groups()) if m else None, 1 if m else 0))
    return out


def current():
    """The version of the C++ header, the reference the others are held to."""
    m = LOCATIONS[0][2].search(read(CONFIG))
    if not m:
        sys.exit(f"{CONFIG}: no BRAINSORT_VERSION")
    return m.group(1)


def check(tag=None):
    v = current()
    bad = []
    for name, got, count in found():
        if got != v:
            bad.append(f"  {name}: {got if count == 1 else f'{count} matches'} (expected {v})")
    if f"\n## {v}\n" not in read(CHANGELOG):
        bad.append(f"  changelog: no section '## {v}'")
    if tag is not None and tag != f"v{v}":
        bad.append(f"  tag {tag} is not v{v}")
    if bad:
        print(f"version {v}: inconsistent", file=sys.stderr)
        print("\n".join(bad), file=sys.stderr)
        return 1
    print(f"version {v}: every location agrees" + (f", tag {tag} matches" if tag else ""))
    return 0


def set_version(new, force):
    m = SEMVER.match(new)
    if not m:
        sys.exit(f"{new}: not MAJOR.MINOR.PATCH")
    old = current()
    if not force and tuple(map(int, m.groups())) <= tuple(map(int, old.split("."))):
        sys.exit(f"{new} is not above the current {old} (--force to set it anyway)")
    # the sources; the single header and the lock file are regenerated below
    for name, path, rx, count in LOCATIONS:
        if path is SINGLE or path is LOCK:
            continue
        text = read(path)
        ms = list(rx.finditer(text))
        if len(ms) != count:
            sys.exit(f"{name}: {len(ms)} matches, expected {count}; fix the file by hand first")
        for mm in reversed(ms):
            text = text[: mm.start(1)] + new + text[mm.end(1):]
        write(path, text)
    text = read(CONFIG)
    text, n = MACROS.subn(f"#define BRAINSORT_VERSION_MAJOR {m.group(1)}\n#define BRAINSORT_VERSION_MINOR {m.group(2)}\n#define BRAINSORT_VERSION_PATCH {m.group(3)}\n", text)
    if n != 1:
        sys.exit(f"{CONFIG}: the numeric macros were not found together")
    write(CONFIG, text)
    # the changelog: the unreleased section becomes this version's
    log = read(CHANGELOG)
    if "\n## Unreleased\n" in log:
        write(CHANGELOG, log.replace("\n## Unreleased\n", f"\n## {new}\n", 1))
    elif f"\n## {new}\n" not in log:
        print(f"note: {CHANGELOG.name} has no '## Unreleased' section; add '## {new}' by hand", file=sys.stderr)
    # the generated files
    subprocess.run([sys.executable, str(ROOT / "scripts/amalgamate.py")], check=True)
    lock = subprocess.run(["cargo", "update", "--workspace", "--offline", "--quiet"], cwd=ROOT / "rust")
    if lock.returncode != 0:
        subprocess.run(["cargo", "update", "--workspace", "--quiet"], cwd=ROOT / "rust", check=True)
    print(f"{old} -> {new}")
    sys.stdout.flush()
    return check()


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0
    if argv[0] == "--check":
        tag = None
        if len(argv) == 3 and argv[1] == "--tag":
            tag = argv[2]
        elif len(argv) != 1:
            sys.exit("usage: version.py --check [--tag vX.Y.Z]")
        return check(tag)
    return set_version(argv[0], "--force" in argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

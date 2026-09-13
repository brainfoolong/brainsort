#!/bin/sh
# The timing run of one machine: every algorithm on every key type, dataset
# and size, then the public-API benchmark. Writes results/<id>.csv,
# results/<id>.meta.json and results/<id>.api.md; scripts/website.py turns
# every such set in results/ into the website.
#
#   sh scripts/bench.sh [sortbench options]      e.g. --reps 11, --pin 3
#   BENCH_SIZES="1000 10000 100000 1000000 10000000" BENCH_API_MAX_N=10000000 sh scripts/bench.sh
#                                                ten million elements as well (an option, about an hour longer)
#
# Environment:
#   BENCH_ID      file id (default: what the binary reports, e.g. linux-x86-64-gcc13)
#   BENCH_HOST    a description of the machine for the stamp (default: none)
#   BENCH_SIZES   element counts, space separated (default "1000 10000 100000 1000000")
#   BENCH_API_MAX_N  largest size of the API benchmark (default 1000000)
#   BUILD_DIR     build directory (default build-linux; it is built first, so a
#                 stale binary can never carry a fresh code fingerprint)
#   BENCH_NO_SITE set to skip rebuilding the website afterwards
set -e
cd "$(dirname "$0")/.."
BUILD_DIR=${BUILD_DIR:-build-linux}
export BUILD_DIR
sh scripts/build-linux.sh
SB="$BUILD_DIR/sortbench"
API="$BUILD_DIR/brainsort_api_bench"
ID=${BENCH_ID:-$("$SB" --print-id)}
SIZES=""
for n in ${BENCH_SIZES:-1000 10000 100000 1000000}; do SIZES="$SIZES --n $n"; done
mkdir -p results

echo "== timing: results/$ID.csv"
# shellcheck disable=SC2086
"$SB" --all-types --all-algos --all-datasets --timing-only $SIZES \
    --id "$ID" --host "${BENCH_HOST:-}" --csv "results/$ID.csv" "$@"

echo "== public API: results/$ID.api.md"
"$API" --max-n "${BENCH_API_MAX_N:-1000000}" --host "${BENCH_HOST:-}" > "results/$ID.api.md"

if [ -z "${BENCH_NO_SITE:-}" ]; then
    if command -v python3 >/dev/null 2>&1; then python3 scripts/website.py
    else echo "python3 not found; run scripts/website.py later to build the website"; fi
fi

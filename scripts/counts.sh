#!/bin/sh
# The deterministic numbers: one counted run per (size, key type, dataset,
# algorithm) cell, no timing. They are the same on every machine.
#
#   sh scripts/counts.sh                 regenerate results/counts.csv, the golden
#                                        file of the test suite (n = 1,000, 10,000, 100,000)
#   COUNTS_SIZES=1000000 sh scripts/counts.sh   one million elements into
#                                        results/counts-1000000.csv (what CI adds for the website)
#
# Environment: COUNTS_SIZES (space separated), COUNTS_OUT (output file),
# BUILD_DIR (default build-linux; built first).
set -e
cd "$(dirname "$0")/.."
BUILD_DIR=${BUILD_DIR:-build-linux}
export BUILD_DIR
sh scripts/build-linux.sh
SIZES_LIST=${COUNTS_SIZES:-1000 10000 100000}
if [ -n "${COUNTS_OUT:-}" ]; then OUT=$COUNTS_OUT
elif [ "$SIZES_LIST" = "1000 10000 100000" ]; then OUT=results/counts.csv
else OUT="results/counts-$(echo "$SIZES_LIST" | tr ' ' '-').csv"; fi
SIZES=""
for n in $SIZES_LIST; do SIZES="$SIZES --n $n"; done
mkdir -p results
# shellcheck disable=SC2086
"$BUILD_DIR/sortbench" --counts-only --all-types --all-algos --all-datasets $SIZES --csv "$OUT" "$@"

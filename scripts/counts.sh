#!/bin/sh
# The deterministic numbers: one counted run per (size, key type, dataset,
# algorithm) cell, no timing. They are the same on every machine.
#
#   sh scripts/counts.sh                 regenerate results/counts.csv, the golden
#                                        file of the test suite (n = 10 to 1,000,000)
#   COUNTS_SIZES=10000000 sh scripts/counts.sh   ten million elements into
#                                        results/counts-10000000.csv (an option, over an hour)
#
# When cargo is installed, the Rust sorts' counts follow into
# results/rust-counts.csv (or rust-counts-<sizes>.csv): what can be counted
# of the Rust ecosystem as shipped, on the same cells.
#
# Environment: COUNTS_SIZES (space separated), COUNTS_OUT (output file),
# COUNTS_NO_RUST (skip the Rust counts), BUILD_DIR (default build-linux;
# built first).
set -e
cd "$(dirname "$0")/.."
BUILD_DIR=${BUILD_DIR:-build-linux}
export BUILD_DIR
sh scripts/build-linux.sh
SIZES_LIST=${COUNTS_SIZES:-10 100 1000 10000 100000 1000000}
if [ -n "${COUNTS_OUT:-}" ]; then OUT=$COUNTS_OUT
elif [ "$SIZES_LIST" = "10 100 1000 10000 100000 1000000" ]; then OUT=results/counts.csv
else OUT="results/counts-$(echo "$SIZES_LIST" | tr ' ' '-').csv"; fi
SIZES=""
for n in $SIZES_LIST; do SIZES="$SIZES --n $n"; done
mkdir -p results
# shellcheck disable=SC2086
"$BUILD_DIR/sortbench" --counts-only --all-types --all-datasets $SIZES --csv "$OUT" "$@"
if [ -z "${COUNTS_NO_RUST:-}" ] && command -v cargo >/dev/null 2>&1; then
    RUST_OUT=$(echo "$OUT" | sed 's#results/counts#results/rust-counts#')
    echo "== the Rust sorts: $RUST_OUT"
    (cd rust && cargo run --release -p brainsort-bench -- --rust-counts --sizes "$(echo "$SIZES_LIST" | tr ' ' ',')" --out "../$RUST_OUT")
fi

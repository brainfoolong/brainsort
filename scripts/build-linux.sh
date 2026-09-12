#!/bin/sh
# Build on Linux, WSL, macOS or an MSYS2 shell. Uses CMake + Ninja when both
# are installed, otherwise plain g++. BUILD_DIR overrides the output directory
# (default build-linux).
set -e
cd "$(dirname "$0")/.."
BUILD_DIR=${BUILD_DIR:-build-linux}
mkdir -p "$BUILD_DIR"
if command -v cmake >/dev/null 2>&1 && command -v ninja >/dev/null 2>&1; then
    [ -f "$BUILD_DIR/CMakeCache.txt" ] || cmake -S . -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD_DIR"
else
    CXX=${CXX:-g++}
    SRC="-DSB_SOURCE_DIR=\"$(pwd)\""
    FLAGS="-std=c++20 -O2 -Wall -Wextra -Iinclude -Ithird_party"
    echo "cmake not found, compiling with $CXX"
    $CXX $FLAGS -DNDEBUG "$SRC" src/bench.cpp -o "$BUILD_DIR/sortbench"
    $CXX $FLAGS -UNDEBUG "$SRC" src/test.cpp  -o "$BUILD_DIR/sortbench_tests"
    $CXX $FLAGS -UNDEBUG -pthread tests/api_tests.cpp -o "$BUILD_DIR/brainsort_tests"
    $CXX $FLAGS -DNDEBUG "$SRC" bench/api_bench.cpp -o "$BUILD_DIR/brainsort_api_bench"
fi
echo "built: $BUILD_DIR/sortbench $BUILD_DIR/sortbench_tests $BUILD_DIR/brainsort_tests $BUILD_DIR/brainsort_api_bench"

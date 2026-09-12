# Build on Windows with CMake + Ninja (MinGW-w64 g++ from MSYS2, or clang).
# BUILD_DIR overrides the output directory (default build-win). A directory
# that was configured with another generator (Visual Studio) is built as is.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$buildDir = if ($env:BUILD_DIR) { $env:BUILD_DIR } else { "build-win" }
if (-not (Test-Path (Join-Path $buildDir "CMakeCache.txt"))) {
    cmake -S . -B $buildDir -G Ninja -DCMAKE_BUILD_TYPE=Release
    if ($LASTEXITCODE -ne 0) { throw "configure failed" }
}
cmake --build $buildDir --config Release
if ($LASTEXITCODE -ne 0) { throw "build failed" }
Write-Host "built: $buildDir\sortbench.exe $buildDir\sortbench_tests.exe $buildDir\brainsort_tests.exe $buildDir\brainsort_api_bench.exe"

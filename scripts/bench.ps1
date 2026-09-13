# The timing run of one Windows machine: every algorithm on every key type,
# dataset and size, then the public-API benchmark. Writes results\<id>.csv,
# results\<id>.meta.json and results\<id>.api.md; scripts\website.py turns every
# such set in results\ into the website.
#
#   .\scripts\bench.ps1 [sortbench options]      e.g. --reps 11
#   $env:BENCH_SIZES = "1000 10000 100000 1000000"; $env:BENCH_API_MAX_N = "1000000"; .\scripts\bench.ps1
#                                                more sizes (an option; the default is 10, 100 and 100,000
#                                                elements and few repetitions)
#
# Environment (same names as bench.sh): BENCH_ID, BENCH_HOST, BENCH_SIZES
# (default "10 100 100000"), BENCH_REPS (default 5), BENCH_API_MAX_N (default: the quick run at 10, 100 and 100000), BUILD_DIR (default build-win, built first; a multi-config
# build such as MSVC keeps its binaries in <BUILD_DIR>\Release), BENCH_NO_SITE, BENCH_NO_RUST
# (skip the Rust port's API benchmark, which runs when cargo is found).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$buildDir = if ($env:BUILD_DIR) { $env:BUILD_DIR } else { "build-win" }
& (Join-Path $PSScriptRoot "build-windows.ps1")
if ($LASTEXITCODE -ne 0) { throw "build failed" }
$bin = if (Test-Path (Join-Path $buildDir "sortbench.exe")) { $buildDir } else { Join-Path $buildDir "Release" }
$sb  = Join-Path $bin "sortbench.exe"
$api = Join-Path $bin "brainsort_api_bench.exe"
$id = if ($env:BENCH_ID) { $env:BENCH_ID } else { (& $sb --print-id).Trim() }
$sizes = @()
foreach ($n in ($(if ($env:BENCH_SIZES) { $env:BENCH_SIZES } else { "10 100 100000" }) -split " ")) { $sizes += @("--n", $n) }
$reps = if ($env:BENCH_REPS) { $env:BENCH_REPS } else { "5" }
$hostText = if ($env:BENCH_HOST) { $env:BENCH_HOST } else { "" }
$apiMax = if ($env:BENCH_API_MAX_N) { $env:BENCH_API_MAX_N } else { "100000" }
New-Item -ItemType Directory -Force results | Out-Null

Write-Host "== timing: results\$id.csv"
& $sb --all-types --all-algos --all-datasets --timing-only @sizes --reps $reps --rounds 1 --id $id --host $hostText --csv "results\$id.csv" @args
if ($LASTEXITCODE -ne 0) { throw "benchmark failed" }

Write-Host "== public API: results\$id.api.md"
if ($env:BENCH_API_MAX_N) { & $api --max-n $apiMax --host $hostText | Out-File -Encoding utf8 "results\$id.api.md" }
else { & $api --quick --host $hostText | Out-File -Encoding utf8 "results\$id.api.md" }
if ($LASTEXITCODE -ne 0) { throw "API benchmark failed" }

if (-not $env:BENCH_NO_RUST -and (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Host "== public API, Rust port: results\$id.rust.api.md"
    Push-Location rust
    & cargo run --release -p brainsort-bench -- --bench --max-n $apiMax --host $hostText --out "..\results\$id.rust.api.md"
    $rc = $LASTEXITCODE
    Pop-Location
    if ($rc -ne 0) { throw "Rust API benchmark failed" }
}

if (-not $env:BENCH_NO_SITE) {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        & python scripts\website.py
        if ($LASTEXITCODE -ne 0) { throw "website build failed" }
    } else {
        Write-Host "python not found; run scripts\website.py later to build the website"
    }
}

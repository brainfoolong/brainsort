# The deterministic numbers on Windows: one counted run per cell, no timing.
# The output is the same file every other machine produces.
#
#   .\scripts\counts.ps1                          regenerate results\counts.csv (n = 10 to 1,000,000)
#   $env:COUNTS_SIZES = "10000000"; .\scripts\counts.ps1   ten million elements into results\counts-10000000.csv (an option, over an hour)
#
# When cargo is installed, the Rust sorts' counts follow into results\rust-counts.csv
# (or rust-counts-<sizes>.csv): what can be counted of the Rust ecosystem as
# shipped, on the same cells.
#
# Environment: COUNTS_SIZES, COUNTS_OUT, COUNTS_NO_RUST (skip the Rust counts),
# BUILD_DIR (default build-win, built first).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$buildDir = if ($env:BUILD_DIR) { $env:BUILD_DIR } else { "build-win" }
& (Join-Path $PSScriptRoot "build-windows.ps1")
if ($LASTEXITCODE -ne 0) { throw "build failed" }
$bin = if (Test-Path (Join-Path $buildDir "sortbench.exe")) { $buildDir } else { Join-Path $buildDir "Release" }
$sizesList = if ($env:COUNTS_SIZES) { $env:COUNTS_SIZES } else { "10 100 1000 10000 100000 1000000" }
$out = if ($env:COUNTS_OUT) { $env:COUNTS_OUT }
       elseif ($sizesList -eq "10 100 1000 10000 100000 1000000") { "results\counts.csv" }
       else { "results\counts-" + ($sizesList -replace " ", "-") + ".csv" }
$sizes = @()
foreach ($n in ($sizesList -split " ")) { $sizes += @("--n", $n) }
New-Item -ItemType Directory -Force results | Out-Null
& (Join-Path $bin "sortbench.exe") --counts-only --all-types --all-datasets @sizes --csv $out @args
if ($LASTEXITCODE -ne 0) { throw "counts run failed" }
if (-not $env:COUNTS_NO_RUST -and (Get-Command cargo -ErrorAction SilentlyContinue)) {
    $rustOut = $out -replace "results\\counts", "results\rust-counts"
    Write-Host "== the Rust sorts: $rustOut"
    Push-Location rust
    & cargo run --release -p brainsort-bench -- --rust-counts --sizes ($sizesList -replace " ", ",") --out (Join-Path ".." $rustOut)
    $rc = $LASTEXITCODE
    Pop-Location
    if ($rc -ne 0) { throw "Rust counts run failed" }
}

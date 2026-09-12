# The deterministic numbers on Windows: one counted run per cell, no timing.
# The output is the same file every other machine produces.
#
#   .\scripts\counts.ps1                          regenerate results\counts.csv (n = 1,000, 10,000, 100,000)
#   $env:COUNTS_SIZES = "1000000"; .\scripts\counts.ps1    one million elements into results\counts-1000000.csv
#
# Environment: COUNTS_SIZES, COUNTS_OUT, BUILD_DIR (default build-win, built first).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$buildDir = if ($env:BUILD_DIR) { $env:BUILD_DIR } else { "build-win" }
& (Join-Path $PSScriptRoot "build-windows.ps1")
if ($LASTEXITCODE -ne 0) { throw "build failed" }
$bin = if (Test-Path (Join-Path $buildDir "sortbench.exe")) { $buildDir } else { Join-Path $buildDir "Release" }
$sizesList = if ($env:COUNTS_SIZES) { $env:COUNTS_SIZES } else { "1000 10000 100000" }
$out = if ($env:COUNTS_OUT) { $env:COUNTS_OUT }
       elseif ($sizesList -eq "1000 10000 100000") { "results\counts.csv" }
       else { "results\counts-" + ($sizesList -replace " ", "-") + ".csv" }
$sizes = @()
foreach ($n in ($sizesList -split " ")) { $sizes += @("--n", $n) }
New-Item -ItemType Directory -Force results | Out-Null
& (Join-Path $bin "sortbench.exe") --counts-only --all-types --all-algos --all-datasets @sizes --csv $out @args
if ($LASTEXITCODE -ne 0) { throw "counts run failed" }

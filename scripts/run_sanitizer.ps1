<#
.SYNOPSIS
  Runs compute-sanitizer (memcheck, racecheck, synccheck, initcheck) on the
  standalone attention and INT4 GEMM binaries and records the summaries in
  results/sanitizer.json.

.EXAMPLE
  .\scripts\run_sanitizer.ps1      # after .\make.ps1 build
#>
$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
$Bins = [ordered]@{
    attention = 'build\cmake\kf_sanitize_attn.exe'   # csrc/micro/sanitize_attn.cu
    gemm      = 'build\cmake\kf_sanitize_gemm.exe'   # csrc/micro/sanitize_gemm.cu
}
$results = [ordered]@{}
foreach ($name in $Bins.Keys) {
    $Exe = Join-Path $Root $Bins[$name]
    if (-not (Test-Path $Exe)) { throw "$Exe not found. Run .\make.ps1 build first." }
    $results[$name] = [ordered]@{}
    foreach ($tool in 'memcheck', 'racecheck', 'synccheck', 'initcheck') {
        $t0 = Get-Date
        $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
        $out = & compute-sanitizer --tool $tool $Exe 2>&1 | ForEach-Object { "$_" }
        $code = $LASTEXITCODE
        $ErrorActionPreference = $prev
        $summary = ($out | Select-String 'SUMMARY' | Select-Object -Last 1).Line -replace '^=+\s*', ''
        $workload = ($out | Select-String 'kf_sanitize_\w+:' | Select-Object -Last 1).Line
        $results[$name][$tool] = [ordered]@{
            exit_code = $code
            clean     = ($code -eq 0)
            summary   = $summary
            workload  = $workload
            seconds   = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
        }
        Write-Host "[$name/$tool] exit $code  $summary"
    }
}

$version = (& compute-sanitizer --version 2>&1 | Select-Object -Last 1)
$doc = [ordered]@{
    tool_version = "$version"
    binaries     = $Bins
    timestamp    = (Get-Date).ToString('s')
    tools        = $results
}
$path = Join-Path $Root 'results\sanitizer.json'
# UTF-8 without BOM (Set-Content -Encoding utf8 in Windows PowerShell adds a BOM).
[System.IO.File]::WriteAllText($path, ($doc | ConvertTo-Json -Depth 5), [System.Text.UTF8Encoding]::new($false))
Write-Host "wrote $path"
if ($results.Values | ForEach-Object { $_.Values } | Where-Object { -not $_.clean }) { throw 'compute-sanitizer reported errors' }

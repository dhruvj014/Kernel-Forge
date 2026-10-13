<#
.SYNOPSIS
  Profiles one attention implementation with Nsight Compute and appends the
  key metrics to results/ncu_summary.json.

.EXAMPLE
  .\scripts\run_ncu.ps1                                   # naive, B=2 H=16 N=2048 D=64 fp16
  .\scripts\run_ncu.ps1 -Impl attn_naive -Label naive_causal -ExtraArgs '--causal'

  Requires GPU performance counter access (NVIDIA Control Panel > Developer >
  Manage GPU Performance Counters > allow all users) or an elevated shell.
#>
param(
    [string]$Impl = 'attn_naive',
    [string]$Label = '',
    [string]$Shape = '--B 2 --H 16 --N 2048 --D 64 --dtype fp16',
    [string]$ExtraArgs = '',
    [string]$Set = 'full'
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
$Py = Join-Path $Root '.venv\Scripts\python.exe'
if (-not $Label) { $Label = $Impl }
$OutDir = Join-Path $Root 'results\ncu'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Rep = Join-Path $OutDir $Label

$targetArgs = @((Join-Path $Root 'bench\profile_target.py'), '--impl', $Impl) + ($Shape -split '\s+') + ($ExtraArgs -split '\s+' | Where-Object { $_ })

# --nvtx-include restricts collection to kernels launched inside the
# "kf_profile" range, so warmup launches and PyTorch setup kernels are skipped.
$ncuArgs = @('--set', $Set, '--import-source', 'yes', '--nvtx', '--nvtx-include', 'kf_profile/',
             '--target-processes', 'all', '-f', '-o', $Rep, $Py) + $targetArgs

Write-Host "ncu $($ncuArgs -join ' ')"
$prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
& ncu @ncuArgs 2>&1 | ForEach-Object { "$_" }
$code = $LASTEXITCODE
$ErrorActionPreference = $prev
if ($code -ne 0) { throw "ncu failed with exit code $code" }

& $Py (Join-Path $Root 'scripts\ncu_summary.py') "$Rep.ncu-rep" --label $Label --impl $Impl --shape "$Shape $ExtraArgs"
if ($LASTEXITCODE -ne 0) { throw "ncu_summary.py failed" }

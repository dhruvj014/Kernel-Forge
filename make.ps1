<#
.SYNOPSIS
  KernelForge task runner for native Windows (the Makefile is the Linux equivalent).

.EXAMPLE
  .\make.ps1 setup    # install pinned deps into .venv and register the package (editable)
  .\make.ps1 build    # build the torch extension and the CMake microbenchmarks
  .\make.ps1 test     # pytest correctness suite
  .\make.ps1 bench    # hardware ceilings + attention benchmarks -> results/*.json
  .\make.ps1 report   # regenerate BENCHMARKS.md from results/*.json
  .\make.ps1 ncu      # Nsight Compute profile of the representative shape
  .\make.ps1 lint     # ruff

  Everything Python runs from .venv. Only the NVIDIA driver, CUDA toolkit,
  Nsight tools and MSVC Build Tools live outside it.
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet('setup', 'build', 'test', 'bench', 'report', 'ncu', 'lint', 'clean', 'all')]
    [string]$Target = 'build',

    # Extra args forwarded to the underlying python script (bench/test/ncu).
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Venv = Join-Path $Root '.venv'
$Py = Join-Path $Venv 'Scripts\python.exe'
$BuildDir = Join-Path $Root 'build'

function Assert-Venv {
    if (-not (Test-Path $Py)) {
        throw "No .venv found at $Venv. Create it with: python -m venv .venv ; then run .\make.ps1 setup (see docs/env.md)."
    }
}

# Imports the MSVC x64 developer environment (cl.exe, link.exe, Windows SDK)
# into this PowerShell session. nvcc on Windows requires cl.exe as its host
# compiler.
function Enter-MsvcEnv {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) { return }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found. Install VS 2022 Build Tools with the C++ workload." }
    $inst = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $inst) { throw "MSVC x64 tools not found. Install the 'Desktop development with C++' workload (see docs/env.md)." }
    $vcvars = Join-Path $inst 'VC\Auxiliary\Build\vcvars64.bat'
    cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
    }
}

function Set-ProjectEnv {
    # venv first so its cmake and ninja win over any copies bundled with VS.
    $env:PATH = "$Venv\Scripts;$env:PATH"
    $env:HF_HOME = Join-Path $Root '.cache\hf'
    $env:DISTUTILS_USE_SDK = '1'      # tell torch.utils.cpp_extension to use the env we just loaded
    $env:TORCH_CUDA_ARCH_LIST = '8.9'
    $env:PYTHONUTF8 = '1'
}

# Runs a native command, streams output (optionally teeing to a log) and fails
# on a non-zero exit code. Native stderr is converted to plain strings so
# Windows PowerShell does not treat compiler warnings as terminating errors.
function Invoke-Native {
    param([string]$Exe, [string[]]$Arguments, [string]$Log)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        if ($Log) {
            # StreamWriter instead of Tee-Object: Tee-Object wraps long lines at
            # the console width, which breaks parsing of ptxas output.
            New-Item -ItemType Directory -Force -Path (Split-Path $Log) | Out-Null
            $sw = [System.IO.StreamWriter]::new($Log, $false)
            try {
                & $Exe @Arguments 2>&1 | ForEach-Object { $line = "$_"; $sw.WriteLine($line); $line }
            } finally { $sw.Dispose() }
        } else {
            & $Exe @Arguments 2>&1 | ForEach-Object { "$_" }
        }
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prev
    }
    if ($code -ne 0) { throw "'$Exe $($Arguments -join ' ')' failed with exit code $code" }
}

function Invoke-Setup {
    Invoke-Native $Py @('-m', 'pip', 'install', '-r', (Join-Path $Root 'requirements.txt'))
    Enter-MsvcEnv
    Invoke-Native $Py @('-m', 'pip', 'install', '--no-build-isolation', '--no-deps', '-e', $Root)
}

function Invoke-Build {
    Enter-MsvcEnv
    Push-Location $Root
    try {
        Invoke-Native $Py @('setup.py', 'build_ext', '--inplace') (Join-Path $BuildDir 'ext_build.log')
        Invoke-Native 'cmake' @('-S', $Root, '-B', (Join-Path $BuildDir 'cmake'), '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release')
        Invoke-Native 'cmake' @('--build', (Join-Path $BuildDir 'cmake')) (Join-Path $BuildDir 'cmake_build.log')
        Invoke-Native $Py @('scripts/ptxas_summary.py', (Join-Path $BuildDir 'ext_build.log'), (Join-Path $BuildDir 'cmake_build.log'))
    } finally { Pop-Location }
}

Assert-Venv
Set-ProjectEnv

switch ($Target) {
    'setup' { Invoke-Setup }
    'build' { Invoke-Build }
    'test' { Push-Location $Root; try { Invoke-Native $Py (@('-m', 'pytest', 'tests') + $Rest) } finally { Pop-Location } }
    'bench' {
        Push-Location $Root
        try {
            Invoke-Native $Py @('bench/hw_ceilings.py')
            Invoke-Native $Py (@('bench/bench_attention.py') + $Rest)
        } finally { Pop-Location }
    }
    'report' { Push-Location $Root; try { Invoke-Native $Py @('scripts/make_report.py') } finally { Pop-Location } }
    'ncu' { & (Join-Path $Root 'scripts\run_ncu.ps1') @Rest }
    'lint' { Push-Location $Root; try { Invoke-Native $Py @('-m', 'ruff', 'check', '.') } finally { Pop-Location } }
    'clean' {
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $BuildDir
        Get-ChildItem (Join-Path $Root 'kernelforge') -Filter '_C*.pyd' | Remove-Item -Force
    }
    'all' { Invoke-Build; & $PSCommandPath test; & $PSCommandPath bench; & $PSCommandPath report }
}

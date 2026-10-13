# Environment

How KernelForge is built, tested and profiled on the development machine, plus every deviation from the original build brief and why.

## Machine

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (Ada Lovelace, sm_89), 8 GB GDDR6 |
| Host | Windows 11, 32 GB RAM |
| Driver | 610.60 |
| CUDA toolkit | 13.0 (nvcc V13.0.48) |
| Host compiler | MSVC 14.44.35207 (Visual Studio 2022 Build Tools, "Desktop development with C++"), Windows SDK 10.0.26100.0 |
| Nsight Compute | 2025.3.0 |
| Python | 3.12.0 from python.org |
| PyTorch | 2.14.1+cu130 |

Measured hardware ceilings (bandwidth, Tensor Core and FP32 throughput) live in `results/hw_ceilings.json`, never in this file.

## Deviation from the brief: native Windows instead of WSL2

The brief recommends WSL2. This project builds and profiles natively on Windows instead:

- Nsight Compute has limited GPU performance-counter support under WSL2. On native Windows it works once counters are enabled (see Profiling below), so the brief's fallback route becomes the main route and there is only one toolchain to keep in sync.
- `make.ps1` is the Windows task runner. The `Makefile` is kept as the Linux equivalent and uses the same scripts.

## What lives where

Only system-level NVIDIA and Microsoft components are installed outside the project:

- NVIDIA driver, CUDA toolkit, Nsight Compute (and later Nsight Systems)
- MSVC Build Tools (nvcc on Windows needs `cl.exe` as its host compiler)

Everything else (torch, ninja, cmake, pytest, ruff, transformers, datasets) is pinned in `requirements.txt` and installed into `.venv`. The Hugging Face cache is project-local (`HF_HOME=.cache\hf`, set by `make.ps1`) so model downloads never touch the user profile.

## Setup from a fresh clone

```powershell
# 1. venv from the python.org interpreter (NOT Anaconda, see Pitfalls)
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv

# 2. pinned deps plus an editable install of the kernelforge package
.\make.ps1 setup

# 3. build the torch extension (kernelforge._C) and CMake microbenchmarks
.\make.ps1 build

# 4. smoke test the toolchain
.\build\cmake\kf_hello.exe
```

`make.ps1` finds MSVC through `vswhere`, imports the `vcvars64.bat` environment into the session, puts `.venv\Scripts` first on PATH (so the venv's cmake and ninja win), and sets `TORCH_CUDA_ARCH_LIST=8.9` and `DISTUTILS_USE_SDK=1`.

## Build outputs

- `kernelforge/_C*.pyd`: the torch extension, built by `setup.py` through `torch.utils.cpp_extension` with ninja.
- `build/cmake/kf_hello.exe`, `build/cmake/kf_bandwidth.exe`: standalone microbenchmarks built by CMake.
- `build/ext_build.log`, `build/cmake_build.log`: full compiler logs including `-Xptxas -v` output.
- `results/ptxas.json`: registers, shared memory and spills per kernel, parsed from those logs by `scripts/ptxas_summary.py` on every build.

Compile flags: `-O3 -lineinfo -Xptxas=-v -std=c++20`, sm_89 only. `--use_fast_math` is off by default and enabled with `KF_FAST_MATH=1`, so the correctness tests see IEEE `expf` and division.

## Profiling

**Working route: Nsight Compute on native Windows.** Verified in Phase 0:

1. Enable counters once: NVIDIA Control Panel > Desktop > Enable Developer Settings, then Developer > Manage GPU Performance Counters > "Allow access to the GPU performance counters to all users". Without this, `ncu` fails with `ERR_NVGPUCTRPERM` unless run from an elevated shell.
2. `ncu --set basic -o results\ncu\hello_smoke .\build\cmake\kf_hello.exe` profiled `vector_add` in 8 passes (Speed Of Light: DRAM throughput about 90% of peak, the expected result for a streaming add).
3. Profiling the torch extension from Python also works: `ncu --nvtx --nvtx-include "kf_profile/" --target-processes all ... .venv\Scripts\python.exe bench\profile_target.py` captured exactly the three naive attention kernels launched inside the `kf_profile` NVTX range and nothing from warmup or torch setup. `scripts/run_ncu.ps1` wraps this and writes the key metrics to `results/ncu_summary.json`.

Quirks:

- `--target-processes all` is required. `.venv\Scripts\python.exe` is a launcher that starts the base interpreter as a child process, and without the flag `ncu` only attaches to the launcher, which never touches the GPU.
- By default `ncu` locks the GPU to its base clock (`--clock-control base`) so repeated profiles are comparable. Kernel durations in `ncu` reports are therefore longer than the boosted timings in benchmark JSON. Benchmarks are the source for speed numbers, `ncu` for the explanation (occupancy, throughput, stalls, bank conflicts).
- The Windows locale prints numbers with Indian digit grouping (`20,07,924`). `scripts/ncu_summary.py` asks for raw CSV with base units and strips separators before parsing.
- Full `.ncu-rep` files are gitignored because of their size. Key metrics are summarized in `results/ncu_summary.json`.

**Compute Sanitizer** (`compute-sanitizer --tool memcheck .\build\cmake\kf_hello.exe`) reports 0 errors. Later phases also run `racecheck` and `synccheck` on small shapes.

**Nsight Systems** (`nsys`) is not on PATH yet. It is only needed for the Phase 5 GPU time share measurement and will be installed and documented then.

## Pitfalls hit during setup

1. **Anaconda Python breaks torch.** A venv made from Anaconda's Python failed on `import torch` with `WinError 1114` (DLL initialization failed in `c10.dll`). The wheel's file hashes matched its RECORD, so the install was intact. The cause is Anaconda's older bundled VC++ runtime (14.29) loading first. Fix: build the venv from the python.org interpreter.
2. **torch 2.14 headers need C++20** (designated initializers, `std::strong_ordering`). Both the extension and CMake use `-std=c++20` / `/std:c++20`.
3. **`AT_DISPATCH_FLOATING_TYPES_AND2` instantiates `double`**, which the kernels do not implement (unresolved `from_float<double>`). `csrc/common/torch_utils.h` defines `KF_DISPATCH_HALF_BF16_FLOAT` covering exactly fp16, bf16 and fp32.
4. **MSVC's `INFINITY` is an overflowing double literal** that nvcc warns about. Device code uses `kf::neg_inf()` (the bit pattern `0xff800000`).
5. **CMake silently targeted sm_75.** CUDA 13 fills in its default architecture during `project()`, so `CMAKE_CUDA_ARCHITECTURES 89` must be set before `project()`.
6. **`Tee-Object` wraps long lines** at the console width, which broke ptxas log parsing. `make.ps1` writes logs with a `StreamWriter` instead.
7. **VS Build Tools via winget failed** (installer exit code 1). Installing the "Desktop development with C++" workload through the Visual Studio Installer GUI worked.

## Benchmarking conditions

Laptop GPUs boost and throttle with temperature and power source. All benchmarks are run plugged in, with Windows power mode set to "Best performance". Every benchmark JSON records `nvidia-smi` clocks, power state, temperature and throttle reasons at the start and end of the run, so throttled runs are visible in the data.

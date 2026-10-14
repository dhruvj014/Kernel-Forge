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
8. **The GPU was capped at 30 W.** The Dell G15 5530 ships with Alienware Command Center in a mode that limits the RTX 4060 to 30 W (vBIOS default 115 W, max 140 W). The first ceilings run measured 16.5 TFLOPS FP16 Tensor Core and 3.3 TFLOPS SGEMM. The giveaway was FP16-accumulate GEMM being only 1.14x faster than FP32-accumulate instead of about 2x, and `nvidia-smi -q -d POWER` showing "Current Power Limit: 30.00 W". After switching AWCC to Performance the enforced limit is 135 W. That run is kept as `results/hw_ceilings_30w.json`, and BENCHMARKS.md shows the comparison. Check the limit with `nvidia-smi --query-gpu=enforced.power.limit --format=csv` before benchmarking.
9. **CUDA sysmem fallback hides OOM.** Under WDDM, the driver's default "CUDA - Sysmem Fallback Policy" spills allocations that do not fit in VRAM into system RAM over PCIe instead of failing. Naive attention at N=8192 (8.6 GB of scores) "ran" at a crawl instead of raising OOM. `kernelforge.bench_utils.cap_to_free_vram()` caps the torch allocator to the VRAM free at startup so this becomes a clean `torch.OutOfMemoryError`. The policy can also be set to "Prefer No Sysmem Fallback" in NVIDIA Control Panel > Manage 3D settings. About 1.3 GiB of the 8 GiB is reserved by Windows and desktop apps, so roughly 6.7 GiB is usable.
10. **The Windows torch wheel has no FlashAttention SDPA backend** ("Torch was not compiled with flash attention"). `SDPBackend.FLASH_ATTENTION` cannot run on this machine. The SDPA baselines are the memory-efficient backend and the cuDNN fused attention backend (cuDNN 9.24), which run for every dtype, head dim and mask in the grid. Availability is probed and recorded in each attention results file.
11. **Python launch overhead leaks into sub-millisecond timings.** Enqueueing one SDPA call from Python costs 0.1 to 0.9 ms (cuDNN builds or looks up an execution plan per call), longer than the kernels at N=512. The GPU then idles between the start event and the kernel, and that idle time was counted, with medians up to 5x too high and fp16 vs bf16 differing by 3x on the same shape. `time_cuda` now enqueues a GPU spin (`torch.cuda._sleep`) sized to 5x the measured host enqueue time (minimum 1 ms) before each timed run, which brought the p10 to p90 spread to about 3%.

## Models and datasets (Phase 5)

`scripts/download_assets.py` fetches the original Hugging Face checkpoints (safetensors, config and tokenizer files only) and the WikiText-2 test split into the project-local cache `.cache\hf` (gitignored, about 5 GB). Delete that folder to reclaim the space.

| asset | size | used for |
|---|---|---|
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 2.05 GiB | perplexity, decode |
| Qwen/Qwen2.5-1.5B | 2.89 GiB | perplexity, decode |
| Salesforce/wikitext, wikitext-2-raw-v1 (test) | a few MB | perplexity, decode prompts |

Ollama's models are not usable here: they are llama.cpp GGUF files that are already quantized (Q4_0 / Q4_K_M), so they provide no FP16 baseline and would mean re-quantizing quantized weights. The evaluation scripts set `HF_HUB_OFFLINE=1` and run from the cache.

GPU time share during decode comes from `torch.profiler` (CUPTI), not Nsight Systems. Both read the same kernel activity records, and the profiler needs no system install.

## Benchmarking conditions

Laptop GPUs boost and throttle with temperature and power source. All benchmarks are run plugged in, with Windows power mode set to "Best performance". Every benchmark JSON records `nvidia-smi` clocks, power state, temperature and throttle reasons at the start and end of the run, so throttled runs are visible in the data.
12. **FlashAttention-2 baseline dropped (time-boxed attempt, Phase 3).** PyPI has only source distributions of `flash-attn` (latest 2.8.3.post2), with no Windows wheel for any version (`pip download flash-attn --only-binary :all:` finds none). A source build compiles hundreds of CUDA kernels with MSVC against torch 2.14 / CUDA 13, a combination the project does not list as supported, and typically takes hours when it works at all on Windows. Unofficial prebuilt Windows wheels exist on GitHub, but running untrusted binaries is out of scope. The attention baselines remain PyTorch SDPA with the cuDNN 9.24 fused attention backend (a FlashAttention-2-class kernel) and the memory-efficient backend.

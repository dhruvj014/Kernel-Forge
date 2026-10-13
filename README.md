# KernelForge

Hand-written CUDA kernels for LLM inference, built from scratch for an RTX 4060 Laptop GPU (Ada, sm_89):

- a fused FlashAttention-style attention kernel (tiling, online softmax, Tensor Cores)
- a fused INT4 weight-only dequantize-GEMM kernel for memory-bound decode

Both are exposed as PyTorch custom ops, validated against FP32 references, benchmarked against cuBLAS and PyTorch SDPA, profiled with Nsight Compute, and integrated into TinyLlama 1.1B and Qwen2.5 1.5B.

**Status:** work in progress. No results are published yet. Every number this project reports will come from `results/*.json` through the generated `BENCHMARKS.md`.

## Rules this project follows

- Kernels are hand-written. cuBLAS, cuDNN, FlashAttention, CUTLASS and Triton appear only as benchmark baselines.
- No fabricated metrics. Hardware ceilings are measured on this machine, not taken from spec sheets.
- Correctness first, then speed. Each optimization is its own benchmarked and profiled commit.

## Quickstart (native Windows)

Prerequisites and the full setup are in [docs/env.md](docs/env.md).

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv   # python.org Python, not Anaconda
.\make.ps1 setup                # pinned deps into .venv, editable install
.\make.ps1 build                # torch extension + CMake microbenchmarks
.\make.ps1 test                 # correctness suite (needs the GPU)
```

On Linux the `Makefile` provides the same targets.

## Layout

```
csrc/         CUDA kernels (common/, attention/, gemm/, micro/) and torch bindings
kernelforge/  Python package: ops, quantizer, modules, benchmark harness
bench/        hardware ceilings, kernel benchmarks, end-to-end evaluation
tests/        correctness tests against FP32 references
scripts/      report generation, Nsight Compute helpers, plots
results/      benchmark JSON (source of every published number)
docs/         environment, design notes per phase, references
```

## References

See [docs/references.md](docs/references.md).

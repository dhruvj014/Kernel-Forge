"""KernelForge: hand-written CUDA kernels for LLM inference.

Kernels live in kernelforge.ops (requires the compiled kernelforge._C
extension). kernelforge.bench_utils has no extension dependency so the
benchmark harness also works for baseline-only runs.
"""

__version__ = "0.1.0"

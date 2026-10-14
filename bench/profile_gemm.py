"""One INT4 GEMM call inside an NVTX range for Nsight Compute (see scripts/run_ncu.ps1 -Target gemm).

Warmup calls run outside the range. L2 is flushed right before the profiled
call, matching the cold-L2 benchmark protocol.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_gemm import int4_fused  # noqa: E402

from kernelforge import quantize as kq  # noqa: E402
from kernelforge.bench_utils import L2Flusher  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--impl", default="int4_fused", choices=["int4_fused", "cublas_fp16"])
    p.add_argument("--M", type=int, default=1)
    p.add_argument("--N", type=int, default=5632)
    p.add_argument("--K", type=int, default=2048)
    p.add_argument("--group", type=int, default=128)
    a = p.parse_args()
    w = (torch.randn(a.N, a.K, device="cuda") * 0.02).half()
    x = torch.randn(a.M, a.K, device="cuda").half()
    qw = kq.quantize_weight(w, a.group)
    fn = (lambda: int4_fused(x, qw)) if a.impl == "int4_fused" else (lambda: F.linear(x, w))
    flush = L2Flusher()
    for _ in range(3):
        fn()
    flush()
    torch.cuda.synchronize()
    with torch.cuda.nvtx.range("kf_profile"):
        fn()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()

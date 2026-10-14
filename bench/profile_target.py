"""Runs one attention call inside an NVTX range so Nsight Compute can profile
exactly that call (scripts/run_ncu.ps1 filters on the "kf_profile" range).
Warmup calls happen outside the range and are not profiled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench_attention import DTYPES, make_impl  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--impl", default="attn_naive")
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--H", type=int, default=16)
    p.add_argument("--N", type=int, default=2048)
    p.add_argument("--D", type=int, default=64)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    args = p.parse_args()

    dtype = DTYPES[args.dtype]
    shape = (args.B, args.H, args.N, args.D)
    q, k, v = (torch.randn(shape, device="cuda", dtype=dtype) for _ in range(3))
    fn = make_impl(args.impl, args.causal)
    for _ in range(3):
        fn(q, k, v)
    torch.cuda.synchronize()
    with torch.cuda.nvtx.range("kf_profile"):
        fn(q, k, v)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()

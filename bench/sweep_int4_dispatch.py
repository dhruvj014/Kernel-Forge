"""Chooses the GEMV/GEMM crossover for ops.int4_matmul from measurements.

Times the GEMV kernel and the Tensor Core GEMM at M = 1..8 on every distinct
layer shape of both models (cold L2, same protocol as bench_gemm.py). The
dispatcher threshold is the largest M at which the GEMV's summed time over
all shapes (what one decode step pays) is lower than the GEMM's.

Output: results/gemm_dispatch.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_gemm import CONFIGS, layer_shapes  # noqa: E402

from kernelforge import ops  # noqa: E402
from kernelforge import quantize as kq  # noqa: E402
from kernelforge.bench_utils import cap_to_free_vram, time_cuda, write_results  # noqa: E402


def main() -> None:
    cap_to_free_vram()
    rows = []
    for model in CONFIGS:
        shapes = {}
        for layer, nk in layer_shapes(model).items():
            shapes.setdefault(nk, []).append(layer)
        for (n, k), layers in shapes.items():
            w = torch.randn(n, k, device="cuda") * 0.02
            qw = kq.quantize_weight(w, 128)
            p, s = qw.packed, qw.scales
            for m in range(1, 9):
                x = torch.randn(m, k, device="cuda").half()
                tv = time_cuda(lambda x=x, p=p, s=s: ops.int4_gemv(x, p, s, 128), flush_l2=True)
                tg = time_cuda(lambda x=x, p=p, s=s: ops.int4_gemm(x, p, s, 128), flush_l2=True)
                rows.append({"model": model, "layer": "/".join(layers), "N": n, "K": k, "M": m,
                             "gemv_ms": tv.median_ms, "gemm_ms": tg.median_ms,
                             "gemv_faster": tv.median_ms < tg.median_ms})
                print(f"{model:13s} {'/'.join(layers):21s} M={m} gemv {tv.median_ms * 1e3:6.1f} us  "
                      f"gemm {tg.median_ms * 1e3:6.1f} us", flush=True)
    by_m = {m: [r for r in rows if r["M"] == m] for m in range(1, 9)}
    wins = {m: sum(r["gemv_faster"] for r in rs) for m, rs in by_m.items()}
    totals = {m: {"gemv_ms": sum(r["gemv_ms"] for r in rs), "gemm_ms": sum(r["gemm_ms"] for r in rs)}
              for m, rs in by_m.items()}
    # Largest M up to which the GEMV is cheaper in total over all layer shapes
    # (what one decode step pays), stopping at the first M where it is not.
    threshold = 0
    for m in range(1, 9):
        if totals[m]["gemv_ms"] >= totals[m]["gemm_ms"]:
            break
        threshold = m
    out = write_results("gemm_dispatch", {"rule": "largest M with sum(gemv) < sum(gemm) over all layer shapes",
                                          "rows": rows, "gemv_wins_by_m": wins, "totals_by_m": totals,
                                          "shapes_per_m": len(by_m[1]), "gemv_max_m": threshold})
    print(f"gemv wins by M: {wins}\nchosen GEMV_MAX_M = {threshold}\nwrote {out}")


if __name__ == "__main__":
    main()

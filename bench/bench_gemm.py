"""INT4 weight-only GEMM benchmark on the real layer shapes of TinyLlama 1.1B and Qwen2.5 1.5B.

Implementations, all producing y = x W^T for x [M, K] fp16:
  int4_fused        KernelForge fused dequant kernels (GEMV for M <= 8, Tensor Core GEMM above)
  cublas_fp16       torch F.linear on the original fp16 weights (cuBLAS)
  dequant_matmul    PyTorch dequantize to fp16, then F.linear: the unfused INT4 path

Every timed run starts with L2 flushed (bench_utils.L2Flusher). A single layer's
weights fit in the 32 MB L2, but in a real decode step the other layers evict
them, so warm-L2 numbers would overstate bandwidth.

Effective bandwidth = (weight bytes + scale bytes + x bytes + y bytes) / time,
reported as a percent of the measured DRAM copy peak (the brief's reference)
and of the measured read-only peak (results/hw_ceilings.json). A GEMV is
almost pure reads, so the read peak is its real roof: cuBLAS FP16 reaches
about 100% of the copy number.

Output: results/gemm_final.json (default).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kernelforge import ops  # noqa: E402
from kernelforge import quantize as kq  # noqa: E402
from kernelforge.bench_utils import (  # noqa: E402
    REPO_ROOT,
    cap_to_free_vram,
    gbps,
    gemm_flops,
    gpu_clock_snapshot,
    load_results,
    tflops,
    time_cuda,
    write_results,
)

CONFIGS = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
}
M_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]


def layer_shapes(model: str) -> dict[str, tuple[int, int]]:
    """(N, K) of each linear layer, read from the model's config.json in the project HF cache."""
    from huggingface_hub import hf_hub_download

    cfg = json.loads(Path(hf_hub_download(CONFIGS[model], "config.json",
                                          cache_dir=REPO_ROOT / ".cache" / "hf" / "hub")).read_text())
    h, inter = cfg["hidden_size"], cfg["intermediate_size"]
    head_dim = h // cfg["num_attention_heads"]
    kv = cfg["num_key_value_heads"] * head_dim
    return {"q_proj": (h, h), "k_proj": (kv, h), "v_proj": (kv, h), "o_proj": (h, h),
            "gate_proj": (inter, h), "up_proj": (inter, h), "down_proj": (h, inter)}


def int4_fused(x, qw):
    if hasattr(ops, "int4_matmul"):
        return ops.int4_matmul(x, qw.packed, qw.scales, qw.group_size)
    if x.shape[0] > 8:
        raise NotImplementedError("Tensor Core INT4 GEMM not built yet")
    return ops.int4_gemv(x, qw.packed, qw.scales, qw.group_size)


def bench_shape(model, layer, n, k, m, group, peak_gbps, read_peak_gbps, iters):
    g = torch.Generator(device="cuda").manual_seed(n * 31 + k)
    w = (torch.randn(n, k, generator=g, device="cuda") * 0.02).half()
    x = torch.randn(m, k, generator=g, device="cuda").half()
    qw = kq.quantize_weight(w, group)
    ref = x.float() @ kq.dequantize(qw, torch.float32).T
    x_bytes, y_bytes = x.numel() * 2, m * n * 2
    impls = {
        "int4_fused": (lambda: int4_fused(x, qw), qw.nbytes),
        "cublas_fp16": (lambda: F.linear(x, w), w.numel() * 2),
        "dequant_matmul": (lambda: F.linear(x, kq.dequantize(qw)), qw.nbytes),
    }
    out = []
    for impl, (fn, wbytes) in impls.items():
        rec = {"model": model, "layer": layer, "N": n, "K": k, "M": m, "group": group, "impl": impl}
        try:
            y = fn()
            if impl != "cublas_fp16":
                rec["max_rel_err_vs_dequant_ref"] = ((y.float() - ref).abs().max() / ref.abs().max()).item()
            t = time_cuda(fn, warmup=10, iters=iters, flush_l2=True)
        except NotImplementedError as exc:
            rec.update(status="unavailable", error=str(exc))
            out.append(rec)
            continue
        moved = wbytes + x_bytes + y_bytes
        rec.update(status="ok", timing=t.to_dict(), median_ms=t.median_ms, bytes_moved=moved,
                   gbps=gbps(moved, t.median_ms), pct_of_peak_dram=100 * gbps(moved, t.median_ms) / peak_gbps,
                   pct_of_read_peak=100 * gbps(moved, t.median_ms) / read_peak_gbps,
                   tflops=tflops(gemm_flops(m, n, k), t.median_ms))
        out.append(rec)
    qerr = kq.quantization_error(w, qw)
    for rec in out:
        rec["quant_rel_fro_err_vs_fp16"] = qerr["rel_fro_err"]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default=",".join(CONFIGS))
    p.add_argument("--m", default=",".join(map(str, M_VALUES)))
    p.add_argument("--group", type=int, default=128)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--out", default="gemm_final")
    args = p.parse_args()

    cap_to_free_vram()
    hw = load_results("hw_ceilings")
    peak = hw["roofline"]["peak_dram_gbps"]  # copy kernel: the brief's reference peak
    read_peak = hw["dram_bandwidth"]["read"]["gbps"]  # read-only kernel: the roof for a GEMV (almost pure reads)
    clocks = gpu_clock_snapshot()
    records = []
    for model in args.models.split(","):
        shapes = layer_shapes(model)
        seen = {}
        for layer, (n, k) in shapes.items():
            seen.setdefault((n, k), []).append(layer)
        # q/o and k/v and gate/up share shapes: benchmark each distinct (N, K) once.
        for (n, k), layers in seen.items():
            name = "/".join(layers)
            for m in map(int, args.m.split(",")):
                for r in bench_shape(model, name, n, k, m, args.group, peak, read_peak, args.iters):
                    records.append(r)
                    if r["status"] == "ok":
                        print(f"{model:13s} {name:21s} N={n:5d} K={k:5d} M={m:3d} {r['impl']:15s} "
                              f"{r['median_ms'] * 1e3:8.1f} us {r['gbps']:6.1f} GB/s ({r['pct_of_peak_dram']:4.1f}%)",
                              flush=True)
                torch.cuda.empty_cache()

    # Speedup of the fused kernel over each baseline, per (model, layer, M).
    idx = {(r["model"], r["layer"], r["M"], r["impl"]): r for r in records if r["status"] == "ok"}
    speedups = []
    for (model, layer, m, impl), r in idx.items():
        if impl != "int4_fused":
            continue
        row = {"model": model, "layer": layer, "N": r["N"], "K": r["K"], "M": m}
        for base in ("cublas_fp16", "dequant_matmul"):
            b = idx.get((model, layer, m, base))
            if b:
                row[f"speedup_vs_{base}"] = b["median_ms"] / r["median_ms"]
        speedups.append(row)
    out = write_results(args.out, {
        "protocol": {"warmup": 10, "iters": args.iters, "l2_flush": True, "launch_guard": True,
                     "bytes_note": "weights (packed int4 + fp16 scales, or fp16) + x + y"},
        "group_size": args.group, "peak_dram_gbps": peak, "peak_dram_read_gbps": read_peak, "clocks_start": clocks,
        "clocks_end": gpu_clock_snapshot(), "records": records, "speedups": speedups,
    })
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

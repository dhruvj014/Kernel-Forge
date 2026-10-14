"""Measures the hardware ceilings that define the roofline for this GPU.

  - Device properties and DRAM bandwidth: build/cmake/kf_bandwidth (our own
    float4 read/copy/write kernels, see csrc/micro/bandwidth.cu).
  - Cross-check bandwidth: torch device-to-device copy.
  - Compute: cuBLAS GEMMs via torch.matmul. FP16 and BF16 with FP32
    accumulation (Tensor Cores), FP32 SGEMM with TF32 disabled (CUDA cores).

Output: results/hw_ceilings.json. These are measured numbers. Spec-sheet
values are never used as ceilings.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kernelforge.bench_utils import (  # noqa: E402
    REPO_ROOT,
    gbps,
    gemm_flops,
    gpu_clock_snapshot,
    tflops,
    time_cuda,
    write_results,
)

GEMM_SIZES = [4096, 6144, 8192]


def run_bandwidth_binary() -> dict:
    exe_dir = REPO_ROOT / "build" / "cmake"
    exe = exe_dir / ("kf_bandwidth.exe" if sys.platform == "win32" else "kf_bandwidth")
    if not exe.exists():
        raise FileNotFoundError(f"{exe} not found. Run the build first (.\\make.ps1 build).")
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True, timeout=600)
    return json.loads(out.stdout)


def torch_copy_bandwidth() -> dict:
    n = 512 << 20  # 512 MiB per buffer, far beyond L2
    src = torch.empty(n, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    t = time_cuda(lambda: dst.copy_(src), warmup=10, iters=50)
    moved = 2.0 * n
    return {"gbps": gbps(moved, t.median_ms), "bytes_moved": moved, "timing": t.to_dict()}


def configure_matmul_precision() -> dict:
    """Forces full-precision accumulation so the ceilings match what our kernels do."""
    m = torch.backends.cuda.matmul
    m.allow_tf32 = False
    m.allow_fp16_reduced_precision_reduction = False
    m.allow_bf16_reduced_precision_reduction = False
    if hasattr(m, "allow_fp16_accumulation"):
        m.allow_fp16_accumulation = False
    return {
        "allow_tf32": m.allow_tf32,
        "allow_fp16_reduced_precision_reduction": m.allow_fp16_reduced_precision_reduction,
        "allow_bf16_reduced_precision_reduction": m.allow_bf16_reduced_precision_reduction,
        "allow_fp16_accumulation": getattr(m, "allow_fp16_accumulation", None),
    }


def gemm_ceiling(dtype: torch.dtype) -> dict:
    per_size = []
    for n in GEMM_SIZES:
        a = torch.randn(n, n, device="cuda", dtype=dtype)
        b = torch.randn(n, n, device="cuda", dtype=dtype)
        t = time_cuda(lambda a=a, b=b: torch.matmul(a, b), warmup=10, iters=50)
        per_size.append(
            {"m": n, "n": n, "k": n, "tflops": tflops(gemm_flops(n, n, n), t.median_ms), "timing": t.to_dict()}
        )
        del a, b
        torch.cuda.empty_cache()
    best = max(per_size, key=lambda r: r["tflops"])
    return {"best_tflops": best["tflops"], "best_size": best["m"], "per_size": per_size}


def fp16_accumulate_ceiling() -> dict | None:
    """FP16 GEMM with FP16 accumulation, for context only (GeForce Ada runs it at 2x the FP32-accumulate rate)."""
    m = torch.backends.cuda.matmul
    if not hasattr(m, "allow_fp16_accumulation"):
        return None
    m.allow_fp16_accumulation = True
    try:
        return gemm_ceiling(torch.float16)
    finally:
        m.allow_fp16_accumulation = False


def main() -> None:
    assert torch.cuda.is_available(), "CUDA GPU required"
    clocks_before = gpu_clock_snapshot()

    bw = run_bandwidth_binary()
    print(f"[bandwidth] read {bw['dram_bandwidth']['read']['gbps']:.1f} GB/s, "
          f"copy {bw['dram_bandwidth']['copy']['gbps']:.1f} GB/s, "
          f"write {bw['dram_bandwidth']['write']['gbps']:.1f} GB/s")
    tcopy = torch_copy_bandwidth()
    print(f"[bandwidth] torch copy_ {tcopy['gbps']:.1f} GB/s")

    precision = configure_matmul_precision()
    gemms = {}
    for name, dt in (
        ("fp16_tc_fp32acc", torch.float16),
        ("bf16_tc_fp32acc", torch.bfloat16),
        ("fp32_simt", torch.float32),
    ):
        gemms[name] = gemm_ceiling(dt)
        print(f"[gemm] {name}: {gemms[name]['best_tflops']:.2f} TFLOPS (n={gemms[name]['best_size']})")
    fp16acc = fp16_accumulate_ceiling()
    if fp16acc:
        gemms["fp16_tc_fp16acc_context_only"] = fp16acc
        print(f"[gemm] fp16_tc_fp16acc (context only): {fp16acc['best_tflops']:.2f} TFLOPS")

    clocks_after = gpu_clock_snapshot()

    peak_bw = bw["dram_bandwidth"]["copy"]["gbps"]
    peak_tc = gemms["fp16_tc_fp32acc"]["best_tflops"]
    peak_fp32 = gemms["fp32_simt"]["best_tflops"]
    roofline = {
        "peak_dram_gbps": peak_bw,
        "peak_dram_source": "kf_bandwidth copy kernel (float4), measured",
        "peak_fp16_tc_tflops": peak_tc,
        "peak_fp16_tc_source": "cuBLAS HGEMM via torch.matmul, FP32 accumulate, measured",
        "peak_fp32_tflops": peak_fp32,
        "peak_fp32_source": "cuBLAS SGEMM via torch.matmul, TF32 off, measured",
        # Arithmetic intensity (FLOP/byte) where a kernel stops being memory bound.
        "ridge_fp16_tc_flop_per_byte": peak_tc * 1e12 / (peak_bw * 1e9),
        "ridge_fp32_flop_per_byte": peak_fp32 * 1e12 / (peak_bw * 1e9),
    }

    out = write_results(
        "hw_ceilings",
        {
            "device": bw["device"],
            "dram_bandwidth": bw["dram_bandwidth"],
            "torch_copy_bandwidth": tcopy,
            "matmul_precision_flags": precision,
            "gemm": gemms,
            "roofline": roofline,
            "clocks_before": clocks_before,
            "clocks_after": clocks_after,
        },
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

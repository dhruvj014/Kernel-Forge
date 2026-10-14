"""Roofline plot of every attention kernel on the measured ceilings of this GPU.

x: arithmetic intensity = attention FLOPs (4*B*H*N^2*D) / DRAM bytes measured
   by Nsight Compute (dram__bytes_read + write, summed over the kernel's launches)
y: achieved TFLOPS = attention FLOPs / benchmark median time
Roofs: measured DRAM copy bandwidth, FP32 CUDA-core peak (SGEMM) and FP16
Tensor Core peak (HGEMM, FP32 accumulate), all from results/hw_ceilings.json.

Representative shape B=2, H=16, N=2048, D=64, fp16, non-causal.
Output: results/plots/roofline.png and results/roofline.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kernelforge.bench_utils import RESULTS_DIR, attn_flops, load_results, write_results  # noqa: E402

SHAPE = {"B": 2, "H": 16, "N": 2048, "D": 64, "causal": False, "dtype": "fp16"}
# label -> (ncu profile label, benchmark impl name, results file holding that impl)
POINTS = {
    "naive": ("naive_fp16", "attn_naive", "attention_naive"),
    "v1 (FP32 tiles)": ("v1_fp16", "attn_v1", "attention_v1"),
    "v2 (WMMA)": ("v2_base_fp16", "attn_v2_base", "attention_final"),
    "v2 + conflict-free": ("v2_layout_fp16", "attn_v2", "attention_final"),
    "v3 (+cp.async)": ("v3_fp16", "attn_v3", "attention_final"),
    "v4 (mma.sync)": ("v4_fp16", "attn_v4", "attention_final"),
    "SDPA cuDNN": ("sdpa_cudnn_fp16", "sdpa_cudnn", "attention_final"),
}


def median_ms(results: dict, impl: str) -> float | None:
    for r in results["records"]:
        if r["impl"] == impl and r["status"] == "ok" and all(r[k] == v for k, v in SHAPE.items()):
            return r["median_ms"]
    return None


def main() -> None:
    hw = load_results("hw_ceilings")["roofline"]
    ncu = load_results("ncu_summary")["profiles"]
    flops = attn_flops(SHAPE["B"], SHAPE["H"], SHAPE["N"], SHAPE["D"], SHAPE["causal"])
    pts = []
    for label, (prof, impl, src) in POINTS.items():
        res = load_results(src)
        ms = median_ms(res, impl) if res else None
        p = ncu.get(prof)
        if ms is None or p is None:
            print(f"skip {label}: missing {'benchmark' if ms is None else 'ncu profile ' + prof}")
            continue
        dram = sum(k.get("dram_bytes") or 0 for k in p["kernels"])
        tf = flops / (ms * 1e-3) / 1e12
        pts.append({"label": label, "impl": impl, "ncu_profile": prof, "median_ms": ms, "tflops": tf,
                    "dram_bytes": dram, "intensity_flop_per_byte": flops / dram,
                    "pct_of_tc_peak": 100 * tf / hw["peak_fp16_tc_tflops"]})
        print(f"{label:20s} AI={flops / dram:9.1f} FLOP/B  {tf:6.2f} TFLOPS")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bw = hw["peak_dram_gbps"] * 1e9
    roofs = [("FP16 Tensor Core (measured)", hw["peak_fp16_tc_tflops"], "#1f77b4"),
             ("FP32 CUDA core (measured)", hw["peak_fp32_tflops"], "#7f7f7f")]
    x = [2.0**i for i in range(-1, 15)]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for name, peak, color in roofs:
        ax.plot(x, [min(peak, bw * xi / 1e12) for xi in x], color=color, linewidth=2, label=name)
    ax.text(x[1], bw * x[1] / 1e12 * 1.3, f"DRAM {hw['peak_dram_gbps']:.0f} GB/s", rotation=33, fontsize=8)
    colors = plt.cm.viridis([i / max(len(pts) - 1, 1) for i in range(len(pts))])
    for p, c in zip(pts, colors, strict=True):
        ax.scatter(p["intensity_flop_per_byte"], p["tflops"], s=60, color=c, edgecolor="black", zorder=5,
                   label=f"{p['label']}: {p['tflops']:.1f} TFLOPS")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP per DRAM byte, measured by ncu)")
    ax.set_ylabel("achieved TFLOPS")
    ax.set_title("Attention on RTX 4060 Laptop (B=2, H=16, N=2048, D=64, fp16)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    out = RESULTS_DIR / "plots" / "roofline.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    write_results("roofline", {"shape": SHAPE, "flops": flops, "ceilings": hw, "points": pts,
                               "plot": "results/plots/roofline.png"})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

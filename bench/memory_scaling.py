"""Peak extra GPU memory vs sequence length: naive (O(N^2)) vs fused (O(N)).

Extra memory = peak allocated during one call minus what was live before it
(inputs excluded). B=2, H=16, D=64, fp16, non-causal. Runs that do not fit in
the VRAM free at startup are recorded as OOM.

Output: results/memory_scaling.json and results/plots/memory_scaling.png.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_attention import make_impl, peak_extra_memory  # noqa: E402

from kernelforge.bench_utils import RESULTS_DIR, attn_scores_bytes, cap_to_free_vram, write_results  # noqa: E402

IMPLS = ["attn_naive", "attn_v1", "sdpa_cudnn"]
SEQ_LENS = [512, 1024, 2048, 4096, 8192, 16384]
B, H, D = 2, 16, 64


def measure() -> list[dict]:
    rows = []
    for n in SEQ_LENS:
        q, k, v = (torch.randn(B, H, n, D, device="cuda", dtype=torch.float16) for _ in range(3))
        for impl in IMPLS:
            fn = make_impl(impl, False)
            rec = {"impl": impl, "N": n, "input_bytes": 3 * q.numel() * q.element_size(),
                   "output_bytes": q.numel() * q.element_size()}
            try:
                rec["peak_extra_bytes"] = peak_extra_memory(lambda fn=fn, q=q, k=k, v=v: fn(q, k, v))
                rec["status"] = "ok"
            except torch.OutOfMemoryError:
                rec["status"] = "oom"
            torch.cuda.empty_cache()
            if impl == "attn_naive":
                rec["scores_matrix_bytes"] = attn_scores_bytes(B, H, n)
            rows.append(rec)
            shown = f"{rec['peak_extra_bytes'] / 2**20:9.1f} MiB" if rec["status"] == "ok" else "      OOM"
            print(f"N={n:6d} {impl:11s} {shown}", flush=True)
        del q, k, v
        torch.cuda.empty_cache()
    return rows


def plot(rows: list[dict], vram_cap: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    styles = {"attn_naive": ("#c0392b", "o", "naive (3 kernels, N x N scores)"),
              "attn_v1": ("#2471a3", "s", "KernelForge v1 (fused)"),
              "sdpa_cudnn": ("#7f8c8d", "^", "PyTorch SDPA cuDNN")}
    for impl, (color, marker, label) in styles.items():
        pts = [(r["N"], r["peak_extra_bytes"] / 2**20) for r in rows if r["impl"] == impl and r["status"] == "ok"]
        if pts:
            ax.plot(*zip(*pts, strict=True), marker=marker, color=color, label=label, linewidth=1.8)
        ooms = [r["N"] for r in rows if r["impl"] == impl and r["status"] == "oom"]
        if ooms:
            ax.scatter(ooms, [vram_cap / 2**20] * len(ooms), marker="x", color=color, s=60,
                       label=f"{impl} OOM", zorder=5)
    ax.axhline(vram_cap / 2**20, color="#444", linestyle="--", linewidth=1)
    ax.text(SEQ_LENS[0], vram_cap / 2**20 * 1.15, "usable VRAM", fontsize=8, color="#444")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length N")
    ax.set_ylabel("extra memory per call (MiB, log)")
    ax.set_title(f"Attention memory vs N (B={B}, H={H}, D={D}, fp16)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = RESULTS_DIR / "plots" / "memory_scaling.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    return out


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    vram = cap_to_free_vram()
    rows = measure()
    png = plot(rows, vram["allocator_cap_bytes"])
    out = write_results("memory_scaling", {"B": B, "H": H, "D": D, "dtype": "fp16", "vram": vram,
                                           "records": rows, "plot": str(png.relative_to(RESULTS_DIR.parent))})
    print(f"wrote {out} and {png}")


if __name__ == "__main__":
    main()

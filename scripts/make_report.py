"""Generates BENCHMARKS.md from results/*.json. Never edit BENCHMARKS.md by hand.

Sections are emitted only when their source JSON exists, so the report grows
phase by phase.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kernelforge.bench_utils import REPO_ROOT, load_results  # noqa: E402

OUT = REPO_ROOT / "BENCHMARKS.md"


def table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def fmt(x, nd=2):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "-"


def section_env(meta: dict) -> str:
    rows = [[k, meta.get(k)] for k in ("gpu", "compute_capability", "sm_count", "driver", "torch",
                                       "torch_cuda", "nvcc", "python", "os", "timestamp")]
    return "## Environment\n\n" + table(["Field", "Value"], rows)


def section_ceilings(hw: dict) -> str:
    d, bw, roof = hw["device"], hw["dram_bandwidth"], hw["roofline"]
    dev_rows = [
        ["SMs", d["sm_count"]],
        ["L2 cache", f"{d['l2_cache_bytes'] / 2**20:.0f} MiB"],
        ["Shared memory per SM", f"{d['smem_per_sm_bytes'] // 1024} KiB"],
        ["Shared memory per block (opt-in max)", f"{d['smem_per_block_optin_bytes'] // 1024} KiB"],
        ["Registers per SM", d["regs_per_sm"]],
        ["Max threads per SM", d["max_threads_per_sm"]],
        ["Max SM clock", f"{d['sm_clock_max_khz'] / 1000:.0f} MHz"],
        ["Memory bus", f"{d['mem_bus_width_bits']}-bit"],
    ]
    meas_rows = [
        ["DRAM read (float4 kernel)", f"{fmt(bw['read']['gbps'], 1)} GB/s"],
        ["DRAM copy (float4 kernel, roofline peak)", f"{fmt(bw['copy']['gbps'], 1)} GB/s"],
        ["DRAM write (float4 kernel)", f"{fmt(bw['write']['gbps'], 1)} GB/s"],
        ["torch copy_ (cross-check)", f"{fmt(hw['torch_copy_bandwidth']['gbps'], 1)} GB/s"],
        ["Theoretical DRAM from clocks (context only)", f"{fmt(d['theoretical_dram_gbps_from_clocks'], 1)} GB/s"],
    ]
    for name, g in hw["gemm"].items():
        meas_rows.append([f"cuBLAS GEMM {name} (n={g['best_size']})", f"{fmt(g['best_tflops'])} TFLOPS"])
    meas_rows += [
        ["Ridge point, FP16 Tensor Core", f"{fmt(roof['ridge_fp16_tc_flop_per_byte'], 1)} FLOP/byte"],
        ["Ridge point, FP32 CUDA core", f"{fmt(roof['ridge_fp32_flop_per_byte'], 1)} FLOP/byte"],
    ]
    return ("## Measured hardware ceilings\n\nSource: `results/hw_ceilings.json` (bench/hw_ceilings.py).\n\n"
            + table(["Property", "Value"], dev_rows) + "\n\n" + table(["Measurement", "Result"], meas_rows))


def section_attention(att: dict, title: str, src: str) -> str:
    recs = att["records"]
    impls = att["grid"]["impls"]
    cfgs = sorted({(r["B"], r["H"], r["N"], r["D"], r["causal"], r["dtype"]) for r in recs},
                  key=lambda c: (c[5], c[4], c[3], c[2]))
    idx = {(r["B"], r["H"], r["N"], r["D"], r["causal"], r["dtype"], r["impl"]): r for r in recs}
    headers = ["dtype", "causal", "B", "H", "N", "D"] + [f"{i} ms (TFLOPS)" for i in impls]
    if "attn_naive" in impls:
        headers += [f"{i} speedup vs naive" for i in impls if i != "attn_naive"]
    rows = []
    for B, H, N, D, causal, dt in cfgs:
        row = [dt, "yes" if causal else "no", B, H, N, D]
        for i in impls:
            r = idx.get((B, H, N, D, causal, dt, i))
            if r is None:
                row.append("-")
            elif r["status"] == "ok":
                row.append(f"{fmt(r['median_ms'], 3)} ({fmt(r['tflops'])})")
            else:
                row.append(r["status"])
        if "attn_naive" in impls:
            naive = idx.get((B, H, N, D, causal, dt, "attn_naive"))
            for i in impls:
                if i == "attn_naive":
                    continue
                r = idx.get((B, H, N, D, causal, dt, i))
                ok = naive and r and naive["status"] == "ok" and r["status"] == "ok"
                row.append(f"{naive['median_ms'] / r['median_ms']:.1f}x" if ok else "-")
        rows.append(row)
    p = att["protocol"]
    note = (f"Median of {p['iters']} timed runs after {p['warmup']} warmup runs. "
            f"TFLOPS use 4*B*H*N^2*D (halved for causal). `oom` means the 8 GB card ran out of memory.")
    return f"## {title}\n\nSource: `{src}`. {note}\n\n" + table(headers, rows)


def section_correctness() -> str | None:
    parts = []
    for path in sorted((REPO_ROOT / "results").glob("correctness_*.json")):
        c = json.loads(path.read_text(encoding="utf-8"))
        rows = [[dt, w["cases"], f"{w['max_abs_err']:.2e}", f"{w['max_rel_err']:.2e}"]
                for dt, w in c["worst_by_dtype"].items()]
        parts.append(f"**{c['suite']}**: {c['passed_cases']}/{c['total_cases']} cases passed "
                     f"(rel err = {c['rel_err_definition']}).\n\n"
                     + table(["dtype", "cases", "worst max abs err", "worst max rel err"], rows))
    return "## Correctness vs FP32 reference\n\n" + "\n\n".join(parts) if parts else None


def section_ncu(ncu: dict) -> str:
    rows = []
    for label, prof in ncu["profiles"].items():
        for k in prof["kernels"]:
            name = str(k["kernel"]).split("(")[0][:48]
            rows.append([label, name, fmt((k.get("duration_ns") or 0) / 1e3, 1),
                         fmt(k.get("achieved_occupancy_pct"), 1), k.get("registers_per_thread"),
                         fmt(k.get("dram_throughput_pct"), 1), fmt(k.get("l2_hit_rate_pct"), 1),
                         ", ".join(s for s, _ in k.get("top_stalls", []))])
    return ("## Nsight Compute summary\n\nSource: `results/ncu_summary.json`.\n\n"
            + table(["profile", "kernel", "us", "occupancy %", "regs", "DRAM %", "L2 hit %", "top stalls"], rows))


def main() -> None:
    parts = ["# KernelForge benchmarks",
             "Generated by `scripts/make_report.py` from `results/*.json`. Do not edit by hand."]
    hw = load_results("hw_ceilings")
    if hw:
        parts += [section_env(hw["meta"]), section_ceilings(hw)]
    att = load_results("attention_naive")
    if att:
        parts.append(section_attention(att, "Attention: naive baseline vs PyTorch SDPA",
                                       "results/attention_naive.json"))
    corr = section_correctness()
    if corr:
        parts.append(corr)
    ncu = load_results("ncu_summary")
    if ncu:
        parts.append(section_ncu(ncu))
    OUT.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

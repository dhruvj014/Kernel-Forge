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
    power = meta.get("power") or {}
    if power:
        rows.append(["power", f"AC: {power.get('on_ac_power')}, plan: {power.get('power_plan')}"])
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


def section_power(hw: dict, hw_low: dict) -> str:
    """Same ceilings measured under the laptop's default 30 W GPU cap, for context."""
    def row(label, fn):
        lo, hi = fn(hw_low), fn(hw)
        return [label, fmt(lo), fmt(hi), f"{hi / lo:.2f}x"]

    rows = [row("DRAM copy (GB/s)", lambda h: h["dram_bandwidth"]["copy"]["gbps"])]
    for name in hw["gemm"]:
        if name in hw_low["gemm"]:
            rows.append(row(f"{name} (TFLOPS)", lambda h, n=name: h["gemm"][n]["best_tflops"]))
    lo_w = hw_low["clocks_after"].get("status", {}).get("power.draw", "-")
    hi_w = hw["clocks_after"].get("status", {}).get("power.draw", "-")
    return ("## Power limit sensitivity\n\n"
            "Source: `results/hw_ceilings_30w.json` (vendor quiet mode, GPU capped at 30 W) vs "
            "`results/hw_ceilings.json` (performance mode, 135 W cap). Power draw at the end of each run: "
            f"{lo_w} vs {hi_w}. Every result in this report uses the 135 W configuration.\n\n"
            + table(["Measurement", "30 W cap", "135 W cap", "ratio"], rows))


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
    note = (f"Median of {p['iters']} timed runs after at least {p['warmup']} warmup runs. "
            "TFLOPS use 4*B*H*N^2*D (halved for causal). `oom` means the run did not fit in the VRAM "
            "free at startup (sysmem fallback disabled through the allocator cap).")
    if p.get("launch_guard"):
        note += " Host launch overhead is excluded by a GPU spin before each timed run."
    avail = att.get("sdpa_backend_availability")
    if avail:
        missing = [k for k, v in avail.items() if not v.startswith("available")]
        if missing:
            note += (f" Not available in this torch build: {', '.join(missing)}. "
                     f"cuDNN version: {att.get('cudnn_version')}.")
    ceil = att.get("peak_fp16_tc_tflops")
    if ceil:
        note += f" Measured FP16 Tensor Core ceiling (FP32 accumulate): {fmt(ceil)} TFLOPS."
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
            name = str(k["kernel"]).split("(")[0].split("::")[-1][:40]
            smem = (k.get("smem_static_per_block_bytes") or 0) + (k.get("smem_dynamic_per_block_bytes") or 0)
            conflicts = (k.get("smem_bank_conflicts_ld") or 0) + (k.get("smem_bank_conflicts_st") or 0)
            rows.append([label, name, fmt((k.get("duration_ns") or 0) / 1e3, 1),
                         fmt(k.get("achieved_occupancy_pct"), 1), fmt(k.get("registers_per_thread"), 0),
                         fmt(smem, 0), fmt(k.get("dram_throughput_pct"), 1), fmt(k.get("l1tex_throughput_pct"), 1),
                         fmt(k.get("l2_hit_rate_pct"), 1),
                         fmt(conflicts, 0), fmt(k.get("tensor_pipe_active_pct"), 1),
                         ", ".join(s for s, _ in k.get("top_stalls", []))])
    return ("## Nsight Compute summary\n\nSource: `results/ncu_summary.json`. Profiled with `--set full` at the "
            "base clock `ncu` locks to, so durations are longer than the benchmark medians.\n\n"
            + table(["profile", "kernel", "us", "occupancy %", "regs", "smem B/block", "DRAM %", "L1/TEX %", "L2 hit %",
                     "smem bank conflicts", "tensor pipe %", "top stalls"], rows))


def main() -> None:
    parts = ["# KernelForge benchmarks",
             "Generated by `scripts/make_report.py` from `results/*.json`. Do not edit by hand."]
    hw = load_results("hw_ceilings")
    if hw:
        parts += [section_env(hw["meta"]), section_ceilings(hw)]
        hw_low = load_results("hw_ceilings_30w")
        if hw_low:
            parts.append(section_power(hw, hw_low))
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

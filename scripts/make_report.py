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


def section_v1(att: dict, naive: dict | None, hw: dict | None) -> str:
    """Fused v1 vs naive (from the Phase 1 run) and SDPA (re-measured in the same run as v1)."""
    key = lambda r: (r["B"], r["H"], r["N"], r["D"], r["causal"], r["dtype"])  # noqa: E731
    idx = {(key(r), r["impl"]): r for r in att["records"]}
    nidx = {key(r): r for r in (naive or {}).get("records", []) if r["impl"] == "attn_naive"}
    fp32_peak = hw["roofline"]["peak_fp32_tflops"] if hw else None
    cfgs = sorted({key(r) for r in att["records"]}, key=lambda c: (c[5], c[4], c[3], c[2]))
    rows = []
    for c in cfgs:
        v1, cud = idx.get((c, "attn_v1")), idx.get((c, "sdpa_cudnn"))
        nv = nidx.get(c)
        ok = lambda r: r is not None and r["status"] == "ok"  # noqa: E731
        rows.append([
            c[5], "yes" if c[4] else "no", c[2], c[3],
            f"{fmt(nv['median_ms'], 2)}" if ok(nv) else (nv["status"] if nv else "-"),
            f"{fmt(v1['median_ms'], 3)} ({fmt(v1['tflops'])})" if ok(v1) else (v1["status"] if v1 else "-"),
            f"{fmt(cud['median_ms'], 3)} ({fmt(cud['tflops'])})" if ok(cud) else "-",
            f"{nv['median_ms'] / v1['median_ms']:.1f}x" if ok(nv) and ok(v1) else "-",
            f"{100 * v1['tflops'] / fp32_peak:.0f}%" if ok(v1) and fp32_peak else "-",
            f"{100 * v1['tflops'] / cud['tflops']:.0f}%" if ok(v1) and ok(cud) else "-",
        ])
    return ("## Attention v1: fused, online softmax, FP32 CUDA cores\n\n"
            "Sources: `results/attention_v1.json` (v1 and SDPA measured in the same run) and "
            "`results/attention_naive.json` (naive). B=2, H=16. v1 does its math as FP32 FMAs on CUDA cores, "
            "so its ceiling is the measured FP32 peak, not the Tensor Core peak.\n\n"
            + table(["dtype", "causal", "N", "D", "naive ms", "v1 ms (TFLOPS)", "sdpa_cudnn ms (TFLOPS)",
                     "v1 speedup vs naive", "v1 % of FP32 peak", "v1 % of cuDNN"], rows))


def section_v1_sweep(sw: dict) -> str:
    rows = [[r["D"], r["config"], r["Br"], r["Bc"], r["registers"], f"{r['smem_bytes'] / 1024:.1f}",
             r["blocks_per_sm"], f"{r['occupancy_pct']:.0f}% ({r['limited_by']})", fmt(r["geomean_tflops"])]
            for r in sw["configs"]]
    best = ", ".join(f"D={d}: Br={b['Br']}, Bc={b['Bc']}" for d, b in sw["best"].items())
    return ("## Attention v1 tile sweep\n\nSource: `results/attention_v1_sweep.json`. fp16, B=2, H=16, 256 threads "
            "per block, geometric mean over N in {1024, 2048, 4096} causal and full. Occupancy is theoretical "
            f"(registers from ptxas, shared memory from the tile sizes). Chosen defaults: {best}.\n\n"
            + table(["D", "config", "Br", "Bc", "regs", "smem KB", "blocks/SM", "occupancy",
                     "geomean TFLOPS"], rows))


def section_memory(mem: dict) -> str:
    impls = list(dict.fromkeys(r["impl"] for r in mem["records"]))
    ns = sorted({r["N"] for r in mem["records"]})
    idx = {(r["impl"], r["N"]): r for r in mem["records"]}
    rows = []
    for n in ns:
        row = [n]
        for i in impls:
            r = idx.get((i, n))
            if r and r["status"] == "ok":
                row.append(f"{r['peak_extra_bytes'] / 2**20:.1f}")
            else:
                row.append(r["status"] if r else "-")
        rows.append(row)
    return (f"## Attention memory vs sequence length\n\nSource: `results/memory_scaling.json`. Extra memory per "
            f"call beyond the inputs, B={mem['B']}, H={mem['H']}, D={mem['D']}, fp16.\n\n"
            f"![memory scaling]({mem['plot'].replace(chr(92), '/')})\n\n"
            + table(["N"] + [f"{i} MiB" for i in impls], rows))


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
            ffma_warp = (k.get("ffma_thread_inst") or 0) / 32
            lds_per_ffma = (k.get("shared_ld_warp_inst") or 0) / ffma_warp if ffma_warp else None
            rows.append([label, name, fmt((k.get("duration_ns") or 0) / 1e3, 1),
                         fmt(k.get("achieved_occupancy_pct"), 1), fmt(k.get("registers_per_thread"), 0),
                         fmt(smem, 0), fmt(k.get("dram_throughput_pct"), 1), fmt(k.get("l1tex_throughput_pct"), 1),
                         fmt(k.get("l2_hit_rate_pct"), 1),
                         fmt(conflicts, 0), fmt(lds_per_ffma, 2), fmt(k.get("tensor_pipe_active_pct"), 1),
                         ", ".join(s for s, _ in k.get("top_stalls", []))])
    return ("## Nsight Compute summary\n\nSource: `results/ncu_summary.json`. Profiled with `--set full` at the "
            "base clock `ncu` locks to, so durations are longer than the benchmark medians.\n\n"
            + table(["profile", "kernel", "us", "occupancy %", "regs", "smem B/block", "DRAM %", "L1/TEX %", "L2 hit %",
                     "smem bank conflicts", "LDS per FFMA", "tensor pipe %", "top stalls"], rows))


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
    v1 = load_results("attention_v1")
    if v1:
        parts.append(section_v1(v1, att, hw))
    sweep = load_results("attention_v1_sweep")
    if sweep:
        parts.append(section_v1_sweep(sweep))
    mem = load_results("memory_scaling")
    if mem:
        parts.append(section_memory(mem))
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

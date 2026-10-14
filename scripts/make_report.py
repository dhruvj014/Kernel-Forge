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


ABLATION = [  # (impl, step description, results file)
    ("attn_naive", "naive: 3 kernels, N x N scores in DRAM", "attention_naive"),
    ("attn_v1", "v1: fused tiles + online softmax, FP32 CUDA cores", "attention_v1"),
    ("attn_v2_base", "v2: WMMA Tensor Cores (fp16 tiles, padded smem)", "attention_final"),
    ("attn_v2", "+ conflict-free softmax/O layout", "attention_final"),
    ("attn_v3", "+ cp.async double-buffered K/V (v3)", "attention_final"),
    ("attn_v4", "v4: mma.sync, softmax + O in registers, ldmatrix, XOR swizzle, tuned tiles", "attention_final"),
]


def _geomean(xs: list[float]) -> float:
    import math

    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def section_ablation(hw: dict | None) -> str | None:
    """Incremental speedup of each optimization at the representative shape and over the common grid."""
    srcs = {s: load_results(s) for _, _, s in ABLATION}
    if not all(srcs.values()):
        return None
    key = lambda r: (r["B"], r["H"], r["N"], r["D"], r["causal"], r["dtype"])  # noqa: E731
    by = {impl: {key(r): r for r in srcs[s]["records"] if r["impl"] == impl and r["status"] == "ok"}
          for impl, _, s in ABLATION}
    rep = (2, 16, 2048, 64, False, "fp16")
    peak = hw["roofline"]["peak_fp16_tc_tflops"] if hw else None
    rows, prev = [], None
    for impl, desc, src in ABLATION:
        r = by[impl].get(rep)
        if r is None:
            continue
        # Grid speedup vs the previous step: geometric mean over configs both measured.
        if prev:
            common = [k for k in by[impl] if k in by[prev]]
            grid = _geomean([by[prev][k]["median_ms"] / by[impl][k]["median_ms"] for k in common])
            step = f"{by[prev][rep]['median_ms'] / r['median_ms']:.2f}x"
            grid_s = f"{grid:.2f}x over {len(common)} configs"
        else:
            step, grid_s = "-", "-"
        rows.append([desc, f"`{src}.json`", fmt(r["median_ms"], 3), fmt(r["tflops"]),
                     f"{100 * r['tflops'] / peak:.0f}%" if peak else "-", step, grid_s])
        prev = impl
    return ("## Attention optimization ablation\n\nEach row adds one optimization to the row above. Representative "
            "shape B=2, H=16, N=2048, D=64, fp16, non-causal. The grid column is the geometric-mean speedup over "
            "the previous step across every (N, D, causal, dtype) both versions ran.\n\n"
            + table(["step", "source", "ms", "TFLOPS", "% of TC peak", "speedup vs previous (rep. shape)",
                     "speedup vs previous (grid)"], rows))


def section_final(att: dict, hw: dict | None) -> str:
    key = lambda r: (r["N"], r["D"], r["causal"], r["dtype"])  # noqa: E731
    idx = {(key(r), r["impl"]): r for r in att["records"] if r["status"] == "ok"}
    cfgs = sorted({key(r) for r in att["records"]}, key=lambda c: (c[3], c[2], c[1], c[0]))
    peak = hw["roofline"]["peak_fp16_tc_tflops"] if hw else None
    rows = []
    for c in cfgs:
        v4, cu, ef = idx.get((c, "attn_v4")), idx.get((c, "sdpa_cudnn")), idx.get((c, "sdpa_efficient"))
        if not v4:
            continue
        rows.append([c[3], "yes" if c[2] else "no", c[0], c[1], f"{fmt(v4['median_ms'], 3)} ({fmt(v4['tflops'])})",
                     f"{fmt(cu['median_ms'], 3)} ({fmt(cu['tflops'])})" if cu else "-",
                     f"{fmt(ef['median_ms'], 3)} ({fmt(ef['tflops'])})" if ef else "-",
                     f"{100 * v4['tflops'] / cu['tflops']:.0f}%" if cu else "-",
                     f"{100 * v4['tflops'] / peak:.0f}%" if peak else "-"])
    ratios = [idx[(c, "attn_v4")]["tflops"] / idx[(c, "sdpa_cudnn")]["tflops"] for c in cfgs
              if (c, "attn_v4") in idx and (c, "sdpa_cudnn") in idx]
    summary = (f"Across {len(ratios)} configs, v4 throughput relative to cuDNN SDPA: geometric mean "
               f"{100 * _geomean(ratios):.0f}%, min {100 * min(ratios):.0f}%, max {100 * max(ratios):.0f}%.")
    return ("## Final attention kernel (v4) vs PyTorch SDPA\n\nSource: `results/attention_final.json` (all kernels "
            f"measured in the same run). B=2, H=16. {summary}\n\n"
            + table(["dtype", "causal", "N", "D", "v4 ms (TFLOPS)", "sdpa_cudnn ms (TFLOPS)",
                     "sdpa_efficient ms (TFLOPS)", "v4 % of cuDNN", "v4 % of TC peak"], rows))


def section_v4_sweep(sw: dict) -> str:
    rows = [[r["D"], r["config"], r["warps"], r["Br"], r["Bc"], r["registers"], f"{r['smem_bytes'] / 1024:.0f}",
             r["blocks_per_sm"], f"{r['occupancy_pct']:.0f}% ({r['limited_by']})", fmt(r["geomean_tflops"])]
            for r in sw["configs"]]
    best = ", ".join(f"D={d}: {b['warps']} warps, Bc={b['Bc']}" for d, b in sw["best"].items())
    return ("## Attention v4 occupancy sweep\n\nSource: `results/attention_v4_sweep.json`. fp16, B=2, H=16, geometric "
            f"mean over N in {{1024, 2048, 4096}} causal and full. Chosen defaults: {best}.\n\n"
            + table(["D", "config", "warps", "Br", "Bc", "regs", "smem KB", "blocks/SM", "occupancy",
                     "geomean TFLOPS"], rows))


def section_roofline(rf: dict) -> str:
    rows = [[p["label"], fmt(p["intensity_flop_per_byte"], 1), fmt(p["tflops"]), f"{p['pct_of_tc_peak']:.0f}%"]
            for p in rf["points"]]
    return ("## Roofline\n\nSource: `results/roofline.json` (scripts/plot_roofline.py). Intensity uses DRAM bytes "
            "measured by Nsight Compute, throughput uses the benchmark median.\n\n"
            "![roofline](results/plots/roofline.png)"
            "\n\n" + table(["kernel", "FLOP per DRAM byte", "TFLOPS", "% of TC peak"], rows))


LAYER_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def section_gemm(gm: dict) -> str:
    """INT4 fused kernels vs cuBLAS FP16 and PyTorch dequant-then-matmul, per layer shape and M."""
    ok = [r for r in gm["records"] if r["status"] == "ok"]
    idx = {(r["model"], r["layer"], r["M"], r["impl"]): r for r in ok}
    models = list(dict.fromkeys(r["model"] for r in ok))
    ms = sorted({r["M"] for r in ok})
    parts = [f"## INT4 weight-only GEMM\n\nSource: `results/gemm_final.json` (bench/bench_gemm.py). Group size "
             f"{gm['group_size']}, symmetric INT4 with fp16 scales, fp16 activations. L2 is flushed (by a read) "
             "before every timed run. Bandwidth counts packed weights + scales + x + y (fp16 weights for cuBLAS). "
             f"Measured peaks: copy {fmt(gm['peak_dram_gbps'], 1)} GB/s (the brief's reference), read-only "
             f"{fmt(gm.get('peak_dram_read_gbps'), 1)} GB/s (the real roof of a GEMV, which is almost pure reads)."]
    rows = []
    for model in models:
        layers = [lay for lay in dict.fromkeys(r["layer"] for r in ok if r["model"] == model)]
        for lay in layers:
            f, c = idx.get((model, lay, 1, "int4_fused")), idx.get((model, lay, 1, "cublas_fp16"))
            d = idx.get((model, lay, 1, "dequant_matmul"))
            if not (f and c):
                continue
            rows.append([model, lay, f"{f['N']} x {f['K']}", fmt(f["median_ms"] * 1e3, 1), fmt(c["median_ms"] * 1e3, 1),
                         fmt(d["median_ms"] * 1e3, 1) if d else "-", f"{c['median_ms'] / f['median_ms']:.2f}x",
                         fmt(f["gbps"], 1), f"{f['pct_of_peak_dram']:.0f}%", f"{f.get('pct_of_read_peak', 0):.0f}%",
                         f"{c['pct_of_peak_dram']:.0f}%"])
    parts.append("### Decode (M = 1) per layer shape\n\n" + table(
        ["model", "layers", "N x K", "INT4 us", "cuBLAS FP16 us", "dequant+matmul us", "speedup vs cuBLAS",
         "INT4 GB/s", "INT4 % copy peak", "INT4 % read peak", "cuBLAS % copy peak"], rows))
    # One decoder layer = q, k, v, o, gate, up, down. Shared shapes are benchmarked once and counted per layer.
    agg = []
    for model in models:
        for m in ms:
            tot = {}
            for impl in ("int4_fused", "cublas_fp16"):
                t = 0.0
                for name in LAYER_ORDER:
                    rec = next((r for (mo, lay, mm, im), r in idx.items()
                                if mo == model and mm == m and im == impl and name in lay.split("/")), None)
                    t = t + rec["median_ms"] if rec and t is not None else None
                tot[impl] = t
            if tot["int4_fused"] and tot["cublas_fp16"]:
                agg.append([model, m, fmt(tot["int4_fused"] * 1e3, 1), fmt(tot["cublas_fp16"] * 1e3, 1),
                            f"{tot['cublas_fp16'] / tot['int4_fused']:.2f}x"])
    parts.append("### All 7 linear layers of one decoder layer, by batch M\n\nSum of per-layer medians "
                 "(q, k, v, o, gate, up, down). M <= 8 uses the GEMV kernel, M > 8 the Tensor Core GEMM.\n\n"
                 + table(["model", "M", "INT4 fused us", "cuBLAS FP16 us", "speedup"], agg))
    return "\n\n".join(parts)


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
    abl = section_ablation(hw)
    if abl:
        parts.append(abl)
    final = load_results("attention_final")
    if final:
        parts.append(section_final(final, hw))
    sw4 = load_results("attention_v4_sweep")
    if sw4:
        parts.append(section_v4_sweep(sw4))
    rf = load_results("roofline")
    if rf:
        parts.append(section_roofline(rf))
    gm = load_results("gemm_final")
    if gm:
        parts.append(section_gemm(gm))
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

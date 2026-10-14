"""Extracts the protocol metrics (build brief section 4) from an .ncu-rep file
and merges them into results/ncu_summary.json under the given label.

Per kernel: duration, achieved occupancy, registers/thread, shared memory per
block, DRAM throughput %, L2 hit rate, shared-memory bank conflicts, tensor
pipe utilization, top-3 warp stall reasons, plus DRAM bytes and FLOP counts
used for the roofline.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kernelforge.bench_utils import RESULTS_DIR, env_metadata  # noqa: E402

# Each key maps to candidate metric names, first match wins. Names differ
# between Nsight Compute versions and architectures (Ada exposes the _v2
# tensor-pipe counters and gpu__dram_throughput instead of dram__throughput).
METRICS = {
    "duration_ns": ["gpu__time_duration.sum"],
    "achieved_occupancy_pct": ["sm__warps_active.avg.pct_of_peak_sustained_active"],
    "theoretical_occupancy_pct": ["sm__maximum_warps_per_active_cycle_pct"],
    "registers_per_thread": ["launch__registers_per_thread"],
    "smem_static_per_block_bytes": ["launch__shared_mem_per_block_static"],
    "smem_dynamic_per_block_bytes": ["launch__shared_mem_per_block_dynamic"],
    "block_size": ["launch__block_size"],
    "grid_size": ["launch__grid_size"],
    "dram_throughput_pct": ["gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
                            "dram__throughput.avg.pct_of_peak_sustained_elapsed"],
    "memory_throughput_pct": ["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"],
    "l1tex_throughput_pct": ["l1tex__throughput.avg.pct_of_peak_sustained_elapsed"],
    "sm_throughput_pct": ["sm__throughput.avg.pct_of_peak_sustained_elapsed"],
    "l2_hit_rate_pct": ["lts__t_sector_hit_rate.pct"],
    "l1_hit_rate_pct": ["l1tex__t_sector_hit_rate.pct"],
    "smem_bank_conflicts_ld": ["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"],
    "smem_bank_conflicts_st": ["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"],
    "tensor_pipe_active_pct": ["sm__pipe_tensor_op_hmma_cycles_active_v2.avg.pct_of_peak_sustained_active",
                               "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active"],
    "dram_bytes_read": ["dram__bytes_read.sum"],
    "dram_bytes_write": ["dram__bytes_write.sum"],
    # Not in --set full as plain sums. scripts/run_ncu.ps1 requests them with --metrics.
    "ffma_thread_inst": ["smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"],
    "fadd_thread_inst": ["smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"],
    "fmul_thread_inst": ["smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"],
}
STALL_RE = re.compile(r"^smsp__average_warps_issue_stalled_(\w+?)_per_issue_active\.ratio$")


def find_ncu() -> str:
    exe = shutil.which("ncu") or shutil.which("ncu.bat")
    if not exe:
        raise FileNotFoundError("ncu not on PATH")
    return exe


def to_num(s: str):
    s = s.replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return s or None


def summarize(rep: Path) -> list[dict]:
    out = subprocess.run(
        [find_ncu(), "--import", str(rep), "--csv", "--page", "raw", "--print-units", "base"],
        capture_output=True, text=True, check=True,
    ).stdout
    # Drop any banner lines before the CSV header.
    start = out.find('"ID"')
    rows = list(csv.reader(io.StringIO(out[start:])))
    header, data = rows[0], rows[2:]  # rows[1] holds units
    kernels = []
    for row in data:
        if len(row) != len(header):
            continue
        rec = dict(zip(header, row, strict=True))
        k = {"kernel": rec.get("Kernel Name"), "id": to_num(rec.get("ID", ""))}
        for key, candidates in METRICS.items():
            for metric in candidates:
                if rec.get(metric, "") != "":
                    k[key] = to_num(rec[metric])
                    break
        stalls = {}
        for col, val in rec.items():
            m = STALL_RE.match(col)
            if m and m.group(1) not in ("not_issued", "selected"):
                v = to_num(val)
                if isinstance(v, float):
                    stalls[m.group(1)] = v
        k["top_stalls"] = sorted(stalls.items(), key=lambda kv: -kv[1])[:3]
        rd, wr = k.get("dram_bytes_read"), k.get("dram_bytes_write")
        if isinstance(rd, float) and isinstance(wr, float):
            k["dram_bytes"] = rd + wr
            fp32_flops = sum(
                (k.get(n) or 0) * (2 if n == "ffma_thread_inst" else 1)
                for n in ("ffma_thread_inst", "fadd_thread_inst", "fmul_thread_inst")
                if isinstance(k.get(n), float)
            )
            k["fp32_flops"] = fp32_flops
            if k["dram_bytes"] > 0:
                k["fp32_arith_intensity_flop_per_byte"] = fp32_flops / k["dram_bytes"]
        kernels.append(k)
    return kernels


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("report", type=Path)
    p.add_argument("--label", required=True)
    p.add_argument("--impl", default="")
    p.add_argument("--shape", default="")
    args = p.parse_args()

    kernels = summarize(args.report)
    total_ns = sum(k.get("duration_ns") or 0 for k in kernels)
    path = RESULTS_DIR / "ncu_summary.json"
    doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"profiles": {}}
    doc["meta"] = env_metadata()
    doc["profiles"][args.label] = {
        "impl": args.impl,
        "shape": args.shape.strip(),
        "report_file": args.report.name,
        "total_duration_ns": total_ns,
        "kernels": kernels,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print(f"[{args.label}] {len(kernels)} kernels, total {total_ns / 1e6:.3f} ms")
    for k in kernels:
        print(f"  {str(k['kernel'])[:60]:60s} {k.get('duration_ns', 0) / 1e3:10.1f} us  "
              f"occ {k.get('achieved_occupancy_pct', 0):5.1f}%  dram {k.get('dram_throughput_pct', 0):5.1f}%  "
              f"l1 {k.get('l1tex_throughput_pct', 0):5.1f}%  "
              f"regs {k.get('registers_per_thread')}  stalls {[s for s, _ in k['top_stalls']]}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

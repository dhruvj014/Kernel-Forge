"""Br x Bc tile sweep for fused attention v1.

For each head dim and each (Br, Bc) config: registers (from results/ptxas.json),
shared memory per block, theoretical blocks per SM and occupancy, and measured
TFLOPS at B=2, H=16, fp16, N in {1024, 2048, 4096}, causal and full.

Output: results/attention_v1_sweep.json. The default config per head dim in
csrc/attention/attn_v1_fp32.cu is the best geometric-mean TFLOPS here.
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_attention import bench_one  # noqa: E402

from kernelforge import ops  # noqa: E402
from kernelforge.bench_utils import RESULTS_DIR, cap_to_free_vram, gpu_clock_snapshot, write_results  # noqa: E402

NT = 256  # threads per block, fixed in attn_v1_fp32.cu


def ptxas_regs(dtype_tag: str, d: int, br: int, bc: int) -> int | None:
    path = RESULTS_DIR / "ptxas.json"
    if not path.exists():
        return None
    for info in json.loads(path.read_text(encoding="utf-8"))["kernels"].values():
        name = info.get("demangled", "")
        if "attn_v1_kernel" in name and f"<{dtype_tag}, (int){d}, (int){br}, (int){bc}," in name:
            return info.get("registers")
    return None


def occupancy(regs: int | None, smem: int, props) -> dict:
    """Theoretical resident blocks per SM from the three per-SM limits."""
    by_threads = props.max_threads_per_multi_processor // NT
    # 1 KB per block is reserved by the runtime on sm_8x.
    by_smem = props.shared_memory_per_multiprocessor // (smem + 1024)
    by_regs = props.regs_per_multiprocessor // (regs * NT) if regs else by_threads
    blocks = min(by_threads, by_smem, by_regs)
    limiter = min((("threads", by_threads), ("smem", by_smem), ("registers", by_regs)), key=lambda kv: kv[1])[0]
    warps = blocks * NT // 32
    max_warps = props.max_threads_per_multi_processor // 32
    return {"blocks_per_sm": blocks, "occupancy_pct": 100.0 * warps / max_warps, "limited_by": limiter}


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    cap_to_free_vram()
    props = torch.cuda.get_device_properties(0)
    clocks = gpu_clock_snapshot()
    rows = []
    for d in (64, 128):
        for cfg, (br, bc) in enumerate(ops.ATTN_V1_CONFIGS):
            smem = 4 * ((br + bc) * (d + 1) + br * (bc + 1))
            regs = ptxas_regs("__half", d, br, bc)
            occ = occupancy(regs, smem, props)
            perf = []
            for n in (1024, 2048, 4096):
                for causal in (False, True):
                    r = bench_one(f"attn_v1_cfg{cfg}", 2, 16, n, d, causal, "fp16", 10, 50)
                    perf.append({"N": n, "causal": causal, "median_ms": r.get("median_ms"),
                                 "tflops": r.get("tflops"), "status": r["status"]})
            ok = [p["tflops"] for p in perf if p["status"] == "ok"]
            geo = math.exp(sum(math.log(t) for t in ok) / len(ok)) if ok else None
            row = {"D": d, "config": cfg, "Br": br, "Bc": bc, "registers": regs, "smem_bytes": smem,
                   **occ, "geomean_tflops": geo, "runs": perf}
            rows.append(row)
            print(f"D={d:3d} cfg{cfg} Br={br:3d} Bc={bc:2d} regs={regs} smem={smem / 1024:5.1f}KB "
                  f"blocks/SM={occ['blocks_per_sm']} occ={occ['occupancy_pct']:4.1f}% ({occ['limited_by']}) "
                  f"geomean={geo:.2f} TFLOPS", flush=True)
    best = {}
    for d in (64, 128):
        cand = [r for r in rows if r["D"] == d and r["geomean_tflops"]]
        b = max(cand, key=lambda r: r["geomean_tflops"])
        best[str(d)] = {"config": b["config"], "Br": b["Br"], "Bc": b["Bc"], "geomean_tflops": b["geomean_tflops"]}
    out = write_results("attention_v1_sweep", {"threads_per_block": NT, "dtype": "fp16", "B": 2, "H": 16,
                                               "clocks_start": clocks, "clocks_end": gpu_clock_snapshot(),
                                               "configs": rows, "best": best})
    print(f"best: {best}\nwrote {out}")


if __name__ == "__main__":
    main()

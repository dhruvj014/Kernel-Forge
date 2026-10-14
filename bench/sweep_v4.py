"""Occupancy and tile sweep for attention v4 (mma.sync): warps per block x Bc.

Per head dim and config: registers (results/ptxas.json), shared memory,
theoretical blocks per SM and occupancy, and measured TFLOPS at B=2, H=16,
fp16, N in {1024, 2048, 4096}, causal and full. The best geometric mean per
head dim becomes the default in csrc/attention/attn_v4_kernel.cuh.

Output: results/attention_v4_sweep.json.
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


def ptxas_regs(d: int, warps: int, bc: int) -> int | None:
    path = RESULTS_DIR / "ptxas.json"
    tag = f"attn_v4_kernel<__half, (int){d}, (int){warps}, (int){bc}>"
    for info in json.loads(path.read_text(encoding="utf-8"))["kernels"].values():
        if tag in info.get("demangled", ""):
            return info.get("registers")
    return None


def occupancy(regs: int | None, smem: int, threads: int, props) -> dict:
    by_threads = props.max_threads_per_multi_processor // threads
    by_smem = props.shared_memory_per_multiprocessor // (smem + 1024)  # 1 KB reserved per block
    by_regs = props.regs_per_multiprocessor // (regs * threads) if regs else by_threads
    blocks = min(by_threads, by_smem, by_regs, 24)
    limiter = min((("threads", by_threads), ("smem", by_smem), ("registers", by_regs)), key=lambda kv: kv[1])[0]
    warps = blocks * threads // 32
    return {"blocks_per_sm": blocks, "occupancy_pct": 100.0 * warps / (props.max_threads_per_multi_processor // 32),
            "limited_by": limiter}


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    cap_to_free_vram()
    props = torch.cuda.get_device_properties(0)
    clocks = gpu_clock_snapshot()
    rows = []
    for d in (64, 128):
        for cfg, (warps, bc) in enumerate(ops.ATTN_V4_CONFIGS):
            smem = (16 * warps + 4 * bc) * d * 2
            regs = ptxas_regs(d, warps, bc)
            occ = occupancy(regs, smem, warps * 32, props)
            runs = []
            for n in (1024, 2048, 4096):
                for causal in (False, True):
                    r = bench_one(f"attn_v4_cfg{cfg}", 2, 16, n, d, causal, "fp16", 10, 50)
                    runs.append({"N": n, "causal": causal, "median_ms": r.get("median_ms"),
                                 "tflops": r.get("tflops"), "status": r["status"]})
            ok = [x["tflops"] for x in runs if x["status"] == "ok"]
            geo = math.exp(sum(math.log(x) for x in ok) / len(ok))
            rows.append({"D": d, "config": cfg, "warps": warps, "Br": 16 * warps, "Bc": bc, "registers": regs,
                         "smem_bytes": smem, **occ, "geomean_tflops": geo, "runs": runs})
            print(f"D={d:3d} cfg{cfg} warps={warps} Bc={bc:2d} regs={regs} smem={smem / 1024:5.1f}KB "
                  f"blocks/SM={occ['blocks_per_sm']} occ={occ['occupancy_pct']:4.1f}% ({occ['limited_by']}) "
                  f"geomean={geo:.2f} TFLOPS", flush=True)
    best = {}
    for d in (64, 128):
        b = max((r for r in rows if r["D"] == d), key=lambda r: r["geomean_tflops"])
        best[str(d)] = {k: b[k] for k in ("config", "warps", "Bc", "geomean_tflops")}
    out = write_results("attention_v4_sweep", {"dtype": "fp16", "B": 2, "H": 16, "clocks_start": clocks,
                                               "clocks_end": gpu_clock_snapshot(), "configs": rows, "best": best})
    print(f"best: {best}\nwrote {out}")


if __name__ == "__main__":
    main()

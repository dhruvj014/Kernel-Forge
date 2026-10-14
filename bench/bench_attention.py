"""Attention benchmark: KernelForge kernels vs PyTorch SDPA backends.

Default grid is the representative slice (B=2, H=16, all N, D in {64, 128},
causal and non-causal, fp16 and bf16). --full runs the brief's complete sweep.

Each SDPA backend is forced with torch.nn.attention.sdpa_kernel so the result
records which backend actually ran. A backend that cannot run a shape is
recorded as "unavailable" with the error, never silently replaced.

Output: results/<out>.json (default attention_naive).
"""

from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kernelforge import ops  # noqa: E402
from kernelforge.bench_utils import (  # noqa: E402
    attn_flops,
    attn_min_bytes,
    attn_scores_bytes,
    cap_to_free_vram,
    gbps,
    gpu_clock_snapshot,
    load_results,
    tflops,
    time_cuda,
    write_results,
)

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

SDPA_BACKENDS = {
    "sdpa_flash": SDPBackend.FLASH_ATTENTION,
    "sdpa_efficient": SDPBackend.EFFICIENT_ATTENTION,
    "sdpa_cudnn": SDPBackend.CUDNN_ATTENTION,
    "sdpa_math": SDPBackend.MATH,
}


def make_impl(name: str, causal: bool):
    if name == "attn_naive":
        return lambda q, k, v: ops.attn_naive(q, k, v, causal=causal)
    if name == "attn_v1":
        return lambda q, k, v: ops.flash_attn_v1(q, k, v, causal=causal)
    if name.startswith("attn_v1_cfg"):
        cfg = int(name.removeprefix("attn_v1_cfg"))
        return lambda q, k, v: ops.flash_attn_v1(q, k, v, causal=causal, config=cfg)
    if name in SDPA_BACKENDS:
        backend = SDPA_BACKENDS[name]

        def run(q, k, v):
            with sdpa_kernel(backend):
                return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

        return run
    raise ValueError(f"unknown impl {name}")


def probe_sdpa_backends() -> dict:
    """Which SDPA backends this torch build can run at all (fp16, D=64, small N).

    The Windows torch wheels are built without the FlashAttention backend, so
    sdpa_flash is recorded as unavailable here rather than silently skipped.
    """
    q = torch.randn(1, 1, 128, 64, device="cuda", dtype=torch.float16)
    out = {}
    for name, backend in SDPA_BACKENDS.items():
        try:
            with sdpa_kernel(backend):
                F.scaled_dot_product_attention(q, q, q)
            torch.cuda.synchronize()
            out[name] = "available"
        except RuntimeError as exc:
            out[name] = f"unavailable: {str(exc).splitlines()[0][:200]}"
    return out


def peak_extra_memory(fn) -> int:
    """Bytes allocated during one call beyond what was live before it."""
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del out
    return peak - base


def bench_one(impl: str, B, H, N, D, causal, dtype_name, warmup, iters) -> dict:
    dtype = DTYPES[dtype_name]
    rec = {"impl": impl, "B": B, "H": H, "N": N, "D": D, "causal": causal, "dtype": dtype_name}
    g = torch.Generator(device="cuda").manual_seed(0)
    try:
        q, k, v = (torch.randn((B, H, N, D), generator=g, device="cuda", dtype=dtype) for _ in range(3))
        fn = make_impl(impl, causal)
        call = lambda: fn(q, k, v)  # noqa: E731
        rec["peak_extra_mem_bytes"] = peak_extra_memory(call)
        t = time_cuda(call, warmup=warmup, iters=iters)
    except torch.OutOfMemoryError:
        rec["status"] = "oom"
        torch.cuda.empty_cache()
        return rec
    except RuntimeError as exc:  # e.g. "No available kernel" for a forced SDPA backend
        rec["status"] = "unavailable"
        rec["error"] = str(exc).splitlines()[0][:300]
        torch.cuda.empty_cache()
        return rec
    finally:
        q = k = v = None  # noqa: F841

    elem = torch.tensor([], dtype=dtype).element_size()
    flops = attn_flops(B, H, N, D, causal)
    rec.update(
        status="ok",
        timing=t.to_dict(),
        median_ms=t.median_ms,
        tflops=tflops(flops, t.median_ms),
        flops=flops,
        min_bytes=attn_min_bytes(B, H, N, D, elem),
        min_bytes_gbps=gbps(attn_min_bytes(B, H, N, D, elem), t.median_ms),
    )
    if impl == "attn_naive":
        rec["scores_matrix_bytes"] = attn_scores_bytes(B, H, N)
    torch.cuda.empty_cache()
    return rec


def build_grid(full: bool, dtypes: list[str]):
    seq_lens = [512, 1024, 2048, 4096, 8192]
    if full:
        bs, hs = [1, 2, 4], [8, 16, 32]
    else:
        bs, hs = [2], [16]
    return list(itertools.product(bs, hs, seq_lens, [64, 128], [False, True], dtypes))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full", action="store_true", help="full brief sweep (B 1-4, H 8-32)")
    p.add_argument("--impls", default="attn_naive,sdpa_efficient,sdpa_cudnn")
    p.add_argument("--dtypes", default="fp16,bf16")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--out", default="attention_naive")
    p.add_argument("--seq-lens", default="", help="comma list to restrict N (default: 512..8192)")
    p.add_argument("--resume", action="store_true", help="keep records already in results/<out>.json")
    args = p.parse_args()

    impls = args.impls.split(",")
    dtypes = args.dtypes.split(",")
    grid = build_grid(args.full, dtypes)
    if args.seq_lens:
        keep = {int(n) for n in args.seq_lens.split(",")}
        grid = [g for g in grid if g[2] in keep]
    # A forced backend that cannot run raises (recorded per record below), and
    # torch also emits one UserWarning per attempt explaining why. Silence those.
    warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\..*|__main__")
    vram = cap_to_free_vram()
    print(f"allocator capped at {vram['allocator_cap_bytes'] / 2**30:.2f} GiB (free VRAM at start)")
    backends = probe_sdpa_backends()
    print(f"sdpa backends: {backends}")
    clocks = gpu_clock_snapshot()
    print(f"clocks: {clocks.get('status')}", flush=True)

    records = []
    if args.resume:
        prev = load_results(args.out)
        records = prev["records"] if prev else []
    done = {(r["impl"], r["B"], r["H"], r["N"], r["D"], r["causal"], r["dtype"]) for r in records}

    for B, H, N, D, causal, dt in grid:
        for impl in impls:
            if (impl, B, H, N, D, causal, dt) in done:
                continue
            r = bench_one(impl, B, H, N, D, causal, dt, args.warmup, args.iters)
            records.append(r)
            tag = f"{impl:15s} B={B} H={H:2d} N={N:5d} D={D:3d} {'causal' if causal else 'full  '} {dt}"
            if r["status"] == "ok":
                print(f"{tag}  {r['median_ms']:9.3f} ms  {r['tflops']:7.2f} TFLOPS  "
                      f"mem+{r['peak_extra_mem_bytes'] / 2**20:8.1f} MiB", flush=True)
            else:
                print(f"{tag}  {r['status']} {r.get('error', '')}", flush=True)
            # Saved after every record so an interrupted run loses nothing (--resume continues it).
            save(args, impls, dtypes, backends, vram, clocks, records, complete=False)

    out = save(args, impls, dtypes, backends, vram, clocks, records, complete=True)
    print(f"wrote {out}")


def save(args, impls, dtypes, backends, vram, clocks, records, complete: bool):
    # Speedup of each baseline over naive on the same config (naive time / impl time).
    by_cfg = {}
    for r in records:
        if r["status"] == "ok":
            by_cfg.setdefault((r["B"], r["H"], r["N"], r["D"], r["causal"], r["dtype"]), {})[r["impl"]] = r
    comparisons = []
    for cfg, impl_recs in by_cfg.items():
        if "attn_naive" not in impl_recs:
            continue
        naive_ms = impl_recs["attn_naive"]["median_ms"]
        row = dict(zip(["B", "H", "N", "D", "causal", "dtype"], cfg, strict=True))
        for name, rr in impl_recs.items():
            if name != "attn_naive":
                row[f"{name}_speedup_over_naive"] = naive_ms / rr["median_ms"]
        comparisons.append(row)

    ceilings = load_results("hw_ceilings")
    return write_results(
        args.out,
        {
            "complete": complete,
            "protocol": {"warmup": args.warmup, "min_warmup_ms": 100.0, "iters": args.iters, "l2_flush": False,
                         "launch_guard": True,
                         "note": "warm-L2 timing, attention working sets exceed L2 at N >= 1024. "
                                 "A GPU spin before each timed run keeps host launch overhead out of the "
                                 "event interval (see kernelforge/bench_utils.time_cuda)."},
            "grid": {"full": args.full, "impls": impls, "dtypes": dtypes},
            "vram": vram,
            "sdpa_backend_availability": backends,
            "cudnn_version": torch.backends.cudnn.version(),
            "peak_fp16_tc_tflops": ceilings["roofline"]["peak_fp16_tc_tflops"] if ceilings else None,
            "clocks_start": clocks,
            "clocks_end": gpu_clock_snapshot(),
            "records": records,
            "comparisons": comparisons,
        },
    )


if __name__ == "__main__":
    main()

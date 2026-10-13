"""Shared benchmark harness. Every number in results/*.json goes through here.

Protocol (build brief section 4): 10 warmup runs, >= 50 timed runs with one
cudaEvent pair per run, report median and p10/p90. GPU clocks are logged at
the start of each benchmark because the laptop RTX 4060 boosts and throttles.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import re
import statistics
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


@dataclass
class Timing:
    median_ms: float
    p10_ms: float
    p90_ms: float
    mean_ms: float
    min_ms: float
    iters: int
    warmup: int
    l2_flushed: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _percentile(sorted_vals: list[float], p: float) -> float:
    idx = round(p * (len(sorted_vals) - 1))
    return sorted_vals[idx]


class L2Flusher:
    """Evicts L2 by zeroing a buffer twice the L2 size before a timed run.

    Used for memory-bound kernels so a small working set cannot be served
    from L2 across repeated iterations and inflate bandwidth.
    """

    def __init__(self, device: torch.device | int = 0):
        props = torch.cuda.get_device_properties(device)
        l2 = getattr(props, "L2_cache_size", 0) or (32 << 20)
        self.buf = torch.empty(2 * l2, dtype=torch.uint8, device=device)

    def __call__(self) -> None:
        self.buf.zero_()


def time_cuda(
    fn: Callable[[], object],
    warmup: int = 10,
    iters: int = 50,
    flush_l2: bool = False,
) -> Timing:
    """Times `fn` (which should only enqueue GPU work) with CUDA events."""
    flusher = L2Flusher() if flush_l2 else None
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if flusher is not None:
            flusher()  # enqueued before the start event, so not timed
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    samples = sorted(s.elapsed_time(e) for s, e in zip(starts, ends, strict=True))
    return Timing(
        median_ms=statistics.median(samples),
        p10_ms=_percentile(samples, 0.10),
        p90_ms=_percentile(samples, 0.90),
        mean_ms=statistics.fmean(samples),
        min_ms=samples[0],
        iters=iters,
        warmup=warmup,
        l2_flushed=flush_l2,
    )


# ---------------------------------------------------------------------------
# FLOP and byte accounting (exact formulas from the build brief)
# ---------------------------------------------------------------------------


def attn_flops(B: int, H: int, N: int, D: int, causal: bool) -> float:
    """Forward FLOPs: QK^T and PV are each 2*N*N*D per head. Causal counts half."""
    flops = 4.0 * B * H * N * N * D
    return flops / 2 if causal else flops


def attn_min_bytes(B: int, H: int, N: int, D: int, elem_bytes: int) -> float:
    """Lower bound on DRAM traffic: read Q, K, V once and write O once."""
    return 4.0 * B * H * N * D * elem_bytes


def attn_scores_bytes(B: int, H: int, N: int) -> float:
    """Size of one FP32 N x N score matrix per head (what naive attention materializes)."""
    return 4.0 * B * H * N * N


def gemm_flops(M: int, N: int, K: int) -> float:
    return 2.0 * M * N * K


def tflops(flops: float, ms: float) -> float:
    return flops / (ms * 1e-3) / 1e12


def gbps(nbytes: float, ms: float) -> float:
    return nbytes / (ms * 1e-3) / 1e9


# ---------------------------------------------------------------------------
# Environment and clock logging
# ---------------------------------------------------------------------------


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def gpu_clock_snapshot() -> dict:
    """Current clocks, power and temperature from nvidia-smi.

    Combines the parsed `nvidia-smi -q -d CLOCK` report with a CSV query for
    power state, power draw and temperature (which explain throttling).
    """
    snap: dict = {"timestamp": _dt.datetime.now().isoformat(timespec="seconds")}

    text = _run(["nvidia-smi", "-q", "-d", "CLOCK"])
    if text:
        clocks: dict = {}
        section = None
        # Layout: 4-space indented section headers ("    Clocks"), 8-space
        # indented "Key : Value" entries underneath.
        for line in text.splitlines():
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            key, _, val = line.strip().partition(":")
            key, val = key.strip(), val.strip()
            if indent == 4 and not val:
                section = re.sub(r"\W+", "_", key.lower()).strip("_")
                clocks[section] = {}
            elif indent >= 8 and section and val:
                clocks[section][re.sub(r"\W+", "_", key.lower()).strip("_")] = val
        snap["clocks"] = {k: v for k, v in clocks.items() if v}

    fields = "pstate,clocks.sm,clocks.mem,clocks.max.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active"
    csv = _run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"])
    if csv:
        vals = [v.strip() for v in csv.strip().splitlines()[0].split(",")]
        snap["status"] = dict(zip(fields.split(","), vals, strict=False))
    return snap


def env_metadata() -> dict:
    props = torch.cuda.get_device_properties(0)
    driver = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    nvcc = _run(["nvcc", "--version"])
    nvcc_rel = None
    if nvcc:
        m = re.search(r"release ([\d.]+), V([\d.]+)", nvcc)
        nvcc_rel = m.group(2) if m else None
    return {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "driver": driver.strip() if driver else None,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nvcc": nvcc_rel,
        "python": platform.python_version(),
        "os": platform.platform(),
        "kf_fast_math": os.environ.get("KF_FAST_MATH", "0") == "1",
    }


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------


def write_results(name: str, payload: dict) -> Path:
    """Writes results/<name>.json with environment metadata attached."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{name}.json"
    doc = {"meta": env_metadata(), **payload}
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return out


def load_results(name: str) -> dict | None:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

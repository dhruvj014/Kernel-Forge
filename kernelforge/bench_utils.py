"""Shared benchmark harness. Every number in results/*.json goes through here.

Protocol (build brief section 4): >= 10 warmup runs (and >= 100 ms of
warmup), >= 50 timed runs with one cudaEvent pair per run, report median and
p10/p90. A GPU spin before each timed run keeps Python launch overhead out of
the measured interval (see time_cuda). GPU clocks are logged at
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
import time
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
    host_enqueue_ms: float  # CPU time to enqueue one fn() call, measured during warmup
    launch_guard_ms: float  # GPU spin before each timed run (0 = disabled)

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


_SLEEP_CYCLES_PER_MS: float | None = None


def _sleep_cycles_per_ms() -> float:
    """Calibrates torch.cuda._sleep (a GPU spin loop counted in SM cycles) to milliseconds."""
    global _SLEEP_CYCLES_PER_MS
    if _SLEEP_CYCLES_PER_MS is None:
        cycles = 5_000_000
        torch.cuda._sleep(cycles)  # first call ramps clocks, not timed
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        torch.cuda._sleep(cycles)
        e.record()
        torch.cuda.synchronize()
        _SLEEP_CYCLES_PER_MS = cycles / s.elapsed_time(e)
    return _SLEEP_CYCLES_PER_MS


def cap_to_free_vram(margin_bytes: int = 256 << 20) -> dict:
    """Makes the torch caching allocator raise OutOfMemoryError at the real VRAM limit.

    On Windows (WDDM) the driver's "CUDA sysmem fallback" policy lets
    allocations that do not fit in VRAM spill into system RAM over PCIe
    instead of failing. A kernel then "runs" at a tiny fraction of its real
    speed and the benchmark records a meaningless number. Capping the
    allocator to the memory that is free at startup turns that case back into
    a clean OOM, which is recorded as such.
    """
    free, total = torch.cuda.mem_get_info()
    cap = max(free - margin_bytes, 0)
    torch.cuda.set_per_process_memory_fraction(cap / total)
    return {"free_bytes_at_start": free, "total_bytes": total, "allocator_cap_bytes": cap}


def time_cuda(
    fn: Callable[[], object],
    warmup: int = 10,
    iters: int = 50,
    flush_l2: bool = False,
    min_warmup_ms: float = 100.0,
    launch_guard: bool = True,
) -> Timing:
    """Times `fn` (which should only enqueue GPU work) with one CUDA event pair per run.

    Launch guard: if enqueueing fn() from Python takes longer than the kernel
    runs, the GPU reaches the start event, idles until the launch arrives, and
    that idle time is counted as kernel time. Measured on SDPA at N=512 this
    inflated medians by up to 5x. So before each timed run we enqueue a GPU
    spin (torch.cuda._sleep) longer than the host enqueue cost. While the GPU
    spins, the CPU enqueues start event, fn() and end event, so the interval
    between the events contains only GPU work.

    Warmup runs at least `warmup` calls and at least `min_warmup_ms` of wall
    time, so short kernels still give the laptop GPU time to reach boost clocks.
    """
    flusher = L2Flusher() if flush_l2 else None
    # Synchronize every warmup call so the wall clock measures GPU time. fn()
    # only enqueues work, so without the sync the loop queued about 1000
    # launches (until the launch queue filled) before 100 ms of wall time had
    # passed, and long kernels spent minutes in warmup.
    done, t0 = 0, time.perf_counter()
    while done < warmup or (time.perf_counter() - t0) * 1e3 < min_warmup_ms:
        fn()
        torch.cuda.synchronize()
        done += 1
        if done >= 10_000:
            break

    # Host enqueue cost per call, measured without synchronizing in between.
    n_probe = 5
    t0 = time.perf_counter()
    for _ in range(n_probe):
        fn()
    host_ms = (time.perf_counter() - t0) * 1e3 / n_probe
    torch.cuda.synchronize()

    # 5x margin: host enqueue cost is spiky (cuDNN plan lookups, GC, OS jitter),
    # and a 3x margin still left one bimodal record at N=512 in the Phase 1 run.
    guard_ms = max(1.0, 5.0 * host_ms + 0.5) if launch_guard else 0.0
    guard_cycles = int(guard_ms * _sleep_cycles_per_ms()) if launch_guard else 0

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if flusher is not None:
            flusher()  # enqueued before the start event, so not timed
        if guard_cycles:
            torch.cuda._sleep(guard_cycles)
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
        warmup=done,
        l2_flushed=flush_l2,
        host_enqueue_ms=host_ms,
        launch_guard_ms=guard_ms,
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


def power_state() -> dict:
    """AC vs battery and the active power plan. Laptop GPUs clock lower on battery."""
    state: dict = {}
    if platform.system() == "Windows":
        # BatteryStatus 2 means "on AC power" in Win32_Battery.
        bat = _run(["powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_Battery | Select-Object -First 1).BatteryStatus"])
        if bat and bat.strip().isdigit():
            state["on_ac_power"] = bat.strip() == "2"
        scheme = _run(["powercfg", "/getactivescheme"])
        if scheme:
            m = re.search(r"\((.+)\)", scheme)
            state["power_plan"] = m.group(1) if m else scheme.strip()
    return state


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
        "power": power_state(),
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

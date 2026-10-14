r"""memcheck workload through the torch extension (covers naive and v1 bindings).

Runs every attention kernel and every v1 tile config once on shapes that hit
ragged tails and causal diagonal tiles, and checks outputs against the FP32
reference:

  compute-sanitizer --tool memcheck --target-processes all .venv\Scripts\python.exe scripts\sanitize_target.py

racecheck, synccheck and initcheck use the standalone binary instead
(scripts/run_sanitizer.ps1, csrc/micro/sanitize_attn.cu). Under racecheck a
torch process spends 20+ minutes of CPU instrumenting torch's own CUDA modules.
--small runs one head at N=70 only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kernelforge import ops  # noqa: E402


def reference(q, k, v, causal):
    s = (q.float() @ k.float().transpose(-1, -2)) / q.shape[-1] ** 0.5
    if causal:
        n = q.shape[-2]
        s = s.masked_fill(torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    return torch.softmax(s, -1) @ v.float()


def main() -> None:
    small = "--small" in sys.argv
    heads, seq_lens = (1, (70,)) if small else (2, (37, 130))
    torch.manual_seed(0)
    worst = 0.0
    runs = 0
    for d in (64, 128):
        for n in seq_lens:
            q, k, v = (torch.randn(1, heads, n, d, device="cuda", dtype=torch.float16) for _ in range(3))
            for causal in (False, True):
                ref = reference(q, k, v, causal)
                outs = [ops.attn_naive(q, k, v, causal=causal)]
                outs += [ops.flash_attn_v1(q, k, v, causal=causal, config=c) for c in range(len(ops.ATTN_V1_CONFIGS))]
                for o in outs:
                    worst = max(worst, (o.float() - ref).abs().max().item())
                    runs += 1
    torch.cuda.synchronize()
    print(f"sanitize_target: {runs} kernel runs, worst abs err {worst:.2e}")
    if worst > 2e-3:
        raise SystemExit("output mismatch")


if __name__ == "__main__":
    main()

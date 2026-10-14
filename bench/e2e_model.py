"""End-to-end batch-1 decode: tokens/sec, GPU time share and memory, FP16 vs KernelForge.

For each model and config: prefill a fixed 128-token WikiText-2 prompt, then
decode 128 tokens greedily with a manual KV-cache loop (DynamicCache), timing
every decode step with a CUDA sync. tokens/sec = 1 / median step time.

GPU time share comes from a separate, profiled run (torch.profiler / CUPTI):
summed CUDA kernel time divided by wall time over the same decode steps.
Profiling adds overhead, so the tokens/sec run is never profiled. In PyTorch
eager mode each step also pays Python and launch overhead for every layer,
which the share makes visible: when the GPU is idle most of the step, faster
kernels move tokens/sec less than they move kernel time.

Configs: fp16 (SDPA), kf_attn, int4 (KFLinearInt4, SDPA), int4_kf_attn.
Output: results/e2e.json.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from ppl_eval import MODELS, wikitext_ids  # noqa: E402

from kernelforge import modules as km  # noqa: E402
from kernelforge.bench_utils import gpu_clock_snapshot, load_results, write_results  # noqa: E402


def module_bytes(model) -> int:
    return sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))


@torch.no_grad()
def decode(model, prompt: torch.Tensor, steps: int, profile: bool = False) -> dict:
    from transformers import DynamicCache

    cache = DynamicCache()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(prompt, past_key_values=cache, use_cache=True)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0
    nxt = out.logits[:, -1:].argmax(-1)
    times, tokens = [], [nxt.item()]
    prof = None
    if profile:
        from torch.profiler import ProfilerActivity, profile

        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        prof.__enter__()
    wall0 = time.perf_counter()
    for _ in range(steps):
        t = time.perf_counter()
        out = model(nxt, past_key_values=cache, use_cache=True)
        nxt = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
        tokens.append(nxt.item())
    wall = time.perf_counter() - wall0
    st = sorted(times)
    res = {"prefill_s": prefill_s, "step_ms_median": 1e3 * statistics.median(times),
           "step_ms_p10": 1e3 * st[len(st) // 10], "step_ms_p90": 1e3 * st[9 * len(st) // 10],
           "tokens_per_s": 1.0 / statistics.median(times), "tokens": tokens}
    if prof is not None:
        prof.__exit__(None, None, None)
        # Sum of device time over CUDA kernels only (excludes memcpy/memset), top-level events.
        kernel_us = sum(e.self_device_time_total for e in prof.key_averages() if e.device_type.name == "CUDA")
        res = {"profiled_wall_s": wall, "gpu_kernel_s": kernel_us / 1e6, "gpu_time_share": kernel_us / 1e6 / wall,
               "profiled_steps": steps}
    return res


@torch.no_grad()
def graph_decode(model, prompt: torch.Tensor, steps: int, profile: bool = False) -> dict:
    """Decode with one CUDA graph per step: StaticCache, decode step captured once, then replayed.

    Replay launches the whole step's kernels with no Python and no per-kernel launch overhead,
    so tokens/sec reflects GPU work. The cache position is a device tensor bumped in place, so the
    captured masks and cache writes follow it. Attention is SDPA here: a static cache needs a
    mask over its unfilled slots, which the KernelForge kernel does not take (see phase5 notes).
    """
    from transformers import StaticCache

    p = prompt.shape[1]
    cache = StaticCache(config=model.config, max_cache_len=p + steps + 8)
    out = model(prompt, cache_position=torch.arange(p, device="cuda"), past_key_values=cache, use_cache=True)
    tok_in = out.logits[:, -1:].argmax(-1).clone()
    tokens = [tok_in.item()]
    pos = torch.tensor([p], device="cuda")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # warmup off the capture stream
        for _ in range(2):
            model(tok_in, cache_position=pos, past_key_values=cache, use_cache=True)
    torch.cuda.current_stream().wait_stream(side)
    # StaticLayer writes new K/V at its own device-side counter (cumulative_length), not at
    # cache_position, and every forward advances it. The two warmup calls left it at p + 2, so
    # without this reset every replayed step wrote its K/V two slots past what the mask (built
    # from cache_position) lets the token see: each token could not attend to itself. Found
    # because graph decode diverged from eager StaticCache decode at the first token.
    for layer in cache.layers:
        layer.cumulative_length.fill_(p)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits = model(tok_in, cache_position=pos, past_key_values=cache, use_cache=True).logits
    prof = None
    if profile:
        from torch.profiler import ProfilerActivity
        from torch.profiler import profile as tprofile

        prof = tprofile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        prof.__enter__()
    times = []
    wall0 = time.perf_counter()
    for _ in range(steps):
        t = time.perf_counter()
        graph.replay()
        tok_in.copy_(logits[:, -1:].argmax(-1))
        pos.add_(1)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
        tokens.append(tok_in.item())
    wall = time.perf_counter() - wall0
    if prof is not None:
        prof.__exit__(None, None, None)
        kernel_us = sum(e.self_device_time_total for e in prof.key_averages() if e.device_type.name == "CUDA")
        return {"graph_gpu_time_share": kernel_us / 1e6 / wall}
    med = statistics.median(times)
    return {"graph_tokens_per_s": 1.0 / med, "graph_step_ms_median": 1e3 * med, "graph_tokens": tokens}


@torch.no_grad()
def static_eager_tokens(model, prompt: torch.Tensor, steps: int) -> list[int]:
    """Greedy tokens from eager decode on a StaticCache: the exact reference for the CUDA-graph path.

    Same cache, masks and kernels as the graph, so the tokens must match exactly. (Eager decode on a
    DynamicCache uses unmasked SDPA, whose rounding differs, so greedy decoding can legitimately drift.)
    """
    from transformers import StaticCache

    p = prompt.shape[1]
    cache = StaticCache(config=model.config, max_cache_len=p + steps + 8)
    out = model(prompt, cache_position=torch.arange(p, device="cuda"), past_key_values=cache, use_cache=True)
    t = out.logits[:, -1:].argmax(-1)
    toks = [t.item()]
    for i in range(steps):
        out = model(t, cache_position=torch.tensor([p + i], device="cuda"), past_key_values=cache, use_cache=True)
        t = out.logits[:, -1:].argmax(-1)
        toks.append(t.item())
    return toks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--group", type=int, default=128)
    p.add_argument("--out", default="e2e")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    prev = load_results(args.out)
    results = prev["results"] if prev and "results" in prev else {}
    clocks = gpu_clock_snapshot()
    for name in args.models.split(","):
        repo = MODELS[name]
        tok = AutoTokenizer.from_pretrained(repo)
        prompt = wikitext_ids(tok)[1000:1000 + args.prompt_len].unsqueeze(0).cuda()
        torch.cuda.reset_peak_memory_stats()
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float16, attn_implementation="sdpa")
        model = model.cuda().eval()
        res = {"repo": repo, "prompt_len": args.prompt_len, "decode_steps": args.steps, "configs": {}}
        fp16_bytes = module_bytes(model)
        for cfg, int4, kf in [("fp16", False, False), ("kf_attn", False, True), ("int4", True, False),
                              ("int4_kf_attn", True, True)]:
            if int4 and not any(isinstance(m_, km.KFLinearInt4) for m_ in model.modules()):
                km.quantize_linears_(model, args.group)
            if kf:
                km.enable_kf_attention(model)
            else:
                model.set_attn_implementation("sdpa")
            decode(model, prompt, 16)  # warmup: cuDNN plans, allocator, clocks
            torch.cuda.reset_peak_memory_stats()
            r = decode(model, prompt, args.steps)
            r["peak_mem_bytes"] = torch.cuda.max_memory_allocated()
            r["model_bytes"] = module_bytes(model)
            r.update(decode(model, prompt, 32, profile=True))
            if not kf:  # graph mode runs SDPA attention, so it is measured for the fp16 and int4 weight configs
                g = graph_decode(model, prompt, args.steps)
                g.update(graph_decode(model, prompt, 32, profile=True))
                # Captured masks and cache writes must follow the device-side position: exact match with
                # eager StaticCache decode proves the graph computes the same thing.
                g["graph_tokens_match_eager"] = g["graph_tokens"] == static_eager_tokens(model, prompt, args.steps)
                r.update(g)
                print(f"{name:13s} {cfg:13s} CUDA graph {g['graph_tokens_per_s']:6.1f} tok/s  "
                      f"GPU share {100 * g['graph_gpu_time_share']:4.1f}%  "
                      f"tokens match eager: {g['graph_tokens_match_eager']}", flush=True)
            res["configs"][cfg] = r
            print(f"{name:13s} {cfg:13s} {r['tokens_per_s']:6.1f} tok/s  step {r['step_ms_median']:6.2f} ms  "
                  f"GPU share {100 * r['gpu_time_share']:4.1f}%  peak {r['peak_mem_bytes'] / 2**30:.2f} GiB  "
                  f"model {r['model_bytes'] / 2**30:.2f} GiB", flush=True)
        res["fp16_model_bytes"] = fp16_bytes
        base = res["configs"]["fp16"]
        res["speedup_vs_fp16"] = {c: v["tokens_per_s"] / base["tokens_per_s"] for c, v in res["configs"].items()}
        res["same_tokens_as_fp16"] = {c: v["tokens"] == base["tokens"] for c, v in res["configs"].items()}
        if "graph_tokens_per_s" in base:
            res["graph_speedup_int4_vs_fp16"] = (res["configs"]["int4"]["graph_tokens_per_s"]
                                                 / base["graph_tokens_per_s"])
        results[name] = res
        del model
        torch.cuda.empty_cache()
        out = write_results(args.out, {"clocks_start": clocks, "clocks_end": gpu_clock_snapshot(), "results": results})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""WikiText-2 perplexity: FP16 baseline vs KernelForge attention vs INT4 weights vs both.

Protocol (build brief section 2.3), identical for every config: concatenate the
raw test split with "\\n\\n", tokenize once, cut into non-overlapping windows of
`--ctx` tokens (default 2048, the last partial window is dropped), and score
every next-token prediction inside each window. Perplexity = exp(mean NLL).

Configs (one model load per model, INT4 applied in place after the FP16 runs):
  fp16            original weights in fp16, PyTorch SDPA attention
  kf_attn         fp16 weights, KernelForge fused attention (ops.flash_attn)
  int4            INT4 weights (kernelforge.quantize, group 128) via KFLinearInt4, SDPA attention
  int4_kf_attn    INT4 weights and KernelForge attention

Output: results/ppl.json.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # everything was fetched by scripts/download_assets.py

from kernelforge import modules as km  # noqa: E402
from kernelforge.bench_utils import gpu_clock_snapshot, load_results, write_results  # noqa: E402

MODELS = {"tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B"}


def wikitext_ids(tokenizer) -> torch.Tensor:
    from datasets import load_dataset

    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    return tokenizer(text, return_tensors="pt").input_ids[0]


@torch.no_grad()
def perplexity(model, ids: torch.Tensor, ctx: int, max_windows: int | None) -> dict:
    n_windows = ids.numel() // ctx
    if max_windows:
        n_windows = min(n_windows, max_windows)
    nll, count = 0.0, 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for w in range(n_windows):
        x = ids[w * ctx:(w + 1) * ctx].unsqueeze(0).cuda()
        logits = model(x).logits[0]
        # Cross entropy in FP32, in chunks so a 150k-word vocabulary does not need a 1+ GB FP32 copy.
        for c in range(0, ctx - 1, 512):
            e = min(c + 512, ctx - 1)
            nll += F.cross_entropy(logits[c:e].float(), x[0, c + 1:e + 1], reduction="sum").item()
        count += ctx - 1
        del logits
    torch.cuda.synchronize()
    mean = nll / count
    return {"ppl": math.exp(mean) if mean < 700 else float("inf"), "mean_nll": mean, "windows": n_windows,
            "tokens_scored": count, "seconds": time.perf_counter() - t0, "finite": math.isfinite(mean)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--group", type=int, default=128)
    p.add_argument("--max-windows", type=int, default=0, help="0 = whole test set")
    p.add_argument("--out", default="ppl")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    prev = load_results(args.out)
    results = prev["results"] if prev and "results" in prev else {}
    clocks = gpu_clock_snapshot()
    for name in args.models.split(","):
        repo = MODELS[name]
        tok = AutoTokenizer.from_pretrained(repo)
        ids = wikitext_ids(tok)
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float16, attn_implementation="sdpa")
        model = model.cuda().eval()
        res = {"repo": repo, "dtype": "float16", "test_tokens": ids.numel(), "ctx": args.ctx, "configs": {}}
        order = [("fp16", False, False), ("kf_attn", False, True), ("int4", True, False), ("int4_kf_attn", True, True)]
        for cfg, int4, kf in order:
            if int4 and not any(isinstance(m_, km.KFLinearInt4) for m_ in model.modules()):
                res["int4_layers_replaced"] = km.quantize_linears_(model, args.group)
                res["group_size"] = args.group
            if kf:
                km.enable_kf_attention(model)
            else:
                model.set_attn_implementation("sdpa")
            km.ATTN_STATS.update(kernelforge_calls=0, sdpa_fallback_calls=0)
            torch.cuda.reset_peak_memory_stats()
            r = perplexity(model, ids, args.ctx, args.max_windows or None)
            r["peak_mem_bytes"] = torch.cuda.max_memory_allocated()
            r["attn_stats"] = dict(km.ATTN_STATS)
            res["configs"][cfg] = r
            print(f"{name:13s} {cfg:13s} ppl {r['ppl']:9.4f}  ({r['windows']} windows, {r['seconds']:.0f} s, "
                  f"attn {r['attn_stats']})", flush=True)
        base = res["configs"]["fp16"]["ppl"]
        res["delta_vs_fp16"] = {c: v["ppl"] - base for c, v in res["configs"].items()}
        results[name] = res
        del model
        torch.cuda.empty_cache()
        out = write_results(args.out, {"protocol": "WikiText-2 raw test, joined with blank lines, non-overlapping "
                                                   f"{args.ctx}-token windows, last partial window dropped",
                                       "clocks_start": clocks, "results": results})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

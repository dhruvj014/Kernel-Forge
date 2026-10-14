# Phase 5 notes: integration, end-to-end evaluation, packaging

Every number below is copied from a file in `results/`, named next to it. `BENCHMARKS.md` has the full tables, and the README's results block is generated from the same JSON.

## What was built

- **Stable API** (`kernelforge/ops.py`): `flash_attn(q, k, v, causal, scale)` and `int4_linear(x, packed, scales, group_size, bias)`, both registered with `torch.library.custom_op` (with fake implementations, so `torch.compile` can trace them), with shape, dtype and device validation and clear errors. Inference only: no autograd formulas are registered.
- **Decode-capable attention** (`csrc/attention/attn_v4_kernel.cuh`): v4 now takes queries [B, H, Nq, D] and keys/values [B, Hk, Nk, D] with H % Hk == 0. Grouped-query attention reads KV head h / (H / Hk) directly, with no `repeat_kv` copies. Nq < Nk is decode or chunked prefill against a KV cache: query i is position Nk - Nq + i, so the causal mask is aligned to the bottom-right corner. The original square interface remains as a wrapper.
- **Hugging Face integration** (`kernelforge/modules.py`): `KFLinearInt4` (drop-in `nn.Linear`), `quantize_linears_` (swaps the 7 decoder linears per layer, keeps embeddings and lm_head in fp16), and `enable_kf_attention`, which registers a `"kernelforge"` function with transformers' attention interface. Anything the kernel does not support (padding masks, dropout, other head dims) falls back to SDPA, and every fallback is counted in `ATTN_STATS` and saved with the results.
- **Evaluation**: `bench/ppl_eval.py` (WikiText-2 perplexity), `bench/e2e_model.py` (batch-1 decode, eager and CUDA graph, GPU time share, memory), `scripts/download_assets.py`.
- **Packaging**: README with a generated results block, `docs/lessons.md`, `.clang-format` (sources formatted once), and CI (`.github/workflows/ci.yml`: ruff, clang-format, nvcc build of every kernel in a CUDA container).

## Perplexity (`results/ppl.json`)

WikiText-2 raw test split, joined with blank lines, tokenized once, scored in non-overlapping 2048-token windows (last partial window dropped), identical for every config. Models in fp16. INT4: round-to-nearest, symmetric, group 128, on all 7 decoder linears of every layer.

| model | FP16 | KernelForge attention | INT4 | INT4 + KF attention |
|---|---|---|---|---|
| TinyLlama 1.1B (166 windows) | 7.9722 | 7.9723 (+0.0001) | 8.7714 (+0.799) | 8.7713 |
| Qwen2.5 1.5B (146 windows) | 9.2649 | 9.2649 (-0.0000) | 10.8537 (+1.589) | 10.8538 |

- **The fused attention kernel is numerically transparent.** It changes perplexity by at most 0.0001, with 3652 (TinyLlama) and 4088 (Qwen) kernel calls and zero SDPA fallbacks.
- **The INT4 cost is the quantizer's, not the kernels'.** The kernels match "dequantize, then matmul" to fp16 rounding (Phase 4). +10% (TinyLlama) and +17% (Qwen) perplexity is the usual price of plain round-to-nearest INT4 on small models, and is what GPTQ (error-compensating rounding) or AWQ (activation-aware scaling) exist to reduce. Both are out of scope for a kernel project, and the format is compatible with either.
- **Qwen2.5 in fp16.** Both models ship in bf16, and the INT4 kernels take fp16 activations. The risk was overflow in fp16, but Qwen2.5 1.5B's perplexity is finite and in line with its published range, so no bf16 path was needed.

## Decode (`results/e2e.json`)

Batch 1, a 128-token WikiText-2 prompt, 128 greedy decode steps, tokens/sec = 1 / median step time. GPU time share = CUDA kernel time / wall time from a separate `torch.profiler` (CUPTI) run, so profiling never touches the tokens/sec run.

**PyTorch eager mode is host bound on this machine.** FP16 TinyLlama decodes at 10.9 tokens/s (92 ms per step) with the GPU busy only 21% of each step. Every op of every layer pays Python dispatch plus WDDM kernel launch overhead. INT4 is *slower* in eager mode (7.6 tokens/s): its linear layers are 3.2x faster on the GPU, but `int4_linear` adds Python per call (reshape, validation, `torch.library` dispatch), and with the GPU idle about 80% of the time, host time is all that matters. Eager numbers are reported but say nothing about the kernels.

**CUDA graphs** capture one decode step and replay it with no Python and no per-kernel launch cost:

| model | FP16 | INT4 | speedup | GPU time share (FP16 / INT4) | weights |
|---|---|---|---|---|---|
| TinyLlama 1.1B | 92.9 tok/s | **212.3 tok/s** | **2.28x** | 91% / 81% | 2.05 GiB to 0.71 GiB |
| Qwen2.5 1.5B | 59.0 tok/s | **122.8 tok/s** | **2.08x** | 93% / 77% | 2.88 GiB to 1.06 GiB |

Why 2.1x to 2.3x end to end when the linear layers are 3.2x faster (Phase 4): the rest of each step stays fp16 and unchanged (the embedding gather, the fp16 lm_head, which reads 131 MB for TinyLlama and 467 MB for Qwen's 151936-word vocabulary on every token, attention over the KV cache, norms and rotary embeddings), and the GPU time share drops to 77 to 81% with INT4 because the remaining per-step overheads become a bigger fraction of a shorter step. Qwen gains less because its lm_head is 3.6x larger and stays fp16.

**Graph correctness.** The graph's tokens match eager decode on the same static KV cache exactly, for every config (`graph_tokens_match_eager` in `results/e2e.json`). That check found a real bug in the first version (see below).

**Peak VRAM** (eager decode): TinyLlama 2.07 GiB (FP16) to 0.75 GiB (INT4), Qwen2.5 2.96 to 1.19 GiB.

## Two decisions and their limits

- **Graph mode runs SDPA attention, not the KernelForge kernel.** CUDA graphs need fixed shapes, so they use transformers' StaticCache, which keeps the full cache length and masks unfilled slots. The fused kernel takes no mask (it handles causal and ragged lengths through Nq/Nk). Supporting it would mean passing the valid key length as a device-side value instead of a host integer. At this context length (128 to 256 tokens) attention is a small share of a decode step, while the linear layers and lm_head dominate. KernelForge attention is exercised end to end in prefill (perplexity, every window) and in eager decode (3652 and 4088 calls, no fallbacks).
- **`torch.profiler` instead of Nsight Systems.** Both read the same CUPTI kernel records. The profiler needs no system install and writes the GPU time share straight into the results JSON.

## The CUDA graph bug

The first CUDA-graph version reported 209 tokens/s for INT4 TinyLlama, but its tokens did not match eager decode. Against eager decode on the *same* static cache (identical kernels and masks) it diverged at the first decode token. transformers' `StaticLayer.update` writes new K/V at its own device-side counter (`cumulative_length`), not at the `cache_position` passed in, and every forward advances it. The two warmup calls before capture left it at prompt + 2, so each replayed step wrote its K/V two slots past what the mask (built from `cache_position`) let the token see, and no token could attend to its own key and value. On real text the effect was small enough that greedy decoding agreed for 14 to 119 tokens. Resetting the counter after warmup fixed it: graph and eager decode now match exactly, and the buggy numbers were discarded.

## Other notes

- **A low first measurement.** The first benchmark in a fresh process can read low while the laptop GPU ramps clocks up from idle (v4 at the representative shape: 24.47 TFLOPS first, then 26.43 and 26.44, vs 26.59 in Phase 3). The final attention benchmark was re-run after the GQA/KV-cache change to v4, so every attention number comes from the committed code.
- **The zero-difference check.** In a short smoke test, KernelForge attention gave logits identical to SDPA to the bit. That was verified rather than assumed: with 17 keys both kernels compute the softmax over one tile in FP32 and round identically, at 300 keys 0.006% of outputs differ by one fp16 ULP, and over full 2048-token windows perplexity differs by 0.0001.
- **Tests:** 1476 pass (159 skipped: fp32 cases of the fp16/bf16-only Tensor Core kernels), including 168 GQA and KV-cache cases for the stable attention API (`results/correctness_flash_attn_general.json`). compute-sanitizer is clean on both standalone binaries after the v4 change (`results/sanitizer.json`).

## Interview questions

**1. Your INT4 kernels are 3x faster, so why was INT4 decode slower than FP16 in PyTorch?**
Because eager-mode decode on this machine is host bound: the GPU is busy only about 21% of each step, and the rest is Python dispatch and kernel launch overhead for roughly a thousand ops. Faster kernels shrink the 21%, and the INT4 path adds a little Python per layer, so tokens/sec went down. With the step captured in a CUDA graph the GPU is busy 77 to 93% of the time, and INT4 decodes 2.28x faster than FP16 on TinyLlama.

**2. Why is the end-to-end decode speedup 2.1x to 2.3x and not the 3.2x of the linear layers?**
Amdahl's law. Only the decoder linears are INT4. The fp16 lm_head (131 MB per token for TinyLlama, 467 MB for Qwen's large vocabulary), attention over the KV cache, the embedding, norms and rotary embeddings are unchanged. Qwen gains less because its lm_head is 3.6x larger.

**3. How do you handle grouped-query attention and KV-cache decode in one kernel?**
Query head h reads KV head h / (H / Hk) through the base pointer, so nothing is copied. For decode, queries are the last Nq of Nk positions, so query i gets absolute position Nk - Nq + i, the causal mask compares key index against that, and causal block skipping uses the same offset. Nq = Nk, H = Hk is ordinary self-attention.

**4. How did you make sure the CUDA-graph numbers were real?**
By checking the graph's tokens against eager decode on the same static cache, which must match exactly. The first version did not: transformers' static cache writes K/V at an internal device-side counter, and my warmup calls had advanced it, so the captured step wrote K/V two slots past the mask. Greedy decoding still produced fluent text for dozens of tokens, so only the exact comparison exposed it.

**5. INT4 raised perplexity 10 to 17%. Is that a kernel problem?**
No. The kernels match "dequantize, then matmul" to fp16 rounding, and the perplexity change is the quantization itself: round-to-nearest, symmetric, group 128. That's the expected cost on 1 to 2B models, and the standard fixes, GPTQ and AWQ, change how the INT4 values and scales are chosen, not the kernel. The packed format works with either.

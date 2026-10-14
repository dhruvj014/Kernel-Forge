# Phase 1 notes: hardware ceilings, harness, naive baseline

Every number below is copied from a file in `results/`, named next to it. `BENCHMARKS.md` has the full tables.

## What was built

- **Measured hardware ceilings** (`bench/hw_ceilings.py`, `csrc/micro/bandwidth.cu`): DRAM bandwidth from our own float4 read, copy and write kernels, plus cuBLAS GEMM throughput for FP16 and BF16 Tensor Cores (FP32 accumulate) and FP32 CUDA cores (TF32 off).
- **Benchmark harness** (`kernelforge/bench_utils.py`): CUDA-event timing with median and p10/p90, a launch-overhead guard, an allocator cap that turns VRAM overflow into a clean OOM, clock, power and temperature logging, and the brief's FLOP and byte formulas.
- **Naive attention** (`csrc/attention/attn_naive.cu`): three kernels (QK^T, row softmax, PV) with FP32 math and fp16, bf16 or fp32 I/O. This is the "before" picture.
- **Correctness suite** (`tests/test_attention.py`): 159 numeric cases plus 8 input-validation tests against an FP32 PyTorch reference.
- **Naive vs SDPA benchmark** (`bench/bench_attention.py`): B=2, H=16, N from 512 to 8192, D in {64, 128}, causal and full, fp16 and bf16.
- **Nsight Compute profile** of the naive kernels at the representative shape (`scripts/run_ncu.ps1`, `scripts/ncu_summary.py`).

## Measured ceilings (`results/hw_ceilings.json`)

| Ceiling | Value | Note |
|---|---|---|
| DRAM copy (roofline memory peak) | 234.5 GB/s | 92% of the 256 GB/s implied by a 128-bit bus at 16 Gbps |
| FP16 Tensor Core, FP32 accumulate (roofline compute peak) | 31.0 TFLOPS | cuBLAS HGEMM, n=4096 |
| FP16 Tensor Core, FP16 accumulate (context only) | 57.1 TFLOPS | 1.84x the FP32-accumulate rate |
| FP32 CUDA core | 8.5 TFLOPS | cuBLAS SGEMM, TF32 off |
| Ridge point, Tensor Core | 132 FLOP/byte | below this intensity a kernel is memory bound |

**Why the compute peak is the FP32-accumulate HGEMM.** Our attention kernels accumulate in FP32. GeForce Ada runs FP16 MMA with FP32 accumulation at half the rate of FP16 accumulation, as the measured 1.84x gap shows. Measuring our kernels against the 57 TFLOPS number would make them look half as efficient as they are.

**Why the largest GEMM is not the best.** n=4096 beats 6144 and 8192 by about 8%. The longer runs hold the GPU at its 135 W cap long enough for clocks to drop. The best sustained result is the honest ceiling for kernels of similar duration.

## Four measurement problems found and fixed

These shaped the harness that every later phase uses.

1. **30 W power cap.** The laptop's vendor app (Alienware Command Center) was in a mode that capped the GPU at 30 W. The giveaway was FP16-accumulate GEMM running only 1.14x faster than FP32-accumulate instead of about 2x. `nvidia-smi -q -d POWER` confirmed "Current Power Limit: 30.00 W". At 30 W vs 135 W: Tensor Core 16.5 vs 31.0 TFLOPS (1.88x), SGEMM 3.3 vs 8.5 TFLOPS (2.6x), DRAM copy 206.6 vs 234.5 GB/s (1.14x). Bandwidth suffers least because moving bytes costs far less power than Tensor Core math. Both runs are kept (`hw_ceilings_30w.json`), and every benchmark JSON now records AC power, power plan, clocks and throttle reasons.
2. **Launch overhead inside the timed interval.** Enqueueing one SDPA call from Python costs 0.1 to 0.9 ms (cuDNN looks up an execution plan on every call), more than the 0.08 to 0.1 ms kernels at N=512. The GPU reached the start event, sat idle until the launch arrived, and the idle time was counted. Medians came out up to 5x too high, and fp16 vs bf16 differed by 3x on identical shapes. Fix: `time_cuda` measures host enqueue cost during warmup, then enqueues a GPU spin (`torch.cuda._sleep`) of 5x that cost before each timed run. While the GPU spins, the CPU queues start event, kernel and end event, so the interval contains only GPU work. A first version with a 3x margin still left one bimodal record, so the margin was raised and the N=512 and 1024 rows re-run. In the final data no record's median sits more than 10% above its p10.
3. **Sysmem fallback instead of OOM.** Naive attention at N=8192 needs 8.6 GB for scores alone. Instead of failing, the Windows driver spilled to system RAM over PCIe and the kernel ran at a crawl. `cap_to_free_vram()` limits the torch allocator to the VRAM free at startup (6.7 of 8 GiB under WDDM), so the run records `oom` in seconds.
4. **No Flash SDPA backend on Windows.** The Windows torch wheel is built without it. The baselines are SDPA's memory-efficient backend and cuDNN 9.24 fused attention, a FlashAttention-2-class kernel. Availability is probed and saved in `results/attention_naive.json`.

## Naive attention design

`S = scale * Q K^T` (one thread per score, FP32), then `P = softmax(S)` (one block per row), then `O = P V` (one thread per output element). The N x N FP32 score matrix lives in global memory between kernels.

- **Safe softmax.** Subtract the row max before `exp`. Without it, scores of about 512 (the `large_constant_offset` test) overflow FP32 `exp`.
- **Causal.** Masked scores are written as `-inf`, which become exact zeros after `exp`, and PV stops summing at the diagonal. Every causal row keeps its diagonal, so the row max is always finite.
- **Block reduction.** Warp shuffles (`__shfl_xor_sync`), one partial per warp through shared memory, then a final shuffle in warp 0. The leading `__syncthreads` protects the scratch buffer when the reduction runs twice back to back (max, then sum).
- **FP32 math for every I/O dtype.** Accuracy is limited only by the final rounding to the output dtype (see below).

## Correctness (`results/correctness_attn_naive.json`)

All 159 numeric cases pass: N from 1 to 1000, including ragged lengths around 16, 32, 64 and 128, D in {64, 128}, causal and full, plus all-equal scores, scores of magnitude ~1000, and a large constant offset.

| dtype | worst max abs error | interpretation |
|---|---|---|
| fp16 | 9.77e-4 = 2^-10 | half an fp16 ULP for outputs in [2, 4) |
| bf16 | 7.81e-3 = 2^-7 | half a bf16 ULP for outputs in [2, 4) |
| fp32 | 8.3e-7 | accumulation-order differences vs cuBLAS |

The fp16 and bf16 errors are exactly the rounding error of storing an FP32 result in the output dtype, so the kernel's math adds no measurable error. Tolerances (2e-3 fp16, 1.6e-2 bf16) are 2x these bounds. Phase 2 and 3 kernels have to meet the same bounds.

## Benchmark results (`results/attention_naive.json`)

Representative shape B=2, H=16, N=2048, D=64, fp16, non-causal:

| impl | median | TFLOPS | % of 31.0 TC peak | extra memory |
|---|---|---|---|---|
| naive | 108.4 ms | 0.32 | 1% | 520 MiB |
| SDPA efficient | 1.46 ms | 23.5 | 76% | 8 MiB |
| SDPA cuDNN | 1.28 ms | 27.0 | 87% | 8 MiB |

- Across the grid, cuDNN is **40x to 85x** faster than naive and peaks at **28.1 TFLOPS (90% of the measured Tensor Core ceiling)** at N=4096, D=128. That's the target for Phases 2 and 3.
- Naive is flat at 0.30 to 0.34 TFLOPS for every shape. Even judged against the FP32 CUDA-core ceiling of 8.5 TFLOPS, that's under 4%.
- **Memory:** naive's extra memory is the score matrix. 512 MiB at N=2048, 2 GiB at N=4096, OOM at N=8192 (8 GiB needed). SDPA's extra memory is just the output, 8 to 64 MiB. That is O(N^2) vs O(N).
- Causal naive takes half the time of full: QK^T skips masked dot products and PV stops at the diagonal, and the TFLOPS formula halves too, so TFLOPS stays flat.

## What the profiler shows (`results/ncu_summary.json`)

`ncu --set full` at the representative shape. `ncu` locks clocks to base, so these durations run about 15% longer than the benchmark median.

| kernel | share of time | DRAM % | L1/TEX % | top stall |
|---|---|---|---|---|
| QK^T | 78% | 3.5 | 98.6 | `lg_throttle` (88% of stall samples) |
| softmax | 4% | 90.3 | 29.2 | `long_scoreboard` |
| PV | 18% | 12.0 | 97.4 | `long_scoreboard`, `lg_throttle` |

**The surprise: naive attention is not DRAM bound.** Its whole-pipeline arithmetic intensity is 14.7 FLOP per DRAM byte (36.4 GFLOP over 2.48 GB), below the FP32 ridge point of 36. A DRAM-only roofline would call it memory bound. But DRAM is nearly idle in the two kernels that take 96% of the time. The bottleneck is the L1/load-store path:

- **QK^T.** `threadIdx.x` walks the key index j, so for a fixed `d` the 32 threads of a warp read `K[j*D + d]`: 32 addresses 128 bytes apart (D=64, fp16). Every 2-byte load instruction touches 32 different cache lines. The lines then stay in L1 (99% hit rate), so DRAM sees almost nothing, but L1 has to serve 32 tag lookups per instruction for 64 bytes of useful data. The load/store queue fills, and warps stall on `lg_throttle` (waiting to issue more loads). The FFMA count matches the formula exactly (8.59e9 = B*H*N^2*D), so no work is wasted. Each FMA just costs about one uncoalesced load.
- **PV.** Coalesced over d, but each FMA needs two loads (a broadcast `P[i][j]` and `V[j][d]`). Again L1 bound, not DRAM bound.
- **softmax.** Pure streaming. It reads and writes the 512 MiB score matrix at 90% of DRAM peak, so it's already near its roof. Its cost exists only because the scores were materialized. The 22k shared-memory bank conflicts come from the block reduction and are negligible.

**What this means for Phase 2.** Tiling Q, K and V through shared memory fixes the QK^T and PV access patterns: each element is loaded from global memory once per tile and then reused from shared memory by many threads, with conflict-free layouts. Fusing the three kernels with online softmax deletes the score matrix, the softmax kernel, and the O(N^2) memory. Phase 3 then moves the two matmuls onto Tensor Cores, which is the only way past the 8.5 TFLOPS CUDA-core ceiling toward cuDNN's 27 TFLOPS.

## Resource usage (`results/ptxas.json`)

QK^T 37 registers, PV 40, softmax 18 with 128 B of shared memory. No spills. With 256 threads per block, registers do not limit occupancy (theoretical 100%).

## Interview questions

**1. Why measure the hardware ceilings instead of using the spec sheet?**
Spec numbers assume ideal clocks and power. On this laptop the vendor power mode alone changed the measured Tensor Core ceiling by 1.88x (30 W vs 135 W). "% of peak" only means something if the peak was measured on the same machine, in the same power state, with the same accumulate precision the kernels use.

**2. Why is the Tensor Core ceiling the FP32-accumulate HGEMM?**
The kernels accumulate in FP32 for accuracy, and GeForce Ada runs FP32-accumulate MMA at half the FP16-accumulate rate (measured 1.84x apart). Using the 57 TFLOPS FP16-accumulate number would understate our efficiency by about 2x.

**3. Naive attention has an arithmetic intensity below the ridge point. Is it memory bound?**
Not in the DRAM sense. Nsight shows DRAM at 3.5% in QK^T, which takes 78% of the time, while L1/TEX is at 98.6% and 88% of stalls are `lg_throttle`. Warps in QK^T read K with a 128-byte stride, so every load instruction touches 32 cache lines. They hit in L1, but L1 throughput is the wall. A DRAM roofline points at the wrong fix. The real fix is coalesced loads and data reuse through shared-memory tiling.

**4. How do you know the kernel is correct, and why those tolerances?**
Every case is compared against an FP32 PyTorch reference with TF32 off. The worst errors are exactly half an output ULP (2^-10 for fp16, 2^-7 for bf16, outputs in [2, 4)), so all the error comes from the final rounding, which no kernel can avoid. Tolerances are 2x those bounds. The grid covers N=1, ragged lengths around tile sizes, causal masks, all-equal scores, and scores large enough to overflow `exp` without max subtraction.

**5. What went wrong in benchmarking and how did you catch it?**
A 30 W power cap, caught because FP16-accumulate GEMM was only 1.14x faster than FP32-accumulate instead of 2x. Python launch overhead counted as kernel time, caught because fp16 and bf16 differed by 3x on identical shapes, and fixed with a GPU spin before each timed run (the guard margin was tightened after one bimodal record showed up). And Windows sysmem fallback, caught because an allocation that should have failed ran at 45 W with full VRAM. Each fix lives in the shared harness, so later phases inherit it.

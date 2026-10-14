# Phase 3 notes: Tensor Cores, conflict-free layouts, cp.async, mma.sync

Every number below is copied from a file in `results/`, named next to it. `BENCHMARKS.md` has the full tables.

**Refreshed in Phase 5.** v4 gained grouped-query attention and KV-cache decode (query length != key length) in Phase 5, so `results/attention_final.json` was re-measured with the committed code and the numbers below are from that run. Changes vs the first Phase 3 run were within run-to-run variation at the representative shape (26.59 then 26.44 TFLOPS).

## Result in one paragraph

The final kernel (v4, `csrc/attention/attn_v4_kernel.cuh`) runs at **26.4 TFLOPS** at the representative shape (B=2, H=16, N=2048, D=64, fp16): 85% of the measured FP16 Tensor Core ceiling (31.0 TFLOPS), 83x faster than the naive baseline, and 98% of PyTorch's cuDNN fused attention (27.0 TFLOPS) measured in the same run. Across all 40 benchmark configs (N from 512 to 8192, D in {64, 128}, causal and full, fp16 and bf16), v4's throughput is a **geometric mean of 100% of cuDNN** (min 82%, max 119%). It is faster than cuDNN on 15 of the 40 and peaks at **27.3 TFLOPS** (N=4096, D=64, fp16). Sources: `results/attention_final.json`, `results/hw_ceilings.json`.

## What was built

| kernel | file | idea |
|---|---|---|
| v2 base | `attn_v2_kernel.cuh` (variant 0) | QK^T and PV on Tensor Cores via WMMA, fp16/bf16 tiles in padded shared memory |
| v2 | same (variant 1) | conflict-free shared-memory access in softmax, P write and O rescale |
| v3 | same (variant 2) | + cp.async double-buffered K/V tiles |
| v4 | `attn_v4_kernel.cuh`, `attn_v4_tile.cuh`, `common/mma_utils.cuh` | inline-PTX `mma.sync` + `ldmatrix`, softmax and O in registers (FlashAttention-2 style warp partitioning), XOR-swizzled unpadded K/V tiles, cp.async, tuned tiles |

All four share one code path per family, so every ablation step is measured on the same code with one feature switched: `ops.flash_attn_v2(conflict_free=False)`, `ops.flash_attn_v2()`, `ops.flash_attn_v3()`, `ops.flash_attn_v4(config=...)`.

## Ablation (`results/attention_final.json`, representative shape, all measured in the same run)

| step | ms | TFLOPS | % of TC peak | speedup vs previous (rep. shape) | geomean speedup vs previous (40 configs) |
|---|---|---|---|---|---|
| naive | 108.44 | 0.32 | 1% | | |
| v1 (Phase 2, FP32 CUDA cores) | 19.51 | 1.76 | 6% | 5.56x | 5.08x (32 configs) |
| v2 base: WMMA | 3.440 | 9.99 | 32% | 5.67x | 4.83x |
| + conflict-free layout | 3.302 | 10.40 | 34% | 1.04x | 1.06x |
| + cp.async double buffering (v3) | 3.015 | 11.40 | 37% | 1.10x | 1.12x |
| v4: mma.sync, registers, swizzle, tuned | 1.299 | 26.44 | 85% | 2.32x | 2.46x |

(naive and v1 come from `attention_naive.json` and `attention_v1.json`. The grid column for v1 covers the 32 configs where naive did not run out of memory.)

## Step 1: v2, Tensor Cores via WMMA

**What.** 4 warps per block, Br = 64, each warp owns 16 query rows. Per key tile each warp computes its 16 x Bc slice of S = Q K^T with 16x16x16 WMMA tiles (fp16/bf16 in, FP32 accumulate) and stores it to shared memory. The warp then runs the online softmax on its rows (2 lanes per row), writes P (fp16/bf16) over the same bytes, rescales its O rows (kept in shared memory as FP32), loads O back as the accumulator, adds P V, and stores again.

**Why the shared-memory round trip.** WMMA accumulator fragments have an unspecified element layout. A thread does not know which row its elements belong to, so per-row max, sum and rescale cannot be done in registers. Everything that is per row goes through shared memory. This is the main limitation v4 removes.

**Why K needs no transpose.** K is stored [key][d]. Read with a "column-major" B fragment, that memory is exactly K^T, so `load_matrix_sync(b_col, Ks, ld)` gives the B operand of Q K^T directly. V ([key][d]) is already the row-major B operand of P V.

**Shared memory and Bc.** D=64 uses Bc=64 (61 KB). At D=128, Bc=64 would need 103,424 bytes, over the 101,376-byte per-block limit, so D=128 uses Bc=32 (76 KB).

**Result.** 9.99 TFLOPS, 5.7x over v1. The Tensor Cores give 3.6x more peak FLOPs than the FP32 pipe, and the MMA reads each fragment once for a 16x16x16 product, which removes v1's one-shared-load-per-FMA bottleneck.

**What the profiler said next** (`ncu_summary.json`, profile `v2_base_fp16`): achieved occupancy 8.3% (one 4-warp block per SM, limited by 61 KB of shared memory), tensor pipe active 15%, 13.2M shared-load and 15.2M shared-store bank conflicts, top stalls `wait`, `short_scoreboard`, `long_scoreboard`. All latency stalls with only 4 warps to hide them.

## Step 2: conflict-free shared-memory layout

**What.** Each row of S/P and O is handled by a lane pair. In the first version lane `half` walked the contiguous half `[half*Bc/2, ...)`. The two halves are Bc/2 = 32 floats apart (a multiple of 32 banks), so the pair always hit the same bank. And with a row stride of 68 floats (4 banks mod 32), rows r and r+8 also collide. The fix interleaves columns (`c = 2*jj + half`) and starts rows 8 to 15 one column pair later. Then the 32 lanes cover banks {4r + 2j + h} for rows 0 to 7 and {4r + 2 + 2j + h} for rows 8 to 15, which are 32 distinct banks on every step. Iteration order is free (each lane visits all its columns anyway), so the rotation costs nothing.

**Why padding and not an XOR swizzle here.** WMMA `load_matrix_sync`/`store_matrix_sync` require a plain strided layout (pointer plus leading dimension), so a swizzled tile cannot be passed to them. Padding (8 halves / 4 floats per row, keeping 16-byte alignment and WMMA's `ld % 8` rule) is the only option for WMMA-touched buffers. v4 uses the swizzle because `ldmatrix` takes per-lane row addresses.

**Result** (`v2_base_fp16` vs `v2_layout_fp16`): shared-load conflicts 13.2M to 4.2M (-68%), store conflicts 15.2M to 8.4M (-45%), kernel time 4.04 to 3.79 ms under `ncu`, 1.04x at the representative shape and 1.07x over the grid. The conflicts left are exactly 2^22 and 2^23. They come from WMMA's own FP32 accumulator `store_matrix_sync`/`load_matrix_sync` on the S and O buffers, whose access pattern the kernel does not control.

## Step 3: v3, cp.async double buffering

**What.** Two K/V buffers. In iteration kt the kernel issues `cp.async` copies of tile kt+1 into the other buffer, commits them as a group, and then `cp.async.wait_group 1`, which waits for everything except the newest group, so tile kt has landed while tile kt+1 is still in flight. A `__syncthreads()` follows, because `wait_group` only covers the calling thread's own copies. Another barrier at the end of the iteration stops a fast warp from prefetching tile kt+2 into the buffer a slow warp is still reading. Rows past N use the zero-fill form (`src-size = 0`), so ragged tiles need no branch around the copy. `.cg` caches the copies in L2 only, since each K/V element is reused from shared memory, not L1.

**What latency it hides.** In v2 every tile started with a synchronous global load. Each thread issued its 16-byte loads, then the whole block waited about one DRAM round trip at a barrier with no math in flight. With cp.async the copy engine fills shared memory directly (no registers, no instruction waiting on the data) while the warps run the MMAs and softmax of the current tile.

**Result.** 1.10x at the representative shape, 1.13x over the grid, and 1.27x at D=128, where each tile is twice as large (quick check before the final run: 8.47 to 10.76 TFLOPS). Shared memory grows to 79 KB (D=64) and 93 KB (D=128), so occupancy stays at one 4-warp block per SM. Profile `v3_fp16`: tensor pipe 18%, occupancy still 8.3%. The remaining stalls are `wait` and `short_scoreboard`, the S/P/O shared-memory round trips that cp.async does not touch.

## Step 4: v4, mma.sync with FlashAttention-2 style warp partitioning

**What changes.** `mma.sync.m16n8k16` (inline PTX) has a documented fragment layout (`common/mma_utils.cuh`): thread `lane` holds rows `lane/4` and `lane/4 + 8`, columns `2*(lane%4)` and `+1` of each 16x8 accumulator tile. So each warp:

1. loads its 16 Q rows into A fragments once (`ldmatrix.x4`), registers for the whole kernel
2. per key tile computes S = Q K^T into registers (`ldmatrix.x4` on K gives B fragments for two 8-key tiles)
3. runs the online softmax on those registers. A row lives in the 4 lanes of a quad, so row max and row sum are two `__shfl_xor_sync` steps (offsets 1 and 2), with no shared memory and no barrier
4. rescales its O accumulators (registers) by `exp2(m_old - m_new)` per row
5. repacks P from the S accumulators into A fragments. The C layout of two adjacent 16x8 tiles is exactly the A layout of one 16x16 tile, so this is a float-to-fp16 pack in registers
6. O += P V with `ldmatrix.x4.trans` on V (stored [key][d], needed as "col" B fragments)

That is FlashAttention-2's partitioning: a warp owns query rows and never shares softmax state, and the only block-level synchronization is around the shared K/V tiles. Shared memory holds just Q (read once) and two K/V stages: 48 KB at the D=64 default instead of v3's 79 KB.

**XOR swizzle instead of padding.** `ldmatrix` reads 8 rows per phase, 16 bytes each. With unpadded 128-byte rows those 8 reads start on the same bank (8-way conflict). Chunk c of row r is stored at chunk position `c ^ (r % 8)` (`attn_v4_tile.cuh`), so the 8 reads land on 8 different 16-byte chunks covering all 32 banks. Unlike padding it costs no shared memory and keeps every chunk 16-byte aligned for `cp.async`. Measured shared-memory bank conflicts at the representative shape: 14k loads and 0 stores (v3: 4.4M and 9.5M).

**Tuning** (`results/attention_v4_sweep.json`, fp16, geomean over N in {1024, 2048, 4096} causal and full):

| D | warps x Bc | regs | smem | blocks/SM (limit) | TFLOPS |
|---|---|---|---|---|---|
| 64 | 4 x 32 | 96 | 24 KB | 4 (smem) | 24.48 |
| 64 | 4 x 64 | 126 | 40 KB | 2 (smem) | 23.36 |
| 64 | 8 x 32 | 100 | 32 KB | 2 (registers) | 22.27 |
| **64** | **8 x 64** | 126 | 48 KB | 2 (smem) | **24.85** |
| **128** | **4 x 32** | 167 | 48 KB | 2 (smem) | **23.79** |
| 128 | 4 x 64 | 166 | 80 KB | 1 (smem) | 20.30 |
| 128 | 8 x 32 | 174 | 64 KB | 1 (smem) | 21.65 |
| 128 | 8 x 64 | 184 | 96 KB | 1 (smem) | 22.47 |

Defaults: 8 warps x Bc=64 at D=64, 4 warps x Bc=32 at D=128. At D=128 the O accumulators alone take 64 registers per thread, so bigger tiles push a block to 166 to 184 registers and 64 to 96 KB of shared memory, one block (at most 8 warps) per SM, and the cost of exposed latency outweighs the better reuse. At D=64 the differences are within 12%, and the largest tile wins because each K/V element feeds more MMAs per barrier.

**Result.** 26.44 TFLOPS, 2.32x over v3 at the representative shape and 2.46x over the grid. Profile `v4_fp16` vs `sdpa_cudnn_fp16` (same shape, both at `ncu`'s locked base clock):

| metric | v4 | cuDNN SDPA |
|---|---|---|
| duration | 1.450 ms | 1.426 ms |
| tensor pipe active | 43.3% | 44.2% |
| top stall | `math_pipe_throttle` | `math_pipe_throttle` |
| registers per thread | 128 | 229 |
| achieved occupancy | 32.7% | 16.7% |
| shared memory per block | 48 KB | 72 KB |
| DRAM throughput | 9.2% | 9.4% |

Both kernels are now limited by the Tensor Core pipe itself (`math_pipe_throttle`: warps are ready but the pipe is busy). The tensor-pipe metric peaks at the FP16-accumulate rate, and GeForce Ada runs FP32-accumulate MMA at half that rate (`hw_ceilings.json`: 57.1 vs 31.0 TFLOPS), so 43.3% of that peak is 80% (using the measured 1.84x ratio) to 87% (using a nominal 2x) of what an FP32-accumulate kernel can reach, consistent with v4 running at 85% of the measured FP32-accumulate ceiling. cuDNN takes a different point in the design space (229 registers, half the occupancy, bigger tiles) and lands within 1.7% of v4 under `ncu`.

## Roofline (`results/roofline.json`, `results/plots/roofline.png`)

| kernel | FLOP per DRAM byte | TFLOPS | % of TC peak |
|---|---|---|---|
| naive | 13.9 | 0.32 | 1% |
| v1 | 599.5 | 1.76 | 6% |
| v2 base | 368.3 | 9.99 | 32% |
| v2 conflict-free | 605.0 | 10.40 | 34% |
| v3 | 601.7 | 11.40 | 37% |
| v4 | 1004.9 | 26.44 | 85% |
| SDPA cuDNN | 1003.6 | 27.04 | 87% |

Intensity uses DRAM bytes measured by `ncu`, so it reflects real traffic (K/V tiles re-read from L2 or DRAM by every query block), not the ideal `Q+K+V+O` minimum. Naive sits on the memory slope at 14 FLOP/byte, below the FP32 ridge (36). Every fused kernel is far right of the Tensor Core ridge (132 FLOP/byte), so from v1 on, the question is never DRAM bandwidth but how close the kernel gets to the compute roof. v1 is pinned under the FP32 roof (8.5 TFLOPS). v2 and v3 break through it but sit about 3x under the Tensor Core roof, latency bound at 8% occupancy. v4 reaches 85%. v4's intensity is higher than v3's because its default tile (Br = 128 rows at D=64) re-reads each K/V tile half as often.

## Where the remaining gap is, and why

v4's geomean over 40 configs is 100% of cuDNN, so there is no overall gap to report, but there is a pattern:

- **D=64: 97% to 119% of cuDNN (faster on 13 of 20).** v4 wins most clearly at small N and causal shapes (for example N=512 causal). A plausible reason, not separately measured: at small N there are few query blocks per head, and v4's 2 blocks per SM (cuDNN: one 8-warp block at 229 registers) keeps more of the 24 SMs busy.
- **D=128: 82% to 107% of cuDNN (faster on 2 of 20), weakest at N=8192.** With 4 warps x Bc=32 the kernel runs 2 blocks per SM, but each 16-row warp re-reads a K/V tile for only 32 keys between barriers. Larger tiles would reuse better, but the sweep shows they lose to register and shared-memory pressure (1 block per SM). Closing this needs techniques v4 does not have: splitting D across warps or keeping Q in shared memory to free 32 registers, a 3-stage cp.async pipeline, and software pipelining of the ldmatrix loads of the next K tile under the current MMAs (register double buffering).
- **No FlashAttention-2 comparison.** The Windows wheel does not exist and a source build is out of scope (`docs/env.md` item 12). cuDNN's fused kernel (`cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma_f16`) is the FlashAttention-2-class reference instead.
- **Forward only, fp16/bf16 only, D in {64, 128}.** No backward pass, no dropout, no variable-length batching, no paged KV cache.

## Correctness and sanitizers

- `tests/test_attention.py`: 1014 tests pass (159 skipped: fp32 cases for the fp16/bf16-only Tensor Core kernels). v2, v3 and v4 run the full edge-case grid (N=1, ragged lengths around 16/32/64/128, causal, all-equal scores, scores of magnitude ~1000), and every v4 config is tested separately (`results/correctness_attn_v4_configs.json`, 224 cases).
- Worst errors (fp16 / bf16): v2 1.02e-3 / 9.93e-3, all within the Phase 1 tolerances (2e-3 / 1.6e-2). They're slightly above v1's half-ULP bound because P is rounded to fp16/bf16 before the PV product, as in FlashAttention.
- `results/sanitizer.json`: memcheck, racecheck, synccheck and initcheck report 0 errors and 0 hazards on `kf_sanitize_attn.exe`, which runs every v1 config, all three v2/v3 variants and all four v4 configs on 1 to 3 key tiles (96 kernel runs, outputs checked against a double-precision reference). racecheck matters most for v3/v4, where a missing barrier around the double-buffered stages would be a write-after-read hazard.

## Resource usage (`results/ptxas.json`)

One spill, found by the clean rebuild that regenerated `ptxas.json`: the v2 conflict-free variant, fp16, D=64 (`attn_v2_kernel<__half, 64, 64, true, 1>`) runs at 64 registers with an 8-byte stack frame (8 bytes spilled and reloaded per thread). The bf16 build of the same variant and every other kernel have no spills. It's the same ptxas heuristic as v1's first build (no explicit minimum blocks per SM in `__launch_bounds__`). The fix is known (`__launch_bounds__(kThreads, 1)`), but the ablation rows were measured with this build, and v2 is not the shipped kernel, so it is recorded rather than re-measured. 8 bytes of local memory per thread normally stays in L1.

v4 defaults (fp16): 128 registers at D=64, 167 at D=128, no spills (126 at D=64 before the Phase 5 GQA and KV-cache change). Shared memory per block: 48 KB at both defaults.

## Interview questions

**1. Why did WMMA stall at 37% of peak, and what did mma.sync change?**
WMMA's accumulator layout is unspecified, so per-row softmax and output rescaling had to go through shared memory every tile: store S, softmax, write P, load P, load O, MMA, store O. That made the kernel latency bound, and its shared-memory footprint (79 KB) allowed one 4-warp block per SM (8% occupancy). mma.sync's layout is documented, so S, P and O stay in registers, row reductions are 2 quad shuffles, and shared memory holds only Q and K/V, 48 KB with 8-warp blocks and 2 blocks per SM. Tensor pipe utilization went from 18% to 43%, about the same as cuDNN (44%).

**2. How do you reuse the S accumulator as the A operand of P V without shared memory?**
In m16n8k16, a C fragment holds rows g and g+8, columns 2t and 2t+1 of a 16x8 tile. An A fragment holds rows g and g+8, columns 2t, 2t+1, 2t+8 and 2t+9 of a 16x16 tile. So the C fragments of two adjacent 8-key score tiles are exactly the A fragment for those 16 keys. After the softmax the kernel packs pairs of floats into fp16x2 registers in that order and feeds them to the next MMA.

**3. Padding vs XOR swizzle: when do you use each?**
WMMA load/store take a pointer and a leading dimension, so their buffers must be plainly strided, and padding (8 halves per row) is the only option. `ldmatrix` takes one row address per lane, so the address can be swizzled: chunk c of row r goes to chunk `c ^ (r % 8)`. The 8 rows read in one phase then hit 8 different 16-byte chunks, which is all 32 banks, with no wasted shared memory and every chunk still 16-byte aligned for cp.async. Measured: v3 (padded) had 4.4M load conflicts, v4 (swizzled) 14k.

**4. Explain the cp.async commit/wait semantics in your pipeline.**
Each iteration issues the copies for tile kt+1, commits them as a group, then waits with `wait_group 1`: at most one group (the one just issued) may still be in flight, so tile kt is complete while kt+1 keeps loading under the math. `wait_group` covers only the calling thread's copies, so a `__syncthreads()` follows before anyone reads the tile. A second barrier at the end of the iteration keeps a fast warp from prefetching kt+2 into the buffer a slow warp is still reading. The last iteration waits with `wait_group 0`.

**5. Why did the best D=128 config use fewer warps and smaller tiles than D=64?**
At D=128 each thread holds 64 FP32 output accumulators and 32 registers of Q fragments. Bigger tiles pushed the kernel to 166 to 184 registers and 64 to 96 KB of shared memory, one block per SM, and the exposed latency cost more than the reuse gained. 4 warps x Bc=32 keeps 2 blocks per SM at 167 registers. That's also where v4 trails cuDNN (82% to 107%), and the fixes would be freeing registers (Q in shared memory, splitting D across warps) and deeper pipelining.

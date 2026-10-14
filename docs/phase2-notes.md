# Phase 2 notes: fused attention v1 (tiling and online softmax on CUDA cores)

Every number below is copied from a file in `results/`, named next to it. `BENCHMARKS.md` has the full tables.

## What was built

- `csrc/attention/attn_v1_fp32.cu`: one fused kernel for attention forward. Tiled, online softmax, FP32 math, fp16/bf16/fp32 I/O, head dim 64 and 128 via templates, five (Br, Bc) tile configs, causal masking with block skipping, ragged N.
- `kernelforge.ops.flash_attn_v1(q, k, v, causal, scale, config)`.
- Tests: v1 runs the full Phase 1 edge-case grid, plus every tile config on lengths below, at and just past tile boundaries (`tests/test_attention.py`).
- `scripts/sanitize_target.py`: small-shape workload for compute-sanitizer memcheck, racecheck and synccheck.
- `bench/sweep_v1_tiles.py` (Br x Bc sweep), `bench/memory_scaling.py` (memory vs N plot), v1 added to `bench/bench_attention.py`.

## The algorithm in my own words

Attention for one query row i is `o_i = sum_j softmax(s_i)_j * v_j` with `s_ij = scale * q_i . k_j`. The softmax needs the row max (for numerical safety) and the row sum (to normalize) over all N keys before any probability is final. The naive kernel gets them by writing all N scores to memory first.

Online softmax removes that dependency. Process the keys in tiles of Bc. After seeing tiles 1..t, keep three things per row:

- `m_t`: the max score seen so far
- `l_t = sum over seen j of exp(s_ij - m_t)`: the normalizer, relative to the current max
- `O_t = sum over seen j of exp(s_ij - m_t) * v_j`: the unnormalized output, relative to the current max

When tile t+1 arrives with scores `s_ij`:

1. `m_new = max(m_t, max_j s_ij)`
2. Every term already in `l_t` and `O_t` was computed as `exp(s - m_t)`. To re-express it relative to `m_new`, multiply by `exp(m_t - m_new)`, because `exp(s - m_t) * exp(m_t - m_new) = exp(s - m_new)`. So `alpha = exp(m_t - m_new)`, `l = alpha * l_t`, `O = alpha * O_t`.
3. Add the new tile: `p_ij = exp(s_ij - m_new)`, `l += sum_j p_ij`, `O += sum_j p_ij * v_j`.

After the last tile, `o_i = O / l`. This is exact, not an approximation: each term ends up multiplied by exactly `exp(s_ij - m_final)` and divided by exactly `sum_j exp(s_ij - m_final)`, the same as the three-pass softmax. Only rounding differs. Since `m_new >= m_t`, `alpha <= 1` and every exponent is `<= 0`, so nothing overflows however large the scores are. On the first tile `m_t = -inf`, so `alpha = exp(-inf) = 0` and the (zero) history drops out.

The N x N score matrix never exists. Each block holds one Br x Bc tile of it in shared memory at a time, so extra memory per call is just the output.

**exp2 instead of exp.** The kernel multiplies scores by `scale * log2(e)` once and uses `exp2f`, since `exp(x) = 2^(x * log2 e)`. `exp2f` is a single MUFU.EX2 instruction. `expf` adds the multiply inside every call. The max, rescale and normalization are unchanged because they're done in the same (log2) units throughout.

## Kernel structure

Grid: `(ceil(N / Br), B*H)`, 256 threads per block. Shared memory, all FP32 and padded:

| buffer | shape | purpose |
|---|---|---|
| `Qs` | Br x (D+1) | the block's query tile, loaded once |
| `KVs` | Bc x (D+1) | the current K tile, then the V tile of the same key range |
| `Ps` | Br x (Bc+1) | `exp2(s - m)` for the PV product |

**Thread mapping.** `TPR = 256 / Br` threads share a query row. Thread `g` of a row computes scores for columns `g, g+TPR, ...` and owns output columns `g, g+TPR, ...`. Row max and row sum are shuffle reductions across those TPR adjacent lanes (`__shfl_xor_sync` with offsets below TPR), so softmax needs no shared memory and no barrier.

**Per key tile:**

1. barrier, load K tile, barrier
2. scores: dot products from shared memory, scaled, masked only if the tile crosses the diagonal or N
3. shuffle max, rescale `l` and the accumulator by `alpha`, write `p` to `Ps`, shuffle sum
4. barrier, load V tile into the same buffer, barrier
5. `acc[j] += sum_c Ps[r][c] * V[c][g + j*TPR]`

**Why four barriers.** Barrier 1 stops a fast warp from overwriting `KVs` with the next K tile while a slow warp is still reading V from the previous one (and on the first tile it publishes `Qs`). Barrier 2 makes the K tile visible before anyone reads it. Barrier 3 ensures all K reads are done before V overwrites the buffer, and that `Ps` is visible. Barrier 4 makes V visible. Sharing one buffer for K and V saves Bc x (D+1) x 4 bytes (16.6 KB at Bc=64, D=64), which is what lets two blocks fit per SM at D=64. The cost is two extra barriers per tile.

**Coalesced loads.** Each thread loads 16 bytes (8 fp16 or 4 fp32 values) with one `uint4` load, and consecutive threads load consecutive 16-byte chunks of the contiguous Br x D tile. A warp reads 512 contiguous bytes per instruction. Values are converted to FP32 on the way into shared memory, so the inner loops never convert.

**Bank conflicts (padding).** Shared memory has 32 banks of 4 bytes. Without padding, rows of 64 or 128 floats start on the same bank, so lanes reading the same column `d` of different rows (Q in the dot product, P in PV) would all hit one bank and serialize. With a row stride of D+1, row r starts at bank `r mod 32`, so the 32/TPR different rows a warp touches are in different banks. K reads are broadcasts or adjacent columns. Phase 3 replaces padding with an XOR swizzle, which matters once the loads become 16-byte vector or `ldmatrix` loads.

**Causal block skipping.** A query tile `[q0, q0+Br)` never needs keys after its last row, so the key loop stops at `ceil(min(q0+Br, N) / Bc)` tiles. Fully masked tiles are never loaded. Only tiles that cross the diagonal (`k0 + Bc - 1 > q0`) or the end of the sequence pay for per-score mask comparisons. Interior tiles skip them.

**Ragged N.** Rows past N are zero-filled on load (predicated), their scores are masked to `-inf`, and output rows past N are not stored. The one dangerous case is a row whose scores are all `-inf` in the first tile: `m_new = -inf` and `exp2(-inf - -inf)` would be NaN. The kernel uses 0 as the exponent base when `m_new` is `-inf`, so those terms are exactly 0. Real rows always see key 0 in the first tile (it's never causally masked), so this only affects padding rows, which are never stored.

**`__launch_bounds__(256, 1)`.** The first build spilled up to 76 bytes per thread in the fp16 and bf16 versions: ptxas capped them at 64 registers while the fp32 versions used 80 or more. The explicit minimum of one block per SM removed every spill. Shared memory already limits these configs to one or two blocks per SM, so a register cap gained nothing.

## Correctness (`results/correctness_attn_v1.json`, `results/correctness_attn_v1_configs.json`)

- The full Phase 1 grid (159 cases) passes with the same worst errors as naive in fp16 (9.77e-4) and bf16 (7.81e-3), exactly half an output ULP. The fused kernel adds no measurable error.
- Every one of the 5 tile configs passes on N in {1, 31, 33, 64, 127, 129, 300}, D in {64, 128}, causal and full (140 cases).
- **One tolerance change, with a reason.** fp32 at scores of magnitude ~1000 differed from the reference by 5.4e-5 (old tolerance 1e-5). At |s| = 1000 an FP32 score already carries ~6e-5 of rounding error, and softmax turns score error into relative error in P. v1 computes `exp2(s * log2 e)` while the reference computes `exp(s)`, so they round differently. The fp32 tolerance now adds `8 * 2^-24 * max|s| * max|v|`, which only matters for large-magnitude scores. Naive passed before only because it uses the same formula as the reference.
- Total suite: 469 tests pass.

## compute-sanitizer (`results/sanitizer.json`)

memcheck, racecheck, synccheck and initcheck all report **0 errors and 0 hazards** on `build/cmake/kf_sanitize_attn.exe`: every config, both head dims, causal and full, N in {37, 70} (1 to 3 key tiles, which exercises the shared K/V buffer reuse that barriers 1 and 3 protect).

**Why a standalone binary.** Under racecheck, a Python process running torch spent over 20 minutes of CPU time instrumenting torch's own CUDA modules (GPU at 11% utilization) and never reached our kernels. Splitting the kernels into `attn_v1_kernel.cuh` (no ATen) let a 150-line C++ program run them alone, with a double-precision CPU reference. racecheck now takes 77 s. memcheck through the torch extension (`scripts/sanitize_target.py`, also covers naive) was clean too.

## Tile sweep (`results/attention_v1_sweep.json`)

| config (Br x Bc) | D=64 TFLOPS | D=128 TFLOPS | blocks/SM (D=64, D=128) |
|---|---|---|---|
| 32 x 32 | 1.43 | 1.53 | 3, 2 |
| 32 x 64 | 1.49 | 1.53 | 2, 1 |
| 64 x 32 | 1.57 | 1.59 | 2, 1 |
| **64 x 64** | **1.69** | **1.66** | 2, 1 |
| 128 x 32 | 1.62 | 1.64 | 1, 1 |

Geometric mean over N in {1024, 2048, 4096}, causal and full, fp16. **64 x 64 is the default for both head dims.** The spread is only 18%, and the config with the highest occupancy (32 x 32 at D=64, 50%) is the slowest. Occupancy is not what limits this kernel. Larger tiles win because each loaded K/V element feeds more score columns (fewer loads per FMA) and there are fewer barriers per FLOP.

## Benchmark results (`results/attention_v1.json`)

v1 and the SDPA baselines were measured in the same run. Naive times come from `results/attention_naive.json`.

| B=2, H=16, N=2048, D=64, fp16 | median | TFLOPS | extra memory |
|---|---|---|---|
| naive | 108.4 ms | 0.32 | 520 MiB |
| **v1** | **19.5 ms** | **1.76** | **8 MiB** |
| SDPA cuDNN | 1.27 ms | 26.95 | 8 MiB |

- **v1 is 4.8x to 5.6x faster than naive for every N >= 1024** (4.2x at N=512), and runs N=8192 and 16384, where naive is out of memory.
- v1 runs at 1.39 to 1.76 TFLOPS, 16 to 21% of the measured FP32 CUDA-core peak (8.5 TFLOPS), and 6 to 11% of cuDNN's Tensor Core throughput.
- Causal is about 2x faster than full at the same N, so TFLOPS stays flat. Block skipping really removes half the work.

## Memory scaling (`results/memory_scaling.json`, `results/plots/memory_scaling.png`)

| N | naive | v1 | SDPA cuDNN |
|---|---|---|---|
| 1024 | 132 MiB | 4 MiB | 4 MiB |
| 4096 | 2064 MiB | 16 MiB | 16 MiB |
| 8192 | OOM | 32 MiB | 32 MiB |
| 16384 | OOM | 64 MiB | 64 MiB |

v1's extra memory is exactly the output tensor: it doubles when N doubles (O(N)), while naive's quadruples (O(N^2)). At N=4096 v1 needs 129x less memory than naive.

## What the profiler shows (`results/ncu_summary.json`, profile `v1_fp16`)

Representative shape, `ncu --set full`:

| metric | v1 | what it means |
|---|---|---|
| L1/TEX (shared memory pipe) throughput | 98.1% | saturated |
| shared-load instructions per FFMA | 1.06 | 4x too many (see below) |
| top stall | `mio_throttle` | warps wait for the shared-memory instruction queue |
| FMA pipe active | 12.9% | the math units are mostly idle |
| DRAM throughput | 1.0% | tiling solved the global-memory problem completely |
| achieved occupancy | 32.9% (theoretical 33.3%) | 2 blocks/SM, limited by 49.9 KB of shared memory |
| FFMA count | 17.19e9 | exactly B*H*N^2*D for QK^T plus the same for PV: no wasted work |
| shared-store bank conflicts | 22.2M | from the tile loader (see below) |
| shared-load bank conflicts | 42k | negligible: padding works for the read patterns |

**v1 is bound by shared-memory instruction issue, and that bound is structural.** One warp-wide FFMA covers 32 lanes, and an Ada SM issues 128 FFMA lanes per clock (4 schedulers x 32). A warp-wide shared load is one 32-lane wavefront, and the SM's shared-memory pipe serves one wavefront per clock. So a kernel needs at most 0.25 shared loads per FFMA to keep the FMA units busy. v1 issues 1.06, so it can't exceed about 25% of the FP32 peak, roughly 2.1 TFLOPS. It reaches 1.76. Register blocking in the score loop (load `q[d]` once, reuse it across all of a thread's score columns) helped. In a first sweep that was stopped before saving, the 32 x 32 and 32 x 64 configs at D=64 both measured about 1.16 TFLOPS without it, against 1.43 and 1.49 with it in the saved sweep. That "before" number is not in `results/`, so treat it as indicative. The PV loop and the remaining K loads still cost about one LDS per FMA.

Two ways out, both deliberately left for Phase 3:

1. **2D register tiles** (each thread computes a small rows x columns block, as in SGEMM) would cut shared loads per FMA toward 0.25 on CUDA cores. Even perfect, that still tops out at the 8.5 TFLOPS FP32 ceiling.
2. **Tensor Cores** do a 16x16x16 matrix multiply per warp instruction from register fragments, with 3.6x the FLOP ceiling (31.0 vs 8.5 TFLOPS). That's v2.

**The store-side bank conflicts** come from `load_tile`. Each thread converts 8 fp16 values and stores 8 consecutive floats into a row with stride D+1. Consecutive threads start 8 floats apart, so lanes 0, 4, 8, ... land in the same bank. Padding fixes column reads but not this store pattern. v2 keeps tiles in fp16 and writes 16-byte vectors with an XOR swizzle, the Phase 3 "conflict-free layout" step.

## Harness fixes this phase

- **Warmup counted wall time while launches were asynchronous**, so for slow kernels the loop queued about 1000 launches (until the queue filled) before 100 ms of wall time passed. v1 at N=8192 ran 1022 warmup calls per config. Timed samples were never affected (they're bracketed by events after a sync), but runs took minutes per config and kept the GPU hotter for longer than intended. `time_cuda` now synchronizes each warmup call. The N=8192 D=128 rows of `attention_v1.json` and later results use the fixed harness. Earlier rows and `attention_naive.json` were measured with the longer warmup.
- `attn_v1_kernel.cuh` split out so the kernels can be reused without ATen (sanitizer binary now, Phase 3 microbenchmarks later).

## Resource usage (`results/ptxas.json`)

Default config (64 x 64), fp16: 96 registers at D=64, 123 at D=128, no spills. Shared memory per block is 48.8 KB at D=64 and 80.8 KB at D=128, so the shared-memory opt-in above 48 KB is required (`cudaFuncSetAttribute`).

## Interview questions

**1. Why is online softmax exact and not an approximation?**
Every term added to the running sum and output is `exp(s - m_old)`. When the max increases, multiplying the history by `exp(m_old - m_new)` turns each of those terms into exactly `exp(s - m_new)`. After the last tile every term is `exp(s - m_final)` and the normalizer is the full sum of those, the same quantities the three-pass softmax computes. Only floating-point rounding differs, and the tests show the error is still exactly half an output ULP.

**2. Why does fused attention use O(N) memory, and what limits N now?**
Each block only ever holds a Br x Bc tile of scores in shared memory, so the N x N matrix is never stored. Extra memory is the output tensor alone: 32 MiB at N=8192 where naive needs 8 GiB. N is now limited by the Q, K, V and O tensors themselves, not by attention.

**3. v1 runs at 21% of FP32 peak. What limits it, and how did you prove it?**
Shared-memory instruction issue. Nsight shows the shared-memory pipe at 98%, top stall `mio_throttle`, and FMA pipe at 13%. The kernel issues 1.06 shared loads per FFMA, but an Ada SM can only issue one 32-lane shared wavefront per clock against 128 FFMA lanes, so above 0.25 loads per FFMA the math units starve. That puts a roughly 25% ceiling on v1, and it's at 21%. DRAM is at 1%, so global memory isn't the problem.

**4. Why did the configuration with the highest occupancy lose the tile sweep?**
Occupancy hides latency, but v1 isn't latency bound, it's throughput bound on the shared-memory pipe. More resident warps just queue more shared loads behind the same pipe. Bigger tiles (64 x 64) reuse each loaded value across more FMAs and run fewer barriers per FLOP, which directly attacks the real bottleneck, even at one third occupancy.

**5. How do you know there are no races around the shared K/V buffer?**
The kernel uses one buffer for K then V, protected by four barriers per tile, and I can explain each one (see "Why four barriers"). compute-sanitizer racecheck, synccheck, memcheck and initcheck all report zero hazards on a workload where every config processes 2 to 3 key tiles, which is exactly where a missing barrier 1 or 3 would show up as a write-after-read hazard. Outputs in that run are also checked against a double-precision reference.

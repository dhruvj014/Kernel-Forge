# Phase 4 notes: fused INT4 dequantize-GEMM

Every number below is copied from a file in `results/`, named next to it. `BENCHMARKS.md` has the full tables.

## Result in one paragraph

At decode (M = 1), the fused INT4 kernels run all 7 linear layers of one decoder layer **3.17x faster than cuBLAS FP16 on TinyLlama 1.1B and 3.15x faster on Qwen2.5 1.5B** (`results/gemm_final.json`). On the large MLP layers they stream weights at **89 to 90% of the measured DRAM copy peak** (84 to 85% of the read-only peak), **3.39x to 3.45x faster than cuBLAS**. INT4 stays faster than cuBLAS FP16 at every batch size tested up to M = 128 (1.03x to 1.09x there), because a Tensor Core GEMM path takes over from M = 8. Weights take 4.125 bits each with group size 128, 3.9x less memory than fp16.

## Why INT4 weights speed up decode (and not prefill)

A linear layer computes `y = x W^T` with W of size N x K. At decode M = 1, so the layer does 2NK FLOPs while reading 2NK bytes of fp16 weights: an arithmetic intensity of 1 FLOP/byte, two orders of magnitude below this GPU's Tensor Core ridge (132 FLOP/byte, `hw_ceilings.json`). The time is set entirely by how fast the weights stream from DRAM. Storing them in 4 bits plus one fp16 scale per 128 weights cuts the bytes to 0.52 per weight, about 3.9x fewer, so a kernel that streams at the same bandwidth is about 3.9x faster. The measured 3.4x on large layers is that bound minus kernel efficiency (89% vs cuBLAS's 100% of the copy peak).

The win only exists if dequantization is fused. The PyTorch path (unpack to fp16, then `F.linear`) writes and re-reads a full fp16 copy of the weights and runs several elementwise kernels. It takes 1.0 to 2.2 ms per layer, 19x to 138x slower than cuBLAS on the original weights.

At prefill (M in the hundreds or thousands) each weight is reused by M rows, so the GEMM is compute bound, and 4-bit storage saves bandwidth that is not the bottleneck. The data shows the advantage shrinking as M grows: 3.17x at M = 1, 1.80x at M = 8, 1.23x at M = 64, 1.03x at M = 128 (TinyLlama, all 7 layers).

## What was built

| piece | file |
|---|---|
| Group-wise symmetric INT4 quantizer, packing, PyTorch dequant reference | `kernelforge/quantize.py` |
| Shared unpack helper (nibbles to half2 without int-to-float instructions) | `csrc/gemm/int4.cuh` |
| Kernel A: GEMV, M = 1..8 | `csrc/gemm/dequant_gemv_kernel.cuh`, `dequant_gemv.cu` |
| Kernel B: Tensor Core GEMM, any M, with split-K | `csrc/gemm/dequant_gemm_tc_kernel.cuh`, `dequant_gemm_tc.cu` |
| M-based dispatcher | `kernelforge.ops.int4_matmul` |
| Layer shapes read from each model's `config.json` | `bench/bench_gemm.py` |

## Quantization format

- Symmetric, per output row and per group of G = 128 (or 64) consecutive k: `scale = absmax / 7` (fp16), `q = clamp(round(w / scale), -8, 7)`. No zero point, so dequantization is one multiply.
- The scale is rounded to fp16 before q is computed, so q is quantized against exactly the scale the kernel will use.
- Packed int32 [N, K/8]: each word holds 8 consecutive k as unsigned nibbles `u = q + 8`, in the order k0 k2 k4 k6 k1 k3 k5 k7 from bit 0.

**Why that nibble order.** `(word >> 4i) & 0x000F000F` leaves u(k 2i) in bits 0-3 and u(k 2i+1) in bits 16-19. OR-ing `0x64006400` turns each 16-bit half into the fp16 number 1024 + u (0x6400 is 1024.0, and in [1024, 2048) the fp16 mantissa step is exactly 1, so the low bits are u). One `__hsub2` of 1032 then gives the half2 (q[2i], q[2i+1]) exactly, already paired the way two consecutive fp16 activations sit in a 32-bit register. Unpacking 8 weights takes 4 shifts, 4 AND/OR pairs and 4 half2 subtractions, with no integer-to-float conversions. A warp's 16-byte loads of a weight row stay contiguous.

**Quantization error** (random Gaussian weights, `gemm_final.json`): relative Frobenius error 11.7% vs the fp16 weights at G = 128, the expected level for round-to-nearest INT4 on Gaussian data. Real-model quality (WikiText-2 perplexity) is Phase 5. Every element is within half a quantization step of the original (`tests/test_quantize.py`).

## Kernel A: GEMV (M = 1..8)

- Each warp owns 2 output rows and walks all of K. Lane l takes 32-weight chunks l, l+32, ... of each row: one 16-byte load per chunk, so each warp load instruction reads 512 contiguous bytes of a weight row.
- 32 divides G, so a chunk lies inside one group and needs one scale.
- The chunk's weights are unpacked once (kept as half2, exact since |q| <= 8) and reused for all M activation rows. Products accumulate in FP32, the group scale is applied once per chunk, and a 5-step `__shfl_xor_sync` reduction finishes each output.
- `__launch_bounds__(128, 3)`: ptxas otherwise capped M = 5 and 6 at 128 registers and spilled 8 to 16 bytes. The explicit minimum of 3 blocks per SM allows 170 registers, so no variant spills (60 to 164 registers).

**Profile** (`ncu_summary.json`, `gemv_m1_gate` vs `cublas_m1_gate`, TinyLlama gate_proj, M = 1): 26.9 us at 88.8% DRAM throughput vs cuBLAS's 97.2 us at 93.4%. Both kernels' top stall is `long_scoreboard`, waiting on global loads, which is what a bandwidth-bound kernel should show.

**Small layers.** k_proj/v_proj (N = 256) reach only 17 to 23% of peak (still 1.4x to 1.8x faster than cuBLAS). 256 rows are 32 blocks of 8 rows on 24 SMs, and the layer streams only about 200 to 270 KB, so the run is a few microseconds of ramp-up and tail. Splitting K across warps for small-N layers would help, and it's listed under future work.

## Kernel B: Tensor Core GEMM (M >= 8)

- Block tile BM x 64 columns, k-tiles of 64, 4 warps, `mma.sync.m16n8k16` (fp16 in, FP32 accumulate).
- Per k-tile: `cp.async` double-buffers the x tile and the packed weight tile (64 rows x 32 bytes), the block dequantizes the weight tile once into a swizzled fp16 shared tile (unpack + one half2 multiply by the row's scale, since a 64-wide k-tile always sits inside one group), then `ldmatrix` + `mma.sync` exactly as in attention v4. W stored [n][k] is already the "col" B operand. DRAM traffic is the packed 4-bit tile, and fp16 weights only exist in shared memory.

Two problems found and fixed with the profiler:

1. **Too few blocks (split-K).** The grid is N/64 x ceil(M/BM) blocks: 4 blocks for a 256-wide k/v projection, 32 for down_proj, on 24 SMs, each walking all of K alone. At M = 16, k/v took 30.7 us vs cuBLAS's 12.3 us. When the grid is small, blocks now take slices of K (aiming for 4 blocks per SM, each slice at least 4 k-tiles), write FP32 partials, and a reduction kernel sums them in fixed order (deterministic). k/v at M = 16 went to 10.2 us, down_proj from 104.4 to 88.1 us (measured during development, before the final run).
2. **A 64-row tile wasted Tensor Core work at small M.** At M = 16, `ncu` (`gemm_m16_gate_bm64`) showed `math_pipe_throttle` as the top stall with the tensor pipe 39% busy: 48 of the 64 rows were zero padding, and M = 16 took as long as M = 64. The warp layout is now templated (BM = 16: 1 x 4 warps of 16 x 16. BM = 32: 2 x 2 of 16 x 32. BM = 64: 2 x 2 of 32 x 32), and the host picks the smallest tile covering M. Same shape after the fix (`gemm_m16_gate`): 70.5 to 47.4 us, tensor pipe 39% to 15%, DRAM throughput 36% to 58%, top stall now `long_scoreboard`, back to waiting on memory as a weight-streaming kernel should.

**Where Kernel B stands.** At M = 16 and 32 the GEMM reaches 35 to 59% of the DRAM copy peak on the q/o, gate/up and down layers, beating cuBLAS by 1.33x to 2.29x instead of the 3.4x the GEMV gets at M = 1 (`gemm_final.json`). The long-K down_proj layers are the weakest (1.33x to 1.52x): N is only 2048 or 1536, so the grid relies on split-K, which adds FP32 partial writes and a reduction pass. Remaining overheads: a 2-stage pipeline with three block barriers per k-tile, and dequantizing through shared memory (a store and an `ldmatrix` reload per weight). The known fix is the Marlin approach: reorder the packed weights offline so each thread loads the exact nibbles of its own mma B fragments straight into registers, with no shared-memory dequant tile, plus a deeper `cp.async` pipeline. That's listed as future work rather than attempted here.

## The dispatcher (`results/gemm_dispatch.json`)

`ops.int4_matmul` uses the GEMV for M <= 7 and the Tensor Core GEMM from M = 8. The threshold was measured, not guessed: `bench/sweep_int4_dispatch.py` times both kernels at M = 1..8 on all 8 distinct layer shapes and picks the largest M at which the GEMV's summed time is lower (what one decode step pays). Summed over the shapes: at M = 7 the GEMV totals 289.9 us vs 318.5 us, at M = 8 it is 322.6 vs 302.1. The GEMV's FP32 math grows linearly with M, so it turns ALU bound well before M = 8, while the GEMM pads M up to its 16-row tile almost for free on Tensor Cores.

## Decode results (`results/gemm_final.json`, M = 1, cold L2)

| model | layers | N x K | INT4 us | cuBLAS FP16 us | speedup | INT4 % copy peak | INT4 % read peak |
|---|---|---|---|---|---|---|---|
| TinyLlama | q/o | 2048 x 2048 | 13.3 | 38.9 | 2.92x | 70% | 66% |
| TinyLlama | k/v | 256 x 2048 | 5.1 | 9.2 | 1.80x | 23% | 22% |
| TinyLlama | gate/up | 5632 x 2048 | 28.7 | 97.3 | 3.39x | 89% | 84% |
| TinyLlama | down | 2048 x 5632 | 28.7 | 98.3 | 3.43x | 89% | 84% |
| Qwen2.5 | q/o | 1536 x 1536 | 9.2 | 23.6 | 2.56x | 57% | 53% |
| Qwen2.5 | k/v | 256 x 1536 | 5.1 | 7.2 | 1.40x | 17% | 16% |
| Qwen2.5 | gate/up | 8960 x 1536 | 33.8 | 115.7 | 3.42x | 90% | 85% |
| Qwen2.5 | down | 1536 x 8960 | 33.8 | 116.7 | 3.45x | 90% | 85% |

All 7 linear layers of one decoder layer, summed: TinyLlama 122.9 vs 389.1 us (3.17x), Qwen2.5 130.0 vs 409.6 us (3.15x). By batch (TinyLlama): 3.17x at M = 1, 2.50x at M = 4, 1.80x at M = 8, 1.60x at M = 32, 1.23x at M = 64, 1.03x at M = 128.

**Why two bandwidth percentages.** The brief asks for a percent of the measured copy peak (234.5 GB/s). A GEMV is almost pure reads, and the measured read-only peak is higher (248.0 GB/s): cuBLAS's fp16 GEMV reaches 100 to 102% of the copy number. The read peak is the honest roof for this kernel, so both are reported.

## Measurement fixes this phase

- **The L2 flush polluted the timing.** The first flusher zeroed a 64 MB buffer, leaving L2 full of dirty lines that the timed kernel then had to write back to DRAM. That added about 8 us to every call: the INT4 GEMV measured 36.9 us and cuBLAS 106.5 us, against 26.9 and 97.2 us of kernel time in `ncu`. The flusher now reads the buffer instead (clean lines are dropped for free), and the same calls measure 28.7 and 98.3 us, `ncu` plus the 2 us cost of an empty event pair. Without any flush INT4 would look even better (14.3 us, since its 5.9 MB of weights fit in L2 while cuBLAS's 23 MB don't), which is why every GEMM run is cold.
- **The PyTorch dequant reference was a Python loop over 8 nibble positions.** It was vectorized (one broadcast shift plus a gather) so the unfused baseline measures PyTorch, not my loop. It's still 19x to 138x slower than cuBLAS, which is the point: dequantization has to be fused.

## Correctness and sanitizers

- `tests/test_quantize.py` (9): exact pack/unpack round trip on CPU and GPU, the documented bit layout, every element within half a step, bits per weight, all-zero groups, and clean errors for K not a multiple of the group, bad group sizes and out-of-range q.
- `tests/test_gemm.py` (284): GEMV at M = 1..8 and GEMM at M in {1, 8, 9, 16, 17, 32, 64, 65, 128} on every layer shape of both models with G = 64 and 128, ragged N for the GEMV, the dispatcher, and clean rejections. The reference is "dequantize in PyTorch, then FP32 matmul". Worst relative error (max abs error / max abs reference): GEMV 4.3e-4, GEMM 4.8e-4, both under fp16's half-ULP of 4.9e-4. The kernels compute the same math as the reference, differing only in accumulation order and the final fp16 rounding.
- `results/sanitizer.json`: memcheck, racecheck, synccheck and initcheck report 0 errors and 0 hazards on `kf_sanitize_gemm.exe` (24 runs: GEMV M = 1..8, the GEMM in all three tile layouts with 4-way split-K, outputs checked against a double-precision reference), and on the attention binary.
- Full suite: 1307 tests pass (159 skipped: fp32 cases of the fp16/bf16-only Tensor Core attention kernels).

## Limitations

- Weight-only INT4 with fp16 activations, symmetric, no zero points, no activation-aware scaling (AWQ) or error-compensating rounding (GPTQ). Quality is measured on real weights in Phase 5.
- The GEMM needs N to be a multiple of 64 (true for every layer of both models), and the GEMV handles any N.
- Small-N layers (k/v, N = 256) and mid-size M (16 to 64) leave bandwidth on the table, see the two future-work items above.

## Interview questions

**1. Why does INT4 weight-only quantization speed up decode but not prefill?**
At M = 1 a linear layer does 2 FLOPs per weight while reading every weight once, so it is DRAM bound, and 4-bit weights move 3.9x fewer bytes. At large M each weight is reused M times, the layer becomes compute bound, and the bandwidth saving stops mattering. Measured on TinyLlama's 7 layers: 3.17x at M = 1, 1.03x at M = 128.

**2. Why must dequantization be fused?**
Unfused, the fp16 weights are written to memory and read back, so the layer moves more bytes than plain fp16. Measured, the PyTorch dequant-then-matmul path is 19x to 138x slower than cuBLAS. The fused kernels unpack in registers (GEMV) or shared memory (GEMM), and only the 4-bit weights ever cross DRAM.

**3. How do you turn 4-bit integers into fp16 cheaply?**
The magic-number trick: put the nibble u in the low mantissa bits of 0x6400 (1024.0). In [1024, 2048) fp16 has a mantissa step of exactly 1, so the bit pattern 0x6400 | u is the number 1024 + u. Subtracting 1032 gives q = u - 8 exactly, two values at a time with one `__hsub2`. The packing order (k0 k2 k4 k6 k1 k3 k5 k7) is chosen so one shift and mask yields a ready half2 pair of consecutive k.

**4. How did you choose between the GEMV and the GEMM?**
By measurement: both kernels at M = 1..8 on every layer shape, picking the largest M at which the GEMV's total time is lower. The GEMV's CUDA-core math grows linearly with M, while the Tensor Core GEMM pads to a 16-row tile almost for free, so the crossover is M = 8.

**5. Your first GEMM was as slow at M = 16 as at M = 64. How did you find out why?**
Nsight showed `math_pipe_throttle` as the top stall: the Tensor Cores were the bottleneck, which makes no sense for a memory-bound shape until you notice 48 of every 64 rows were zero padding. The 64-row tile did 4x the necessary MMAs. Templating the warp layout to use 16- or 32-row tiles cut the kernel from 70.5 to 47.4 us, and the top stall became `long_scoreboard` (waiting on memory), the right bottleneck for this kernel.

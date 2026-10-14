# Lessons learned

The five hardest bugs and surprises in KernelForge, and how they were found. Several were measurement bugs, not kernel bugs. On a laptop GPU, a benchmark harness is a piece of code that needs debugging too.

## 1. The GPU was running at 30 W, and the giveaway was a ratio, not a number

**Symptom.** The first hardware-ceiling run measured 16.5 TFLOPS for FP16 Tensor Core GEMM and 3.3 TFLOPS for FP32 SGEMM. Both looked plausible in isolation for a laptop part.

**What exposed it.** FP16 GEMM with FP16 accumulation was only 1.14x faster than with FP32 accumulation. On GeForce Ada that ratio should be about 2x, because FP32-accumulate MMA runs at half rate. A ratio that's off is harder to explain away than an absolute number. `nvidia-smi -q -d POWER` then showed "Current Power Limit: 30.00 W" against a 115 W default: the laptop vendor's control app was in a quiet mode.

**Fix and lesson.** At the 135 W performance-mode cap the same runs measured 31.0 and 8.5 TFLOPS (1.9x and 2.6x higher), and the FP16-accumulate ratio became 1.84x. Every benchmark JSON now logs AC power, power plan, clocks and throttle reasons. Sanity-check ratios that the architecture fixes (accumulate modes, read vs copy bandwidth), not just absolute numbers.

## 2. Naive attention looked memory bound on the roofline, but DRAM was idle

**Symptom.** Naive attention's arithmetic intensity is 14.7 FLOP per DRAM byte, under the FP32 ridge point of 36, so a DRAM roofline calls it memory bound. The obvious "fix" would be reducing DRAM traffic.

**What exposed it.** Nsight Compute showed DRAM throughput at 3.5% in the QK^T kernel, which takes 78% of the time, while L1/TEX was at 98.6% and 88% of warp stalls were `lg_throttle` (load/store queue full). Each warp read K with a 128-byte stride, so every 2-byte load instruction touched 32 cache lines. They hit in L1, but L1 throughput was the wall.

**Lesson.** The DRAM roofline is one roof among several. The same pattern came back in v1 (fused, CUDA cores): DRAM at 1% and `mio_throttle` as top stall, because it issued 1.06 shared-memory loads per FFMA against an issue budget of 0.25. That pointed straight at Tensor Cores, which feed the math from register fragments, rather than at more tuning of v1.

## 3. A benchmark harness can create speedups that don't exist, four different ways

All four were found because a result looked inconsistent, and all four were fixed in the shared harness (`kernelforge/bench_utils.py`) so every later number inherited the fix.

- **Python launch overhead inside the timed interval.** Enqueueing one cuDNN SDPA call costs up to 0.9 ms of Python, longer than the 0.1 ms kernel, so the GPU reached the start event and idled. fp16 and bf16 timings differed by 3x on identical shapes. Fix: a GPU spin (`torch.cuda._sleep`) sized to 5x the measured host enqueue time before each timed run.
- **Warmup measured wall time while launches were asynchronous.** For slow kernels the loop queued about 1000 launches before 100 ms had passed, and v1 at N = 8192 spent minutes per config in warmup. Fix: synchronize each warmup call.
- **Windows sysmem fallback.** Naive attention at N = 8192 needed 8.6 GB in an 8 GB card. Instead of failing, the driver spilled to system RAM over PCIe, and the kernel "ran" at 45 W with full VRAM. Fix: cap the torch allocator to the VRAM free at startup, so it raises a clean OOM.
- **The L2 flush polluted the timing.** Zeroing a 64 MB buffer to flush L2 left it full of dirty lines, and the timed INT4 GEMV paid about 8 us writing them back (36.9 us measured vs 26.9 us kernel time in `ncu`, cuBLAS 106.5 vs 97.2). A read-based flush matches `ncu`. Without any flush the INT4 kernel would have looked 2x better again, because its 5.9 MB of weights fit in L2 and cuBLAS's 23 MB don't.

**Lesson.** Cross-check benchmark timings against `ncu` kernel durations. When they disagree by more than the event overhead (about 2 us here), the harness is measuring something other than the kernel.

## 4. The INT4 GEMM was as slow at M = 16 as at M = 64

**Symptom.** The Tensor Core dequant GEMM took 68.6 us at M = 16 and 76.8 us at M = 64 on TinyLlama's gate_proj. For a weight-streaming kernel, time should barely depend on M, but it shouldn't be slow at M = 16 either.

**What exposed it.** First, small grids: k/v projections (N = 256) launched 4 blocks on 24 SMs, fixed with split-K (30.7 to 10.2 us). Then Nsight Compute at M = 16 showed `math_pipe_throttle` as the top stall with the tensor pipe 39% busy: compute bound, for a shape that should be memory bound. The block tile was 64 rows tall, so 48 of every 64 rows were zero padding and the kernel did 4x the necessary MMAs. Templating the warp layout so the tile follows M (16, 32 or 64 rows) cut the kernel from 70.5 to 47.4 us, and the top stall became `long_scoreboard`, waiting on memory, the expected bottleneck.

**Lesson.** When a stall reason doesn't fit the kernel's expected regime, the kernel is doing work you didn't intend. Here the profiler pointed at wasted MMAs before any arithmetic did.

## 5. Tools that can't see your kernel, and a result that looked too good

- **racecheck never reached the kernels.** compute-sanitizer racecheck on a Python + torch process spent over 20 minutes of CPU instrumenting torch's own CUDA modules, with the GPU at 11% utilization, and never got to the attention kernels. The fix was structural: every kernel family lives in a torch-free header (`attn_v1_kernel.cuh`, `attn_v4_kernel.cuh`, `dequant_gemm_tc_kernel.cuh`, ...), so small standalone binaries (`kf_sanitize_attn`, `kf_sanitize_gemm`) run racecheck in about a minute. The same headers let CI compile every kernel without torch.
- **ptxas spilled registers it didn't have to.** v1's fp16 builds, two GEMV variants and one v2 variant were capped at 64 or 128 registers and spilled, while sibling instantiations used more. ptxas picks a register target from an occupancy heuristic when `__launch_bounds__` has no minimum-blocks argument. An explicit `__launch_bounds__(threads, min_blocks)` fixed each one. `results/ptxas.json`, regenerated on every build, is what made these visible.
- **A CUDA graph that was 2.2x faster and subtly wrong.** Eager decode in PyTorch on Windows ran at about 10 tokens/s with the GPU busy only 20% of each step, so the decode benchmark captures each step in a CUDA graph. The first version looked great, but its tokens differed from eager decode. Against eager decode on the *same* static KV cache (identical kernels and masks) it diverged at the very first token. The cause: transformers' `StaticLayer` writes new K/V at its own device-side counter, not at the `cache_position` passed in, and the two warmup calls before capture had advanced that counter by 2. Every replayed step wrote its K/V two slots beyond what the mask let the token see, so no token could attend to itself. On real text the damage was small enough that greedy decoding agreed for 14 to 119 tokens, which is exactly why it needed an exact reference to catch. Resetting the counter after warmup made graph and eager decode identical token for token, and the corrected (slower) numbers replaced the buggy ones.
- **Exactly zero difference.** Plugged into TinyLlama, the KernelForge attention produced logits identical to PyTorch SDPA to the last bit. That's suspicious for two different kernels, so it was checked layer by layer and then in isolation: at 17 tokens both compute the softmax over a single tile in FP32 and round identically, while at 300 tokens 0.006% of outputs differ by one fp16 ULP. Over full 2048-token WikiText-2 windows, perplexity differs by 0.0001. Real, but worth proving.

// Fused INT4 dequantize + GEMV for small batch (decode), M = 1..8.
//
//   y[m, n] = sum_k x[m, k] * q[n, k] * scale[n, k / G]
//
// At decode the weight matrix is read once per token for very little math
// (2*M FLOPs per weight), so the kernel is bound by DRAM bandwidth and the goal
// is to stream the packed weights at close to the measured peak. 4-bit weights
// cut that traffic about 4x vs fp16, and dequantizing in registers means the
// fp16 weights never exist in memory.
//
// Work split: each warp owns ROWS output rows and walks all of K. Lane l reads
// 32-weight chunks l, l+32, l+64, ... of each row: one 16-byte load per chunk,
// so a warp reads 512 contiguous bytes of a weight row per instruction. Each
// chunk lies inside one quantization group (32 divides G), so it needs a single
// scale. The chunk's weights are unpacked once and reused for all M activation
// rows. Partial dot products accumulate in FP32 per lane, and a 5-step
// warp-shuffle reduction finishes each output.
//
// Kernel and raw-pointer launcher only (no ATen), shared by the torch wrapper
// (dequant_gemv.cu) and the standalone sanitizer binary.
#pragma once

#include <stdexcept>

#include "common/check.cuh"
#include "common/cuda_utils.cuh"
#include "gemm/int4.cuh"

namespace kf::gemv {

constexpr int kWarps = 4;
constexpr int kRows = 2;  // output rows per warp

// __launch_bounds__(128, 3): at least 3 blocks (12 warps) per SM, which caps
// registers at 65536 / (3 * 128) = 170. Without the explicit minimum, ptxas
// capped M=5 and M=6 at 128 registers and spilled 8 to 16 bytes. 12 warps x
// 2 rows x 512 bytes keep about 12 KB of weight loads in flight per SM.
template <int M>
__global__ void __launch_bounds__(kWarps * 32, 3)
    dequant_gemv_kernel(const __half* __restrict__ x, const uint4* __restrict__ wq, const __half* __restrict__ scales,
                        __half* __restrict__ y, int N, int K, int group) {
  const int lane = threadIdx.x % 32;
  const int n0 = (blockIdx.x * kWarps + threadIdx.x / 32) * kRows;
  if (n0 >= N) return;        // whole warp exits together, no shuffle hazard
  const int chunks = K / 32;  // 32 weights = one uint4 of packed words
  const int groups = K / group;
  const int chunks_per_group = group / 32;

  float acc[M][kRows];
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[m][r] = 0.f;

  for (int c = lane; c < chunks; c += 32) {
    // Dequantize this chunk for every owned row: 16 half2 values per row. Kept
    // as half2 (exact: |q| <= 8) and widened to FP32 at the FMA, which halves
    // the registers vs storing float2. The ALU work is cheap next to the DRAM
    // time per chunk, and the first float2 version spilled at M=2, 5 and 6.
    __half2 w[kRows][16];
    float s[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      const int n = min(n0 + r, N - 1);  // a missing last row repeats row N-1, never stored
      const uint4 packed = wq[size_t(n) * chunks + c];
      s[r] = __half2float(scales[size_t(n) * groups + c / chunks_per_group]);
      const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        __half2 h[4];
        kf::int4::unpack8(words[j], h);
#pragma unroll
        for (int i = 0; i < 4; ++i) w[r][4 * j + i] = h[i];
      }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
      // 32 activations = 4 x 16-byte loads, from L1/L2 (x is tiny and shared by all warps).
      const uint4* xp = reinterpret_cast<const uint4*>(x + size_t(m) * K) + 4 * c;
      float part[kRows] = {};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint4 xv = xp[j];
        const __half2* xh = reinterpret_cast<const __half2*>(&xv);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float2 xf = __half22float2(xh[i]);
#pragma unroll
          for (int r = 0; r < kRows; ++r) {
            const float2 wf = __half22float2(w[r][4 * j + i]);
            part[r] = fmaf(xf.x, wf.x, fmaf(xf.y, wf.y, part[r]));
          }
        }
      }
#pragma unroll
      for (int r = 0; r < kRows; ++r) acc[m][r] = fmaf(s[r], part[r], acc[m][r]);
    }
  }

#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      float v = acc[m][r];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
      if (lane == 0 && n0 + r < N) y[size_t(m) * N + n0 + r] = __float2half_rn(v);
    }
}

template <int M>
void launch(const __half* x, const uint4* wq, const __half* s, __half* y, int N, int K, int group,
            cudaStream_t stream) {
  const int rows_per_block = kWarps * kRows;
  dequant_gemv_kernel<M><<<kf::ceil_div(N, rows_per_block), kWarps * 32, 0, stream>>>(x, wq, s, y, N, K, group);
  CUDA_CHECK_LAST();
}

// M in [1, 8]. wq is the packed weight matrix viewed as uint4 (4 words = 32 weights).
inline void dispatch(int M, const __half* x, const uint4* wq, const __half* s, __half* y, int N, int K, int group,
                     cudaStream_t stream) {
  switch (M) {
    case 1:
      return launch<1>(x, wq, s, y, N, K, group, stream);
    case 2:
      return launch<2>(x, wq, s, y, N, K, group, stream);
    case 3:
      return launch<3>(x, wq, s, y, N, K, group, stream);
    case 4:
      return launch<4>(x, wq, s, y, N, K, group, stream);
    case 5:
      return launch<5>(x, wq, s, y, N, K, group, stream);
    case 6:
      return launch<6>(x, wq, s, y, N, K, group, stream);
    case 7:
      return launch<7>(x, wq, s, y, N, K, group, stream);
    case 8:
      return launch<8>(x, wq, s, y, N, K, group, stream);
    default:
      throw std::invalid_argument("int4 GEMV handles M in [1, 8]");
  }
}

}  // namespace kf::gemv

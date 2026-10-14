// Fused INT4 dequantize + GEMM on Tensor Cores for larger batch (M = 16..128).
//
//   y[M, N] = x[M, K] * W_hat^T,  W_hat[n, k] = q[n, k] * scale[n, k / G]
//
// Block tile 64 (M) x 64 (N), k-tiles of 64, 4 warps in a 2 x 2 grid, each
// warp computing 32 x 32 with mma.sync m16n8k16 (fp16 in, FP32 accumulate).
// Per k-tile:
//   1. cp.async (double buffered) brings the 64 x 64 x tile and the packed
//      64 x 64 INT4 weight tile (64 rows x 32 bytes) into shared memory
//   2. the block dequantizes the weight tile once into a swizzled fp16 tile
//      (unpack8 + one half2 multiply by the row's scale; a 64-wide k-tile always
//      lies inside one group because G is 64 or 128)
//   3. ldmatrix + mma.sync, exactly as in attention v4: x rows are the A
//      operand, and W stored [n][k] is already the "col" B operand
// Weight traffic from DRAM is the packed 4-bit tile. The fp16 weights only
// ever exist in shared memory.
//
// Split-K: the grid is N/64 x ceil(M/64) blocks, which is only 4 blocks for a
// 256-wide k/v projection and 32 for down_proj at M <= 64, on 24 SMs. Each
// block then walks all of K alone, so the GPU is mostly idle. When the grid is
// small, blockIdx.z takes a contiguous slice of the k-tiles, writes an FP32
// partial [split][M][N], and splitk_reduce sums the slices into y (in a fixed
// order, so results are deterministic).
//
// Kernels and raw-pointer launchers only (no ATen), shared by the torch
// wrapper (dequant_gemm_tc.cu) and the standalone sanitizer binary.
#pragma once

#include <algorithm>
#include <cstddef>

#include "attention/attn_v4_tile.cuh"
#include "common/check.cuh"
#include "common/cp_async.cuh"
#include "common/cuda_utils.cuh"
#include "common/mma_utils.cuh"
#include "gemm/int4.cuh"

namespace kf::gemm_tc {

using kf::attn_v4::swz;
constexpr int BN = 64, BK = 64, kThreads = 128;

// Warp layouts: WM x WN warps (4 total), each computing MT x NT mma tiles of
// 16 x 8. BM = 16 * WM * MT rows per block, BN = 8 * WN * NT = 64 columns.
// The host picks the smallest BM that covers M: with a fixed 64-row tile, M=16
// spent 4x the Tensor Core work on zero rows (ncu: math_pipe_throttle top
// stall, M=16 as slow as M=64).
template <int WM, int WN, int MT, int NT>
struct Layout {
  static constexpr int BM = 16 * WM * MT;
  static_assert(WM * WN == 4 && 8 * WN * NT == BN && NT % 2 == 0, "4 warps covering 64 columns");
};

template <int WM, int WN, int MT, int NT>
__global__ void __launch_bounds__(kThreads)
    dequant_gemm_tc_kernel(const __half* __restrict__ x, const uint32_t* __restrict__ wq,
                           const __half* __restrict__ scales, __half* __restrict__ y,
                           float* __restrict__ partial, int M, int N, int K, int group,
                           int kt_per_split) {
  constexpr int BM = Layout<WM, WN, MT, NT>::BM;
  __shared__ __align__(128) __half As[2][BM * BK];         // x tiles, swizzled
  __shared__ __align__(128) uint32_t Wp[2][BN * BK / 8];   // packed weight tiles, [n][8 words]
  __shared__ __align__(128) __half Bs[BN * BK];            // dequantized weight tile, swizzled

  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int g = lane >> 2, t = lane & 3;
  const int wm = warp / WN, wn = warp % WN;  // warp's (16 MT) x (8 NT) sub-tile
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int kwords = K / 8, groups = K / group;
  // This block's k-tile range: [kt_begin, kt_begin + num_kt).
  const int kt_begin = blockIdx.z * kt_per_split;
  const int num_kt = min(kt_per_split, K / BK - kt_begin);

  auto issue = [&](int kt, int st) {
    const int k0 = (kt_begin + kt) * BK;
    for (int i = tid; i < BM * BK / 8; i += kThreads) {  // BM x 8 chunks of x
      const int r = i / (BK / 8), c = (i % (BK / 8)) * 8;
      const bool ok = m0 + r < M;  // rows past M are zero-filled
      kf::cp_async16(&As[st][swz<BK>(r, c)], ok ? x + size_t(m0 + r) * K + k0 + c : x, ok);
    }
    {  // 128 weight chunks of 16 bytes (4 words = 32 k), one per thread
      const int r = tid / 2, c = (tid % 2) * 4;
      kf::cp_async16(&Wp[st][r * 8 + c], wq + size_t(n0 + r) * kwords + k0 / 8 + c, true);
    }
    kf::cp_async_commit();
  };

  float acc[MT][NT][4] = {};  // [m16 tile][n8 tile][fragment]
  issue(0, 0);
  for (int kt = 0; kt < num_kt; ++kt) {
    const int st = kt & 1;
    if (kt + 1 < num_kt) {
      issue(kt + 1, st ^ 1);
      kf::cp_async_wait<1>();
    } else {
      kf::cp_async_wait<0>();
    }
    __syncthreads();  // tile kt landed (all threads' copies), and Bs is free

    {  // Dequantize: thread owns row n = tid / 2, words 4*(tid % 2) .. +3 (32 k values)
      const int r = tid / 2, w0 = (tid % 2) * 4;
      const __half s = scales[size_t(n0 + r) * groups + ((kt_begin + kt) * BK) / group];
      const __half2 s2 = __halves2half2(s, s);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        __half2 h[4];
        kf::int4::unpack8(Wp[st][r * 8 + w0 + j], h);
#pragma unroll
        for (int i = 0; i < 4; ++i) h[i] = __hmul2(h[i], s2);
        *reinterpret_cast<uint4*>(&Bs[swz<BK>(r, (w0 + j) * 8)]) = *reinterpret_cast<uint4*>(h);
      }
    }
    __syncthreads();  // Bs complete

#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      uint32_t a[MT][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
        kf::mma::ldmatrix_x4(a[mt], &As[st][swz<BK>(wm * 16 * MT + mt * 16 + (lane & 15), kk * 16 + (lane >> 4) * 8)]);
#pragma unroll
      for (int nb2 = 0; nb2 < NT / 2; ++nb2) {
        uint32_t b[4];
        const int n = wn * 8 * NT + nb2 * 16 + (lane & 7) + (lane >> 4) * 8;
        kf::mma::ldmatrix_x4(b, &Bs[swz<BK>(n, kk * 16 + ((lane >> 3) & 1) * 8)]);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          kf::mma::Mma<__half>::run(acc[mt][2 * nb2], a[mt], b[0], b[1]);
          kf::mma::Mma<__half>::run(acc[mt][2 * nb2 + 1], a[mt], b[2], b[3]);
        }
      }
    }
    __syncthreads();  // done with As[st] (refilled next iteration) and Bs
  }

#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int m = m0 + wm * 16 * MT + mt * 16 + g + 8 * h;
      if (m >= M) continue;
#pragma unroll
      for (int nb = 0; nb < NT; ++nb) {
        const int n = n0 + wn * 8 * NT + nb * 8 + 2 * t;
        if (partial)
          *reinterpret_cast<float2*>(partial + (size_t(blockIdx.z) * M + m) * N + n) =
              make_float2(acc[mt][nb][2 * h], acc[mt][nb][2 * h + 1]);
        else
          *reinterpret_cast<__half2*>(y + size_t(m) * N + n) =
              __floats2half2_rn(acc[mt][nb][2 * h], acc[mt][nb][2 * h + 1]);
      }
    }
}

// y[i] = sum over splits of partial[s][i], fixed order, FP32 then one rounding to fp16.
__global__ void splitk_reduce(const float* __restrict__ partial, __half* __restrict__ y, int splits, size_t mn) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < mn; i += size_t(gridDim.x) * blockDim.x) {
    float v = 0.f;
    for (int s = 0; s < splits; ++s) v += partial[s * mn + i];
    y[i] = __float2half_rn(v);
  }
}


// Rows per block for a given M (see Layout): the smallest tile that covers M.
inline int block_m(int M) { return M <= 16 ? 16 : (M <= 32 ? 32 : 64); }

// Split-K factor: aim for at least 4 blocks per SM in total while giving each
// split at least 4 k-tiles, so the cp.async pipeline still has a few
// iterations to overlap. Returns (splits, k-tiles per split).
inline void choose_splits(int M, int N, int K, int sm_count, int& splits, int& kt_per_split) {
  const int tiles = (N / BN) * kf::ceil_div(M, block_m(M));
  const int num_kt = K / BK;
  splits = std::max(1, std::min(kf::ceil_div(4 * sm_count, tiles), num_kt / 4));
  kt_per_split = kf::ceil_div(num_kt, splits);
  splits = kf::ceil_div(num_kt, kt_per_split);  // no empty splits
}

// FP32 workspace the caller must provide (0 when no split-K is needed).
inline size_t workspace_floats(int M, int N, int K, int sm_count) {
  int splits, kt;
  choose_splits(M, N, K, sm_count, splits, kt);
  return splits > 1 ? size_t(splits) * M * N : 0;
}

template <int WM, int WN, int MT, int NT>
void launch(const __half* x, const uint32_t* wq, const __half* scales, __half* y, float* ws, int M, int N,
            int K, int group, int sm_count, cudaStream_t stream) {
  int splits, kt_per_split;
  choose_splits(M, N, K, sm_count, splits, kt_per_split);
  dim3 grid(N / BN, kf::ceil_div(M, Layout<WM, WN, MT, NT>::BM), splits);
  dequant_gemm_tc_kernel<WM, WN, MT, NT><<<grid, kThreads, 0, stream>>>(
      x, wq, scales, y, splits > 1 ? ws : nullptr, M, N, K, group, kt_per_split);
  CUDA_CHECK_LAST();
  if (splits > 1) {
    const size_t mn = size_t(M) * N;
    splitk_reduce<<<static_cast<int>(std::min<size_t>(kf::ceil_div(mn, size_t(256)), 1024)), 256, 0, stream>>>(
        ws, y, splits, mn);
    CUDA_CHECK_LAST();
  }
}

// Requires N % 64 == 0, K % 64 == 0, and ws with workspace_floats(...) floats.
inline void run(const __half* x, const uint32_t* wq, const __half* scales, __half* y, float* ws, int M, int N,
                int K, int group, int sm_count, cudaStream_t stream) {
  if (M <= 16) launch<1, 4, 1, 2>(x, wq, scales, y, ws, M, N, K, group, sm_count, stream);
  else if (M <= 32) launch<2, 2, 1, 4>(x, wq, scales, y, ws, M, N, K, group, sm_count, stream);
  else launch<2, 2, 2, 4>(x, wq, scales, y, ws, M, N, K, group, sm_count, stream);
}

}  // namespace kf::gemm_tc

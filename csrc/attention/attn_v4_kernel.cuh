// Fused attention v4: mma.sync Tensor Cores with FlashAttention-2 style warp
// partitioning. Everything per row lives in registers.
//
// Each warp owns 16 query rows for the whole kernel. Per key tile it computes
// S = Q K^T into mma accumulators, applies the online softmax to those
// registers (row max and sum are shuffles across the 4 lanes of a quad, see
// the fragment layout in common/mma_utils.cuh), rescales its O accumulators in
// registers, repacks P from the S accumulators into A fragments, and runs
// P V. Compared with v2/v3 (WMMA):
//   - no S, P or O buffers in shared memory, no fragment <-> smem round trips
//   - shared memory holds only Q (read into registers once) and the double
//     buffered K/V tiles: 40 KB at D=64 instead of 79 KB, so 2 blocks fit per SM
//   - K/V tiles are unpadded with an XOR swizzle, read with ldmatrix
//   - cp.async double buffering as in v3
#pragma once

#include <stdexcept>
#include <string>

#include "attention/attn_v4_tile.cuh"
#include "common/check.cuh"
#include "common/mma_utils.cuh"
#include "common/types.cuh"

namespace kf::attn_v4 {

constexpr float kLog2e = 1.4426950408889634f;

// Problem shape. Queries [B, H, Nq, D], keys/values [B, Hk, Nk, D] with
// H % Hk == 0 (grouped-query attention: query head h reads K/V head
// h / (H / Hk), no copies). Nq != Nk is decode or chunked prefill against a
// KV cache: query i sits at absolute position Nk - Nq + i, so causal masking
// is aligned to the bottom-right corner (query i sees keys 0 .. Nk - Nq + i).
struct Shape {
  int B, H, Hk, Nq, Nk;
};

template <int D, int kWarps, int Bc>
constexpr size_t smem_bytes() {
  return size_t(16 * kWarps + 2 * 2 * Bc) * D * 2;  // Q + 2 stages x (K + V)
}

template <typename T, int D, int kWarps, int Bc>
__global__ void __launch_bounds__(kWarps * 32)
    attn_v4_kernel(const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v, T* __restrict__ o,
                   Shape sh, float scale_log2, bool causal) {
  constexpr int NT = kWarps * 32;
  constexpr int Br = 16 * kWarps;
  constexpr int KD = D / 16;  // k-steps of Q K^T over the head dim
  constexpr int NB = Bc / 8;  // 16x8 score tiles per warp per key tile
  constexpr int OB = D / 8;   // 16x8 output tiles per warp
  using M = kf::mma::Mma<T>;

  extern __shared__ __align__(128) unsigned char smem_raw[];
  T* Qs = reinterpret_cast<T*>(smem_raw);
  T* KV = Qs + Br * D;
  auto Ks = [&](int st) { return KV + st * 2 * Bc * D; };
  auto Vs = [&](int st) { return KV + st * 2 * Bc * D + Bc * D; };

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int g = lane >> 2;  // fragment row group
  const int t = lane & 3;   // thread in group
  const int q0 = blockIdx.x * Br;
  const int wr0 = q0 + warp * 16;   // first query row of this warp
  const int Nq = sh.Nq, N = sh.Nk;  // N: key length (the loop bound)
  const int qoff = N - Nq;          // absolute position of query row 0
  const int b = blockIdx.y / sh.H, h = blockIdx.y % sh.H;
  const size_t qhead = size_t(blockIdx.y) * Nq * D;
  const size_t kvhead = (size_t(b) * sh.Hk + h / (sh.H / sh.Hk)) * N * D;

  int num_kt = kf::ceil_div(N, Bc);
  if (causal) num_kt = min(num_kt, kf::ceil_div(min(q0 + Br, Nq) + qoff, Bc));

  load_tile_async<T, D, Br, NT>(Qs, q + qhead, q0, Nq);
  load_tile_async<T, D, Bc, NT>(Ks(0), k + kvhead, 0, N);
  load_tile_async<T, D, Bc, NT>(Vs(0), v + kvhead, 0, N);
  kf::cp_async_commit();

  uint32_t qf[KD][4];                           // this warp's Q rows as A fragments, loaded once
  float m[2] = {kf::neg_inf(), kf::neg_inf()};  // rows g and g+8
  float l[2] = {0.f, 0.f};
  float oacc[OB][4];
#pragma unroll
  for (int ob = 0; ob < OB; ++ob)
#pragma unroll
    for (int e = 0; e < 4; ++e) oacc[ob][e] = 0.f;

  for (int kt = 0; kt < num_kt; ++kt) {
    const int k0 = kt * Bc;
    const int st = kt & 1;
    if (kt + 1 < num_kt) {  // prefetch the next tile, wait for this one (see v3)
      load_tile_async<T, D, Bc, NT>(Ks(st ^ 1), k + kvhead, k0 + Bc, N);
      load_tile_async<T, D, Bc, NT>(Vs(st ^ 1), v + kvhead, k0 + Bc, N);
      kf::cp_async_commit();
      kf::cp_async_wait<1>();
    } else {
      kf::cp_async_wait<0>();
    }
    __syncthreads();

    if (kt == 0) {
      // ldmatrix x4 addresses: lanes 0-15 give rows 0-15 at column 16kk,
      // lanes 16-31 the same rows at column 16kk+8 -> a0..a3 in order.
#pragma unroll
      for (int kk = 0; kk < KD; ++kk)
        kf::mma::ldmatrix_x4(qf[kk], Qs + swz<D>(warp * 16 + (lane & 15), kk * 16 + (lane >> 4) * 8));
    }

    // S = Q K^T. One ldmatrix x4 on K yields B fragments for two 8-key tiles:
    // matrices = (keys 0-7, d 0-7), (keys 0-7, d 8-15), (keys 8-15, d 0-7),
    // (keys 8-15, d 8-15). K rows are [key][d], already the "col" B layout.
    float s[NB][4];
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
      for (int e = 0; e < 4; ++e) s[nb][e] = 0.f;
#pragma unroll
    for (int kk = 0; kk < KD; ++kk) {
#pragma unroll
      for (int nb2 = 0; nb2 < NB / 2; ++nb2) {
        uint32_t b[4];
        const int key = nb2 * 16 + (lane & 7) + (lane >> 4) * 8;
        const int d = kk * 16 + ((lane >> 3) & 1) * 8;
        kf::mma::ldmatrix_x4(b, Ks(st) + swz<D>(key, d));
        M::run(s[2 * nb2], qf[kk], b[0], b[1]);
        M::run(s[2 * nb2 + 1], qf[kk], b[2], b[3]);
      }
    }

    // Online softmax in registers. Element e of tile nb is row g + 8*(e>>1),
    // key k0 + 8*nb + 2t + (e&1). A row is spread over the 4 lanes of a quad.
    const bool need_mask = (k0 + Bc > N) || (causal && k0 + Bc - 1 > wr0 + qoff);
    float mx[2] = {kf::neg_inf(), kf::neg_inf()};
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float x = s[nb][e] * scale_log2;
        if (need_mask) {
          const int key = k0 + nb * 8 + 2 * t + (e & 1);
          const int row = wr0 + g + 8 * (e >> 1) + qoff;  // absolute query position
          if (key >= N || (causal && key > row)) x = kf::neg_inf();
        }
        s[nb][e] = x;
        mx[e >> 1] = fmaxf(mx[e >> 1], x);
      }
    float base[2], alpha[2], sum[2] = {0.f, 0.f};
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], 1));
      mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], 2));
      const float m_new = fmaxf(m[r], mx[r]);
      base[r] = (m_new == kf::neg_inf()) ? 0.f : m_new;  // fully masked rows: avoid inf - inf
      alpha[r] = exp2f(m[r] - base[r]);
      m[r] = m_new;
    }
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float p = exp2f(s[nb][e] - base[e >> 1]);
        s[nb][e] = p;
        sum[e >> 1] += p;
      }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      sum[r] += __shfl_xor_sync(0xffffffff, sum[r], 1);
      sum[r] += __shfl_xor_sync(0xffffffff, sum[r], 2);
      l[r] = l[r] * alpha[r] + sum[r];
    }
#pragma unroll
    for (int ob = 0; ob < OB; ++ob) {
      oacc[ob][0] *= alpha[0];
      oacc[ob][1] *= alpha[0];
      oacc[ob][2] *= alpha[1];
      oacc[ob][3] *= alpha[1];
    }

    // O += P V. The C fragments of score tiles 2kk and 2kk+1 are exactly the
    // A fragment of P for keys 16kk..16kk+15, so P never leaves registers.
    // V is [key][d]; ldmatrix .trans turns 8x8 blocks of it into "col" B
    // fragments: matrices = (keys 0-7, d 0-7), (keys 8-15, d 0-7),
    // (keys 0-7, d 8-15), (keys 8-15, d 8-15).
#pragma unroll
    for (int kk = 0; kk < Bc / 16; ++kk) {
      const uint32_t a[4] = {M::pack(s[2 * kk][0], s[2 * kk][1]), M::pack(s[2 * kk][2], s[2 * kk][3]),
                             M::pack(s[2 * kk + 1][0], s[2 * kk + 1][1]), M::pack(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
      for (int ob2 = 0; ob2 < OB / 2; ++ob2) {
        uint32_t b[4];
        const int key = kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        const int d = ob2 * 16 + (lane >> 4) * 8;
        kf::mma::ldmatrix_x4_trans(b, Vs(st) + swz<D>(key, d));
        M::run(oacc[2 * ob2], a, b[0], b[1]);
        M::run(oacc[2 * ob2 + 1], a, b[2], b[3]);
      }
    }
    __syncthreads();  // the next iteration prefetches into this stage
  }

  // Normalize and store: each lane writes 2 adjacent outputs per 8-column tile.
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int row = wr0 + g + 8 * r;
    if (row >= Nq) continue;
    const float inv_l = 1.f / l[r];
    T* orow = o + qhead + size_t(row) * D;
#pragma unroll
    for (int ob = 0; ob < OB; ++ob)
      *reinterpret_cast<uint32_t*>(orow + ob * 8 + 2 * t) =
          M::pack(oacc[ob][2 * r] * inv_l, oacc[ob][2 * r + 1] * inv_l);
  }
}

template <typename T, int D, int kWarps, int Bc>
void launch(const T* q, const T* k, const T* v, T* o, Shape sh, float scale_log2, bool causal, cudaStream_t stream) {
  constexpr size_t smem = smem_bytes<D, kWarps, Bc>();
  auto kernel = attn_v4_kernel<T, D, kWarps, Bc>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)));
  dim3 grid(kf::ceil_div(sh.Nq, 16 * kWarps), sh.B * sh.H);
  kernel<<<grid, kWarps * 32, smem, stream>>>(q, k, v, o, sh, scale_log2, causal);
  CUDA_CHECK_LAST();
}

// Configs for the occupancy sweep: (warps per block, Bc). config -1 = tuned default.
constexpr int kNumConfigs = 4;

template <typename T, int D>
void dispatch_d(int config, const T* q, const T* k, const T* v, T* o, Shape sh, float scale_log2, bool causal,
                cudaStream_t stream) {
  switch (config) {
    case 0:
      return launch<T, D, 4, 32>(q, k, v, o, sh, scale_log2, causal, stream);
    case 1:
      return launch<T, D, 4, 64>(q, k, v, o, sh, scale_log2, causal, stream);
    case 2:
      return launch<T, D, 8, 32>(q, k, v, o, sh, scale_log2, causal, stream);
    case 3:
      return launch<T, D, 8, 64>(q, k, v, o, sh, scale_log2, causal, stream);
    default:
      throw std::invalid_argument("attn_v4 config must be in [0, 4), got " + std::to_string(config));
  }
}

template <typename T>
void dispatch(int config, int D, const T* q, const T* k, const T* v, T* o, Shape sh, float scale_log2, bool causal,
              cudaStream_t stream) {
  // Defaults from bench/sweep_v4.py (results/attention_v4_sweep.json): at D=64,
  // 8 warps x Bc=64 (2 blocks/SM). At D=128, 4 warps x Bc=32: larger tiles push
  // registers (166-184) and smem (64-96 KB) to 1 block per SM.
  if (config < 0) config = (D == 64) ? 3 : 0;
  if (D == 64) return dispatch_d<T, 64>(config, q, k, v, o, sh, scale_log2, causal, stream);
  if (D == 128) return dispatch_d<T, 128>(config, q, k, v, o, sh, scale_log2, causal, stream);
  throw std::invalid_argument("attn_v4 supports D in {64, 128}, got " + std::to_string(D));
}

// Square self-attention over BH independent heads (the original interface).
template <typename T>
void dispatch(int config, int D, const T* q, const T* k, const T* v, T* o, int BH, int N, float scale_log2, bool causal,
              cudaStream_t stream) {
  dispatch<T>(config, D, q, k, v, o, Shape{BH, 1, 1, N, N}, scale_log2, causal, stream);
}

}  // namespace kf::attn_v4

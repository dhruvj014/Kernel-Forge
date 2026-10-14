// Fused attention v1: tiling plus online softmax, FP32 math on CUDA cores.
//
// One thread block owns one (batch*head, query tile of Br rows). It loads the
// Q tile into shared memory once, then streams K and V tiles of Bc rows
// through shared memory. For every key tile it computes the Br x Bc score
// block, updates a running row max m and normalizer l, rescales the output
// accumulator by exp(m_old - m_new), and adds P V. The N x N score matrix is
// never written to global memory, so extra memory is O(Br*Bc) per block
// instead of O(N^2) per head.
//
// Thread layout: TPR = NT / Br threads cooperate on one query row. Thread g of
// a row owns score columns {g, g+TPR, g+2*TPR, ...} and output columns
// {g, g+TPR, ...}. Row reductions (max, sum) are shuffles across those TPR
// adjacent lanes, so no shared memory or __syncthreads is needed for softmax.
//
// Math is FP32 regardless of I/O dtype. Tiles are converted to FP32 when they
// land in shared memory. Phase 3 replaces the two matmuls with Tensor Cores.
//
// Kernels and launchers only (no ATen), so both the torch extension
// (attn_v1_fp32.cu) and the standalone sanitizer binary
// (csrc/micro/sanitize_attn.cu) can include them.
#pragma once

#include <stdexcept>
#include <string>

#include "common/check.cuh"
#include "common/cuda_utils.cuh"
#include "common/types.cuh"

namespace kf::attn_v1 {

// exp(x) = exp2(x * log2(e)). Folding log2(e) into the softmax scale once lets
// the inner loop use exp2f, which maps to a single MUFU.EX2 instruction.
constexpr float kLog2e = 1.4426950408889634f;

// Copies ROWS x D elements starting at row `row0` into a padded FP32 shared
// tile with row stride LD. Global loads are 16 bytes per thread (8 halves or 4
// floats) and consecutive threads read consecutive 16-byte chunks, so a warp
// reads 512 contiguous bytes per instruction (fully coalesced). Rows past N
// are zero-filled (predicated load), which keeps their scores finite; they are
// masked to -inf later.
template <typename T, int D, int LD, int ROWS, int NT>
__device__ __forceinline__ void load_tile(float* __restrict__ dst, const T* __restrict__ src, int row0, int N) {
  constexpr int VEC = 16 / sizeof(T);
  static_assert(D % VEC == 0, "D must be a multiple of the vector width");
  constexpr int kChunks = ROWS * D / VEC;
  for (int i = threadIdx.x; i < kChunks; i += NT) {
    const int e = i * VEC;
    const int rr = e / D;
    const int cc = e % D;
    const int gr = row0 + rr;
    float vals[VEC];
    if (gr < N) {
      const uint4 raw = *reinterpret_cast<const uint4*>(src + static_cast<size_t>(gr) * D + cc);
      const T* t = reinterpret_cast<const T*>(&raw);
#pragma unroll
      for (int u = 0; u < VEC; ++u) vals[u] = kf::to_float(t[u]);
    } else {
#pragma unroll
      for (int u = 0; u < VEC; ++u) vals[u] = 0.f;
    }
#pragma unroll
    for (int u = 0; u < VEC; ++u) dst[rr * LD + cc + u] = vals[u];
  }
}

// Reduction across the TPR adjacent lanes that share a query row. TPR is a
// power of two <= 32, and a row's lanes are aligned to TPR within the warp,
// so xor offsets below TPR never leave the group.
template <int TPR>
__device__ __forceinline__ float group_max(float v) {
#pragma unroll
  for (int off = TPR / 2; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
  return v;
}
template <int TPR>
__device__ __forceinline__ float group_sum(float v) {
#pragma unroll
  for (int off = TPR / 2; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
  return v;
}

// __launch_bounds__(NT, 1): only promise one resident block per SM. Without
// the explicit minimum, ptxas capped the fp16/bf16 instantiations at 64
// registers and spilled up to 76 bytes (results/ptxas.json), while the fp32
// ones used 80+. Shared memory (50 to 97 KB per block) already limits these
// configs to 1 or 2 blocks per SM, so a register cap buys no occupancy.
template <typename T, int D, int Br, int Bc, int NT>
__global__ void __launch_bounds__(NT, 1)
    attn_v1_kernel(const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v, T* __restrict__ o, int N,
                   float scale_log2, bool causal) {
  constexpr int TPR = NT / Br;   // threads per query row
  constexpr int SPT = Bc / TPR;  // score columns per thread
  constexpr int OPT = D / TPR;   // output columns per thread
  static_assert(NT % Br == 0 && TPR <= 32 && (TPR & (TPR - 1)) == 0, "TPR must be a power of two <= 32");
  static_assert(Bc % TPR == 0 && D % TPR == 0, "tile must split evenly");
  // +1 float of padding per row. Without it, rows D=64 or 128 floats apart
  // map to the same bank, and the lanes of a warp that read the same column
  // of different rows (Q in the dot product, P in PV) would serialize.
  constexpr int LD = D + 1;
  constexpr int LDP = Bc + 1;

  extern __shared__ float smem[];
  float* Qs = smem;           // Br x LD
  float* KVs = Qs + Br * LD;  // Bc x LD, holds K then V of the same tile
  float* Ps = KVs + Bc * LD;  // Br x LDP, exp(S - m) for the PV product

  const int tid = threadIdx.x;
  const int r = tid / TPR;  // query row within the tile
  const int g = tid % TPR;  // this thread's slot within the row group
  const int q0 = blockIdx.x * Br;
  const int qi = q0 + r;  // global query index (may be >= N on the last tile)
  const size_t head_off = static_cast<size_t>(blockIdx.y) * N * D;
  const T* qh = q + head_off;
  const T* kh = k + head_off;
  const T* vh = v + head_off;

  load_tile<T, D, LD, Br, NT>(Qs, qh, q0, N);

  float m = kf::neg_inf();  // running row max (in log2 units)
  float l = 0.f;            // running sum of exp2(s - m)
  float acc[OPT];
#pragma unroll
  for (int j = 0; j < OPT; ++j) acc[j] = 0.f;

  // Causal block skipping: key tiles that start after the last query row of
  // this tile are entirely above the diagonal, so they are never loaded.
  int num_kt = kf::ceil_div(N, Bc);
  if (causal) num_kt = min(num_kt, kf::ceil_div(min(q0 + Br, N), Bc));

  for (int kt = 0; kt < num_kt; ++kt) {
    const int k0 = kt * Bc;
    // Barrier 1: every thread finished the previous tile's PV (reads of KVs
    // and Ps) before KVs is overwritten. On kt=0 it also publishes Qs.
    __syncthreads();
    load_tile<T, D, LD, Bc, NT>(KVs, kh, k0, N);
    __syncthreads();  // barrier 2: K tile visible to all threads

    // Masking is only needed on tiles that cross the diagonal or the end of
    // the sequence. Interior tiles skip the per-score comparisons.
    const bool need_mask = (k0 + Bc > N) || (causal && k0 + Bc - 1 > q0);

    // Register blocking: d is the outer loop, so q[r][d] is read from shared
    // memory once and reused for all SPT score columns this thread owns. With
    // the loops the other way round (j outer, d inner), every FMA needed two
    // shared-memory loads and the kernel was shared-memory bound (about 1.2
    // TFLOPS at D=64 in the first sweep). Now it is (1 + SPT) loads per SPT FMAs.
    float s[SPT];
#pragma unroll
    for (int j = 0; j < SPT; ++j) s[j] = 0.f;
    const float* qrow = Qs + r * LD;
    const float* kcol = KVs + g * LD;
#pragma unroll 8
    for (int d = 0; d < D; ++d) {
      const float qd = qrow[d];
#pragma unroll
      for (int j = 0; j < SPT; ++j) s[j] = fmaf(qd, kcol[j * TPR * LD + d], s[j]);
    }
    float tile_max = kf::neg_inf();
#pragma unroll
    for (int j = 0; j < SPT; ++j) {
      float sv = s[j] * scale_log2;
      if (need_mask) {
        const int kj = k0 + g + j * TPR;
        if (kj >= N || (causal && kj > qi)) sv = kf::neg_inf();
      }
      s[j] = sv;
      tile_max = fmaxf(tile_max, sv);
    }
    tile_max = group_max<TPR>(tile_max);

    // Online softmax update. m_new can only be -inf for padding rows whose
    // scores are all masked. Using 0 as the base then gives exp2(-inf) = 0
    // everywhere instead of exp2(-inf - -inf) = NaN.
    const float m_new = fmaxf(m, tile_max);
    const float base = (m_new == kf::neg_inf()) ? 0.f : m_new;
    const float alpha = exp2f(m - base);  // rescales history, 0 on first tile
    float row_sum = 0.f;
#pragma unroll
    for (int j = 0; j < SPT; ++j) {
      const float p = exp2f(s[j] - base);
      Ps[r * LDP + g + j * TPR] = p;
      row_sum += p;
    }
    row_sum = group_sum<TPR>(row_sum);
    l = l * alpha + row_sum;
    m = m_new;
#pragma unroll
    for (int j = 0; j < OPT; ++j) acc[j] *= alpha;

    // Barrier 3: all K reads are done (KVs can take V) and P is visible.
    __syncthreads();
    load_tile<T, D, LD, Bc, NT>(KVs, vh, k0, N);
    __syncthreads();  // barrier 4: V tile visible

    // acc[j] += sum_c P[r][c] * V[c][g + j*TPR]. P[r][c] is a broadcast
    // within the row group, and the TPR lanes read adjacent V columns.
    const float* prow = Ps + r * LDP;
    for (int c = 0; c < Bc; ++c) {
      const float p = prow[c];
      const float* vrow = KVs + c * LD + g;
#pragma unroll
      for (int j = 0; j < OPT; ++j) acc[j] = fmaf(p, vrow[j * TPR], acc[j]);
    }
  }

  if (qi < N) {
    // l >= 1 for every real row: the row max term contributes exp2(0) = 1.
    const float inv_l = 1.f / l;
    T* orow = o + head_off + static_cast<size_t>(qi) * D;
#pragma unroll
    for (int j = 0; j < OPT; ++j) orow[g + j * TPR] = kf::from_float<T>(acc[j] * inv_l);
  }
}

template <int D, int Br, int Bc>
constexpr size_t smem_bytes() {
  return sizeof(float) * (static_cast<size_t>(Br + Bc) * (D + 1) + static_cast<size_t>(Br) * (Bc + 1));
}

template <typename T, int D, int Br, int Bc>
void launch(const T* q, const T* k, const T* v, T* o, int BH, int N, float scale_log2, bool causal,
            cudaStream_t stream) {
  constexpr int NT = 256;
  constexpr size_t smem = smem_bytes<D, Br, Bc>();
  auto kernel = attn_v1_kernel<T, D, Br, Bc, NT>;
  // Above 48 KB of dynamic shared memory a kernel must opt in explicitly.
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
  dim3 grid(kf::ceil_div(N, Br), BH);
  kernel<<<grid, NT, smem, stream>>>(q, k, v, o, N, scale_log2, causal);
  CUDA_CHECK_LAST();
}

// Tile configurations swept in Phase 2 (index = `config` argument).
// -1 picks the default chosen by that sweep for the head dim.
template <typename T, int D>
void dispatch_config(int config, const T* q, const T* k, const T* v, T* o, int BH, int N, float scale_log2, bool causal,
                     cudaStream_t stream) {
  switch (config) {
    case 0:
      return launch<T, D, 32, 32>(q, k, v, o, BH, N, scale_log2, causal, stream);
    case 1:
      return launch<T, D, 32, 64>(q, k, v, o, BH, N, scale_log2, causal, stream);
    case 2:
      return launch<T, D, 64, 32>(q, k, v, o, BH, N, scale_log2, causal, stream);
    case 3:
      return launch<T, D, 64, 64>(q, k, v, o, BH, N, scale_log2, causal, stream);
    case 4:
      return launch<T, D, 128, 32>(q, k, v, o, BH, N, scale_log2, causal, stream);
    default:
      throw std::invalid_argument("unknown attn_v1 config " + std::to_string(config));
  }
}

}  // namespace kf::attn_v1

// Fused attention v2: the two matmuls on Tensor Cores via WMMA.
//
// Same online-softmax algorithm as v1 (see docs/phase2-notes.md). What changes:
//   - Q K^T and P V run as 16x16x16 WMMA tiles (fp16 or bf16 in, FP32 accumulate).
//   - Tiles stay in fp16/bf16 in shared memory (half the bytes of v1's FP32 tiles).
//   - 4 warps per block, each warp owns 16 query rows. A warp's scores, softmax
//     statistics and output rows are private to it, so only the shared K/V
//     tile loads need block-wide barriers. Everything else uses __syncwarp.
//
// The fragment-to-shared-memory handoff: WMMA accumulator fragments have an
// unspecified element layout, so a thread cannot tell which row an element
// belongs to. Softmax needs per-row max/sum, and the output accumulator needs
// per-row rescaling by exp(m_old - m_new). So each tile goes:
//   S fragments --store--> S (FP32, smem) --softmax by rows--> P (fp16/bf16, smem,
//   written over S) --load--> P fragments --mma with V--> O fragments
// and O lives in shared memory (FP32): before each PV product the warp
// rescales its O rows in smem, loads them as the accumulator, adds P V, and
// stores back. v4 (mma.sync) removes this round trip by using the documented
// register layout.
#pragma once

#include <mma.h>

#include <stdexcept>
#include <string>

#include "common/check.cuh"
#include "common/cp_async.cuh"
#include "common/cuda_utils.cuh"
#include "common/types.cuh"

namespace kf::attn_v2 {

namespace wmma = nvcuda::wmma;

constexpr float kLog2e = 1.4426950408889634f;
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kBr = 16 * kWarps;  // 64 query rows per block
// Row padding: 8 elements = 16 bytes keeps every row 16-byte aligned (vector
// loads, WMMA needs ld % 8 == 0 for 16-bit types) and shifts consecutive rows
// by 4 banks so the 8 rows a fragment load touches are spread out.
constexpr int kPad = 8;
constexpr int kPadF = 4;  // FP32 buffers: 16 bytes of padding per row

template <typename T>
struct Frag;  // WMMA needs the element type spelled out per dtype
template <>
struct Frag<__half> {
  using a_row = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>;
  using b_col = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major>;
  using b_row = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major>;
};
template <>
struct Frag<__nv_bfloat16> {
  using a_row = wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>;
  using b_col = wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major>;
  using b_row = wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major>;
};
using Acc = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

// kStages = 1: one K/V buffer (v2). kStages = 2: double buffered for cp.async (v3).
template <int D, int Bc, int kStages = 1>
struct Smem {
  static constexpr int ld = D + kPad;     // Q, K, V row stride (elements)
  static constexpr int lds = Bc + kPadF;  // S row stride (floats)
  static constexpr int ldp = 2 * lds;     // P row stride (16-bit elements, aliases S)
  static constexpr int ldo = D + kPadF;   // O row stride (floats)
  static constexpr size_t q_bytes = size_t(kBr) * ld * 2;
  static constexpr size_t kv_bytes = size_t(Bc) * ld * 2;
  static constexpr size_t s_bytes = size_t(kBr) * lds * 4;
  static constexpr size_t o_bytes = size_t(kBr) * ldo * 4;
  static constexpr size_t total = q_bytes + kStages * 2 * kv_bytes + s_bytes + o_bytes;
};

// ROWS x D tile of 16-bit elements, global -> shared, 16 bytes per thread per
// step, rows past N zero-filled.
template <typename T, int D, int ROWS>
__device__ __forceinline__ void load_tile(T* __restrict__ dst, const T* __restrict__ src, int row0, int N) {
  constexpr int VEC = 8;
  constexpr int kChunks = ROWS * D / VEC;
  constexpr int ld = D + kPad;
  for (int i = threadIdx.x; i < kChunks; i += kThreads) {
    const int rr = (i * VEC) / D;
    const int cc = (i * VEC) % D;
    uint4 val = make_uint4(0, 0, 0, 0);
    if (row0 + rr < N) val = *reinterpret_cast<const uint4*>(src + size_t(row0 + rr) * D + cc);
    *reinterpret_cast<uint4*>(dst + rr * ld + cc) = val;
  }
}

// Same tile copy issued as cp.async (no registers, no waiting). Rows past N
// use the zero-fill form, with a valid in-bounds address that is never read.
template <typename T, int D, int ROWS>
__device__ __forceinline__ void load_tile_async(T* __restrict__ dst, const T* __restrict__ src, int row0, int N) {
  constexpr int VEC = 8;
  constexpr int kChunks = ROWS * D / VEC;
  constexpr int ld = D + kPad;
  for (int i = threadIdx.x; i < kChunks; i += kThreads) {
    const int rr = (i * VEC) / D;
    const int cc = (i * VEC) % D;
    const bool ok = row0 + rr < N;
    kf::cp_async16(dst + rr * ld + cc, ok ? src + size_t(row0 + rr) * D + cc : src, ok);
  }
}

// Which column a lane touches at step j of a row loop over 2*kCols columns,
// shared by the lane pair (2r, 2r+1) that owns row r.
//   kCF = false: lane `half` takes the contiguous half [half*kCols, ...). The
//     two halves are kCols floats apart, a multiple of 32 banks, so the pair
//     always collides, and row strides of 68 floats wrap every 8 rows.
//   kCF = true (conflict-free): columns are interleaved (2*jj + half), and rows
//     8..15 start one column pair later. With a row stride that is 4 banks mod
//     32, the 32 lanes then hit 32 distinct banks on every step.
template <bool kCF, int kCols>
__device__ __forceinline__ int lane_col(int j, int half, int row) {
  if constexpr (kCF) {
    int jj = j + (row >> 3);
    if (jj >= kCols) jj -= kCols;
    return 2 * jj + half;
  } else {
    return half * kCols + j;
  }
}

template <typename T, int D, int Bc, bool kCF, int kStages>
__global__ void __launch_bounds__(kThreads)
    attn_v2_kernel(const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v, T* __restrict__ o, int N,
                   float scale_log2, bool causal) {
  using L = Smem<D, Bc, kStages>;
  using F = Frag<T>;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  T* Qs = reinterpret_cast<T*>(smem_raw);
  // Stage st holds K at kv_base + st * 2 * kv_bytes and V right after it.
  unsigned char* kv_base = smem_raw + L::q_bytes;
  float* Ss = reinterpret_cast<float*>(kv_base + kStages * 2 * L::kv_bytes);
  float* Os = reinterpret_cast<float*>(kv_base + kStages * 2 * L::kv_bytes + L::s_bytes);

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int q0 = blockIdx.x * kBr;
  const size_t head = size_t(blockIdx.y) * N * D;

  // This warp's 16 rows of S/P and O.
  float* Sw = Ss + warp * 16 * L::lds;
  T* Pw = reinterpret_cast<T*>(Sw);
  float* Ow = Os + warp * 16 * L::ldo;

  // Softmax bookkeeping: 2 lanes per row, lane pair (2r, 2r+1) owns row r of
  // the warp's 16 and splits its columns in half. Both lanes keep m and l.
  const int row = lane / 2;
  const int half = lane % 2;
  const int qi = q0 + warp * 16 + row;
  float m = kf::neg_inf();
  float l = 0.f;

  int num_kt = kf::ceil_div(N, Bc);
  if (causal) num_kt = min(num_kt, kf::ceil_div(min(q0 + kBr, N), Bc));
  auto Kst = [&](int st) { return reinterpret_cast<T*>(kv_base + st * 2 * L::kv_bytes); };
  auto Vst = [&](int st) { return reinterpret_cast<T*>(kv_base + st * 2 * L::kv_bytes + L::kv_bytes); };

  if constexpr (kStages == 2) {
    // Prologue: Q and the first K/V tile form group 0.
    load_tile_async<T, D, kBr>(Qs, q + head, q0, N);
    load_tile_async<T, D, Bc>(Kst(0), k + head, 0, N);
    load_tile_async<T, D, Bc>(Vst(0), v + head, 0, N);
    kf::cp_async_commit();
  } else {
    load_tile<T, D, kBr>(Qs, q + head, q0, N);
  }
  for (int i = lane; i < 16 * D; i += 32) Ow[(i / D) * L::ldo + i % D] = 0.f;

  for (int kt = 0; kt < num_kt; ++kt) {
    const int k0 = kt * Bc;
    const T* Ks = Kst(kStages == 2 ? (kt & 1) : 0);
    const T* Vs = Vst(kStages == 2 ? (kt & 1) : 0);
    if constexpr (kStages == 2) {
      // Prefetch tile kt+1 into the other stage, then wait only for tile kt:
      // wait_group 1 leaves the just-issued group in flight, so its global
      // loads overlap all the math below. The other stage was last read in
      // iteration kt-1, which ended with a barrier, so overwriting it is safe.
      if (kt + 1 < num_kt) {
        const int st = (kt + 1) & 1;
        load_tile_async<T, D, Bc>(Kst(st), k + head, k0 + Bc, N);
        load_tile_async<T, D, Bc>(Vst(st), v + head, k0 + Bc, N);
        kf::cp_async_commit();
        kf::cp_async_wait<1>();
      } else {
        kf::cp_async_wait<0>();
      }
      __syncthreads();  // every thread's copies for tile kt (and Q) are visible
    } else {
      __syncthreads();  // all warps done with the previous K/V tile (and Q is visible on kt=0)
      load_tile<T, D, Bc>(const_cast<T*>(Ks), k + head, k0, N);
      load_tile<T, D, Bc>(const_cast<T*>(Vs), v + head, k0, N);
      __syncthreads();  // K/V visible
    }

    // S = Q K^T for this warp's 16 rows: Bc/16 output fragments, D/16 k-steps.
    // K is stored [key][d] row-major, which is K^T in column-major: the B
    // operand of Q K^T is loaded col_major straight from Ks, no transpose.
#pragma unroll
    for (int n = 0; n < Bc / 16; ++n) {
      Acc s_acc;
      wmma::fill_fragment(s_acc, 0.f);
#pragma unroll
      for (int kk = 0; kk < D / 16; ++kk) {
        typename F::a_row a;
        typename F::b_col b;
        wmma::load_matrix_sync(a, Qs + (warp * 16) * L::ld + kk * 16, L::ld);
        wmma::load_matrix_sync(b, Ks + (n * 16) * L::ld + kk * 16, L::ld);
        wmma::mma_sync(s_acc, a, b, s_acc);
      }
      wmma::store_matrix_sync(Sw + n * 16, s_acc, L::lds, wmma::mem_row_major);
    }
    __syncwarp();

    // Online softmax on row `row`, columns [half*Bc/2, half*Bc/2 + Bc/2).
    constexpr int kCols = Bc / 2;
    float s[kCols];
    float tmax = kf::neg_inf();
    const bool need_mask = (k0 + Bc > N) || (causal && k0 + Bc - 1 > q0 + warp * 16);
#pragma unroll
    for (int j = 0; j < kCols; ++j) {
      const int c = lane_col<kCF, kCols>(j, half, row);
      float sv = Sw[row * L::lds + c] * scale_log2;
      const int kj = k0 + c;
      if (need_mask && (kj >= N || (causal && kj > qi))) sv = kf::neg_inf();
      s[j] = sv;
      tmax = fmaxf(tmax, sv);
    }
    tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffff, tmax, 1));
    const float m_new = fmaxf(m, tmax);
    const float base = (m_new == kf::neg_inf()) ? 0.f : m_new;
    const float alpha = exp2f(m - base);
    // Every lane has read its S values into registers before any lane writes
    // P over the same bytes (P aliases S, and a P row is half as long).
    __syncwarp();
    float rsum = 0.f;
#pragma unroll
    for (int j = 0; j < kCols; ++j) {
      const float p = exp2f(s[j] - base);
      rsum += p;
      Pw[row * L::ldp + lane_col<kCF, kCols>(j, half, row)] = kf::from_float<T>(p);
    }
    rsum += __shfl_xor_sync(0xffffffff, rsum, 1);
    l = l * alpha + rsum;
    m = m_new;
    // Rescale this row's output accumulator, each lane half of the D columns.
#pragma unroll
    for (int j = 0; j < D / 2; ++j) Ow[row * L::ldo + lane_col<kCF, D / 2>(j, half, row)] *= alpha;
    __syncwarp();

    // O += P V: D/16 output fragments, Bc/16 k-steps. V is [key][d] row-major,
    // exactly the B operand layout for P V.
#pragma unroll
    for (int n = 0; n < D / 16; ++n) {
      Acc o_acc;
      wmma::load_matrix_sync(o_acc, Ow + n * 16, L::ldo, wmma::mem_row_major);
#pragma unroll
      for (int kk = 0; kk < Bc / 16; ++kk) {
        typename F::a_row a;
        typename F::b_row b;
        wmma::load_matrix_sync(a, Pw + kk * 16, L::ldp);
        wmma::load_matrix_sync(b, Vs + (kk * 16) * L::ld + n * 16, L::ld);
        wmma::mma_sync(o_acc, a, b, o_acc);
      }
      wmma::store_matrix_sync(Ow + n * 16, o_acc, L::ldo, wmma::mem_row_major);
    }
    __syncwarp();
    // Double buffering: the next iteration prefetches into this tile's stage,
    // so every warp must be done reading it first.
    if constexpr (kStages == 2) __syncthreads();
  }

  if (qi < N) {
    const float inv_l = 1.f / l;
    T* orow = o + head + size_t(qi) * D;
#pragma unroll
    for (int j = 0; j < D / 2; ++j) {
      const int c = lane_col<kCF, D / 2>(j, half, row);
      orow[c] = kf::from_float<T>(Ow[row * L::ldo + c] * inv_l);
    }
  }
}

template <typename T, int D, int Bc, bool kCF, int kStages>
void launch(const T* q, const T* k, const T* v, T* o, int BH, int N, float scale_log2, bool causal,
            cudaStream_t stream) {
  constexpr size_t smem = Smem<D, Bc, kStages>::total;
  auto kernel = attn_v2_kernel<T, D, Bc, kCF, kStages>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)));
  dim3 grid(kf::ceil_div(N, kBr), BH);
  kernel<<<grid, kThreads, smem, stream>>>(q, k, v, o, N, scale_log2, causal);
  CUDA_CHECK_LAST();
}

// Bc = 64 at D=64 (61 KB smem), Bc = 32 at D=128 (76 KB): Bc = 64 at D=128
// would need 103,424 bytes, over the 101,376-byte (99 KB) per-block limit.
// Ablation variants (one kernel template, so each step is measured on the same code):
//   0: v2 as first written (bank-conflicting softmax layout, synchronous loads)
//   1: v2 + conflict-free softmax layout
//   2: v3 = variant 1 + cp.async double-buffered K/V (D=64: 79 KB smem, D=128: 93 KB)
template <typename T, int D, int Bc>
void dispatch_d(int variant, const T* q, const T* k, const T* v, T* o, int BH, int N, float scale_log2, bool causal,
                cudaStream_t stream) {
  switch (variant) {
    case 0:
      return launch<T, D, Bc, false, 1>(q, k, v, o, BH, N, scale_log2, causal, stream);
    case 1:
      return launch<T, D, Bc, true, 1>(q, k, v, o, BH, N, scale_log2, causal, stream);
    case 2:
      return launch<T, D, Bc, true, 2>(q, k, v, o, BH, N, scale_log2, causal, stream);
    default:
      throw std::invalid_argument("attn_v2 variant must be 0, 1 or 2");
  }
}

template <typename T>
void dispatch(int variant, int D, const T* q, const T* k, const T* v, T* o, int BH, int N, float scale_log2,
              bool causal, cudaStream_t stream) {
  if (D == 64) return dispatch_d<T, 64, 64>(variant, q, k, v, o, BH, N, scale_log2, causal, stream);
  if (D == 128) return dispatch_d<T, 128, 32>(variant, q, k, v, o, BH, N, scale_log2, causal, stream);
  throw std::invalid_argument("attn_v2 supports D in {64, 128}, got " + std::to_string(D));
}

}  // namespace kf::attn_v2

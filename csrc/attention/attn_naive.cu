// Naive attention: the "before" picture for every later kernel.
//
//   kernel 1: S = scale * Q K^T          (writes B*H*N*N FP32 scores to DRAM)
//   kernel 2: P = softmax(S) row-wise    (reads and rewrites all of S)
//   kernel 3: O = P V                    (reads all of P again)
//
// Correct but deliberately unoptimized: no shared-memory tiling, one thread
// per output element, and the full N x N score matrix round-trips through
// global memory three times. That N^2 traffic and N^2 memory footprint is
// exactly what the fused kernels in Phase 2 and 3 remove.
//
// Layout: q, k, v, o are contiguous [B, H, N, D]. Math is FP32 regardless of
// the I/O dtype.
#include "common/check.cuh"
#include "common/cuda_utils.cuh"
#include "common/torch_utils.h"
#include "common/types.cuh"

namespace {

constexpr int kQkTile = 16;  // 16x16 = 256 threads per block for kernel 1

template <typename T>
__global__ void qk_scores_kernel(const T* __restrict__ q, const T* __restrict__ k, float* __restrict__ s, int N, int D,
                                 float scale, bool causal) {
  // threadIdx.x walks key index j so consecutive threads write consecutive
  // score addresses (coalesced stores). The K reads are strided by D across
  // the warp, which is one of the inefficiencies this baseline keeps.
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  const int bh = blockIdx.z;
  if (i >= N || j >= N) return;

  float* s_row = s + (static_cast<size_t>(bh) * N + i) * N;
  if (causal && j > i) {
    s_row[j] = kf::neg_inf();  // exp(-inf) = 0 in the softmax pass
    return;
  }
  const T* q_row = q + (static_cast<size_t>(bh) * N + i) * D;
  const T* k_row = k + (static_cast<size_t>(bh) * N + j) * D;
  float acc = 0.f;
  for (int d = 0; d < D; ++d) acc += kf::to_float(q_row[d]) * kf::to_float(k_row[d]);
  s_row[j] = acc * scale;
}

// Block-wide reduction: shuffle within each warp, then one value per warp
// through shared memory, then a final shuffle in warp 0. `scratch` must hold
// 32 floats. The leading __syncthreads protects scratch when the function is
// called twice in a row. `identity` fills the lanes of warp 0 that have no
// partial (fewer than 32 warps), so it must be neutral for `op`.
template <typename Op>
__device__ float block_reduce(float val, float* scratch, Op op, float identity) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int num_warps = (blockDim.x + 31) >> 5;
  for (int off = 16; off > 0; off >>= 1) val = op(val, __shfl_xor_sync(0xffffffff, val, off));
  __syncthreads();
  if (lane == 0) scratch[warp] = val;
  __syncthreads();
  if (warp == 0) {
    val = lane < num_warps ? scratch[lane] : identity;
    for (int off = 16; off > 0; off >>= 1) val = op(val, __shfl_xor_sync(0xffffffff, val, off));
    if (lane == 0) scratch[0] = val;
  }
  __syncthreads();
  return scratch[0];
}

struct MaxOp {
  __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};
struct SumOp {
  __device__ float operator()(float a, float b) const { return a + b; }
};

// One block per score row. Classic 3-pass safe softmax: max, sum of
// exp(x - max), normalize. Subtracting the row max keeps exp() from
// overflowing for large-magnitude scores.
__global__ void softmax_rows_kernel(float* __restrict__ s, int N) {
  __shared__ float scratch[32];
  float* row = s + static_cast<size_t>(blockIdx.x) * N;

  float m = kf::neg_inf();
  for (int j = threadIdx.x; j < N; j += blockDim.x) m = fmaxf(m, row[j]);
  m = block_reduce(m, scratch, MaxOp{}, kf::neg_inf());

  // Every row has at least one finite score (causal keeps the diagonal), so
  // m is finite and masked entries become exp(-inf) = 0.
  float sum = 0.f;
  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    float e = expf(row[j] - m);
    row[j] = e;
    sum += e;
  }
  sum = block_reduce(sum, scratch, SumOp{}, 0.f);

  const float inv = 1.f / sum;
  for (int j = threadIdx.x; j < N; j += blockDim.x) row[j] *= inv;
}

constexpr int kPvThreadsX = 32;  // walks head dim d (coalesced V and O)
constexpr int kPvThreadsY = 8;   // walks query rows

template <typename T>
__global__ void pv_kernel(const float* __restrict__ p, const T* __restrict__ v, T* __restrict__ o, int N, int D,
                          bool causal) {
  const int i = blockIdx.x * blockDim.y + threadIdx.y;
  const int bh = blockIdx.y;
  if (i >= N) return;

  const float* p_row = p + (static_cast<size_t>(bh) * N + i) * N;
  const T* v_bh = v + static_cast<size_t>(bh) * N * D;
  T* o_row = o + (static_cast<size_t>(bh) * N + i) * D;
  // Causal rows have exact zeros past the diagonal, so stop the sum there.
  const int j_end = causal ? i + 1 : N;

  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float acc = 0.f;
    for (int j = 0; j < j_end; ++j) acc += p_row[j] * kf::to_float(v_bh[static_cast<size_t>(j) * D + d]);
    o_row[d] = kf::from_float<T>(acc);
  }
}

}  // namespace

at::Tensor attn_naive(at::Tensor q, at::Tensor k, at::Tensor v, bool causal, double scale) {
  KF_CHECK_INPUT(q);
  KF_CHECK_INPUT(k);
  KF_CHECK_INPUT(v);
  TORCH_CHECK(q.dim() == 4, "q must be [B, H, N, D], got ", q.sizes());
  TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes(), "q, k, v must have the same shape, got ", q.sizes(),
              ", ", k.sizes(), ", ", v.sizes());
  TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(),
              "q, k, v must have the same dtype");
  TORCH_CHECK(scale > 0.0, "scale must be positive");

  const int B = static_cast<int>(q.size(0));
  const int H = static_cast<int>(q.size(1));
  const int N = static_cast<int>(q.size(2));
  const int D = static_cast<int>(q.size(3));
  TORCH_CHECK(N >= 1 && D >= 1, "N and D must be >= 1");
  TORCH_CHECK(B * H <= 65535, "B*H must fit in gridDim.z/y (<= 65535)");

  c10::cuda::CUDAGuard guard(q.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // The O(N^2) intermediate. Allocated through the torch caching allocator so
  // an oversized request raises torch.OutOfMemoryError in Python.
  auto s = at::empty({static_cast<int64_t>(B) * H, N, N}, q.options().dtype(at::kFloat));
  auto o = at::empty_like(q);

  KF_DISPATCH_HALF_BF16_FLOAT(q.scalar_type(), "attn_naive", [&] {
    using T = kf::cuda_t<scalar_t>;
    const T* qp = reinterpret_cast<const T*>(q.data_ptr<scalar_t>());
    const T* kp = reinterpret_cast<const T*>(k.data_ptr<scalar_t>());
    const T* vp = reinterpret_cast<const T*>(v.data_ptr<scalar_t>());
    T* op = reinterpret_cast<T*>(o.data_ptr<scalar_t>());
    float* sp = s.data_ptr<float>();

    dim3 qk_block(kQkTile, kQkTile);
    dim3 qk_grid(kf::ceil_div(N, kQkTile), kf::ceil_div(N, kQkTile), B * H);
    qk_scores_kernel<T><<<qk_grid, qk_block, 0, stream>>>(qp, kp, sp, N, D, static_cast<float>(scale), causal);
    CUDA_CHECK_LAST();

    softmax_rows_kernel<<<B * H * N, 256, 0, stream>>>(sp, N);
    CUDA_CHECK_LAST();

    dim3 pv_block(kPvThreadsX, kPvThreadsY);
    dim3 pv_grid(kf::ceil_div(N, kPvThreadsY), B * H);
    pv_kernel<T><<<pv_grid, pv_block, 0, stream>>>(sp, vp, op, N, D, causal);
    CUDA_CHECK_LAST();
  });
  return o;
}

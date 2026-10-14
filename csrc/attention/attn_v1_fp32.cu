// Torch entry point for fused attention v1. Kernels: attn_v1_kernel.cuh.
#include "attention/attn_v1_kernel.cuh"
#include "common/torch_utils.h"

using kf::attn_v1::dispatch_config;
using kf::attn_v1::kLog2e;

int attn_v1_num_configs() { return 5; }

at::Tensor attn_v1(at::Tensor q, at::Tensor k, at::Tensor v, bool causal,
                   double scale, int64_t config) {
  KF_CHECK_INPUT(q);
  KF_CHECK_INPUT(k);
  KF_CHECK_INPUT(v);
  TORCH_CHECK(q.dim() == 4, "q must be [B, H, N, D], got ", q.sizes());
  TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes(),
              "q, k, v must have the same shape");
  TORCH_CHECK(k.scalar_type() == q.scalar_type() &&
                  v.scalar_type() == q.scalar_type(),
              "q, k, v must have the same dtype");
  TORCH_CHECK(scale > 0.0, "scale must be positive");
  const int B = static_cast<int>(q.size(0));
  const int H = static_cast<int>(q.size(1));
  const int N = static_cast<int>(q.size(2));
  const int D = static_cast<int>(q.size(3));
  TORCH_CHECK(D == 64 || D == 128, "attn_v1 supports head dim 64 or 128, got ", D);
  TORCH_CHECK(N >= 1, "N must be >= 1");
  TORCH_CHECK(B * H <= 65535, "B*H must be <= 65535 (gridDim.y)");
  for (const auto* t : {&q, &k, &v}) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0,
                "attn_v1 needs 16-byte aligned inputs (vectorized loads)");
  }

  // Defaults from the Phase 2 Br/Bc sweep (docs/phase2-notes.md).
  if (config < 0) config = 3;  // Br=64, Bc=64 won for both D=64 and D=128

  c10::cuda::CUDAGuard guard(q.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto o = at::empty_like(q);
  const float scale_log2 = static_cast<float>(scale) * kLog2e;
  const int cfg = static_cast<int>(config);

  KF_DISPATCH_HALF_BF16_FLOAT(q.scalar_type(), "attn_v1", [&] {
    using T = kf::cuda_t<scalar_t>;
    const T* qp = reinterpret_cast<const T*>(q.data_ptr<scalar_t>());
    const T* kp = reinterpret_cast<const T*>(k.data_ptr<scalar_t>());
    const T* vp = reinterpret_cast<const T*>(v.data_ptr<scalar_t>());
    T* op = reinterpret_cast<T*>(o.data_ptr<scalar_t>());
    if (D == 64)
      dispatch_config<T, 64>(cfg, qp, kp, vp, op, B * H, N, scale_log2, causal, stream);
    else
      dispatch_config<T, 128>(cfg, qp, kp, vp, op, B * H, N, scale_log2, causal, stream);
  });
  return o;
}

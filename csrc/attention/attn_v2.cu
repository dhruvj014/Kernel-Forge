// Torch entry point for fused attention v2 (WMMA Tensor Cores). Kernels: attn_v2_kernel.cuh.
#include "attention/attn_v2_kernel.cuh"
#include "common/torch_utils.h"

at::Tensor attn_v2(at::Tensor q, at::Tensor k, at::Tensor v, bool causal, double scale,
                   int64_t variant) {
  KF_CHECK_INPUT(q);
  KF_CHECK_INPUT(k);
  KF_CHECK_INPUT(v);
  TORCH_CHECK(q.dim() == 4, "q must be [B, H, N, D], got ", q.sizes());
  TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes(), "q, k, v must have the same shape");
  TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(),
              "q, k, v must have the same dtype");
  TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
              "attn_v2 (Tensor Cores) supports fp16 and bf16, got ", q.scalar_type());
  TORCH_CHECK(scale > 0.0, "scale must be positive");
  const int B = static_cast<int>(q.size(0));
  const int H = static_cast<int>(q.size(1));
  const int N = static_cast<int>(q.size(2));
  const int D = static_cast<int>(q.size(3));
  TORCH_CHECK(D == 64 || D == 128, "attn_v2 supports head dim 64 or 128, got ", D);
  TORCH_CHECK(N >= 1 && B * H <= 65535, "need N >= 1 and B*H <= 65535");
  for (const auto* t : {&q, &k, &v})
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0, "attn_v2 needs 16-byte aligned inputs");

  c10::cuda::CUDAGuard guard(q.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto o = at::empty_like(q);
  const float scale_log2 = static_cast<float>(scale) * kf::attn_v2::kLog2e;
  AT_DISPATCH_SWITCH(q.scalar_type(), "attn_v2",
                     AT_DISPATCH_CASE(at::kHalf, [&] {
                       using T = kf::cuda_t<scalar_t>;
                       kf::attn_v2::dispatch<T>(static_cast<int>(variant), D, reinterpret_cast<const T*>(q.data_ptr()),
                                                reinterpret_cast<const T*>(k.data_ptr()),
                                                reinterpret_cast<const T*>(v.data_ptr()),
                                                reinterpret_cast<T*>(o.data_ptr()), B * H, N, scale_log2,
                                                causal, stream);
                     }) AT_DISPATCH_CASE(at::kBFloat16, [&] {
                       using T = kf::cuda_t<scalar_t>;
                       kf::attn_v2::dispatch<T>(static_cast<int>(variant), D, reinterpret_cast<const T*>(q.data_ptr()),
                                                reinterpret_cast<const T*>(k.data_ptr()),
                                                reinterpret_cast<const T*>(v.data_ptr()),
                                                reinterpret_cast<T*>(o.data_ptr()), B * H, N, scale_log2,
                                                causal, stream);
                     }));
  return o;
}

// Torch entry point for fused attention v4 (mma.sync Tensor Cores). Kernels: attn_v4_kernel.cuh.
//
// q: [B, H, Nq, D], k and v: [B, Hk, Nk, D] with H % Hk == 0 (grouped-query
// attention) and Nq <= Nk when causal (decode / prefill against a KV cache,
// causal mask aligned to the bottom-right corner). Nq == Nk, H == Hk is
// ordinary self-attention.
#include "attention/attn_v4_kernel.cuh"
#include "common/torch_utils.h"

at::Tensor attn_v4(at::Tensor q, at::Tensor k, at::Tensor v, bool causal, double scale, int64_t config) {
  KF_CHECK_INPUT(q);
  KF_CHECK_INPUT(k);
  KF_CHECK_INPUT(v);
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, v must be 4-D [B, heads, N, D]");
  TORCH_CHECK(k.sizes() == v.sizes(), "k and v must have the same shape, got ", k.sizes(), " and ", v.sizes());
  TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(),
              "q, k, v must have the same dtype");
  TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
              "attn_v4 (Tensor Cores) supports fp16 and bf16, got ", q.scalar_type());
  TORCH_CHECK(scale > 0.0, "scale must be positive");
  const int B = static_cast<int>(q.size(0)), H = static_cast<int>(q.size(1));
  const int Nq = static_cast<int>(q.size(2)), D = static_cast<int>(q.size(3));
  const int Hk = static_cast<int>(k.size(1)), Nk = static_cast<int>(k.size(2));
  TORCH_CHECK(k.size(0) == B && k.size(3) == D, "k/v must match q in batch and head dim");
  TORCH_CHECK(Hk >= 1 && H % Hk == 0, "query heads (", H, ") must be a multiple of key/value heads (", Hk, ")");
  TORCH_CHECK(D == 64 || D == 128, "attn_v4 supports head dim 64 or 128, got ", D);
  TORCH_CHECK(Nq >= 1 && Nk >= 1, "sequence lengths must be >= 1");
  TORCH_CHECK(!causal || Nq <= Nk, "causal attention needs Nq <= Nk (queries are the last Nq positions)");
  TORCH_CHECK(B * H <= 65535, "B*H must be <= 65535");
  for (const auto* t : {&q, &k, &v})
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0, "attn_v4 needs 16-byte aligned inputs");

  c10::cuda::CUDAGuard guard(q.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto o = at::empty_like(q);
  const float scale_log2 = static_cast<float>(scale) * kf::attn_v4::kLog2e;
  const kf::attn_v4::Shape sh{B, H, Hk, Nq, Nk};
  AT_DISPATCH_SWITCH(
      q.scalar_type(), "attn_v4", AT_DISPATCH_CASE(at::kHalf, [&] {
        using T = kf::cuda_t<scalar_t>;
        kf::attn_v4::dispatch<T>(static_cast<int>(config), D, reinterpret_cast<const T*>(q.data_ptr()),
                                 reinterpret_cast<const T*>(k.data_ptr()), reinterpret_cast<const T*>(v.data_ptr()),
                                 reinterpret_cast<T*>(o.data_ptr()), sh, scale_log2, causal, stream);
      }) AT_DISPATCH_CASE(at::kBFloat16, [&] {
        using T = kf::cuda_t<scalar_t>;
        kf::attn_v4::dispatch<T>(static_cast<int>(config), D, reinterpret_cast<const T*>(q.data_ptr()),
                                 reinterpret_cast<const T*>(k.data_ptr()), reinterpret_cast<const T*>(v.data_ptr()),
                                 reinterpret_cast<T*>(o.data_ptr()), sh, scale_log2, causal, stream);
      }));
  return o;
}

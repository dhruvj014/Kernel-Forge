// Torch entry point for the INT4 dequant GEMV. Kernel: dequant_gemv_kernel.cuh.
#include "common/torch_utils.h"
#include "gemm/dequant_gemv_kernel.cuh"

void int4_check_inputs(const at::Tensor& x, const at::Tensor& packed, const at::Tensor& scales, int64_t group) {
  KF_CHECK_INPUT(x);
  KF_CHECK_INPUT(packed);
  KF_CHECK_INPUT(scales);
  TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kHalf, "x must be fp16 [M, K]");
  TORCH_CHECK(packed.dim() == 2 && packed.scalar_type() == at::kInt, "packed must be int32 [N, K / 8]");
  TORCH_CHECK(scales.dim() == 2 && scales.scalar_type() == at::kHalf, "scales must be fp16 [N, K / group]");
  TORCH_CHECK(group == 64 || group == 128, "group size must be 64 or 128, got ", group);
  const int64_t K = x.size(1), N = packed.size(0);
  TORCH_CHECK(K % group == 0, "K=", K, " is not a multiple of the group size ", group);
  TORCH_CHECK(packed.size(1) * 8 == K, "packed is [", N, ", ", packed.size(1), "], expected [N, K/8] with K=", K);
  TORCH_CHECK(scales.size(0) == N && scales.size(1) == K / group, "scales must be [N, K / group]");
  for (const auto* t : {&x, &packed})
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0, "INT4 kernels need 16-byte aligned inputs");
}

at::Tensor int4_gemv(at::Tensor x, at::Tensor packed, at::Tensor scales, int64_t group) {
  int4_check_inputs(x, packed, scales, group);
  const int M = static_cast<int>(x.size(0)), K = static_cast<int>(x.size(1));
  const int N = static_cast<int>(packed.size(0));
  TORCH_CHECK(M >= 1 && M <= 8, "int4_gemv handles M in [1, 8], got ", M, " (use int4_gemm)");
  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto y = at::empty({M, N}, x.options());
  auto* xp = reinterpret_cast<const __half*>(x.data_ptr());
  auto* wp = reinterpret_cast<const uint4*>(packed.data_ptr());
  auto* sp = reinterpret_cast<const __half*>(scales.data_ptr());
  auto* yp = reinterpret_cast<__half*>(y.data_ptr());
  const int g = static_cast<int>(group);
  kf::gemv::dispatch(M, xp, wp, sp, yp, N, K, g, stream);
  return y;
}

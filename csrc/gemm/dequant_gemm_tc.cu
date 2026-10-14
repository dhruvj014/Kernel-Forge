// Torch entry point for the INT4 Tensor Core GEMM. Kernels: dequant_gemm_tc_kernel.cuh.
#include "common/torch_utils.h"
#include "gemm/dequant_gemm_tc_kernel.cuh"

void int4_check_inputs(const at::Tensor& x, const at::Tensor& packed, const at::Tensor& scales, int64_t group);

at::Tensor int4_gemm(at::Tensor x, at::Tensor packed, at::Tensor scales, int64_t group) {
  using namespace kf::gemm_tc;
  int4_check_inputs(x, packed, scales, group);
  const int M = static_cast<int>(x.size(0)), K = static_cast<int>(x.size(1));
  const int N = static_cast<int>(packed.size(0));
  TORCH_CHECK(N % BN == 0, "int4_gemm needs N to be a multiple of ", BN, ", got ", N);
  TORCH_CHECK(K % BK == 0, "int4_gemm needs K to be a multiple of ", BK, ", got ", K);
  TORCH_CHECK(M >= 1 && M <= 65535 * 16, "bad M ", M);
  c10::cuda::CUDAGuard guard(x.device());
  int sm = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, x.get_device()));
  auto y = at::empty({M, N}, x.options());
  // Split-K partials go through the torch caching allocator.
  const size_t ws_floats = workspace_floats(M, N, K, sm);
  at::Tensor ws =
      ws_floats ? at::empty({static_cast<int64_t>(ws_floats)}, x.options().dtype(at::kFloat)) : at::Tensor();
  run(reinterpret_cast<const __half*>(x.data_ptr()), reinterpret_cast<const uint32_t*>(packed.data_ptr()),
      reinterpret_cast<const __half*>(scales.data_ptr()), reinterpret_cast<__half*>(y.data_ptr()),
      ws_floats ? ws.data_ptr<float>() : nullptr, M, N, K, static_cast<int>(group), sm,
      at::cuda::getCurrentCUDAStream());
  return y;
}

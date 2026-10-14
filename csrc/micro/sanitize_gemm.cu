// Standalone workload for compute-sanitizer on the INT4 kernels (GEMV and the
// Tensor Core GEMM with split-K), without PyTorch. See sanitize_attn.cu for
// why a standalone binary.
//
// Random q in [-8, 7] and fp16 scales are packed in the layout from
// kernelforge/quantize.py (nibbles k0 k2 k4 k6 k1 k3 k5 k7 from bit 0), and
// every output is checked against a double-precision CPU reference.
// K = 1024 with N = 128 makes the GEMM grid small enough to use 4 split-K slices.
//
//   compute-sanitizer --tool racecheck build\cmake\kf_sanitize_gemm.exe
#include <cmath>
#include <cstdio>
#include <vector>

#include "common/cuda_utils.cuh"
#include "gemm/dequant_gemm_tc_kernel.cuh"
#include "gemm/dequant_gemv_kernel.cuh"

namespace {

struct Lcg {
  uint32_t s;
  uint32_t next() { return s = s * 1664525u + 1013904223u; }
  float uniform() { return (static_cast<float>(next() >> 8) / 16777216.f) * 2.f - 1.f; }
};

constexpr int kNibbleK[8] = {0, 2, 4, 6, 1, 3, 5, 7};  // k offset stored at nibble i

}  // namespace

int main() {
  constexpr int N = 128, K = 1024;
  int sm = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0));
  float worst = 0.f;
  int runs = 0;
  for (int group : {64, 128}) {
    Lcg rng{static_cast<uint32_t>(group)};
    std::vector<int> q(size_t(N) * K);
    std::vector<__half> scales(size_t(N) * (K / group));
    std::vector<uint32_t> packed(size_t(N) * K / 8, 0);
    for (auto& v : q) v = static_cast<int>(rng.next() >> 28) - 8;  // [-8, 7]
    for (auto& s : scales) s = __float2half(0.01f + 0.02f * std::fabs(rng.uniform()));
    for (int n = 0; n < N; ++n)
      for (int w = 0; w < K / 8; ++w)
        for (int i = 0; i < 8; ++i)
          packed[size_t(n) * K / 8 + w] |= uint32_t(q[size_t(n) * K + 8 * w + kNibbleK[i]] + 8) << (4 * i);

    kf::DeviceBuffer<uint32_t> dw(packed.size());
    kf::DeviceBuffer<__half> ds(scales.size());
    CUDA_CHECK(cudaMemcpy(dw.get(), packed.data(), dw.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ds.get(), scales.data(), ds.bytes(), cudaMemcpyHostToDevice));

    for (int M : {1, 3, 8, 9, 16, 17, 33, 64, 65}) {
      std::vector<__half> x(size_t(M) * K), y(size_t(M) * N);
      for (auto& v : x) v = __float2half(rng.uniform());
      std::vector<double> ref(size_t(M) * N, 0.0);
      double ref_max = 0;
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
          double acc = 0;
          for (int k = 0; k < K; ++k)
            acc += double(__half2float(x[size_t(m) * K + k])) * q[size_t(n) * K + k] *
                   __half2float(scales[size_t(n) * (K / group) + k / group]);
          ref[size_t(m) * N + n] = acc;
          ref_max = std::max(ref_max, std::fabs(acc));
        }
      kf::DeviceBuffer<__half> dx(x.size()), dy(y.size());
      CUDA_CHECK(cudaMemcpy(dx.get(), x.data(), dx.bytes(), cudaMemcpyHostToDevice));
      const size_t ws_floats = kf::gemm_tc::workspace_floats(M, N, K, sm);
      kf::DeviceBuffer<float> ws(ws_floats);
      for (int kernel = 0; kernel < 2; ++kernel) {  // 0: GEMV (M <= 8 only), 1: Tensor Core GEMM
        if (kernel == 0 && M > 8) continue;
        CUDA_CHECK(cudaMemset(dy.get(), 0, dy.bytes()));
        if (kernel == 0)
          kf::gemv::dispatch(M, dx.get(), reinterpret_cast<const uint4*>(dw.get()), ds.get(), dy.get(), N, K, group,
                             nullptr);
        else
          kf::gemm_tc::run(dx.get(), dw.get(), ds.get(), dy.get(), ws.get(), M, N, K, group, sm, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(y.data(), dy.get(), dy.bytes(), cudaMemcpyDeviceToHost));
        float err = 0.f;
        for (size_t i = 0; i < y.size(); ++i)
          err = std::max(err, static_cast<float>(std::fabs(__half2float(y[i]) - ref[i]) / ref_max));
        worst = std::max(worst, err);
        ++runs;
        if (err > 2e-3f)
          std::printf("MISMATCH %s group=%d M=%d rel err %.3e\n", kernel ? "gemm" : "gemv", group, M, err);
      }
    }
  }
  std::printf("kf_sanitize_gemm: %d kernel runs, worst rel err %.3e -> %s\n", runs, worst,
              worst <= 2e-3f ? "OK" : "FAIL");
  return worst <= 2e-3f ? 0 : 1;
}

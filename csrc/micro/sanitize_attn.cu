// Standalone workload for compute-sanitizer (memcheck, racecheck, synccheck)
// on the fused attention kernels, without PyTorch.
//
// Why a separate binary: under racecheck, a Python + torch process spent over
// 20 minutes of CPU time instrumenting torch's CUDA modules before reaching
// our kernels. This binary loads only our kernels, so racecheck finishes in
// seconds. It runs every tile config, both head dims, causal and full, on
// ragged lengths that give 1 to 3 key tiles (enough to expose a missing
// barrier around the shared K/V buffer), and checks each output against a
// double-precision CPU reference.
//
//   compute-sanitizer --tool racecheck build\cmake\kf_sanitize_attn.exe
#include <cmath>
#include <cstdio>
#include <vector>

#include "attention/attn_v1_kernel.cuh"
#include "common/cuda_utils.cuh"

namespace {

// Deterministic uniform values in [-2, 2) from a 32-bit LCG.
struct Lcg {
  uint32_t s;
  float next() {
    s = s * 1664525u + 1013904223u;
    return (static_cast<float>(s >> 8) / 16777216.f) * 4.f - 2.f;
  }
};

void reference(const std::vector<float>& q, const std::vector<float>& k,
               const std::vector<float>& v, std::vector<float>& o, int BH,
               int N, int D, bool causal) {
  const double scale = 1.0 / std::sqrt(static_cast<double>(D));
  std::vector<double> s(N);
  for (int bh = 0; bh < BH; ++bh) {
    const size_t off = static_cast<size_t>(bh) * N * D;
    for (int i = 0; i < N; ++i) {
      double m = -1e300;
      const int jend = causal ? i + 1 : N;
      for (int j = 0; j < jend; ++j) {
        double dot = 0;
        for (int d = 0; d < D; ++d) dot += double(q[off + i * D + d]) * k[off + j * D + d];
        s[j] = dot * scale;
        m = std::max(m, s[j]);
      }
      double l = 0;
      for (int j = 0; j < jend; ++j) l += (s[j] = std::exp(s[j] - m));
      for (int d = 0; d < D; ++d) {
        double acc = 0;
        for (int j = 0; j < jend; ++j) acc += s[j] * v[off + j * D + d];
        o[off + i * D + d] = static_cast<float>(acc / l);
      }
    }
  }
}

}  // namespace

int main() {
  constexpr int BH = 2;
  constexpr int kConfigs = 5;
  float worst = 0.f;
  int runs = 0;
  for (int D : {64, 128}) {
    for (int N : {37, 70}) {
      const size_t n = static_cast<size_t>(BH) * N * D;
      std::vector<float> hq(n), hk(n), hv(n), ref(n), out(n);
      std::vector<__half> q16(n), k16(n), v16(n), o16(n);
      Lcg rng{static_cast<uint32_t>(N * 131 + D)};
      // Round inputs to fp16 first so the reference sees exactly what the kernel sees.
      for (size_t i = 0; i < n; ++i) {
        q16[i] = __float2half(rng.next()); hq[i] = __half2float(q16[i]);
        k16[i] = __float2half(rng.next()); hk[i] = __half2float(k16[i]);
        v16[i] = __float2half(rng.next()); hv[i] = __half2float(v16[i]);
      }
      kf::DeviceBuffer<__half> dq(n), dk(n), dv(n), dout(n);
      CUDA_CHECK(cudaMemcpy(dq.get(), q16.data(), dq.bytes(), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dk.get(), k16.data(), dk.bytes(), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dv.get(), v16.data(), dv.bytes(), cudaMemcpyHostToDevice));
      const float scale_log2 = kf::attn_v1::kLog2e / std::sqrt(static_cast<float>(D));
      for (bool causal : {false, true}) {
        reference(hq, hk, hv, ref, BH, N, D, causal);
        for (int cfg = 0; cfg < kConfigs; ++cfg) {
          CUDA_CHECK(cudaMemset(dout.get(), 0, dout.bytes()));
          if (D == 64)
            kf::attn_v1::dispatch_config<__half, 64>(cfg, dq.get(), dk.get(), dv.get(), dout.get(),
                                                     BH, N, scale_log2, causal, nullptr);
          else
            kf::attn_v1::dispatch_config<__half, 128>(cfg, dq.get(), dk.get(), dv.get(), dout.get(),
                                                      BH, N, scale_log2, causal, nullptr);
          CUDA_CHECK(cudaDeviceSynchronize());
          CUDA_CHECK(cudaMemcpy(o16.data(), dout.get(), dout.bytes(), cudaMemcpyDeviceToHost));
          float err = 0.f;
          for (size_t i = 0; i < n; ++i) err = std::max(err, std::fabs(__half2float(o16[i]) - ref[i]));
          worst = std::max(worst, err);
          ++runs;
          if (err > 2e-3f)
            std::printf("MISMATCH D=%d N=%d causal=%d config=%d err=%.3e\n", D, N, causal, cfg, err);
        }
      }
    }
  }
  std::printf("kf_sanitize_attn: %d kernel runs, worst abs err %.3e -> %s\n", runs, worst,
              worst <= 2e-3f ? "OK" : "FAIL");
  return worst <= 2e-3f ? 0 : 1;
}

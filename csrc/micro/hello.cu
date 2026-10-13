// Toolchain smoke test: a vector add that is also the target for the first
// Nsight Compute run (see docs/env.md).
#include <cstdio>
#include <vector>

#include "common/cuda_utils.cuh"

__global__ void vector_add(const float* __restrict__ a,
                           const float* __restrict__ b, float* __restrict__ c,
                           int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  constexpr int n = 1 << 24;
  std::vector<float> ha(n), hb(n), hc(n);
  for (int i = 0; i < n; ++i) {
    ha[i] = static_cast<float>(i);
    hb[i] = 2.0f * static_cast<float>(i);
  }

  kf::DeviceBuffer<float> da(n), db(n), dc(n);
  CUDA_CHECK(cudaMemcpy(da.get(), ha.data(), da.bytes(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db.get(), hb.data(), db.bytes(), cudaMemcpyHostToDevice));

  constexpr int block = 256;
  vector_add<<<kf::ceil_div(n, block), block>>>(da.get(), db.get(), dc.get(), n);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(hc.data(), dc.get(), dc.bytes(), cudaMemcpyDeviceToHost));

  int bad = 0;
  for (int i = 0; i < n; ++i) bad += (hc[i] != 3.0f * static_cast<float>(i));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major,
              prop.minor, prop.multiProcessorCount);
  std::printf("vector_add n=%d mismatches=%d -> %s\n", n, bad,
              bad == 0 ? "OK" : "FAIL");
  return bad == 0 ? 0 : 1;
}

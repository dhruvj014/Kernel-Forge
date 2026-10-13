// Measures achievable DRAM bandwidth and prints device properties as JSON.
// bench/hw_ceilings.py runs this binary and stores the result in
// results/hw_ceilings.json. These measured numbers, not the spec sheet, are
// the memory roof for every roofline and "% of peak" figure in the project.
#include <cstdio>
#include <string>
#include <vector>

#include "common/cuda_utils.cuh"
#include "common/timer.cuh"

namespace {

// float4 loads: each thread moves 16 bytes per instruction, so a warp issues
// 512 contiguous bytes, i.e. four full 128-byte cache lines per load. Scalar
// float loads would need 4x the instructions for the same traffic.
__global__ void read_kernel(const float4* __restrict__ in, size_t n4,
                            float* __restrict__ sink) {
  float acc = 0.f;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    float4 v = in[i];
    acc += v.x + v.y + v.z + v.w;
  }
  // Data-dependent store that never happens in practice. It stops the
  // compiler from proving the loads are dead and deleting them.
  if (acc == 1234567.0f) sink[0] = acc;
}

__global__ void copy_kernel(const float4* __restrict__ in,
                            float4* __restrict__ out, size_t n4) {
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    out[i] = in[i];
  }
}

__global__ void write_kernel(float4* __restrict__ out, size_t n4) {
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  float4 v = make_float4(1.f, 2.f, 3.f, 4.f);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    out[i] = v;
  }
}

int attr(cudaDeviceAttr a, int dev) {
  int v = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&v, a, dev));
  return v;
}

struct Result {
  double gbps;
  kf::TimingStats t;
  int grid;
};

// Tries several grid sizes (in multiples of one full wave) and keeps the best.
// A grid-stride loop makes any grid size correct, the sweep only picks the
// fastest launch shape.
template <typename Launch>
Result best_over_grids(Launch&& launch, double bytes, int wave) {
  Result best{0.0, {}, 0};
  for (int mult : {1, 2, 4, 8}) {
    int grid = wave * mult;
    auto t = kf::time_gpu([&] { launch(grid); }, 10, 50);
    double gbps = bytes / (t.median_ms * 1e-3) / 1e9;
    if (gbps > best.gbps) best = {gbps, t, grid};
  }
  return best;
}

std::string json_result(const char* name, const Result& r, double bytes) {
  char buf[512];
  std::snprintf(buf, sizeof(buf),
                "\"%s\": {\"gbps\": %.2f, \"median_ms\": %.4f, \"p10_ms\": "
                "%.4f, \"p90_ms\": %.4f, \"bytes_moved\": %.0f, \"grid\": %d}",
                name, r.gbps, r.t.median_ms, r.t.p10_ms, r.t.p90_ms, bytes,
                r.grid);
  return buf;
}

}  // namespace

int main() {
  int dev = 0;
  CUDA_CHECK(cudaSetDevice(dev));
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));

  constexpr int block = 256;
  int blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, copy_kernel, block, 0));
  int wave = blocks_per_sm * p.multiProcessorCount;

  // 1 GiB total footprint: far larger than L2 (tens of MB), so every pass
  // streams from DRAM and L2 hits cannot inflate the result.
  const size_t bytes_per_buf = size_t(512) << 20;
  const size_t n4 = bytes_per_buf / sizeof(float4);
  kf::DeviceBuffer<float4> a(n4), b(n4);
  kf::DeviceBuffer<float> sink(1);
  CUDA_CHECK(cudaMemset(a.get(), 0, a.bytes()));

  double rd_bytes = static_cast<double>(bytes_per_buf);
  double cp_bytes = 2.0 * bytes_per_buf;  // one read plus one write per element
  double wr_bytes = static_cast<double>(bytes_per_buf);

  Result rd = best_over_grids(
      [&](int g) {
        read_kernel<<<g, block>>>(a.get(), n4, sink.get());
        CUDA_CHECK_LAST();
      },
      rd_bytes, wave);
  Result cp = best_over_grids(
      [&](int g) {
        copy_kernel<<<g, block>>>(a.get(), b.get(), n4);
        CUDA_CHECK_LAST();
      },
      cp_bytes, wave);
  Result wr = best_over_grids(
      [&](int g) {
        write_kernel<<<g, block>>>(b.get(), n4);
        CUDA_CHECK_LAST();
      },
      wr_bytes, wave);

  int clock_khz = attr(cudaDevAttrClockRate, dev);
  int mem_clock_khz = attr(cudaDevAttrMemoryClockRate, dev);
  int bus_width = attr(cudaDevAttrGlobalMemoryBusWidth, dev);
  // Reported only for context: 2 transfers per clock (DDR) * bus bytes.
  double theoretical_gbps = 2.0 * mem_clock_khz * 1e3 * (bus_width / 8.0) / 1e9;

  std::printf("{\n");
  std::printf("  \"device\": {\"name\": \"%s\", \"cc\": \"%d.%d\", ", p.name,
              p.major, p.minor);
  std::printf("\"sm_count\": %d, \"total_global_mem_bytes\": %zu, ",
              p.multiProcessorCount, p.totalGlobalMem);
  std::printf("\"l2_cache_bytes\": %d, \"persisting_l2_max_bytes\": %d, ",
              p.l2CacheSize, p.persistingL2CacheMaxSize);
  std::printf("\"smem_per_block_bytes\": %zu, \"smem_per_block_optin_bytes\": %zu, ",
              p.sharedMemPerBlock, p.sharedMemPerBlockOptin);
  std::printf("\"smem_per_sm_bytes\": %zu, \"regs_per_sm\": %d, \"regs_per_block\": %d, ",
              p.sharedMemPerMultiprocessor, p.regsPerMultiprocessor,
              p.regsPerBlock);
  std::printf("\"max_threads_per_sm\": %d, \"max_threads_per_block\": %d, ",
              p.maxThreadsPerMultiProcessor, p.maxThreadsPerBlock);
  std::printf("\"max_blocks_per_sm\": %d, \"warp_size\": %d, ",
              p.maxBlocksPerMultiProcessor, p.warpSize);
  std::printf("\"sm_clock_max_khz\": %d, \"mem_clock_max_khz\": %d, ",
              clock_khz, mem_clock_khz);
  std::printf("\"mem_bus_width_bits\": %d, \"theoretical_dram_gbps_from_clocks\": %.2f},\n",
              bus_width, theoretical_gbps);
  std::printf("  \"dram_bandwidth\": {\n    \"buffer_bytes\": %zu, \"block\": %d, \"wave_blocks\": %d,\n",
              bytes_per_buf, block, wave);
  std::printf("    %s,\n", json_result("read", rd, rd_bytes).c_str());
  std::printf("    %s,\n", json_result("copy", cp, cp_bytes).c_str());
  std::printf("    %s\n  }\n}\n", json_result("write", wr, wr_bytes).c_str());
  return 0;
}

#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

#include "check.cuh"

namespace kf {

// cudaEvent-based timer. Events are recorded on the stream, so the measured
// interval is GPU execution time, not host launch overhead.
class GpuTimer {
 public:
  explicit GpuTimer(cudaStream_t stream = nullptr) : stream_(stream) {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  GpuTimer(const GpuTimer&) = delete;
  GpuTimer& operator=(const GpuTimer&) = delete;

  void start() { CUDA_CHECK(cudaEventRecord(start_, stream_)); }
  void stop() { CUDA_CHECK(cudaEventRecord(stop_, stream_)); }

  float elapsed_ms() {
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }

 private:
  cudaStream_t stream_;
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
};

struct TimingStats {
  float median_ms;
  float p10_ms;
  float p90_ms;
  float min_ms;
};

inline TimingStats summarize(std::vector<float> samples) {
  std::sort(samples.begin(), samples.end());
  auto pct = [&](double p) {
    size_t idx = static_cast<size_t>(p * (samples.size() - 1) + 0.5);
    return samples[idx];
  };
  return {pct(0.5), pct(0.1), pct(0.9), samples.front()};
}

// Runs `fn` (which must only enqueue GPU work) `warmup` times untimed, then
// `iters` times with one event pair per iteration.
template <typename Fn>
TimingStats time_gpu(Fn&& fn, int warmup = 10, int iters = 50,
                     cudaStream_t stream = nullptr) {
  for (int i = 0; i < warmup; ++i) fn();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  GpuTimer timer(stream);
  std::vector<float> samples;
  samples.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    timer.start();
    fn();
    timer.stop();
    samples.push_back(timer.elapsed_ms());
  }
  return summarize(std::move(samples));
}

}  // namespace kf

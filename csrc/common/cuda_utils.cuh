#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

#include "check.cuh"

namespace kf {

__host__ __device__ constexpr inline int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

__host__ __device__ constexpr inline size_t ceil_div(size_t a, size_t b) {
  return (a + b - 1) / b;
}

// Owning, move-only device allocation. Used by the standalone C++ binaries.
// The torch extension allocates through the PyTorch caching allocator instead.
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t count) : count_(count) {
    if (count_ > 0) CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
  }
  ~DeviceBuffer() {
    if (ptr_) cudaFree(ptr_);  // never throw from a destructor
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : ptr_(std::exchange(other.ptr_, nullptr)),
        count_(std::exchange(other.count_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr_) cudaFree(ptr_);
      ptr_ = std::exchange(other.ptr_, nullptr);
      count_ = std::exchange(other.count_, 0);
    }
    return *this;
  }

  T* get() const { return ptr_; }
  size_t size() const { return count_; }
  size_t bytes() const { return count_ * sizeof(T); }

 private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

}  // namespace kf

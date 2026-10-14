#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

// Errors throw instead of calling exit() so the same macros work in the
// standalone microbenchmarks and inside the torch extension, where pybind11
// turns std::runtime_error into a Python RuntimeError.
#define CUDA_CHECK(expr)                                                                                      \
  do {                                                                                                        \
    cudaError_t kf_err_ = (expr);                                                                             \
    if (kf_err_ != cudaSuccess) {                                                                             \
      std::ostringstream kf_msg_;                                                                             \
      kf_msg_ << "CUDA error " << cudaGetErrorName(kf_err_) << " (" << cudaGetErrorString(kf_err_) << ") at " \
              << __FILE__ << ":" << __LINE__ << " in `" #expr "`";                                            \
      throw std::runtime_error(kf_msg_.str());                                                                \
    }                                                                                                         \
  } while (0)

// Kernel launches do not return an error code. Launch-configuration errors are
// only visible through cudaGetLastError, so call this right after every launch.
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

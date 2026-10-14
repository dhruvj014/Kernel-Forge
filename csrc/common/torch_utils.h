#pragma once

// Lean ATen headers only. .cu files should not include torch/extension.h,
// which pulls in pybind11 and the Python C API.
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace kf {

// Maps torch element types to the CUDA-native types the kernels are written
// against. c10::Half and __half share the same 16-bit layout (likewise for
// BFloat16), so reinterpret_cast of data pointers is safe.
template <typename T>
struct CudaType {
  using type = T;
};
template <>
struct CudaType<at::Half> {
  using type = __half;
};
template <>
struct CudaType<at::BFloat16> {
  using type = __nv_bfloat16;
};

template <typename T>
using cuda_t = typename CudaType<T>::type;

}  // namespace kf

// Dispatches over exactly the dtypes the kernels implement (fp16, bf16,
// fp32). AT_DISPATCH_FLOATING_TYPES would also instantiate double.
#define KF_DISPATCH_HALF_BF16_FLOAT(TYPE, NAME, ...)                                                       \
  AT_DISPATCH_SWITCH(TYPE, NAME,                                                                           \
                     AT_DISPATCH_CASE(at::kHalf, __VA_ARGS__) AT_DISPATCH_CASE(at::kBFloat16, __VA_ARGS__) \
                         AT_DISPATCH_CASE(at::kFloat, __VA_ARGS__))

#define KF_CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define KF_CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define KF_CHECK_INPUT(x) \
  KF_CHECK_CUDA(x);       \
  KF_CHECK_CONTIG(x)

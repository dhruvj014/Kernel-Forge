#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace kf {

// Kernels use the CUDA-native element types (__half, __nv_bfloat16) rather
// than c10::Half / c10::BFloat16 so the same code can later feed WMMA
// fragments and intrinsics directly. All math happens in FP32 after loading.

// Bit pattern for -inf. MSVC's INFINITY macro is an overflowing double
// literal, which nvcc warns about.
__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T from_float(float x);

template <>
__device__ __forceinline__ float from_float<float>(float x) {
  return x;
}
template <>
__device__ __forceinline__ __half from_float<__half>(float x) {
  return __float2half_rn(x);
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
  return __float2bfloat16_rn(x);
}

}  // namespace kf

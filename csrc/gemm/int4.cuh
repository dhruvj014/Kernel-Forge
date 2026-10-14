#pragma once

// INT4 unpacking shared by the dequant GEMV and GEMM kernels. The packed
// layout is defined in kernelforge/quantize.py: one 32-bit word holds k0..k7
// as unsigned nibbles u = q + 8, ordered k0 k2 k4 k6 k1 k3 k5 k7 from bit 0.

#include <cuda_fp16.h>

#include <cstdint>

namespace kf::int4 {

// One word -> 4 half2 values (q[k0], q[k1]), (q[k2], q[k3]), ... exactly.
//
// (w >> 4i) & 0x000F000F leaves u(k 2i) in bits 0-3 and u(k 2i+1) in bits
// 16-19. OR-ing 0x6400 into each 16-bit half builds the fp16 number 1024 + u:
// 0x6400 is 1024.0, and in [1024, 2048) the fp16 mantissa step is exactly 1,
// so the low mantissa bits are u itself. Subtracting 1032 (= 1024 + 8) then
// gives q = u - 8 exactly. No integer-to-float conversion instructions.
__device__ __forceinline__ void unpack8(uint32_t w, __half2 (&out)[4]) {
  const __half2 bias = __halves2half2(__ushort_as_half(0x6408), __ushort_as_half(0x6408));  // 1032
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t v = ((w >> (4 * i)) & 0x000F000Fu) | 0x64006400u;
    out[i] = __hsub2(*reinterpret_cast<const __half2*>(&v), bias);
  }
}

}  // namespace kf::int4

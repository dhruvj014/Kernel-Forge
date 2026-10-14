#pragma once

// Inline-PTX wrappers for warp-level Tensor Core MMA (mma.sync m16n8k16) and
// ldmatrix, used by attention v4.
//
// Why not WMMA: WMMA fragments have an unspecified element layout, so a kernel
// cannot apply per-row softmax or rescaling to them in registers. mma.sync
// fragments have the layout documented in the PTX ISA ("Matrix Fragments for
// mma.m16n8k16"). With groupID g = lane / 4 and thread-in-group t = lane % 4:
//   A (16x16, row-major), 4 x 32-bit regs, each holding 2 consecutive 16-bit values:
//     a0: row g,   cols 2t, 2t+1        a1: row g+8, cols 2t, 2t+1
//     a2: row g,   cols 2t+8, 2t+9      a3: row g+8, cols 2t+8, 2t+9
//   B (16x8, "col" = each column's k values contiguous), 2 regs:
//     b0: k 2t, 2t+1 of column g        b1: k 2t+8, 2t+9 of column g
//   C/D (16x8, FP32), 4 regs:
//     c0, c1: row g, cols 2t, 2t+1      c2, c3: row g+8, cols 2t, 2t+1
// The C layout of two adjacent 16x8 tiles is exactly the A layout of one
// 16x16 tile, which is what lets P = softmax(S) feed the PV product straight
// from registers.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace kf::mma {

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Four 8x8 16-bit matrices from shared memory. Lanes 8i..8i+7 supply the row
// addresses of matrix i, and register i receives matrix i in the mma fragment
// layout (thread l holds row l/4, elements 2*(l%4) and 2*(l%4)+1).
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* row_ptr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_addr(row_ptr)));
}

// Same, transposing each 8x8 matrix on the way into registers. Used for V,
// which is stored [key][d] but must be the "col" B operand of P V.
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const void* row_ptr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_addr(row_ptr)));
}

template <typename T>
struct Mma;

template <>
struct Mma<__half> {
  // D = A * B + C, 16x8x16, fp16 inputs, FP32 accumulate.
  __device__ __forceinline__ static void run(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
  __device__ __forceinline__ static uint32_t pack(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
  }
};

template <>
struct Mma<__nv_bfloat16> {
  __device__ __forceinline__ static void run(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
  __device__ __forceinline__ static uint32_t pack(float lo, float hi) {
    __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
  }
};

}  // namespace kf::mma

#pragma once

// Shared-memory tile layout for attention v4: unpadded rows of D 16-bit
// elements with an XOR swizzle on 16-byte chunks.
//
// ldmatrix reads 8 rows per phase, 16 bytes from each. With D = 64 a row is
// 128 bytes, so without a swizzle those 8 reads (same column, consecutive
// rows) all start on the same bank and serialize 8 ways. Storing chunk c of
// row r at chunk position c ^ (r % 8) puts the 8 reads at 8 different chunk
// positions, which together cover all 32 banks. Unlike padding, the swizzle
// wastes no shared memory and keeps every chunk 16-byte aligned for cp.async.

#include "common/cp_async.cuh"
#include "common/cuda_utils.cuh"

namespace kf::attn_v4 {

// Element offset of (row, col) in a swizzled [rows][D] tile. `col` is a
// multiple of 8 wherever it is used (chunk granularity).
template <int D>
__device__ __forceinline__ int swz(int row, int col) {
  return row * D + (((col >> 3) ^ (row & 7)) << 3) + (col & 7);
}

// ROWS x D tile, global -> shared via cp.async, 16 bytes per copy, rows past N
// zero-filled. The caller commits and waits.
template <typename T, int D, int ROWS, int NT>
__device__ __forceinline__ void load_tile_async(T* __restrict__ dst, const T* __restrict__ src, int row0, int N) {
  constexpr int kChunksPerRow = D / 8;
  for (int i = threadIdx.x; i < ROWS * kChunksPerRow; i += NT) {
    const int rr = i / kChunksPerRow;
    const int cc = (i % kChunksPerRow) * 8;
    const bool ok = row0 + rr < N;
    kf::cp_async16(dst + swz<D>(rr, cc), ok ? src + size_t(row0 + rr) * D + cc : src, ok);
  }
}

}  // namespace kf::attn_v4

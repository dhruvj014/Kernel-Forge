#pragma once

// Thin wrappers over the sm_80+ asynchronous global -> shared copy (cp.async).
//
// cp.async moves data from global memory straight into shared memory without
// passing through registers, and the issuing thread does not wait for it. That
// lets a kernel start loading the next tile and keep computing on the current
// one. Completion is tracked in groups:
//   cp_async_commit()   closes the group of copies this thread issued so far
//   cp_async_wait<N>()  blocks until at most N of this thread's groups are
//                       still in flight
// wait only covers the calling thread's own copies, so a __syncthreads() is
// still needed afterwards before other threads may read the data.

#include <cuda_runtime.h>

namespace kf {

// 16-byte copy with zero fill. When `pred` is false the source size is 0, so
// nothing is read from global memory and the 16 destination bytes are zeroed.
// That is how ragged tiles (rows past N) are padded without a branch around
// the copy. `.cg` caches in L2 only, since each K/V element is used once per
// block from shared memory.
__device__ __forceinline__ void cp_async16(void* smem_dst, const void* gmem_src, bool pred) {
  const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
  const int src_bytes = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(gmem_src), "r"(src_bytes));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

}  // namespace kf

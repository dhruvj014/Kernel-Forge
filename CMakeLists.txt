cmake_minimum_required(VERSION 3.24)

# RTX 4060 (Ada Lovelace). Must be set before project(): CMake fills in the
# compiler's default architecture (sm_75 for CUDA 13) during language enable.
# Override with -DCMAKE_CUDA_ARCHITECTURES=... if needed.
set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to build for")

project(kernelforge LANGUAGES CXX CUDA)

# Standalone C++ microbenchmarks only. The PyTorch extension is built by
# setup.py (torch.utils.cpp_extension) so it links against the exact torch in
# the venv.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# -lineinfo maps SASS back to source lines in Nsight Compute.
# -Xptxas -v prints registers and shared memory per kernel at build time.
set(KF_CUDA_FLAGS -O3 -lineinfo -Xptxas=-v)

function(kf_micro name src)
  add_executable(${name} ${src})
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/csrc)
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${KF_CUDA_FLAGS}>)
endfunction()

kf_micro(kf_hello csrc/micro/hello.cu)
kf_micro(kf_bandwidth csrc/micro/bandwidth.cu)
kf_micro(kf_sanitize_attn csrc/micro/sanitize_attn.cu)

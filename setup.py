"""Builds the kernelforge._C torch extension.

Run through make.ps1 (Windows) or the Makefile (Linux), which set up the
compiler environment. Must be built with --no-build-isolation so it compiles
against the torch already installed in the venv.

Environment knobs:
  KF_FAST_MATH=1   add --use_fast_math (off by default so correctness tests
                   see IEEE exp/div behaviour)
"""

import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).parent.resolve()

# Only build for the GPU we actually have (Ada, sm_89). torch reads this to
# emit the -gencode flags, so we do not pass -arch ourselves.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

fast_math = os.environ.get("KF_FAST_MATH", "0") == "1"

sources = [
    "csrc/bindings.cpp",
    "csrc/attention/attn_naive.cu",
]

nvcc_flags = [
    "-O3",
    "-lineinfo",  # source-level attribution in Nsight Compute
    "-Xptxas=-v",  # print registers / smem / spills per kernel
    "-std=c++20",  # torch >= 2.10 headers use C++20 (designated initializers, <=>)
]
if fast_math:
    nvcc_flags.append("--use_fast_math")

if os.name == "nt":
    cxx_flags = ["/O2", "/std:c++20"]
else:
    cxx_flags = ["-O3", "-std=c++20"]

setup(
    name="kernelforge",
    version="0.1.0",
    packages=find_packages(include=["kernelforge", "kernelforge.*"]),
    ext_modules=[
        CUDAExtension(
            name="kernelforge._C",
            sources=sources,
            include_dirs=[str(ROOT / "csrc")],
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)

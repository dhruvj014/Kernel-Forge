#include <torch/extension.h>

torch::Tensor attn_naive(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal, double scale);

torch::Tensor attn_v1(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal, double scale, int64_t config);
int attn_v1_num_configs();
torch::Tensor attn_v2(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal, double scale, int64_t variant);
torch::Tensor attn_v4(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal, double scale, int64_t config);
torch::Tensor int4_gemv(torch::Tensor x, torch::Tensor packed, torch::Tensor scales, int64_t group);
torch::Tensor int4_gemm(torch::Tensor x, torch::Tensor packed, torch::Tensor scales, int64_t group);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "KernelForge hand-written CUDA kernels (low-level bindings, use kernelforge.ops)";
  m.def("attn_naive", &attn_naive, "Naive 3-kernel attention baseline: q, k, v [B, H, N, D]", py::arg("q"),
        py::arg("k"), py::arg("v"), py::arg("causal"), py::arg("scale"));
  m.def("attn_v1", &attn_v1, "Fused attention v1 (tiling + online softmax, FP32 CUDA cores): q, k, v [B, H, N, D]",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal"), py::arg("scale"), py::arg("config") = -1);
  m.def("attn_v1_num_configs", &attn_v1_num_configs);
  m.def("attn_v2", &attn_v2, "Fused attention v2 (WMMA Tensor Cores, fp16/bf16): q, k, v [B, H, N, D]", py::arg("q"),
        py::arg("k"), py::arg("v"), py::arg("causal"), py::arg("scale"), py::arg("variant") = 1);
  m.def("attn_v4", &attn_v4,
        "Fused attention v4 (mma.sync Tensor Cores, register softmax, fp16/bf16): q, k, v [B, H, N, D]", py::arg("q"),
        py::arg("k"), py::arg("v"), py::arg("causal"), py::arg("scale"), py::arg("config") = -1);
  m.def("int4_gemv", &int4_gemv, "Fused INT4 dequant GEMV, M in [1, 8]: y[M, N] = x[M, K] @ dequant(packed, scales)^T",
        py::arg("x"), py::arg("packed"), py::arg("scales"), py::arg("group"));
  m.def("int4_gemm", &int4_gemm,
        "Fused INT4 dequant GEMM on Tensor Cores (any M, N % 64 == 0): y[M, N] = x[M, K] @ dequant(packed, scales)^T",
        py::arg("x"), py::arg("packed"), py::arg("scales"), py::arg("group"));
}

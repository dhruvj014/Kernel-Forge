#include <torch/extension.h>

torch::Tensor attn_naive(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                         bool causal, double scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "KernelForge hand-written CUDA kernels (low-level bindings, use kernelforge.ops)";
  m.def("attn_naive", &attn_naive,
        "Naive 3-kernel attention baseline: q, k, v [B, H, N, D]",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal"),
        py::arg("scale"));
}

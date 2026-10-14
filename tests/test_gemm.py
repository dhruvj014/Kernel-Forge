"""Fused INT4 kernels vs "dequantize in PyTorch, then matmul" (FP32 reference).

The fused kernels compute exactly the same math as the reference, so they should
match up to accumulation order and the final rounding to fp16. Tolerance:
max |out - ref| <= 2e-3 * max |ref| (fp16 rounding alone is 2^-11 ~ 4.9e-4 relative).
Quantization error vs the original fp16 weights is a separate question,
covered in tests/test_quantize.py and reported by bench/bench_gemm.py.
"""

import pytest
import torch

from kernelforge import quantize as kq

ops = pytest.importorskip("kernelforge.ops")

REL_TOL = 2e-3

# (N, K) of every linear layer in TinyLlama 1.1B and Qwen2.5 1.5B (from their config.json).
LAYER_SHAPES = {
    "tinyllama_qo": (2048, 2048), "tinyllama_kv": (256, 2048), "tinyllama_gate_up": (5632, 2048),
    "tinyllama_down": (2048, 5632), "qwen_qo": (1536, 1536), "qwen_kv": (256, 1536),
    "qwen_gate_up": (8960, 1536), "qwen_down": (1536, 8960),
}


def make_case(m, n, k, group, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(n, k, generator=g, device="cuda") * 0.02
    x = torch.randn(m, k, generator=g, device="cuda").half()
    qw = kq.quantize_weight(w, group)
    ref = x.float() @ kq.dequantize(qw, torch.float32).T
    return x, qw, ref


def check(error_log, suite, fn, m, n, k, group, case):
    x, qw, ref = make_case(m, n, k, group, seed=m * 7 + n + k)
    out = fn(x, qw.packed, qw.scales, group)
    assert out.shape == (m, n) and out.dtype == torch.float16
    err = (out.float() - ref).abs().max().item()
    scale = ref.abs().max().item()
    passed = err <= REL_TOL * scale
    error_log.add(suite, case=case, dtype="float16", shape=[m, n, k], causal=False, group=group,
                  max_abs_err=err, max_rel_err=err / scale, atol=REL_TOL * scale, passed=passed)
    assert passed, f"{case}: max abs err {err:.3e} vs ref scale {scale:.3e}"


@pytest.mark.parametrize("group", [64, 128])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("layer", list(LAYER_SHAPES))
def test_gemv_layer_shapes(error_log, layer, m, group):
    n, k = LAYER_SHAPES[layer]
    check(error_log, "int4_gemv", ops.int4_gemv, m, n, k, group, layer)


@pytest.mark.parametrize("n", [1, 7, 9, 129])
def test_gemv_ragged_n(error_log, n):
    """N not a multiple of the 8 rows a block owns: the last warp's spare rows must not be written."""
    check(error_log, "int4_gemv", ops.int4_gemv, 3, n, 256, 128, f"ragged_n{n}")


def test_gemv_rejects_k_not_multiple_of_group():
    x = torch.randn(1, 200, device="cuda").half()
    packed = torch.zeros(16, 25, dtype=torch.int32, device="cuda")
    scales = torch.ones(16, 2, dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="not a multiple of group_size"):
        ops.int4_gemv(x, packed, scales, 128)


def test_gemv_rejects_large_m():
    x, qw, _ = make_case(9, 64, 256, 128)
    with pytest.raises(ValueError, match=r"M in \[1, 8\]"):
        ops.int4_gemv(x, qw.packed, qw.scales, 128)


def test_gemv_rejects_shape_mismatch():
    x, qw, _ = make_case(1, 64, 256, 128)
    with pytest.raises(ValueError, match="K/8"):
        ops.int4_gemv(x[:, :128].contiguous(), qw.packed, qw.scales, 128)


@pytest.mark.parametrize("group", [64, 128])
@pytest.mark.parametrize("m", [1, 8, 9, 16, 17, 32, 64, 65, 128])
@pytest.mark.parametrize("layer", list(LAYER_SHAPES))
def test_gemm_tc_layer_shapes(error_log, layer, m, group):
    """Tensor Core kernel at every M, including M not a multiple of the 64-row tile."""
    n, k = LAYER_SHAPES[layer]
    check(error_log, "int4_gemm", ops.int4_gemm, m, n, k, group, layer)


@pytest.mark.parametrize("m", [1, 8, 9, 128])
def test_matmul_dispatcher(error_log, m):
    check(error_log, "int4_matmul", ops.int4_matmul, m, 2048, 2048, 128, "dispatch")


def test_gemm_rejects_n_not_multiple_of_64():
    x, qw, _ = make_case(16, 100, 256, 128)
    with pytest.raises(ValueError, match="multiple of 64"):
        ops.int4_gemm(x, qw.packed, qw.scales, 128)

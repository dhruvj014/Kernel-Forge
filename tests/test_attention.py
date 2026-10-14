"""Correctness of KernelForge attention kernels against an FP32 PyTorch reference.

Tolerances are max absolute error on unit-normal inputs. They were tightened
from the brief's starting point (1e-2) after measuring the naive kernel. The
measured worst case per dtype is written to results/correctness_attn_naive.json.
"""

import math

import pytest
import torch

ops = pytest.importorskip("kernelforge.ops")

# Error budget is dominated by rounding the FP32 result to the output dtype
# (fp16 has a 10-bit mantissa, bf16 has 7), not by the kernel's FP32 math.
ATOL = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 1.6e-2}

DTYPES = [torch.float16, torch.bfloat16, torch.float32]
HEAD_DIMS = [64, 128]
# Includes N=1, values just below/at/above common tile sizes (16, 32, 64, 128)
# and lengths that are not a multiple of anything, to exercise ragged edges.
SEQ_LENS = [1, 7, 16, 31, 63, 64, 65, 127, 128, 129, 257, 1000]
B, H = 2, 3

IMPLS = {"attn_naive": ops.attn_naive, "attn_v1": ops.flash_attn_v1}


def reference_attention(q, k, v, causal: bool, scale: float) -> torch.Tensor:
    """Math-only FP32 attention: softmax(Q K^T * scale + mask) V."""
    qf, kf, vf = q.float(), k.float(), v.float()
    s = (qf @ kf.transpose(-1, -2)) * scale
    if causal:
        n = q.shape[-2]
        mask = torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1)
        s = s.masked_fill(mask, float("-inf"))
    return torch.softmax(s, dim=-1) @ vf


def make_qkv(n, d, dtype, seed=0, scale_q=1.0, scale_k=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape = (B, H, n, d)
    q = (torch.randn(shape, generator=g, device="cuda") * scale_q).to(dtype)
    k = (torch.randn(shape, generator=g, device="cuda") * scale_k).to(dtype)
    v = torch.randn(shape, generator=g, device="cuda").to(dtype)
    return q, k, v


def check(error_log, impl_name, case, q, k, v, causal, atol):
    scale = 1.0 / math.sqrt(q.shape[-1])
    out = IMPLS[impl_name](q, k, v, causal=causal)
    ref = reference_attention(q, k, v, causal, scale)
    assert out.shape == q.shape and out.dtype == q.dtype
    assert torch.isfinite(out).all(), "non-finite values in output"
    diff = (out.float() - ref).abs()
    max_abs = diff.max().item()
    max_rel = max_abs / max(ref.abs().max().item(), 1e-12)
    # Softmax turns an absolute score error e into a relative error ~e in P.
    # In FP32, a score of magnitude |s| carries rounding error ~|s| * 2^-24
    # (and the exp vs exp2 formulations round differently), so the output
    # error budget grows with max|s| * max|v|. Negligible for unit-scale
    # scores, it dominates the fp32 bound only for the |s| ~ 1000 case.
    s_max = ((q.float() @ k.float().transpose(-1, -2)).abs().max().item()) * scale
    atol = atol + 8 * 2.0**-24 * s_max * v.float().abs().max().item()
    passed = max_abs <= atol
    error_log.add(
        impl_name,
        case=case,
        dtype=str(q.dtype).replace("torch.", ""),
        shape=list(q.shape),
        causal=causal,
        max_abs_err=max_abs,
        max_rel_err=max_rel,
        atol=atol,
        passed=passed,
    )
    assert passed, f"max abs err {max_abs:.3e} > atol {atol:.1e} (rel {max_rel:.3e})"


@pytest.mark.parametrize("impl", list(IMPLS))
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("d", HEAD_DIMS)
@pytest.mark.parametrize("n", SEQ_LENS)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda t: str(t).replace("torch.", ""))
def test_random_grid(error_log, impl, dtype, n, d, causal):
    q, k, v = make_qkv(n, d, dtype, seed=n * 1000 + d)
    check(error_log, impl, "random", q, k, v, causal, ATOL[dtype])


@pytest.mark.parametrize("impl", list(IMPLS))
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda t: str(t).replace("torch.", ""))
def test_all_equal_scores(error_log, impl, dtype, causal):
    """Q = 0 makes every score equal, so each row of P is uniform."""
    q, k, v = make_qkv(129, 64, dtype, seed=1)
    q = torch.zeros_like(q)
    check(error_log, impl, "all_equal_scores", q, k, v, causal, ATOL[dtype])


@pytest.mark.parametrize("impl", list(IMPLS))
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda t: str(t).replace("torch.", ""))
def test_large_magnitude(error_log, impl, dtype, causal):
    """Scores of order +-1000: exp() overflows unless the row max is subtracted."""
    q, k, v = make_qkv(200, 64, dtype, seed=2, scale_q=20.0, scale_k=20.0)
    check(error_log, impl, "large_magnitude", q, k, v, causal, ATOL[dtype])


@pytest.mark.parametrize("impl", list(IMPLS))
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda t: str(t).replace("torch.", ""))
def test_large_constant_offset(error_log, impl, dtype):
    """All scores equal and large (512): naive exp(512) overflows FP32 without max subtraction."""
    q = torch.full((B, H, 96, 64), 8.0, device="cuda").to(dtype)
    k = torch.full((B, H, 96, 64), 8.0, device="cuda").to(dtype)
    v = torch.randn((B, H, 96, 64), device="cuda").to(dtype)
    check(error_log, impl, "large_constant_offset", q, k, v, False, ATOL[dtype])


@pytest.mark.parametrize("impl", list(IMPLS))
def test_custom_scale(error_log, impl):
    q, k, v = make_qkv(100, 64, torch.float16, seed=3)
    out = IMPLS[impl](q, k, v, causal=False, scale=0.05)
    ref = reference_attention(q, k, v, False, 0.05)
    assert (out.float() - ref).abs().max().item() <= ATOL[torch.float16]


@pytest.mark.parametrize("config", range(len(ops.ATTN_V1_CONFIGS)))
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("d", HEAD_DIMS)
@pytest.mark.parametrize("n", [1, 31, 33, 64, 127, 129, 300])
def test_v1_every_tile_config(error_log, config, causal, d, n):
    """Every (Br, Bc) config on lengths below, at and just past tile boundaries."""
    q, k, v = make_qkv(n, d, torch.float16, seed=n * 7 + d + config)
    scale = 1.0 / math.sqrt(d)
    out = ops.flash_attn_v1(q, k, v, causal=causal, config=config)
    ref = reference_attention(q, k, v, causal, scale)
    max_abs = (out.float() - ref).abs().max().item()
    error_log.add("attn_v1_configs", case=f"config{config}", dtype="float16", shape=list(q.shape),
                  causal=causal, max_abs_err=max_abs, max_rel_err=max_abs / ref.abs().max().item(),
                  atol=ATOL[torch.float16], passed=max_abs <= ATOL[torch.float16])
    assert max_abs <= ATOL[torch.float16], f"config {ops.ATTN_V1_CONFIGS[config]}: {max_abs:.3e}"


def test_v1_rejects_unsupported_head_dim():
    q = torch.randn(1, 2, 8, 96, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="head dim"):
        ops.flash_attn_v1(q, q, q)


def test_v1_rejects_bad_config():
    q = torch.randn(1, 2, 8, 64, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="config"):
        ops.flash_attn_v1(q, q, q, config=99)


class TestValidation:
    def test_shape_mismatch(self):
        q = torch.randn(1, 2, 8, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 2, 9, 64, device="cuda", dtype=torch.float16)
        with pytest.raises(ValueError, match="same shape"):
            ops.attn_naive(q, k, k)

    def test_cpu_tensor(self):
        q = torch.randn(1, 2, 8, 64, dtype=torch.float16)
        with pytest.raises(ValueError, match="CUDA"):
            ops.attn_naive(q, q, q)

    def test_bad_rank(self):
        q = torch.randn(2, 8, 64, device="cuda", dtype=torch.float16)
        with pytest.raises(ValueError, match=r"\[B, H, N, D\]"):
            ops.attn_naive(q, q, q)

    def test_non_contiguous(self):
        q = torch.randn(1, 8, 2, 64, device="cuda", dtype=torch.float16).transpose(1, 2)
        with pytest.raises(ValueError, match="contiguous"):
            ops.attn_naive(q, q, q)

    def test_bad_dtype(self):
        q = torch.ones(1, 2, 8, 64, device="cuda", dtype=torch.int32)
        with pytest.raises(TypeError, match="not supported"):
            ops.attn_naive(q, q, q)

    def test_mixed_dtype(self):
        q = torch.randn(1, 2, 8, 64, device="cuda", dtype=torch.float16)
        with pytest.raises(TypeError, match="share a dtype"):
            ops.attn_naive(q, q.to(torch.bfloat16), q)

    def test_bad_scale(self):
        q = torch.randn(1, 2, 8, 64, device="cuda", dtype=torch.float16)
        with pytest.raises(ValueError, match="positive"):
            ops.attn_naive(q, q, q, scale=-1.0)

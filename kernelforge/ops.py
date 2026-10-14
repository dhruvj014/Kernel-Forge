"""Python entry points for the KernelForge CUDA kernels.

Inference only: none of these ops define a backward pass.
"""

from __future__ import annotations

import math

import torch

try:
    from kernelforge import _C
except ImportError as exc:  # pragma: no cover - depends on local build state
    _C = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

SUPPORTED_ATTN_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _ext():
    if _C is None:
        raise ImportError(
            "kernelforge._C is not built. Run `.\\make.ps1 build` (Windows) or `make build`."
        ) from _IMPORT_ERROR
    return _C


def _check_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(t).__name__}")
        if not t.is_cuda:
            raise ValueError(f"{name} must be on a CUDA device, got {t.device}")
        if t.dim() != 4:
            raise ValueError(f"{name} must be [B, H, N, D], got shape {tuple(t.shape)}")
        if t.dtype not in SUPPORTED_ATTN_DTYPES:
            raise TypeError(f"{name} dtype {t.dtype} not supported, use one of {SUPPORTED_ATTN_DTYPES}")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous (call .contiguous() first)")
    if not (q.shape == k.shape == v.shape):
        raise ValueError(
            f"q, k, v must have the same shape, got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}"
        )
    if not (q.dtype == k.dtype == v.dtype):
        raise TypeError(f"q, k, v must share a dtype, got {q.dtype}, {k.dtype}, {v.dtype}")
    if not (q.device == k.device == v.device):
        raise ValueError("q, k, v must be on the same device")


def _resolve_scale(scale: float | None, head_dim: int) -> float:
    if scale is None:
        return 1.0 / math.sqrt(head_dim)
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    return float(scale)


def attn_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Naive 3-kernel attention (QK^T, softmax, PV) that materializes the N x N scores.

    q, k, v: contiguous [B, H, N, D] CUDA tensors (fp16, bf16 or fp32).
    Returns o with the same shape and dtype. Memory use grows as B*H*N^2*4 bytes.
    """
    _check_qkv(q, k, v)
    return _ext().attn_naive(q, k, v, bool(causal), _resolve_scale(scale, q.shape[-1]))


ATTN_V1_HEAD_DIMS = (64, 128)
# (Br, Bc) per config index, mirrors dispatch_config in csrc/attention/attn_v1_fp32.cu.
ATTN_V1_CONFIGS = ((32, 32), (32, 64), (64, 32), (64, 64), (128, 32))


def flash_attn_v1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: float | None = None,
    config: int = -1,
) -> torch.Tensor:
    """Fused attention v1: tiled, online softmax, FP32 math on CUDA cores. Never materializes N x N.

    q, k, v: contiguous [B, H, N, D] CUDA tensors (fp16, bf16 or fp32), D in {64, 128}.
    config: index into ATTN_V1_CONFIGS for the (Br, Bc) tile sweep, -1 for the tuned default.
    """
    _check_qkv(q, k, v)
    if q.shape[-1] not in ATTN_V1_HEAD_DIMS:
        raise ValueError(f"flash_attn_v1 supports head dim {ATTN_V1_HEAD_DIMS}, got {q.shape[-1]}")
    if not -1 <= config < len(ATTN_V1_CONFIGS):
        raise ValueError(f"config must be -1 or in [0, {len(ATTN_V1_CONFIGS)}), got {config}")
    return _ext().attn_v1(q, k, v, bool(causal), _resolve_scale(scale, q.shape[-1]), int(config))


TC_ATTN_DTYPES = (torch.float16, torch.bfloat16)


def _check_tc(name: str, q: torch.Tensor) -> None:
    if q.dtype not in TC_ATTN_DTYPES:
        raise TypeError(f"{name} runs on Tensor Cores and supports {TC_ATTN_DTYPES}, got {q.dtype}")
    if q.shape[-1] not in ATTN_V1_HEAD_DIMS:
        raise ValueError(f"{name} supports head dim {ATTN_V1_HEAD_DIMS}, got {q.shape[-1]}")


def flash_attn_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: float | None = None,
    conflict_free: bool = True,
) -> torch.Tensor:
    """Fused attention v2: both matmuls on Tensor Cores via WMMA, online softmax in FP32.

    q, k, v: contiguous [B, H, N, D] CUDA tensors, fp16 or bf16, D in {64, 128}.
    conflict_free=False selects the first version (bank-conflicting softmax layout), kept for the ablation.
    """
    _check_qkv(q, k, v)
    _check_tc("flash_attn_v2", q)
    return _ext().attn_v2(q, k, v, bool(causal), _resolve_scale(scale, q.shape[-1]), int(bool(conflict_free)))


def flash_attn_v3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Fused attention v3: v2 (WMMA, conflict-free layout) plus cp.async double-buffered K/V tiles.

    q, k, v: contiguous [B, H, N, D] CUDA tensors, fp16 or bf16, D in {64, 128}.
    """
    _check_qkv(q, k, v)
    _check_tc("flash_attn_v3", q)
    return _ext().attn_v2(q, k, v, bool(causal), _resolve_scale(scale, q.shape[-1]), 2)


# (warps per block, Bc) per config index, mirrors dispatch_d in csrc/attention/attn_v4_kernel.cuh.
ATTN_V4_CONFIGS = ((4, 32), (4, 64), (8, 32), (8, 64))


def flash_attn_v4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: float | None = None,
    config: int = -1,
) -> torch.Tensor:
    """Fused attention v4: mma.sync Tensor Cores, softmax and O accumulators in registers.

    q, k, v: contiguous [B, H, N, D] CUDA tensors, fp16 or bf16, D in {64, 128}.
    config: index into ATTN_V4_CONFIGS, -1 for the tuned default.
    """
    _check_qkv(q, k, v)
    _check_tc("flash_attn_v4", q)
    if not -1 <= config < len(ATTN_V4_CONFIGS):
        raise ValueError(f"config must be -1 or in [0, {len(ATTN_V4_CONFIGS)}), got {config}")
    return _ext().attn_v4(q, k, v, bool(causal), _resolve_scale(scale, q.shape[-1]), int(config))


def _check_int4(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, group_size: int) -> None:
    if not isinstance(x, torch.Tensor) or not x.is_cuda or x.dtype != torch.float16 or x.dim() != 2:
        raise TypeError("x must be a 2-D fp16 CUDA tensor [M, K]")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if packed.dtype != torch.int32 or scales.dtype != torch.float16:
        raise TypeError("packed must be int32 and scales fp16 (see kernelforge.quantize)")
    k = x.shape[1]
    if k % group_size != 0:
        raise ValueError(f"K={k} is not a multiple of group_size={group_size}")
    if packed.shape[1] * 8 != k:
        raise ValueError(f"packed has K/8={packed.shape[1]} columns but x has K={k}")


def int4_gemv(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Fused INT4 dequant GEMV for M in [1, 8]: x [M, K] fp16 @ W_hat^T -> [M, N] fp16."""
    _check_int4(x, packed, scales, group_size)
    if not 1 <= x.shape[0] <= 8:
        raise ValueError(f"int4_gemv handles M in [1, 8], got M={x.shape[0]}")
    return _ext().int4_gemv(x, packed, scales, int(group_size))


GEMV_MAX_M = 7  # measured crossover, results/gemm_dispatch.json (bench/sweep_int4_dispatch.py)


def int4_gemm(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Fused INT4 dequant GEMM on Tensor Cores (mma.sync), any M, N a multiple of 64."""
    _check_int4(x, packed, scales, group_size)
    if packed.shape[0] % 64 != 0:
        raise ValueError(f"int4_gemm needs N to be a multiple of 64, got N={packed.shape[0]}")
    return _ext().int4_gemm(x, packed, scales, int(group_size))


def int4_matmul(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """y = x @ W_hat^T for INT4 weights from kernelforge.quantize, choosing the kernel by M.

    M <= 7 (decode): the GEMV kernel, a pure weight stream with FP32 CUDA-core math.
    M >= 8: the Tensor Core GEMM. The GEMV's FP32 math grows linearly with M while the GEMM pads M up
    to a 16-row tile almost for free, so from M=8 the GEMM is cheaper in total over the layer shapes
    of both models (results/gemm_dispatch.json).
    """
    if x.shape[0] <= GEMV_MAX_M:
        return int4_gemv(x, packed, scales, group_size)
    return int4_gemm(x, packed, scales, group_size)


# ---------------------------------------------------------------------------
# Stable public API (Phase 5). Inference only: no autograd formulas are
# registered, so calling these on tensors that require grad raises in backward.
# ---------------------------------------------------------------------------


def _check_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> None:
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not isinstance(t, torch.Tensor) or not t.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if t.dim() != 4:
            raise ValueError(f"{name} must be 4-D [B, heads, N, D], got shape {tuple(t.shape)}")
        if t.dtype not in TC_ATTN_DTYPES:
            raise TypeError(f"{name} must be fp16 or bf16, got {t.dtype}")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous (call .contiguous() first)")
    if not (q.dtype == k.dtype == v.dtype):
        raise TypeError(f"q, k, v must share a dtype, got {q.dtype}, {k.dtype}, {v.dtype}")
    if k.shape != v.shape:
        raise ValueError(f"k and v must have the same shape, got {tuple(k.shape)} and {tuple(v.shape)}")
    (b, h, nq, d), (bk, hk, nk, dk) = q.shape, k.shape
    if b != bk or d != dk:
        raise ValueError(f"q {tuple(q.shape)} and k {tuple(k.shape)} must match in batch and head dim")
    if d not in ATTN_V1_HEAD_DIMS:
        raise ValueError(f"head dim must be one of {ATTN_V1_HEAD_DIMS}, got {d}")
    if h % hk != 0:
        raise ValueError(f"query heads ({h}) must be a multiple of key/value heads ({hk})")
    if causal and nq > nk:
        raise ValueError(f"causal attention needs Nq <= Nk (queries are the last Nq positions), got {nq} > {nk}")


@torch.library.custom_op("kernelforge::flash_attn", mutates_args=())
def _flash_attn_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, scale: float) -> torch.Tensor:
    return _ext().attn_v4(q, k, v, causal, scale, -1)


@_flash_attn_op.register_fake
def _(q, k, v, causal, scale):
    return torch.empty_like(q)


def flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Fused attention forward (the v4 mma.sync kernel), the stable entry point.

    q: [B, H, Nq, D], k and v: [B, Hk, Nk, D], fp16 or bf16, contiguous, D in {64, 128}.
    H must be a multiple of Hk (grouped-query attention, no K/V copies). Nq may be smaller
    than Nk (decode or chunked prefill against a KV cache): with causal=True, query i is
    treated as position Nk - Nq + i and sees keys 0 .. Nk - Nq + i.
    Returns [B, H, Nq, D] in the input dtype. Inference only (no backward).
    """
    _check_attn(q, k, v, causal)
    return _flash_attn_op(q, k, v, bool(causal), _resolve_scale(scale, q.shape[-1]))


@torch.library.custom_op("kernelforge::int4_matmul", mutates_args=())
def _int4_matmul_op(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, group_size: int) -> torch.Tensor:
    return int4_matmul(x, packed, scales, group_size)


@_int4_matmul_op.register_fake
def _(x, packed, scales, group_size):
    return x.new_empty(x.shape[0], packed.shape[0])


def int4_linear(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, group_size: int = 128,
                bias: torch.Tensor | None = None) -> torch.Tensor:
    """y = x @ W_hat^T (+ bias) for INT4 weights from kernelforge.quantize. x: [..., K] fp16.

    Leading dimensions are flattened into M, so [B, T, K] works like nn.Linear. The kernel is
    chosen by M (int4_matmul). Inference only (no backward).
    """
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    _check_int4(x2, packed, scales, group_size)
    y = _int4_matmul_op(x2, packed, scales, int(group_size))
    if bias is not None:
        y = y + bias
    return y.reshape(*lead, packed.shape[0])

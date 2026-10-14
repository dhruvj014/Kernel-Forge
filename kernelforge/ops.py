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

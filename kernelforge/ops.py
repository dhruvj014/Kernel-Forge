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

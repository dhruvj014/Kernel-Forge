"""Symmetric group-wise INT4 weight quantization and the packed layout the kernels read.

Weights follow nn.Linear: W is [N, K] (out_features x in_features) and y = x W^T.

Quantization (per output row n, per group of G consecutive k):
    scale[n, g] = max |W[n, gG:(g+1)G]| / 7          (stored as fp16)
    q[n, k]     = clamp(round(W[n, k] / scale), -8, 7)  (int4, two's complement range)
    W_hat       = q * scale
Symmetric with no zero point, so dequantization is one multiply. The scale maps
the group's absmax to +-7, so -8 is reachable only through rounding at the edge.

Packed layout: int32 [N, K / 8]. Word j of row n holds k = 8j .. 8j+7 as
unsigned nibbles u = q + 8 in [0, 15], in this bit order:

    bits  0-3: k0   4-7: k2   8-11: k4  12-15: k6
    bits 16-19: k1  20-23: k3  24-27: k5  28-31: k7

Why this order: in the kernel, (word >> 4i) & 0x000F000F leaves k(2i) in the
low 16 bits and k(2i+1) in the high 16 bits. OR-ing in 0x64006400 turns each
16-bit half into the fp16 value 1024 + u exactly, so one subtraction of 1032
gives a half2 of (q[2i], q[2i+1]), already paired the way consecutive fp16
activations are loaded. Unpacking 8 weights costs 4 shifts, 4 AND/OR pairs and
4 half2 subtractions, and a warp's 16-byte loads of a row stay contiguous.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

DEFAULT_GROUP_SIZE = 128
SUPPORTED_GROUP_SIZES = (64, 128)
_NIBBLE_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)  # k offset stored at nibble position i (bits 4i..4i+3)
_K_TO_POS = torch.tensor([_NIBBLE_ORDER.index(k) for k in range(8)])  # nibble position holding k offset k


@dataclass
class QuantizedWeight:
    packed: torch.Tensor  # int32 [N, K // 8]
    scales: torch.Tensor  # fp16 [N, K // group_size]
    group_size: int
    shape: tuple[int, int]  # (N, K) of the original weight

    @property
    def nbytes(self) -> int:
        return self.packed.numel() * 4 + self.scales.numel() * 2


def _check_shape(weight: torch.Tensor, group_size: int) -> tuple[int, int]:
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2-D [N, K], got shape {tuple(weight.shape)}")
    if group_size not in SUPPORTED_GROUP_SIZES:
        raise ValueError(f"group_size must be one of {SUPPORTED_GROUP_SIZES}, got {group_size}")
    n, k = weight.shape
    if k % group_size != 0:
        raise ValueError(f"K={k} is not a multiple of group_size={group_size}. Pad K or pick another group size.")
    return n, k


def quantize_int4(weight: torch.Tensor, group_size: int = DEFAULT_GROUP_SIZE) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (q, scales): q int8 [N, K] in [-8, 7], scales fp16 [N, K / group_size]."""
    n, k = _check_shape(weight, group_size)
    w = weight.float().reshape(n, k // group_size, group_size)
    absmax = w.abs().amax(dim=-1, keepdim=True)
    # The scale is rounded to fp16 before quantizing, so q is computed with the
    # exact scale the kernel will use. All-zero groups get scale 1 (q = 0).
    scale = (absmax / 7.0).clamp_min(torch.finfo(torch.float16).tiny).to(torch.float16).float()
    scale = torch.where(absmax == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(w / scale), -8, 7).to(torch.int8).reshape(n, k)
    return q, scale.reshape(n, k // group_size).to(torch.float16)


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """int8 [N, K] in [-8, 7] -> int32 [N, K / 8] in the kernel layout (see module docstring)."""
    if q.dim() != 2 or q.shape[1] % 8 != 0:
        raise ValueError(f"q must be [N, K] with K a multiple of 8, got {tuple(q.shape)}")
    if q.min() < -8 or q.max() > 7:
        raise ValueError("q values must lie in [-8, 7]")
    n, k = q.shape
    u = (q.to(torch.int32) + 8).reshape(n, k // 8, 8)
    word = torch.zeros(n, k // 8, dtype=torch.int32, device=q.device)
    for pos, koff in enumerate(_NIBBLE_ORDER):
        word |= u[..., koff] << (4 * pos)
    return word


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of pack_int4: int32 [N, K / 8] -> int8 [N, K] in [-8, 7]."""
    n, kw = packed.shape
    # One broadcast shift extracts all 8 nibbles (positions 0..7), then a gather
    # puts them back in k order. Vectorized so the "dequantize in PyTorch, then
    # matmul" baseline is a few kernels, not a Python loop of 8.
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device)
    # The arithmetic right shift sign-extends bit 31, so mask after shifting.
    nib = ((packed.unsqueeze(-1) >> shifts) & 0xF).to(torch.int8) - 8  # [n, kw, position]
    return nib[..., _K_TO_POS.to(packed.device)].reshape(n, kw * 8)


def quantize_weight(weight: torch.Tensor, group_size: int = DEFAULT_GROUP_SIZE) -> QuantizedWeight:
    """Quantize and pack an nn.Linear weight [N, K] for the KernelForge INT4 kernels."""
    n, k = _check_shape(weight, group_size)
    q, scales = quantize_int4(weight, group_size)
    return QuantizedWeight(packed=pack_int4(q).contiguous(), scales=scales.contiguous(),
                           group_size=group_size, shape=(n, k))


def dequantize(qw: QuantizedWeight, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Reference dequantization in PyTorch: W_hat = q * scale, [N, K]."""
    n, k = qw.shape
    q = unpack_int4(qw.packed).reshape(n, k // qw.group_size, qw.group_size)
    if dtype == torch.float16:  # |q| <= 8 and q*scale are exact enough in fp16, skip the FP32 round trip
        return (q.half() * qw.scales.unsqueeze(-1)).reshape(n, k)
    return (q.float() * qw.scales.float().unsqueeze(-1)).reshape(n, k).to(dtype)


def quantization_error(weight: torch.Tensor, qw: QuantizedWeight) -> dict:
    """Error of the INT4 approximation vs the original weights (independent of any kernel)."""
    w = weight.float()
    err = (dequantize(qw, torch.float32) - w)
    return {
        "max_abs_err": err.abs().max().item(),
        "rmse": err.pow(2).mean().sqrt().item(),
        "rel_fro_err": (err.norm() / w.norm().clamp_min(1e-30)).item(),
        "bits_per_weight": 8.0 * qw.nbytes / w.numel(),
    }

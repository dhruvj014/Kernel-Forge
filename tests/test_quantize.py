"""INT4 quantizer and packed layout (kernelforge.quantize). Pure PyTorch, no kernels."""

import pytest
import torch

from kernelforge import quantize as kq


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pack_unpack_roundtrip_is_exact(device):
    g = torch.Generator(device=device).manual_seed(0)
    q = torch.randint(-8, 8, (64, 256), generator=g, device=device, dtype=torch.int8)
    assert torch.equal(kq.unpack_int4(kq.pack_int4(q)), q)


def test_bit_layout_matches_docstring():
    """k0..k7 = 0..7 (stored as q+8 = 8..15) must land at nibbles in order k0,k2,k4,k6,k1,k3,k5,k7."""
    q = torch.arange(0, 8, dtype=torch.int8).reshape(1, 8)
    word = kq.pack_int4(q).item() & 0xFFFFFFFF
    nibbles = [(word >> (4 * i)) & 0xF for i in range(8)]
    assert nibbles == [8, 10, 12, 14, 9, 11, 13, 15]
    # The kernel's unpack: (word >> 4i) & 0x000F000F holds (k 2i, k 2i+1) in its two 16-bit halves.
    for i in range(4):
        pair = (word >> (4 * i)) & 0x000F000F
        assert (pair & 0xF, pair >> 16) == (2 * i + 8, 2 * i + 1 + 8)


@pytest.mark.parametrize("group_size", [64, 128])
def test_quantization_error_is_bounded(group_size):
    g = torch.Generator().manual_seed(1)
    w = torch.randn(256, 1024, generator=g) * 0.02
    qw = kq.quantize_weight(w, group_size)
    w_hat = kq.dequantize(qw, torch.float32)
    # Round-to-nearest: each element is off by at most half a quantization step.
    step = qw.scales.float().repeat_interleave(group_size, dim=1)
    assert ((w_hat - w).abs() <= 0.5 * step + 1e-8).all()
    stats = kq.quantization_error(w, qw)
    assert stats["rel_fro_err"] < 0.15
    # 4 bits per weight plus one fp16 scale per group.
    assert stats["bits_per_weight"] == pytest.approx(4 + 16 / group_size)


def test_absmax_maps_to_seven_and_zero_groups_are_safe():
    w = torch.zeros(2, 128)
    w[0, 5] = -3.5
    w[0, 9] = 1.0
    q, s = kq.quantize_int4(w, 128)
    assert q[0, 5].item() == -7 and s[0, 0].item() == pytest.approx(0.5, rel=1e-3)
    assert (q[1] == 0).all() and s[1, 0].item() == 1.0  # all-zero group: scale 1, no division by zero


def test_rejects_k_not_multiple_of_group():
    with pytest.raises(ValueError, match="not a multiple of group_size"):
        kq.quantize_weight(torch.randn(16, 200), 128)


def test_rejects_bad_group_size_and_rank():
    with pytest.raises(ValueError, match="group_size"):
        kq.quantize_weight(torch.randn(16, 256), 32)
    with pytest.raises(ValueError, match="2-D"):
        kq.quantize_weight(torch.randn(4, 16, 256), 128)


def test_pack_rejects_out_of_range():
    with pytest.raises(ValueError, match=r"\[-8, 7\]"):
        kq.pack_int4(torch.full((1, 8), 9, dtype=torch.int8))

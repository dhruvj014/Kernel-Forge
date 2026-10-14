"""Drop-in modules that put the KernelForge kernels inside Hugging Face models.

- KFLinearInt4: nn.Linear replacement holding group-wise INT4 weights, forward via ops.int4_linear.
- quantize_linears_: swap a model's decoder linear layers (q, k, v, o, gate, up, down) for KFLinearInt4.
- enable_kf_attention: register the fused attention kernel (ops.flash_attn) with transformers'
  attention interface under the name "kernelforge" and switch a model to it.

Inference only. Nothing here defines a backward pass.
"""

from __future__ import annotations

import torch
from torch import nn

from kernelforge import ops
from kernelforge import quantize as kq

DECODER_LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class KFLinearInt4(nn.Module):
    """INT4 weight-only nn.Linear: y = x W_hat^T + b with W_hat dequantized inside the kernel.

    Weights are stored packed (int32 [N, K/8]) with fp16 scales [N, K/G], 4 + 16/G bits per weight.
    Activations must be fp16.
    """

    def __init__(self, qw: kq.QuantizedWeight, bias: torch.Tensor | None):
        super().__init__()
        self.out_features, self.in_features = qw.shape
        self.group_size = qw.group_size
        self.register_buffer("packed", qw.packed)
        self.register_buffer("scales", qw.scales)
        self.register_buffer("bias", None if bias is None else bias.detach().half())

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = kq.DEFAULT_GROUP_SIZE) -> KFLinearInt4:
        qw = kq.quantize_weight(linear.weight.detach(), group_size)
        return cls(qw, linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float16:
            raise TypeError(f"KFLinearInt4 takes fp16 activations, got {x.dtype}")
        return ops.int4_linear(x, self.packed, self.scales, self.group_size, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, group_size={self.group_size}, int4"


@torch.no_grad()
def quantize_linears_(model: nn.Module, group_size: int = kq.DEFAULT_GROUP_SIZE,
                      names: tuple[str, ...] = DECODER_LINEARS) -> int:
    """Replace every nn.Linear whose attribute name is in `names` with KFLinearInt4, in place.

    The embedding and lm_head stay fp16 (common practice: they are not decoder matmuls, and
    lm_head quantization hurts quality most). Returns the number of layers replaced.
    """
    targets = [(parent, name, child) for parent in model.modules() for name, child in parent.named_children()
               if name in names and isinstance(child, nn.Linear)]
    for parent, name, child in targets:
        setattr(parent, name, KFLinearInt4.from_linear(child, group_size))
        del child
    torch.cuda.empty_cache()
    return len(targets)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

KF_ATTN_NAME = "kernelforge"
ATTN_STATS = {"kernelforge_calls": 0, "sdpa_fallback_calls": 0}


def _kf_attention_forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None,
                          is_causal=None, **kwargs):
    """transformers attention function: query [B, H, Nq, D], key/value [B, Hk, Nk, D] (KV heads not repeated).

    Uses ops.flash_attn when it applies: no padding mask, no dropout, head dim 64/128, fp16/bf16.
    The causal mask is bottom-right aligned, which is exactly the KV-cache convention, so the same
    call serves prefill (Nq = Nk) and decode (Nq = 1). Anything else falls back to transformers'
    SDPA path, and every fallback is counted in ATTN_STATS so results can't hide it.
    """
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    supported = (attention_mask is None and dropout == 0.0 and query.shape[-1] in ops.ATTN_V1_HEAD_DIMS
                 and query.dtype in ops.TC_ATTN_DTYPES and not kwargs.get("output_attentions", False))
    if not supported:
        ATTN_STATS["sdpa_fallback_calls"] += 1
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout,
                                      scaling=scaling, is_causal=is_causal, **kwargs)
    ATTN_STATS["kernelforge_calls"] += 1
    out = ops.flash_attn(query.contiguous(), key.contiguous(), value.contiguous(), causal=causal, scale=scaling)
    return out.transpose(1, 2).contiguous(), None  # [B, Nq, H, D], same as sdpa_attention_forward


def register_kf_attention() -> str:
    """Registers the "kernelforge" attention function and its mask builder with transformers."""
    from transformers import AttentionInterface
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

    AttentionInterface.register(KF_ATTN_NAME, _kf_attention_forward)
    # The flash-attention mask builder returns None when there is no padding (pure causal), which
    # is what lets the fused kernel run. With padding it returns a 2-D mask and we fall back.
    ALL_MASK_ATTENTION_FUNCTIONS.register(KF_ATTN_NAME, ALL_MASK_ATTENTION_FUNCTIONS["flash_attention_2"])
    return KF_ATTN_NAME


def enable_kf_attention(model: nn.Module) -> nn.Module:
    """Switch a loaded transformers model to the KernelForge attention kernel."""
    register_kf_attention()
    model.set_attn_implementation(KF_ATTN_NAME)
    return model

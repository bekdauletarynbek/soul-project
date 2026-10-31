"""Fused residual add for Qwen2 decoder layers.

Every decoder layer does `hidden_states = residual + hidden_states` twice (once after
attention, once after the MLP) -- 56 calls per forward pass on Qwen2.5-1.5B's 28 layers, each
over the full [batch, seq, hidden] tensor. The gradient of a sum is the identity (both inputs
get the incoming gradient unchanged), so backward needs no kernel at all -- only forward
launches one.

There is no narrower hook for this than `Qwen2DecoderLayer.forward` itself: unlike RMSNorm,
RoPE or the MLP's activation, the add is inline in that method, not a separate function or
class method. The patch below is the real method body from transformers v5.16.1's
modeling_qwen2.py (checked against the actual source, not reconstructed from memory) with the
two `residual + hidden_states` lines replaced by the fused kernel; everything else -- the calls
into self.input_layernorm / self.self_attn / self.mlp, which stay under whatever other patches
are active on them -- is unchanged. Verified that `GradientCheckpointingLayer.__call__`
(the base class) falls through to a plain `super().__call__(...)` -- i.e. normal polymorphic
`self.forward(...)` dispatch -- whenever gradient checkpointing is off, which is this task's
baseline (`no gradient checkpointing` per the README), so the monkeypatched class method is
actually what runs.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024


@triton.jit
def _add_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, a + b, mask=mask)


class _ResidualAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        a, b = a.contiguous(), b.contiguous()
        out = torch.empty_like(a)
        n = a.numel()
        _add_kernel[(triton.cdiv(n, BLOCK),)](a, b, out, n, BLOCK=BLOCK)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, grad_out


def residual_add(a, b):
    return _ResidualAdd.apply(a, b)


def patch_qwen2_decoder_layer():
    from transformers.models.qwen2 import modeling_qwen2

    original_forward = modeling_qwen2.Qwen2DecoderLayer.forward

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
               past_key_values=None, use_cache=False, position_embeddings=None, **kwargs):
        if not hidden_states.is_cuda:
            return original_forward(self, hidden_states, attention_mask, position_ids,
                                    past_key_values, use_cache, position_embeddings, **kwargs)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual_add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual_add(residual, hidden_states)
        return hidden_states

    modeling_qwen2.Qwen2DecoderLayer.forward = forward

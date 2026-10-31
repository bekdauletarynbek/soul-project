"""Fused RoPE (rotary position embedding) application for Qwen2, forward and backward in
Triton. Applied to both q and k, once per attention layer (28 layers on Qwen2.5-1.5B) --
28 forward calls + 28 backward calls per training step, each one on the full [batch, heads,
seq, head_dim] q and k tensors.

The reference (`transformers.models.qwen2.modeling_qwen2.apply_rotary_pos_emb`, verified
against the real v5.16.1 source, not from memory):

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

`rotate_half` alone allocates a new tensor via `torch.cat`, then there are 2 multiplies and
1 add per tensor (q and k) -- this fuses all of it into one kernel launch per tensor each
direction, with no `cat` intermediate.

Per half-dim pair (q1, q2) this is exactly a 2D rotation by the per-position angle encoded in
(cos, sin): [out1; out2] = [[c, -s], [s, c]] @ [q1; q2]. Rotation matrices are orthogonal, so
the backward is the inverse rotation (R^T = R(-angle)) applied to the incoming gradient:
dx1 = g1*c + g2*s, dx2 = -g1*s + g2*c. This needs only cos/sin saved from forward, not q/k
themselves -- cheaper than saving the inputs.

Not a `@use_kernel_forward_from_hub` concern: that decorator (and `@use_kernelized_func` on
Qwen2Attention) is a no-op identity decorator unless the separate `kernels` PyPI package is
installed (verified in `transformers.integrations.hub_kernels`, v5.16.1), which it is not in
this task's pinned dependencies -- so patching the module-level `apply_rotary_pos_emb` works
exactly like patching a plain function.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_fwd_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr,
                     stride_xb, stride_xh, stride_xs,
                     stride_cb, stride_cs,
                     n_heads, seq_len, half_dim,
                     BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    s = pid % seq_len
    rest = pid // seq_len
    h = rest % n_heads
    b = rest // n_heads

    cols = tl.arange(0, BLOCK)
    mask = cols < half_dim

    x_row = x_ptr + b * stride_xb + h * stride_xh + s * stride_xs
    x1 = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_row + half_dim + cols, mask=mask, other=0.0).to(tl.float32)

    cs_row = cos_ptr + b * stride_cb + s * stride_cs
    sn_row = sin_ptr + b * stride_cb + s * stride_cs
    c = tl.load(cs_row + cols, mask=mask, other=0.0).to(tl.float32)
    sn = tl.load(sn_row + cols, mask=mask, other=0.0).to(tl.float32)

    out1 = x1 * c - x2 * sn
    out2 = x2 * c + x1 * sn

    out_row = out_ptr + b * stride_xb + h * stride_xh + s * stride_xs
    tl.store(out_row + cols, out1.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(out_row + half_dim + cols, out2.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rope_bwd_kernel(dout_ptr, cos_ptr, sin_ptr, dx_ptr,
                     stride_xb, stride_xh, stride_xs,
                     stride_cb, stride_cs,
                     n_heads, seq_len, half_dim,
                     BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    s = pid % seq_len
    rest = pid // seq_len
    h = rest % n_heads
    b = rest // n_heads

    cols = tl.arange(0, BLOCK)
    mask = cols < half_dim

    dout_row = dout_ptr + b * stride_xb + h * stride_xh + s * stride_xs
    g1 = tl.load(dout_row + cols, mask=mask, other=0.0).to(tl.float32)
    g2 = tl.load(dout_row + half_dim + cols, mask=mask, other=0.0).to(tl.float32)

    cs_row = cos_ptr + b * stride_cb + s * stride_cs
    sn_row = sin_ptr + b * stride_cb + s * stride_cs
    c = tl.load(cs_row + cols, mask=mask, other=0.0).to(tl.float32)
    sn = tl.load(sn_row + cols, mask=mask, other=0.0).to(tl.float32)

    dx1 = g1 * c + g2 * sn
    dx2 = -g1 * sn + g2 * c

    dx_row = dx_ptr + b * stride_xb + h * stride_xh + s * stride_xs
    tl.store(dx_row + cols, dx1.to(dx_ptr.dtype.element_ty), mask=mask)
    tl.store(dx_row + half_dim + cols, dx2.to(dx_ptr.dtype.element_ty), mask=mask)


def _cos_batch_stride(cos, batch):
    """cos/sin come in at `position_ids.shape[0]`, which `Qwen2Model.forward` sets to the real
    batch size only if the caller passes `position_ids` explicitly. TRL's SFTTrainer (unpacked
    mode, this task's setup) does not, so `Qwen2Model.forward` falls back to
    `torch.arange(seq_len).unsqueeze(0)` -- batch 1 -- and relies on ordinary PyTorch
    broadcasting (`q * cos.unsqueeze(1)`) to stretch it over the real batch. A stride of 0 does
    the same thing by hand: `b * 0 == 0` for every b, so every program reads row 0 instead of
    indexing past the single row actually allocated. Without this, forward silently read
    out-of-bounds memory for every batch index beyond 0 (NaN from the very first step, not a
    gradual drift -- caught by @ML bisecting on a real T4 run after RMSNorm+CE+RoPE alone
    still produced NaN)."""
    return cos.stride(0) if cos.shape[0] == batch else 0


def _launch_fwd(x, cos, sin):
    b, h, s, d = x.shape
    half = d // 2
    block = triton.next_power_of_2(half)
    out = torch.empty_like(x)
    stride_cb = _cos_batch_stride(cos, b)
    _rope_fwd_kernel[(b * h * s,)](
        x, cos, sin, out,
        x.stride(0), x.stride(1), x.stride(2),
        stride_cb, cos.stride(1),
        h, s, half, BLOCK=block,
    )
    return out


def _launch_bwd(dout, cos, sin, shape):
    b, h, s, d = shape
    half = d // 2
    block = triton.next_power_of_2(half)
    dx = torch.empty(shape, device=dout.device, dtype=dout.dtype)
    stride_cb = _cos_batch_stride(cos, b)
    _rope_bwd_kernel[(b * h * s,)](
        dout, cos, sin, dx,
        dx.stride(0), dx.stride(1), dx.stride(2),
        stride_cb, cos.stride(1),
        h, s, half, BLOCK=block,
    )
    return dx


class _RotaryEmbed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        q, k = q.contiguous(), k.contiguous()
        cos, sin = cos.contiguous(), sin.contiguous()
        q_out = _launch_fwd(q, cos, sin)
        k_out = _launch_fwd(k, cos, sin)
        ctx.save_for_backward(cos, sin)
        ctx.q_shape, ctx.k_shape = q.shape, k.shape
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq_out, dk_out):
        cos, sin = ctx.saved_tensors
        dq = _launch_bwd(dq_out.contiguous(), cos, sin, ctx.q_shape)
        dk = _launch_bwd(dk_out.contiguous(), cos, sin, ctx.k_shape)
        return dq, dk, None, None


def rotary_embed(q, k, cos, sin):
    return _RotaryEmbed.apply(q, k, cos, sin)


def patch_qwen2_rope():
    """Replace the module-level `apply_rotary_pos_emb` that Qwen2Attention.forward calls by
    unqualified name. Only handles the actual call shape used in this file (q, k as
    [batch, heads, seq, head_dim], cos/sin as [batch, seq, head_dim], unsqueeze_dim=1);
    anything else falls back to the original so it cannot silently miscompute."""
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.apply_rotary_pos_emb

    def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        if not q.is_cuda or q.dim() != 4 or cos.dim() != 3 or unsqueeze_dim != 1:
            return original(q, k, cos, sin, unsqueeze_dim)
        return rotary_embed(q, k, cos, sin)

    modeling_qwen2.apply_rotary_pos_emb = apply_rotary_pos_emb

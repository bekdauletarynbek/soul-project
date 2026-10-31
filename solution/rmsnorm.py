"""Fused RMSNorm for Qwen2 (Qwen2RMSNorm), forward and backward in Triton.

Qwen2's RMSNorm runs once per decoder layer input, once per decoder layer post-attention, and
once at the end of the model (28 layers on Qwen2.5-1.5B -> 57 calls per forward pass, each one
again in the backward pass). The reference implementation (identical across
Qwen2/Llama/Mistral in transformers) is five separate PyTorch ops plus the fp16<->fp32 casts
around them:

    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)

Each op reads and writes a full [*, hidden] tensor; PyTorch allocates an intermediate for
every one of them. This fuses the whole row (mean of squares, rsqrt, two multiplies, the
casts) into one kernel launch per direction, with no extra intermediate tensors, which is
exactly the "normalizations in fp32" + "dtype casts" share of the profile in the brief
(about 44% of a training step together with activations and LoRA). The norm weight in this
setup is part of the frozen base model (`prepare_model_for_kbit_training` keeps it in fp32,
but QLoRA does not train it) -- backward skips the weight-gradient kernel entirely when
`weight.requires_grad` is False, which is the common case here and saves the one part of the
backward that needs atomics across every row.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd_kernel(x_ptr, weight_ptr, out_ptr, rstd_ptr, n_cols, eps,
                        BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(rstd_ptr + row, rstd)

    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    xhat = x * rstd
    out = xhat * w
    tl.store(out_ptr + row * n_cols + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rmsnorm_bwd_kernel(dy_ptr, x_ptr, weight_ptr, rstd_ptr, dx_ptr, dw_ptr, n_cols,
                        COMPUTE_DW: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)

    xhat = x * rstd
    g = dy * w
    # d(xhat_i)/d(x_k) = rstd * (delta_ik - xhat_i * xhat_k / n_cols) -- standard RMSNorm
    # backward (same identity used by Llama/Unsloth-style fused RMSNorm kernels). dx does not
    # go through the `.to(input_dtype)` cast below (a dtype cast's backward is a lossless
    # pass-through in autograd), so it correctly uses the full-precision xhat.
    mean_g_xhat = tl.sum(g * xhat, axis=0) / n_cols
    dx = rstd * (g - xhat * mean_g_xhat)
    tl.store(dx_ptr + row * n_cols + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)

    if COMPUTE_DW:
        # The reference rounds xhat to the input dtype BEFORE multiplying by weight
        # (`weight * hidden_states.to(input_dtype)`), so dweight = dy * xhat is only correct
        # per the chain rule if xhat is rounded the same way first -- using the full-precision
        # xhat here (as an earlier version did) silently disagreed with the real autograd graph
        # whenever the compute dtype loses precision relative to fp32 (fp16: ~0.1% per value,
        # enough to show up as a multi-percent mismatch after summing over every row).
        xhat_rounded = xhat.to(x_ptr.dtype.element_ty).to(tl.float32)
        tl.atomic_add(dw_ptr + cols, dy * xhat_rounded, mask=mask)


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        orig_shape = x.shape
        x2d = x.contiguous().view(-1, orig_shape[-1])
        n_rows, n_cols = x2d.shape
        out = torch.empty_like(x2d)
        rstd = torch.empty(n_rows, device=x.device, dtype=torch.float32)
        block = triton.next_power_of_2(n_cols)
        _rmsnorm_fwd_kernel[(n_rows,)](x2d, weight, out, rstd, n_cols, eps, BLOCK_SIZE=block)
        ctx.save_for_backward(x2d, weight, rstd)
        ctx.orig_shape, ctx.block, ctx.eps = orig_shape, block, eps
        return out.view(orig_shape)

    @staticmethod
    def backward(ctx, grad_out):
        x2d, weight, rstd = ctx.saved_tensors
        n_rows, n_cols = x2d.shape
        dy = grad_out.contiguous().view(n_rows, n_cols)
        dx = torch.empty_like(x2d)
        compute_dw = ctx.needs_input_grad[1]
        dw = torch.zeros(n_cols, device=x2d.device, dtype=torch.float32) if compute_dw else dx
        _rmsnorm_bwd_kernel[(n_rows,)](dy, x2d, weight, rstd, dx, dw, n_cols,
                                       COMPUTE_DW=compute_dw, BLOCK_SIZE=ctx.block)
        dweight = dw.to(weight.dtype) if compute_dw else None
        return dx.view(ctx.orig_shape), dweight, None


def rmsnorm(x, weight, eps):
    return _RMSNorm.apply(x, weight, eps)


def patch_qwen2_rmsnorm():
    """Send Qwen2RMSNorm.forward through the fused kernel. Patches the class, so every
    instance (input_layernorm, post_attention_layernorm, the final model.norm) goes through
    it -- 57 calls per forward pass on Qwen2.5-1.5B's 28 layers."""
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2RMSNorm.forward

    def forward(self, hidden_states):
        if not hidden_states.is_cuda:
            return original(self, hidden_states)
        return rmsnorm(hidden_states, self.weight, self.variance_epsilon)

    modeling_qwen2.Qwen2RMSNorm.forward = forward

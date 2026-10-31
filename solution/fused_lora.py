"""Fused LoRA projection (the A then B matmul pair peft's vanilla LoRA does on q_proj and v_proj,
28 layers each) for a 4-bit-quantized base layer, forward and the grad-x part of backward in
Triton.

`peft.tuners.lora.bnb.Linear4bit.forward` (verified against the real peft==0.20.0 source, not
memory -- the vanilla-LoRA branch, no DoRA/variant, which is this task's config) does, per active
adapter:

    output = lora_B(lora_A(dropout(x))) * scaling
    result = result + output

two separate `nn.Linear` calls (each its own cuBLAS GEMM + kernel launch) plus a scale, with an
(M, r) intermediate (`lora_A(...)`) going to global memory and back for no reason other than that
they are two different `nn.Module`s. Since r=16 is tiny next to in_features (1536) and out_features
(1536 for q_proj, 256 for v_proj here), one Triton program per input row computes both matmuls: a
loop over in_features building the (r,) intermediate `h`, then a loop over out_features turning `h`
into the row's output.

Backward: `grad_h = (grad_out @ B) * scaling` and the weight gradients `grad_A = grad_h.T @ x`,
`grad_B = (grad_out * scaling).T @ h` are left as plain `torch.matmul` -- they reduce over the
whole batch into A/B's weight gradients (shared across every row), which in Triton means atomic
adds into a small buffer; not worth the extra correctness risk here for a result this small
((r, in_features) and (out_features, r), r=16). What IS fused in Triton is `grad_x` (the gradient
that must keep flowing back through the frozen base model to reach every earlier layer's own LoRA
adapters): per row, independent of every other row, no atomics -- the same two-phase-per-row shape
as the forward kernel, just A and the (already-computed) `grad_h` instead of A and x.

`dropout(x)` stays a plain `torch.nn.Dropout` call before this, untouched -- its own RNG draws, so
the loss curve does not move relative to the base run (same seed, same draws).

Hooked in by replacing `peft.tuners.lora.bnb.Linear4bit.forward` (a class method, same technique as
`solution/swiglu.py`): the surrounding branches (disabled adapters, merged weights, mixed-batch
inference, DoRA/other variants) are copied from the real source unchanged and simply call through
to the saved original; only the vanilla-LoRA line is replaced.

Not run on a GPU here -- written against the real peft==0.20.0 source (downloaded for this, see
WORK_LOGS) for the exact attribute names, scaling formula and control flow;
`solution/test_fused_lora.py` checks the kernel against plain PyTorch, but the Triton launches
themselves need a real CUDA run (Kaggle T4) before this is fully trusted.
"""

import torch
import triton
import triton.language as tl

_ORIGINAL_LINEAR4BIT_FORWARD = None  # filled in by patch_lora(), used for every branch but vanilla


@triton.jit
def _lora_fwd_kernel(x_ptr, a_ptr, b_ptr, h_ptr, out_ptr, scaling, in_features, out_features, r,
                     stride_xm, stride_xk, stride_ar, stride_ak, stride_bo, stride_br,
                     stride_hm, stride_om, stride_oo,
                     BLOCK_K: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_O: tl.constexpr):
    """One program per row of x: h = x @ A.T (phase 1, also stored for backward's plain-torch
    grad_A/grad_B), out = h @ B.T * scaling (phase 2)."""
    m = tl.program_id(0)
    r_idx = tl.arange(0, BLOCK_R)
    r_mask = r_idx < r

    h = tl.zeros((BLOCK_R,), dtype=tl.float32)
    for k0 in range(0, in_features, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < in_features
        x_blk = tl.load(x_ptr + m * stride_xm + k_idx * stride_xk, mask=k_mask,
                        other=0.0).to(tl.float32)
        a_blk = tl.load(a_ptr + r_idx[:, None] * stride_ar + k_idx[None, :] * stride_ak,
                        mask=r_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        h += tl.sum(a_blk * x_blk[None, :], axis=1)

    tl.store(h_ptr + m * stride_hm + r_idx, h, mask=r_mask)

    for o0 in range(0, out_features, BLOCK_O):
        o_idx = o0 + tl.arange(0, BLOCK_O)
        o_mask = o_idx < out_features
        b_blk = tl.load(b_ptr + o_idx[:, None] * stride_bo + r_idx[None, :] * stride_br,
                        mask=o_mask[:, None] & r_mask[None, :], other=0.0).to(tl.float32)
        out_blk = tl.sum(b_blk * h[None, :], axis=1) * scaling
        tl.store(out_ptr + m * stride_om + o_idx * stride_oo,
                 out_blk.to(out_ptr.dtype.element_ty), mask=o_mask)


@triton.jit
def _lora_bwd_x_kernel(grad_h_ptr, a_ptr, grad_x_ptr, in_features, r,
                       stride_ghm, stride_ar, stride_ak, stride_gxm, stride_gxk,
                       BLOCK_K: tl.constexpr, BLOCK_R: tl.constexpr):
    """One program per row: grad_x = grad_h @ A -- independent per row, no atomics needed."""
    m = tl.program_id(0)
    r_idx = tl.arange(0, BLOCK_R)
    r_mask = r_idx < r
    gh = tl.load(grad_h_ptr + m * stride_ghm + r_idx, mask=r_mask, other=0.0).to(tl.float32)

    for k0 in range(0, in_features, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < in_features
        a_blk = tl.load(a_ptr + r_idx[:, None] * stride_ar + k_idx[None, :] * stride_ak,
                        mask=r_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        gx_blk = tl.sum(a_blk * gh[:, None], axis=0)
        tl.store(grad_x_ptr + m * stride_gxm + k_idx * stride_gxk,
                 gx_blk.to(grad_x_ptr.dtype.element_ty), mask=k_mask)


class _FusedLoRA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight_a, weight_b, scaling):
        x = x.contiguous()
        weight_a = weight_a.contiguous()  # (r, in_features)
        weight_b = weight_b.contiguous()  # (out_features, r)
        m_rows, in_features = x.shape
        r, in_features_a = weight_a.shape
        out_features, r_b = weight_b.shape
        assert in_features == in_features_a and r == r_b

        h = torch.empty(m_rows, r, device=x.device, dtype=torch.float32)
        out = torch.empty(m_rows, out_features, device=x.device, dtype=x.dtype)
        block_r = triton.next_power_of_2(r)
        _lora_fwd_kernel[(m_rows,)](
            x, weight_a, weight_b, h, out, scaling, in_features, out_features, r,
            x.stride(0), x.stride(1), weight_a.stride(0), weight_a.stride(1),
            weight_b.stride(0), weight_b.stride(1), h.stride(0), out.stride(0), out.stride(1),
            BLOCK_K=128, BLOCK_R=block_r, BLOCK_O=128,
        )
        ctx.save_for_backward(x, weight_a, weight_b, h)
        ctx.scaling = scaling
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight_a, weight_b, h = ctx.saved_tensors
        scaling = ctx.scaling
        grad_out = grad_out.contiguous()
        m_rows, in_features = x.shape
        r = weight_a.shape[0]

        # Weight gradients: shared across every row, reduced over the whole (micro-)batch -- a
        # plain, well-tested matmul rather than a Triton kernel with atomics into a buffer this
        # small (r=16 wide).
        grad_h = (grad_out.float() @ weight_b.float()) * scaling          # (m_rows, r)
        grad_weight_a = grad_h.t() @ x.float()                            # (r, in_features)
        grad_weight_b = (grad_out.float() * scaling).t() @ h              # (out_features, r)

        grad_x = torch.empty_like(x)
        block_r = triton.next_power_of_2(r)
        _lora_bwd_x_kernel[(m_rows,)](
            grad_h.contiguous(), weight_a, grad_x, in_features, r,
            grad_h.stride(0), weight_a.stride(0), weight_a.stride(1), grad_x.stride(0),
            grad_x.stride(1), BLOCK_K=128, BLOCK_R=block_r,
        )
        return (grad_x, grad_weight_a.to(weight_a.dtype), grad_weight_b.to(weight_b.dtype), None)


def fused_lora(x, weight_a, weight_b, scaling):
    return _FusedLoRA.apply(x, weight_a, weight_b, scaling)


def _vanilla_lora_forward(self, x):
    """The part of `Linear4bit.forward` this file changes: `result` starts as the frozen base
    layer's output (untouched, not ours to fuse -- bitsandbytes already fuses its own dequant+GEMM,
    see NF4_RESEARCH.md), then every active vanilla-LoRA adapter adds its contribution. Mirrors the
    real method's order line for line (cast before the variant-vs-vanilla branch, `x` reassigned by
    the cast exactly as the original does) so dropout's RNG draws and float rounding match the base
    run -- only the vanilla branch's matmul pair is replaced."""
    result = self.base_layer(x)
    result = result.clone()
    for active_adapter in self.active_adapters:
        if active_adapter not in self.lora_A.keys():
            continue
        lora_A = self.lora_A[active_adapter]
        lora_B = self.lora_B[active_adapter]
        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]

        requires_conversion = not torch.is_autocast_enabled()
        if requires_conversion:
            expected_dtype = result.dtype
            x = self._cast_input_dtype(x, lora_A.weight.dtype)

        if active_adapter in self.lora_variant:  # DoRA or another variant: not our kernel's job
            result = self.lora_variant[active_adapter].forward(
                self, active_adapter=active_adapter, x=x, result=result)
            if requires_conversion:
                result = result.to(expected_dtype)
            continue

        dropped = dropout(x)
        # hidden_states arrives as (batch, seq, in_features) from Qwen2Attention -- flatten to
        # (rows, in_features) for the kernel, reshape back after. Anything that does not look
        # like that (dim < 2, or a shape mismatch the fallback would handle anyway) goes through
        # the original, unfused computation instead of guessing.
        if not dropped.is_cuda or dropped.dim() < 2 or dropped.shape[-1] != lora_A.weight.shape[1]:
            output = lora_B(lora_A(dropped)) * scaling
        else:
            orig_shape = dropped.shape
            flat = dropped.reshape(-1, orig_shape[-1])
            output = fused_lora(flat, lora_A.weight, lora_B.weight, scaling)
            output = output.reshape(*orig_shape[:-1], output.shape[-1])
            if lora_B.bias is not None:
                output = output + lora_B.bias
        if requires_conversion:
            output = output.to(expected_dtype)
        result = result + output
    return result


def patch_lora():
    """Replace `Linear4bit.forward` class-wide. `disable_adapters`/`merged`/mixed-batch (the
    `adapter_names` kwarg) branches call straight through to the real, saved original -- only the
    plain single-adapter-inference path (what training actually uses here, LoRA on q_proj/v_proj,
    no merging, no batched multi-adapter inference) is replaced.

    `peft.tuners.lora.bnb` itself does `import bitsandbytes` at module load, and
    `peft.tuners.lora.bnb.Linear4bit` only exists inside peft's own `if is_bnb_4bit_available():`
    guard -- either can be missing in a fake/mocked environment (check 10's regression tests may
    run with mocked transformers/peft modules, per the README) or on a machine without
    bitsandbytes at all (confirmed on this repo's own CPU test venv, no GPU). Skip patching rather
    than crash `apply()` -- same spirit as the README's
    `if not isinstance(self.model, torch.nn.Module): return` guard example."""
    global _ORIGINAL_LINEAR4BIT_FORWARD
    try:
        from peft.tuners.lora import bnb as lora_bnb
    except ImportError:
        return
    if not hasattr(lora_bnb, "Linear4bit"):
        return

    if _ORIGINAL_LINEAR4BIT_FORWARD is None:
        _ORIGINAL_LINEAR4BIT_FORWARD = lora_bnb.Linear4bit.forward

    def forward(self, x, *args, **kwargs):
        adapter_names = kwargs.get("adapter_names", None)
        if self.disable_adapters or adapter_names is not None or self.merged or not x.is_cuda:
            return _ORIGINAL_LINEAR4BIT_FORWARD(self, x, *args, **kwargs)
        self._check_forward_args(x, *args, **kwargs)
        return _vanilla_lora_forward(self, x)

    lora_bnb.Linear4bit.forward = forward

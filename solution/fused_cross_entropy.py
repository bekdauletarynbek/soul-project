"""Fused cross-entropy over Qwen2.5's 151,936-token vocabulary, forward and backward in Triton,
chunked over the vocabulary so the (tokens, vocab) logits are never upcast to fp32 as one block.

`transformers.loss.loss_utils.ForCausalLMLoss` -- the loss Soup's `Qwen2ForCausalLM` uses by
default -- does `logits.float()` and then `nn.functional.cross_entropy` over the whole flattened
(tokens, vocab) tensor. For our shapes (batch 4, up to 512 tokens -> up to 2048 rows, vocab
151,936) that fp32 copy alone is up to ~1.2 GB, and autograd keeps at least as much again for the
backward pass -- on a 15 GB T4 already carrying a 4-bit model, that is a sizeable share of the 44%
of step time the README attributes to element-wise ops and dtype casts, and part of the 7% it
measures for softmax/cross-entropy itself.

This computes the identical quantity -- log-sum-exp(row) - row[target], reduced the same way
`transformers.loss.loss_utils.fixed_cross_entropy` reduces it -- a BLOCK_V-wide tile of the
vocabulary at a time, one Triton program per row, with an online (running max + running sum)
softmax so the largest buffer anywhere is (tokens,), not (tokens, vocab). Idea: the same
online-softmax trick flash-attention uses for its running max/sum over KV blocks, applied to
cross-entropy instead of attention -- this is our own implementation from first principles (not a
copy of Liger's or Unsloth's fused CE, which this task does not allow importing anyway).

Hooked in by overwriting `transformers.loss.loss_utils.LOSS_MAPPING["ForCausalLM"]`: it is a plain
module-level dict, looked up by `PreTrainedModel.loss_function` on every call (see
`modeling_utils.py`), so this is a narrow, officially-supported hook -- no subclassing or
reimplementing `Qwen2ForCausalLM.forward` needed. Not covered by a GPU here: wrote and read this
against the actual transformers 5.16.1 source (downloaded for this, see WORK_LOGS), but could not
run it end to end -- the Triton kernels need a run on a CUDA GPU (Kaggle T4) before this is fully
trusted; `solution/test_fused_cross_entropy.py` is ready to confirm it there.
"""

import torch
import triton
import triton.language as tl

BLOCK_V = 4096  # vocab tile; 151,936 / 4096 ~= 38 tiles, keeps each block's registers small

_ORIGINAL_FOR_CAUSAL_LM_LOSS = None  # filled in by patch_cross_entropy(), used as the CPU fallback


@triton.jit
def _ce_fwd_kernel(logits_ptr, targets_ptr, loss_ptr, lse_ptr, stride_row, vocab_size,
                   ignore_index, BLOCK_V: tl.constexpr):
    """One program per row: online log-sum-exp over the vocabulary, plus the target logit."""
    row = tl.program_id(0)
    target = tl.load(targets_ptr + row)
    row_ptr = logits_ptr + row * stride_row

    m = float("-inf")
    s = 0.0
    target_logit = 0.0
    for start in range(0, vocab_size, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < vocab_size
        x = tl.load(row_ptr + offs, mask=mask, other=float("-inf")).to(tl.float32)
        block_max = tl.max(x, axis=0)
        new_m = tl.maximum(m, block_max)
        s = s * tl.exp(m - new_m) + tl.sum(tl.where(mask, tl.exp(x - new_m), 0.0), axis=0)
        m = new_m
        target_logit += tl.sum(tl.where(mask & (offs == target), x, 0.0), axis=0)

    lse = m + tl.log(s)
    valid = target != ignore_index
    tl.store(loss_ptr + row, tl.where(valid, lse - target_logit, 0.0))
    tl.store(lse_ptr + row, lse)


@triton.jit
def _ce_bwd_kernel(logits_ptr, targets_ptr, lse_ptr, grad_out_ptr, grad_logits_ptr, stride_row,
                   vocab_size, ignore_index, BLOCK_V: tl.constexpr):
    """softmax(row) - one_hot(target), scaled by the upstream per-row gradient; 0 on ignored rows.

    Re-reads logits in the same BLOCK_V tiles instead of saving the forward's (tokens, vocab)
    softmax for backward -- the usual recompute-over-store trade a fused kernel makes."""
    row = tl.program_id(0)
    target = tl.load(targets_ptr + row)
    valid = target != ignore_index
    lse = tl.load(lse_ptr + row)
    grad_scale = tl.load(grad_out_ptr + row).to(tl.float32)
    row_ptr = logits_ptr + row * stride_row
    out_ptr = grad_logits_ptr + row * stride_row

    for start in range(0, vocab_size, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < vocab_size
        x = tl.load(row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        softmax = tl.exp(x - lse)
        onehot = (offs == target).to(tl.float32)
        grad = tl.where(valid, (softmax - onehot) * grad_scale, 0.0)
        tl.store(out_ptr + offs, grad.to(out_ptr.dtype.element_ty), mask=mask)


class _FusedCrossEntropy(torch.autograd.Function):
    """Per-row loss (tokens,), unreduced -- the caller sums/divides, so any reduction (mean, or
    sum/num_items_in_batch) gets the right gradient through ordinary autograd on that last step."""

    @staticmethod
    def forward(ctx, logits, targets, ignore_index):
        logits = logits.contiguous()
        n_rows, vocab_size = logits.shape
        loss = torch.empty(n_rows, device=logits.device, dtype=torch.float32)
        lse = torch.empty(n_rows, device=logits.device, dtype=torch.float32)
        _ce_fwd_kernel[(n_rows,)](logits, targets, loss, lse, logits.stride(0), vocab_size,
                                  ignore_index, BLOCK_V=BLOCK_V)
        ctx.save_for_backward(logits, targets, lse)
        ctx.ignore_index = ignore_index
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        logits, targets, lse = ctx.saved_tensors
        n_rows, vocab_size = logits.shape
        grad_logits = torch.empty_like(logits)
        _ce_bwd_kernel[(n_rows,)](logits, targets, lse, grad_loss.contiguous(), grad_logits,
                                  logits.stride(0), vocab_size, ctx.ignore_index, BLOCK_V=BLOCK_V)
        return grad_logits, None, None


def fused_for_causal_lm_loss(logits, labels, vocab_size, num_items_in_batch=None,
                             ignore_index=-100, shift_labels=None, **kwargs):
    """Drop-in replacement for `transformers.loss.loss_utils.ForCausalLMLoss` -- same signature,
    same shifting and reduction, logits never upcast to fp32 as a whole tensor."""
    if not logits.is_cuda:
        return _ORIGINAL_FOR_CAUSAL_LM_LOSS(logits, labels, vocab_size,
                                            num_items_in_batch=num_items_in_batch,
                                            ignore_index=ignore_index, shift_labels=shift_labels,
                                            **kwargs)
    if shift_labels is None:
        labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    flat_logits = logits.view(-1, vocab_size)
    flat_targets = shift_labels.view(-1).to(flat_logits.device)

    per_row_loss = _FusedCrossEntropy.apply(flat_logits, flat_targets, ignore_index)

    if num_items_in_batch is not None:
        denom = num_items_in_batch.to(per_row_loss.device) \
            if torch.is_tensor(num_items_in_batch) else num_items_in_batch
    else:
        denom = (flat_targets != ignore_index).sum().clamp(min=1)
    return per_row_loss.sum() / denom


def patch_cross_entropy():
    """Overwrite the one dict entry `PreTrainedModel.loss_function` reads for any *ForCausalLM
    model whose config does not set its own `loss_type` -- Qwen2ForCausalLM included. A live dict
    mutation, not a copy: `transformers/modeling_utils.py` imports the same `LOSS_MAPPING` object
    and looks a fresh entry up on every call, so this needs no per-model-instance patching."""
    global _ORIGINAL_FOR_CAUSAL_LM_LOSS
    from transformers.loss import loss_utils

    if _ORIGINAL_FOR_CAUSAL_LM_LOSS is None:
        _ORIGINAL_FOR_CAUSAL_LM_LOSS = loss_utils.ForCausalLMLoss
    loss_utils.LOSS_MAPPING["ForCausalLM"] = fused_for_causal_lm_loss

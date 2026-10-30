"""Correctness of the fused cross-entropy kernel against `F.cross_entropy` + `logits.float()`,
forward and backward, for shapes close to the real one (batch*seq rows, 151,936-token vocab).

Run on a GPU: `python -m pytest solution/test_fused_cross_entropy.py -q`.
"""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-2, torch.float32: 1e-3}  # looser than swiglu's: summed over a
# 151,936-wide vocab, fp16 rounding in the loaded logits adds up more than an element-wise op does


def _reference(logits, targets, ignore_index):
    logits32 = logits.float()
    return F.cross_entropy(logits32, targets, ignore_index=ignore_index, reduction="none")


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("rows,vocab", [(8, 151936), (37, 1001)])
def test_fused_ce_matches_pytorch(dtype, rows, vocab):
    from solution.fused_cross_entropy import _FusedCrossEntropy

    torch.manual_seed(0)
    ignore_index = -100
    logits = torch.randn(rows, vocab, device="cuda", dtype=dtype, requires_grad=True)
    targets = torch.randint(0, vocab, (rows,), device="cuda")
    targets[0] = ignore_index  # make sure the ignored row stays loss 0 and grad 0

    ref_loss = _reference(logits, targets, ignore_index)
    ref_loss.sum().backward()
    ref_grad = logits.grad.clone()
    logits.grad = None

    tol = TOLERANCE[dtype]
    out = _FusedCrossEntropy.apply(logits, targets, ignore_index)
    torch.testing.assert_close(out, ref_loss, rtol=tol, atol=tol)
    assert out[0].item() == 0.0, "ignored row must carry zero loss"

    out.sum().backward()
    torch.testing.assert_close(logits.grad, ref_grad, rtol=tol, atol=tol)
    assert torch.all(logits.grad[0] == 0), "ignored row must carry zero gradient"


def test_fused_for_causal_lm_loss_matches_mean_and_sum_reduction():
    """The public entry point `fused_for_causal_lm_loss` -- shift, flatten, both reduction modes
    of `transformers.loss.loss_utils.fixed_cross_entropy` (plain mean, and sum/num_items_in_batch)."""
    from solution.fused_cross_entropy import fused_for_causal_lm_loss

    torch.manual_seed(0)
    batch, seq, vocab = 2, 16, 1001
    ignore_index = -100
    logits = torch.randn(batch, seq, vocab, device="cuda", dtype=torch.float16, requires_grad=True)
    labels = torch.randint(0, vocab, (batch, seq), device="cuda")
    labels[:, : seq // 2] = ignore_index  # half the tokens masked, like the prompt half of a row

    padded = F.pad(labels, (0, 1), value=ignore_index)
    shift_labels = padded[..., 1:].contiguous()
    flat_logits = logits.view(-1, vocab).float()
    flat_targets = shift_labels.view(-1)

    mean_ref = F.cross_entropy(flat_logits, flat_targets, ignore_index=ignore_index,
                               reduction="mean")
    mean_out = fused_for_causal_lm_loss(logits, labels, vocab)
    torch.testing.assert_close(mean_out, mean_ref, rtol=2e-2, atol=2e-2)

    num_items = (flat_targets != ignore_index).sum()
    sum_ref = F.cross_entropy(flat_logits, flat_targets, ignore_index=ignore_index,
                              reduction="sum") / num_items
    sum_out = fused_for_causal_lm_loss(logits, labels, vocab, num_items_in_batch=num_items)
    torch.testing.assert_close(sum_out, sum_ref, rtol=2e-2, atol=2e-2)

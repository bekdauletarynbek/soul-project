"""Correctness of the fused RoPE kernel against the plain PyTorch reference (the exact
`apply_rotary_pos_emb` from transformers v5.16.1's modeling_qwen2.py), forward and backward,
for both q (more heads) and k (fewer heads, GQA) shapes. Run on a GPU: `python -m pytest
solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}


def reference_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def reference_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (reference_rotate_half(q) * sin)
    k_embed = (k * cos) + (reference_rotate_half(k) * sin)
    return q_embed, k_embed


def make_cos_sin(cos_batch, seq_len, head_dim, device, dtype):
    """cos_batch=1 reproduces what `Qwen2Model.forward` actually passes down when the caller
    doesn't supply `position_ids` (TRL's SFTTrainer in unpacked mode never does): shape
    (1, seq_len, head_dim), meant to broadcast over the real batch -- not pre-expanded to it.
    Expanding it here (as an earlier version of this test did) would hide exactly the
    out-of-bounds-read regression this test now exists to catch."""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[None, :, :].expand(cos_batch, -1, -1).to(dtype).contiguous()
    sin = emb.sin()[None, :, :].expand(cos_batch, -1, -1).to(dtype).contiguous()
    return cos, sin


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("batch,q_heads,kv_heads,seq_len,head_dim,cos_batch", [
    (4, 12, 2, 37, 128, 1),  # Qwen2.5-1.5B-like shape (GQA) with the real cos/sin broadcast (batch 1)
    (2, 12, 2, 37, 128, 2),  # the other legal case: position_ids given explicitly, cos/sin match batch
    (1, 2, 2, 8, 16, 1),
])
def test_rope_matches_pytorch(dtype, batch, q_heads, kv_heads, seq_len, head_dim, cos_batch):
    from solution.rope import rotary_embed

    torch.manual_seed(0)
    cos, sin = make_cos_sin(cos_batch, seq_len, head_dim, "cuda", dtype)
    q = torch.randn(batch, q_heads, seq_len, head_dim, device="cuda", dtype=dtype,
                    requires_grad=True)
    k = torch.randn(batch, kv_heads, seq_len, head_dim, device="cuda", dtype=dtype,
                    requires_grad=True)
    dq_grad = torch.randn_like(q)
    dk_grad = torch.randn_like(k)

    q_ref = q.detach().clone().requires_grad_()
    k_ref = k.detach().clone().requires_grad_()
    q_embed_ref, k_embed_ref = reference_apply_rotary_pos_emb(q_ref, k_ref, cos, sin)
    q_embed_ref.backward(dq_grad, retain_graph=True)
    k_embed_ref.backward(dk_grad)

    q_embed, k_embed = rotary_embed(q, k, cos, sin)
    q_embed.backward(dq_grad, retain_graph=True)
    k_embed.backward(dk_grad)

    tol = TOLERANCE[dtype]
    torch.testing.assert_close(q_embed.float(), q_embed_ref.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(k_embed.float(), k_embed_ref.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(q.grad.float(), q_ref.grad.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(k.grad.float(), k_ref.grad.float(), rtol=tol, atol=tol)

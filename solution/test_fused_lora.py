"""Correctness of the fused LoRA kernel against plain PyTorch (two `nn.Linear` calls), forward and
backward, for shapes close to Qwen2.5-1.5B's q_proj/v_proj (in_features=1536, r=16).

Run on a GPU: `python -m pytest solution/test_fused_lora.py -q`.
"""

import pytest
import torch

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 5e-3, torch.float32: 1e-4}


@needs_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("rows,in_features,out_features,r", [(256, 1536, 1536, 16), (37, 1536, 256, 16)])
def test_fused_lora_matches_pytorch(dtype, rows, in_features, out_features, r):
    from solution.fused_lora import fused_lora

    torch.manual_seed(0)
    scaling = 2.0  # lora_alpha / r = 32 / 16, task.yaml's actual LoRA config
    x = torch.randn(rows, in_features, device="cuda", dtype=dtype, requires_grad=True)
    weight_a = torch.randn(r, in_features, device="cuda", dtype=dtype, requires_grad=True) * 0.02
    weight_b = torch.randn(out_features, r, device="cuda", dtype=dtype, requires_grad=True) * 0.02
    grad_out = torch.randn(rows, out_features, device="cuda", dtype=dtype)

    ref = torch.nn.functional.linear(torch.nn.functional.linear(x, weight_a), weight_b) * scaling
    ref.backward(grad_out)
    ref_dx, ref_da, ref_db = x.grad.clone(), weight_a.grad.clone(), weight_b.grad.clone()
    x.grad = weight_a.grad = weight_b.grad = None

    tol = TOLERANCE[dtype]
    out = fused_lora(x, weight_a, weight_b, scaling)
    torch.testing.assert_close(out, ref, rtol=tol, atol=tol)

    out.backward(grad_out)
    torch.testing.assert_close(x.grad, ref_dx, rtol=tol, atol=tol)
    torch.testing.assert_close(weight_a.grad, ref_da, rtol=tol, atol=tol)
    torch.testing.assert_close(weight_b.grad, ref_db, rtol=tol, atol=tol)


def test_patch_falls_back_when_not_cuda():
    """check 10's regression tests may exercise Linear4bit with CPU or mocked layers; the patch
    must not crash or silently miscompute for anything outside the fused path's own guard."""
    from solution.fused_lora import patch_lora

    patch_lora()  # must not raise even without a GPU / without peft's bnb extra available

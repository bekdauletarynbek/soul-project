"""Correctness of the fused RMSNorm against the plain PyTorch reference (the same formula
Qwen2RMSNorm uses), forward and backward, including the frozen-weight path QLoRA actually
takes. Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}
EPS = 1e-6


def reference_rmsnorm(x, weight, eps):
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(input_dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 1536), (3, 77, 1001)])
def test_rmsnorm_matches_pytorch(dtype, shape):
    from solution.rmsnorm import rmsnorm

    torch.manual_seed(0)
    hidden = shape[-1]
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(hidden, device="cuda", dtype=torch.float32, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    ref = reference_rmsnorm(x_ref, weight_ref, EPS)
    ref.backward(grad)

    out = rmsnorm(x, weight, EPS)
    out.backward(grad)

    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(x.grad.float(), x_ref.grad.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(weight.grad.float(), weight_ref.grad.float(), rtol=tol, atol=tol)


def test_rmsnorm_frozen_weight_still_backprops_dx():
    """The real setup: the norm weight is part of the frozen base model (requires_grad=False),
    only dx needs to flow back to the LoRA adapters upstream."""
    from solution.rmsnorm import rmsnorm

    torch.manual_seed(0)
    shape, hidden = (2, 16, 64), 64
    x = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    weight = torch.randn(hidden, device="cuda", dtype=torch.float32, requires_grad=False)
    grad = torch.randn(shape, device="cuda", dtype=torch.float16)

    x_ref = x.detach().clone().requires_grad_()
    ref = reference_rmsnorm(x_ref, weight, EPS)
    ref.backward(grad)

    out = rmsnorm(x, weight, EPS)
    out.backward(grad)

    assert weight.grad is None
    tol = TOLERANCE[torch.float16]
    torch.testing.assert_close(x.grad.float(), x_ref.grad.float(), rtol=tol, atol=tol)

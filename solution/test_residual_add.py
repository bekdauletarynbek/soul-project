"""Correctness of the fused residual add, forward and backward, against a plain `a + b`.
Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 1536), (3, 77, 1001)])
def test_residual_add_matches_pytorch(dtype, shape):
    from solution.residual_add import residual_add

    torch.manual_seed(0)
    a = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    b = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    a_ref = a.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()
    ref = a_ref + b_ref
    ref.backward(grad)

    out = residual_add(a, b)
    out.backward(grad)

    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(a.grad.float(), a_ref.grad.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(b.grad.float(), b_ref.grad.float(), rtol=tol, atol=tol)

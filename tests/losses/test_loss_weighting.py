import math

import pytest
import torch
from torch.testing import assert_close

from torchebm.losses.functional import adaptive_weight, pseudo_huber_loss


def _inputs():
    g = torch.Generator().manual_seed(0)
    u = torch.randn(16, 4, 8, 8, generator=g, dtype=torch.float64, requires_grad=True)
    target = torch.randn(16, 4, 8, 8, generator=g, dtype=torch.float64)
    target[:4] = u[:4].detach() + 1e-3
    return u, target


def _meanflow_reference(u, u_tgt, norm_p, norm_eps):
    r"""Loss of `MeanFlow.forward` in Gsunshine/meanflow (d70cb55), in PyTorch."""
    loss = (u - u_tgt) ** 2
    loss = torch.sum(loss, dim=(1, 2, 3))
    adp_wt = (loss + norm_eps) ** norm_p
    loss = loss / adp_wt.detach()
    return loss.mean()


def _ect_reference(D_yt, D_yr, c):
    r"""Pseudo-Huber metric of `ECTLoss.__call__` in locuslab/ect (4311059), in PyTorch."""
    loss = (D_yt - D_yr) ** 2
    loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)
    if c > 0:
        loss = torch.sqrt(loss + c**2) - c
    else:
        loss = torch.sqrt(loss)
    return loss.mean()


@pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
def test_adaptive_weight_matches_meanflow(p):
    u, target = _inputs()
    reference = _meanflow_reference(u, target, p, 0.01)
    reference_grad = torch.autograd.grad(reference, u)[0]
    squared_error = (u - target).square().flatten(1).sum(1)
    loss = (adaptive_weight(squared_error, p=p) * squared_error).mean()
    assert_close(loss, reference)
    assert_close(torch.autograd.grad(loss, u)[0], reference_grad)


def test_adaptive_weight_is_detached():
    squared_error = torch.rand(8, requires_grad=True)
    assert not adaptive_weight(squared_error).requires_grad


@pytest.mark.parametrize("c", [0.03, 0.5])
def test_pseudo_huber_matches_ect(c):
    u, target = _inputs()
    reference = _ect_reference(u, target, c)
    reference_grad = torch.autograd.grad(reference, u)[0]
    loss = pseudo_huber_loss(u, target, c=c)
    assert_close(loss, reference)
    assert_close(torch.autograd.grad(loss, u)[0], reference_grad)


def test_pseudo_huber_default_c_is_ict():
    u, target = _inputs()
    c = 0.00054 * math.sqrt(4 * 8 * 8)
    assert_close(pseudo_huber_loss(u, target), _ect_reference(u, target, c))


def test_pseudo_huber_weight_and_reduction():
    u, target = _inputs()
    weight = torch.rand(16, dtype=torch.float64)
    per_sample = pseudo_huber_loss(u, target, c=0.1, reduction="none")
    assert per_sample.shape == (16,)
    assert_close(
        pseudo_huber_loss(u, target, c=0.1, weight=weight, reduction="sum"),
        (per_sample * weight).sum(),
    )


def test_compile():
    u, target = _inputs()
    u = u.detach()

    def loss_fn(u, target):
        squared_error = (u - target).square().flatten(1).sum(1)
        return (adaptive_weight(squared_error) * squared_error).mean() + pseudo_huber_loss(u, target)

    assert_close(torch.compile(loss_fn, fullgraph=True)(u, target), loss_fn(u, target))

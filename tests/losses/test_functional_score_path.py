r"""Single-process parity of the functional score path against autograd."""

import pytest
import torch
from torch import nn

from torchebm.core import BaseModel
from torchebm.losses import (
    DenoisingScoreMatching,
    ScoreMatching,
    SlicedScoreMatching,
)


class MLPEnergy(BaseModel):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _loss_and_grads(loss_fn, model, x, seed):
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    loss = loss_fn(x)
    loss.backward()
    grads = [
        None if p.grad is None else p.grad.clone() for p in model.parameters()
    ]
    return loss.detach(), grads


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (DenoisingScoreMatching, {"noise_scale": 0.1}),
        (ScoreMatching, {"hessian_method": "approx"}),
        (SlicedScoreMatching, {"n_projections": 4}),
    ],
)
def test_functional_matches_autograd(cls, kwargs):
    torch.manual_seed(0)
    model = MLPEnergy()
    x = torch.randn(8, 4)
    loss_a, grads_a = _loss_and_grads(
        cls(model=model, use_autograd=True, **kwargs), model, x, seed=7
    )
    loss_f, grads_f = _loss_and_grads(
        cls(model=model, use_autograd=False, **kwargs), model, x, seed=7
    )
    assert torch.allclose(loss_a, loss_f, atol=1e-6)
    for ga, gf in zip(grads_a, grads_f):
        if ga is None:
            assert gf is None
        else:
            assert torch.allclose(ga, gf, atol=1e-6)


def test_functional_model_is_structural_only():
    torch.manual_seed(0)
    model = MLPEnergy()
    template = MLPEnergy()
    x = torch.randn(8, 4)
    kwargs = {"noise_scale": 0.1, "use_autograd": False}
    loss_t, grads_t = _loss_and_grads(
        DenoisingScoreMatching(model=model, functional_model=template, **kwargs),
        model,
        x,
        seed=7,
    )
    loss_d, grads_d = _loss_and_grads(
        DenoisingScoreMatching(model=model, **kwargs), model, x, seed=7
    )
    assert torch.allclose(loss_t, loss_d)
    for gt, gd in zip(grads_t, grads_d):
        if gt is None:
            assert gd is None
        else:
            assert torch.allclose(gt, gd)
    assert all(p.grad is None for p in template.parameters())


def test_functional_model_not_registered():
    loss_fn = DenoisingScoreMatching(
        model=MLPEnergy(), use_autograd=False, functional_model=MLPEnergy()
    )
    assert not any(k.startswith("_functional_model") for k in loss_fn.state_dict())
    assert len(list(loss_fn.parameters())) == len(list(loss_fn.model.parameters()))


def test_functional_works_on_parameterless_model():
    from torchebm.core import DoubleWellModel

    x = torch.randn(8, 2)
    loss_a, _ = _loss_and_grads(
        DenoisingScoreMatching(model=DoubleWellModel(), noise_scale=0.1),
        DoubleWellModel(),
        x,
        seed=7,
    )
    loss_f, _ = _loss_and_grads(
        DenoisingScoreMatching(
            model=DoubleWellModel(), noise_scale=0.1, use_autograd=False
        ),
        DoubleWellModel(),
        x,
        seed=7,
    )
    assert torch.allclose(loss_a, loss_f, atol=1e-6)

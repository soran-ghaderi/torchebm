"""Functional loss preparation and reduction APIs."""

import copy
from typing import Any, Optional

import pytest
import torch
from torch import nn

from torchebm.core import BaseCoupling, CouplingResult
from torchebm.losses import (
    FlowMatchingLoss,
    prepare_flow_matching,
    weighted_mse_loss,
)
from torchebm.losses.loss_utils import get_interpolant


class WeightedReverseCoupling(BaseCoupling):
    def couple(
        self,
        x0: torch.Tensor,
        x1: Optional[torch.Tensor] = None,
        *,
        generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> CouplingResult:
        x1 = self._require_x1(x1)
        weights = torch.arange(1, x0.shape[0] + 1, device=x0.device, dtype=x0.dtype)
        return CouplingResult(x0, x1.flip(0), weights=weights)


class RandomPermutationCoupling(BaseCoupling):
    def couple(
        self,
        x0: torch.Tensor,
        x1: Optional[torch.Tensor] = None,
        *,
        generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> CouplingResult:
        x1 = self._require_x1(x1)
        indices = torch.randperm(x0.shape[0], generator=generator)
        return CouplingResult(x0, x1[indices])


class TimeField(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.linear(x) + t.unsqueeze(-1)


def _fixed_t(t: torch.Tensor):
    return lambda batch, *, device, dtype, generator: t.to(device=device, dtype=dtype)


def test_external_flow_evaluation_matches_class_value_and_gradients():
    torch.manual_seed(7)
    x1 = torch.randn(4, 3)
    x0 = torch.randn(4, 3)
    t = torch.tensor([0.1, 0.3, 0.6, 0.9])
    coupling = WeightedReverseCoupling()
    model = TimeField(dim=3)
    external_model = copy.deepcopy(model)
    loss_weight = lambda value: value + 0.25

    class_loss = FlowMatchingLoss(
        model=model,
        coupling=coupling,
        t_sampler=_fixed_t(t),
        loss_weight_fn=loss_weight,
    )(x1, x0=x0)

    batch = prepare_flow_matching(
        x1,
        interpolant=get_interpolant("linear"),
        coupling=coupling,
        x0=x0,
        t=t,
    )
    prediction = external_model(batch.xt, batch.model_time)
    external_loss = weighted_mse_loss(
        prediction,
        batch.target,
        weights=batch.weights,
        loss_weights=loss_weight(batch.t),
    )

    assert torch.equal(class_loss, external_loss)

    class_loss.backward()
    external_loss.backward()
    for class_param, external_param in zip(
        model.parameters(), external_model.parameters()
    ):
        assert torch.equal(class_param.grad, external_param.grad)


def test_prepare_flow_matching_preserves_rng_order():
    x1 = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    expected_generator = torch.Generator().manual_seed(19)
    actual_generator = torch.Generator().manual_seed(19)
    coupling = RandomPermutationCoupling()
    interpolant = get_interpolant("linear")

    expected_x0 = torch.randn_like(x1, generator=expected_generator)
    expected_coupled = coupling(expected_x0, x1, generator=expected_generator)
    expected_t = torch.rand(4, generator=expected_generator)
    expected_xt, expected_target = interpolant.interpolate(
        expected_coupled.x0, expected_coupled.x1, expected_t
    )

    batch = prepare_flow_matching(
        x1,
        interpolant=interpolant,
        coupling=coupling,
        generator=actual_generator,
    )

    assert torch.equal(batch.xt, expected_xt)
    assert torch.equal(batch.target, expected_target)
    assert torch.equal(batch.t, expected_t)
    assert torch.equal(actual_generator.get_state(), expected_generator.get_state())


def test_explicit_source_and_time_do_not_advance_generator():
    x1 = torch.randn(4, 3)
    x0 = torch.randn_like(x1)
    t = torch.rand(4)
    generator = torch.Generator().manual_seed(23)
    initial_state = generator.get_state()

    batch = prepare_flow_matching(
        x1,
        interpolant=get_interpolant("linear"),
        coupling=WeightedReverseCoupling(),
        x0=x0,
        t=t,
        generator=generator,
    )

    assert torch.equal(batch.t, t)
    assert torch.equal(batch.model_time, t)
    assert torch.equal(generator.get_state(), initial_state)


def test_weighted_mse_loss_keeps_objective_and_coupling_weights_distinct():
    prediction = torch.tensor([[1.0, 3.0], [2.0, 6.0]])
    target = torch.zeros_like(prediction)
    coupling_weights = torch.tensor([1.0, 3.0])
    loss_weights = torch.tensor([2.0, 0.5])

    per_sample = prediction.square().mean(dim=1) * loss_weights
    expected = (coupling_weights * per_sample).sum() / coupling_weights.sum()

    assert torch.equal(
        weighted_mse_loss(
            prediction,
            target,
            weights=coupling_weights,
            loss_weights=loss_weights,
        ),
        expected,
    )
    assert torch.equal(
        weighted_mse_loss(
            prediction, target, loss_weights=loss_weights, reduction="none"
        ),
        per_sample,
    )


def test_prepare_flow_matching_validates_explicit_shapes():
    x1 = torch.randn(4, 3)
    interpolant = get_interpolant("linear")
    coupling = WeightedReverseCoupling()

    with pytest.raises(ValueError, match="x0 shape"):
        prepare_flow_matching(
            x1, interpolant=interpolant, coupling=coupling, x0=torch.randn(3, 3)
        )
    with pytest.raises(ValueError, match="t shape"):
        prepare_flow_matching(
            x1,
            interpolant=interpolant,
            coupling=coupling,
            x0=torch.randn_like(x1),
            t=torch.rand(4, 1),
        )


def test_weighted_mse_loss_rejects_unknown_reduction():
    with pytest.raises(ValueError, match="Unknown reduction"):
        weighted_mse_loss(
            torch.zeros(2, 1),
            torch.zeros(2, 1),
            reduction="sum",  # type: ignore[arg-type]
        )

"""Closed-form scores of the analytic models under `prediction_type`."""

import pytest
import torch

from torchebm.core import (
    AckleyModel,
    AnalyticModel,
    BoothModel,
    DoubleWellModel,
    GaussianModel,
    HarmonicModel,
    HimmelblauModel,
    RastriginModel,
    RosenbrockModel,
    StyblinskiTangModel,
)
from torchebm.samplers import LangevinDynamics

MEAN = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64)
COV = torch.tensor(
    [[2.0, 0.3, 0.0], [0.3, 1.0, 0.2], [0.0, 0.2, 1.5]], dtype=torch.float64
)

MODELS = [
    pytest.param(lambda **kw: DoubleWellModel(2.0, 1.0, **kw), 3, id="double_well"),
    pytest.param(lambda **kw: GaussianModel(MEAN, COV, **kw), 3, id="gaussian"),
    pytest.param(lambda **kw: HarmonicModel(k=2.0, **kw), 3, id="harmonic"),
    pytest.param(lambda **kw: RosenbrockModel(**kw), 4, id="rosenbrock"),
    pytest.param(lambda **kw: AckleyModel(**kw), 3, id="ackley"),
    pytest.param(lambda **kw: RastriginModel(**kw), 3, id="rastrigin"),
    pytest.param(lambda **kw: HimmelblauModel(**kw), 2, id="himmelblau"),
    pytest.param(lambda **kw: BoothModel(**kw), 2, id="booth"),
    pytest.param(lambda **kw: StyblinskiTangModel(**kw), 3, id="styblinski_tang"),
]


def inputs(dim):
    generator = torch.Generator().manual_seed(0)
    return torch.randn(6, dim, generator=generator, dtype=torch.float64)


@pytest.mark.parametrize("factory, dim", MODELS)
def test_closed_form_score_matches_autograd(factory, dim):
    x = inputs(dim)
    score = factory(dtype=torch.float64, prediction_type="score")(x)
    reference = -factory(dtype=torch.float64).gradient(x)
    assert score.shape == x.shape
    assert torch.allclose(score, reference, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("factory, dim", MODELS)
def test_tuple_prediction_returns_energy_and_score(factory, dim):
    x = inputs(dim)
    energy, score = factory(dtype=torch.float64, prediction_type=("energy", "score"))(x)
    assert torch.equal(energy, factory(dtype=torch.float64)(x))
    assert torch.equal(score, factory(dtype=torch.float64, prediction_type="score")(x))


@pytest.mark.parametrize("factory, dim", MODELS)
def test_score_needs_no_autograd(factory, dim):
    model = factory(dtype=torch.float64, prediction_type="score")
    x = inputs(dim)
    with torch.inference_mode():
        score = model(x)
    assert torch.equal(score, model(x))
    assert torch.allclose(model(x[0]), score[:1])


@pytest.mark.parametrize("factory, dim", MODELS)
def test_score_compiles_without_graph_breaks(factory, dim):
    model = factory(dtype=torch.float64, prediction_type=("energy", "score"))
    x = inputs(dim)
    energy, score = torch.compile(model, fullgraph=True)(x)
    reference_energy, reference_score = model(x)
    assert torch.allclose(energy, reference_energy)
    assert torch.allclose(score, reference_score)


def test_langevin_consumes_the_closed_form_score():
    x = torch.randn(8, 3, generator=torch.Generator().manual_seed(1))
    sampler = LangevinDynamics(step_size=0.05)
    from_score = sampler(
        DoubleWellModel(prediction_type="score"),
        x,
        n_steps=5,
        generator=torch.Generator().manual_seed(2),
    )
    from_energy = sampler(
        DoubleWellModel(), x, n_steps=5, generator=torch.Generator().manual_seed(2)
    )
    assert torch.allclose(from_score, from_energy, atol=1e-6)


def test_analytic_model_requires_energy_and_score():
    class EnergyOnly(AnalyticModel):
        def _energy(self, x):
            return x.square().sum(dim=-1)

    with pytest.raises(TypeError):
        EnergyOnly()

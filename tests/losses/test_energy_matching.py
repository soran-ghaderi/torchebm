"""Tests for the Energy Matching (EM) loss.

Based on "Energy Matching: Unifying Flow Matching and Energy-Based Models
for Generative Modeling" (arXiv:2504.10612). Key mechanisms tested:

1. Flow term: -grad V regressed onto the coupled displacement, time-gated.
2. Two-phase training: lambda_cd = 0 skips the Langevin chains entirely.
3. Contrastive term: split negatives, trimmed mean, stability clamp.
"""

import unittest.mock

import pytest
import torch
import torch.nn as nn

from torchebm.core import (
    BaseCoupling,
    BaseModel,
    ConstantScheduler,
    CouplingResult,
    TemperatureScheduler,
)
from torchebm.losses import EnergyMatchingLoss
from torchebm.losses.loss_utils import (
    compute_flow_weight,
    get_interpolant,
    trimmed_mean,
)
from torchebm.samplers import LangevinDynamics


class QuadraticPotential(BaseModel):
    """V(x) = 0.5 ||x||^2, so -grad V = -x (closed form)."""

    def forward(self, x):
        return 0.5 * x.flatten(1).square().sum(dim=1)


class MLPPotential(BaseModel):
    """Small learnable potential.

    The output layer has no bias: the flow term supervises grad V only, and
    grad V is invariant to additive constants, so a final bias would
    (correctly) receive no gradient in the warm-up phase.
    """

    def __init__(self, dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 32), nn.SiLU(), nn.Linear(32, 1, bias=False)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CountingSampler(LangevinDynamics):
    """LangevinDynamics that records per-call batch size and noise schedule."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def sample(self, x=None, n_steps=100, **kwargs):
        self.calls.append(
            {
                "batch": x.shape[0],
                "n_steps": n_steps,
                "noise_scheduler": type(self.schedulers["noise_scale"]).__name__,
            }
        )
        return super().sample(x=x, n_steps=n_steps, **kwargs)


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ]
)
def device(request):
    return torch.device(request.param)


def make_loss(model=None, **kwargs):
    """EM loss with cheap Langevin defaults for tests."""
    model = model if model is not None else QuadraticPotential()
    defaults = dict(n_langevin_steps=5, langevin_dt=0.01)
    defaults.update(kwargs)
    return EnergyMatchingLoss(model=model, **defaults)


# Shapes and finiteness
# =====================


@pytest.mark.parametrize("coupling", ["independent", "ot"])
def test_em_loss_scalar_and_finite(coupling, device):
    loss_fn = make_loss(coupling=coupling, lambda_cd=2.0, device=device)
    x = torch.randn(16, 2, device=device)
    loss = loss_fn(x)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_em_loss_image_shaped_input():
    loss_fn = make_loss(coupling="ot", lambda_cd=0.0)
    x = torch.randn(8, 3, 4, 4)
    loss = loss_fn(x)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


# Flow term math
# ==============


def test_em_warmup_manual_verification():
    """Warm-up loss equals the hand-computed gated flow-matching objective.

    With V(x) = 0.5||x||^2: -grad V(x_t) = -x_t, target u_t = x1 - x0,
    loss = mean(w(t) * mean_flat((-x_t - u_t)^2)).
    """
    batch, dim = 4, 3
    x1 = torch.randn(batch, dim)
    fixed_x0 = torch.randn(batch, dim)
    fixed_t = torch.rand(batch)

    with unittest.mock.patch(
        "torchebm.losses.energy_matching.torch.randn_like",
        return_value=fixed_x0,
    ), unittest.mock.patch(
        "torchebm.losses.energy_matching.torch.rand",
        return_value=fixed_t,
    ):
        loss_fn = make_loss(
            coupling="independent",
            sigma=0.0,
            lambda_cd=0.0,
            flow_weight_cutoff=0.8,
            device="cpu",
        )
        loss = loss_fn(x1)

    xt, ut = get_interpolant("linear").interpolate(fixed_x0, x1, fixed_t)
    w = compute_flow_weight(fixed_t, cutoff=0.8)
    expected = (w * ((-xt - ut).square().mean(dim=1))).mean()
    assert torch.allclose(loss, expected, atol=1e-5)


def test_em_flow_weight_cutoff_one_disables_gating():
    batch, dim = 4, 3
    x1 = torch.randn(batch, dim)
    fixed_x0 = torch.randn(batch, dim)
    fixed_t = torch.rand(batch)

    with unittest.mock.patch(
        "torchebm.losses.energy_matching.torch.randn_like",
        return_value=fixed_x0,
    ), unittest.mock.patch(
        "torchebm.losses.energy_matching.torch.rand",
        return_value=fixed_t,
    ):
        loss_fn = make_loss(
            coupling="independent",
            sigma=0.0,
            lambda_cd=0.0,
            flow_weight_cutoff=1.0,
            device="cpu",
        )
        loss = loss_fn(x1)

    xt, ut = get_interpolant("linear").interpolate(fixed_x0, x1, fixed_t)
    expected = (-xt - ut).square().mean(dim=1).mean()
    assert torch.allclose(loss, expected, atol=1e-5)


def test_compute_flow_weight_values():
    t = torch.tensor([0.0, 0.8, 0.9, 1.0])
    w = compute_flow_weight(t, cutoff=0.8)
    assert torch.allclose(w, torch.tensor([1.0, 1.0, 0.5, 0.0]), atol=1e-6)
    assert torch.equal(compute_flow_weight(t, cutoff=1.0), torch.ones(4))


def test_trimmed_mean_values():
    values = torch.tensor([1.0, 2.0, 3.0, 100.0])
    assert trimmed_mean(values, 0.25) == pytest.approx(2.0)
    assert trimmed_mean(values, 0.0) == pytest.approx(26.5)
    with pytest.raises(ValueError, match="trim_fraction"):
        trimmed_mean(values, 1.0)


# Two-phase behavior
# ==================


def test_em_warmup_skips_sampler():
    model = QuadraticPotential()
    sampler = CountingSampler(model=model, step_size=0.01)
    loss_fn = make_loss(model=model, sampler=sampler, lambda_cd=0.0)
    loss_fn(torch.randn(8, 2))
    assert sampler.calls == []


def test_em_cd_phase_runs_two_chains_with_split_and_schedules():
    model = QuadraticPotential()
    sampler = CountingSampler(model=model, step_size=0.01)
    loss_fn = make_loss(
        model=model,
        sampler=sampler,
        lambda_cd=2.0,
        noise_fraction=0.5,
        n_langevin_steps=5,
    )
    terms = loss_fn.training_losses(torch.randn(8, 2))

    assert len(sampler.calls) == 2
    noise_call, data_call = sampler.calls
    assert noise_call["batch"] == 4 and data_call["batch"] == 4
    assert noise_call["n_steps"] == 5 and data_call["n_steps"] == 5
    # Noise-initialized chain sweeps epsilon(t); data chain holds eps_max.
    assert noise_call["noise_scheduler"] == "TemperatureScheduler"
    assert data_call["noise_scheduler"] == "ConstantScheduler"
    assert terms["negatives"].shape == (8, 2)


def test_em_lambda_cd_setter_switches_phase():
    model = QuadraticPotential()
    sampler = CountingSampler(model=model, step_size=0.01)
    loss_fn = make_loss(model=model, sampler=sampler, lambda_cd=0.0)

    loss_fn(torch.randn(8, 2))
    assert len(sampler.calls) == 0

    loss_fn.lambda_cd = 2.0
    loss_fn(torch.randn(8, 2))
    assert len(sampler.calls) == 2
    assert loss_fn.lambda_cd == 2.0


def test_em_noise_fraction_extremes():
    model = QuadraticPotential()
    sampler = CountingSampler(model=model, step_size=0.01)
    loss_fn = make_loss(
        model=model, sampler=sampler, lambda_cd=1.0, noise_fraction=1.0
    )
    loss_fn(torch.randn(8, 2))
    assert len(sampler.calls) == 1
    assert sampler.calls[0]["noise_scheduler"] == "TemperatureScheduler"

    sampler.calls.clear()
    loss_fn2 = make_loss(
        model=model, sampler=sampler, lambda_cd=1.0, noise_fraction=0.0
    )
    loss_fn2(torch.randn(8, 2))
    assert len(sampler.calls) == 1
    assert sampler.calls[0]["noise_scheduler"] == "ConstantScheduler"


# Contrastive term
# ================


def test_em_negatives_are_detached():
    loss_fn = make_loss(model=MLPPotential(), lambda_cd=2.0)
    terms = loss_fn.training_losses(torch.randn(8, 2))
    assert terms["negatives"].requires_grad is False
    assert terms["negatives"].grad_fn is None


def test_em_cd_clamp_floors_term():
    """When lambda_cd * cd_value < -cd_clamp, the term is exactly -cd_clamp."""
    loss_fn = make_loss(lambda_cd=2.0, cd_clamp=0.02)
    x = torch.randn(8, 2) * 0.1  # low-energy data
    far_negatives = torch.full((8, 2), 10.0)  # very high energy

    with unittest.mock.patch.object(
        loss_fn, "_sample_negatives", return_value=far_negatives
    ):
        terms = loss_fn.training_losses(x)

    assert terms["cd_value"] < 0
    assert terms["cd_loss"].item() == pytest.approx(-0.02)


def test_em_cd_clamp_none_disables_floor():
    loss_fn = make_loss(lambda_cd=2.0, cd_clamp=None)
    x = torch.randn(8, 2) * 0.1
    far_negatives = torch.full((8, 2), 10.0)

    with unittest.mock.patch.object(
        loss_fn, "_sample_negatives", return_value=far_negatives
    ):
        terms = loss_fn.training_losses(x)

    assert terms["cd_loss"].item() == pytest.approx(
        2.0 * terms["cd_value"].item(), rel=1e-5
    )
    assert terms["cd_loss"].item() < -0.02


def test_em_training_losses_keys():
    loss_fn = make_loss(lambda_cd=2.0)
    terms = loss_fn.training_losses(torch.randn(8, 2))
    assert set(terms) == {"loss", "flow_loss", "cd_loss", "cd_value", "negatives"}

    warmup = make_loss(lambda_cd=0.0)
    terms = warmup.training_losses(torch.randn(8, 2))
    assert set(terms) == {"loss", "flow_loss", "cd_loss"}
    assert terms["cd_loss"].item() == 0.0


# Gradient flow and training
# ==========================


@pytest.mark.parametrize("lambda_cd", [0.0, 2.0])
def test_em_gradient_flow(lambda_cd):
    model = MLPPotential()
    loss_fn = make_loss(model=model, lambda_cd=lambda_cd)
    loss = loss_fn(torch.randn(8, 2))
    loss.backward()

    grads = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_em_smoke_training():
    """Short two-phase run on a Gaussian mixture stays finite."""
    from torchebm.datasets import GaussianMixtureDataset

    data = GaussianMixtureDataset(
        n_samples=256, n_components=4, std=0.05, seed=0
    ).get_data()
    model = MLPPotential()
    loss_fn = make_loss(
        model=model, coupling="ot", lambda_cd=0.0, n_langevin_steps=5
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(50):
        if step == 30:
            loss_fn.lambda_cd = 2.0
        batch = data[torch.randint(len(data), (32,))]
        loss = loss_fn(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        assert torch.isfinite(loss)


# Constructor validation and misc
# ===============================


def test_em_invalid_args():
    model = QuadraticPotential()
    with pytest.raises(ValueError, match="noise_fraction"):
        EnergyMatchingLoss(model=model, noise_fraction=1.5)
    with pytest.raises(ValueError, match="cd_trim_fraction"):
        EnergyMatchingLoss(model=model, cd_trim_fraction=1.0)
    with pytest.raises(ValueError, match="cd_clamp"):
        EnergyMatchingLoss(model=model, cd_clamp=-0.1)
    with pytest.raises(ValueError, match="langevin_dt"):
        EnergyMatchingLoss(model=model, langevin_dt=0.0)


def test_em_scheduled_params():
    loss_fn = make_loss(
        sigma=ConstantScheduler(0.05),
        lambda_cd=ConstantScheduler(1.0),
    )
    assert loss_fn.sigma == pytest.approx(0.05)
    assert loss_fn.lambda_cd == pytest.approx(1.0)
    loss_fn.sigma = 0.2
    assert loss_fn.sigma == pytest.approx(0.2)


def test_em_auto_built_sampler_uses_dt():
    loss_fn = make_loss(langevin_dt=0.02)
    assert isinstance(loss_fn.sampler, LangevinDynamics)
    assert loss_fn.sampler.get_scheduled_value("step_size") == pytest.approx(0.02)


def test_em_repr():
    text = repr(make_loss())
    assert "EnergyMatchingLoss" in text
    assert "ExactOTCoupling" in text


# Arbitrary source (x0 override)
# ==============================


def test_em_x0_override_scalar_and_finite(device):
    loss_fn = make_loss(coupling="ot", lambda_cd=2.0, device=device)
    x1 = torch.randn(16, 2, device=device)
    x0 = torch.randn(16, 2, device=device) + 5.0
    loss = loss_fn(x1, x0=x0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_em_x0_shape_mismatch_raises():
    loss_fn = make_loss(lambda_cd=0.0)
    with pytest.raises(ValueError, match="x0 shape"):
        loss_fn(torch.randn(16, 2), x0=torch.randn(8, 2))


def test_em_negatives_start_from_source():
    # With a distinctive far-away source and only 5 quadratic-descent steps,
    # sweep-chain negatives must still sit near the source, not near N(0, I).
    model = QuadraticPotential()
    sampler = CountingSampler(model=model, step_size=0.01)
    loss_fn = make_loss(
        model=model, sampler=sampler, lambda_cd=2.0, noise_fraction=1.0
    )
    x1 = torch.zeros(12, 2)
    x0 = torch.full((12, 2), 30.0)
    terms = loss_fn.training_losses(x1, x0=x0)
    assert terms["negatives"].mean() > 20.0
    assert sampler.calls[0]["batch"] == 12


def test_em_x0_gradient_flow():
    model = MLPPotential()
    loss_fn = make_loss(model=model, lambda_cd=0.0)
    loss = loss_fn(torch.randn(16, 2), x0=torch.randn(16, 2) + 3.0)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum() > 0 for g in grads)


# Weighted couplings (per-pair weights on CouplingResult)
# =======================================================


class _StubWeightedCoupling(BaseCoupling):
    """Identity pairing that attaches fixed per-pair weights."""

    def __init__(self, weights):
        self._weights = weights

    def couple(self, x0, x1=None, **kwargs):
        return CouplingResult(x0, x1, weights=self._weights.to(x0.device))


def test_em_flow_loss_consumes_coupling_weights():
    """With half-zero weights, the flow loss equals the weighted mean."""
    torch.manual_seed(0)
    batch = 8
    weights = torch.tensor([1.0] * 4 + [0.0] * 4)
    model = QuadraticPotential()

    x1 = torch.randn(batch, 2)
    x0 = torch.randn(batch, 2)

    weighted = make_loss(
        model=model, coupling=_StubWeightedCoupling(weights), lambda_cd=0.0
    )
    uniform = make_loss(model=model, coupling="independent", lambda_cd=0.0)

    with unittest.mock.patch(
        "torch.rand", side_effect=lambda *a, **k: torch.full(a, 0.5, **k)
    ), unittest.mock.patch("torch.randn_like", side_effect=lambda x: torch.zeros_like(x)):
        lw = weighted.training_losses(x1, x0=x0)["flow_loss"]
        lu_terms = []
        for i in range(4):  # weighted mean over the first half only
            lu_terms.append(
                uniform.training_losses(x1[i : i + 1], x0=x0[i : i + 1])["flow_loss"]
            )
    assert lw.item() == pytest.approx(torch.stack(lu_terms).mean().item(), rel=1e-5)


def test_em_uniform_weights_match_plain_mean():
    """All-ones weights reproduce the unweighted flow loss exactly."""
    torch.manual_seed(0)
    model = QuadraticPotential()
    x1 = torch.randn(8, 2)
    x0 = torch.randn(8, 2)

    ones = make_loss(
        model=model, coupling=_StubWeightedCoupling(torch.ones(8)), lambda_cd=0.0
    )
    plain = make_loss(model=model, coupling="independent", lambda_cd=0.0)

    with unittest.mock.patch(
        "torch.rand", side_effect=lambda *a, **k: torch.full(a, 0.5, **k)
    ), unittest.mock.patch("torch.randn_like", side_effect=lambda x: torch.zeros_like(x)):
        lw = ones.training_losses(x1, x0=x0)["flow_loss"]
        lp = plain.training_losses(x1, x0=x0)["flow_loss"]
    assert lw.item() == pytest.approx(lp.item(), rel=1e-6)

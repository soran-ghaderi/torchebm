r"""Library-wide explicit-RNG contract (GitHub #245).

Guards that every stochastic component accepts a `torch.Generator`, that the
same seed reproduces results exactly, that independent seeds decorrelate, and
that `generator=None` still consumes the global RNG so existing seeded code is
unaffected.
"""

import pytest
import torch
import torch.nn as nn

from torchebm.core import BaseModel
from torchebm.couplings import SinkhornCoupling
from torchebm.losses import (
    ContrastiveDivergence,
    DenoisingScoreMatching,
    EnergyMatchingLoss,
    EquilibriumMatchingLoss,
    ScoreMatching,
    SlicedScoreMatching,
)
from torchebm.samplers import (
    FlowSampler,
    GradientDescentSampler,
    HamiltonianMonteCarlo,
    LangevinDynamics,
    NesterovSampler,
)

DIM = 2
BATCH = 8


class QuadraticEnergy(BaseModel):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, x):
        return self.scale * (x**2).sum(dim=-1)


class VelocityField(nn.Module):
    def __init__(self, dim=DIM):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x, t=None):
        return self.lin(x)


def gen(seed, device="cpu"):
    return torch.Generator(device=device).manual_seed(seed)


def make_sampler(name):
    if name == "langevin":
        return LangevinDynamics(model=QuadraticEnergy(), step_size=1e-2)
    if name == "hmc":
        return HamiltonianMonteCarlo(model=QuadraticEnergy(), step_size=1e-2)
    if name == "gradient_descent":
        return GradientDescentSampler(model=QuadraticEnergy(), step_size=1e-2)
    if name == "nesterov":
        return NesterovSampler(model=QuadraticEnergy(), step_size=1e-2)
    if name == "flow_sde":
        return FlowSampler(model=VelocityField(), interpolant="linear", mode="sde")
    raise AssertionError(name)


SAMPLERS = ["langevin", "hmc", "gradient_descent", "nesterov", "flow_sde"]


@pytest.mark.parametrize("name", SAMPLERS)
def test_sampler_same_generator_seed_is_reproducible(name):
    sampler = make_sampler(name)
    kwargs = dict(dim=DIM, n_samples=BATCH, n_steps=5)
    a = sampler.sample(**kwargs, generator=gen(0))
    b = sampler.sample(**kwargs, generator=gen(0))
    assert torch.equal(a, b)


@pytest.mark.parametrize("name", SAMPLERS)
def test_sampler_different_generator_seeds_decorrelate(name):
    sampler = make_sampler(name)
    kwargs = dict(dim=DIM, n_samples=BATCH, n_steps=5)
    a = sampler.sample(**kwargs, generator=gen(0))
    b = sampler.sample(**kwargs, generator=gen(1))
    assert not torch.equal(a, b)


@pytest.mark.parametrize("name", SAMPLERS)
def test_sampler_generator_none_uses_global_rng(name):
    """generator=None must keep consuming the global RNG (back-compat)."""
    sampler = make_sampler(name)
    kwargs = dict(dim=DIM, n_samples=BATCH, n_steps=5)
    torch.manual_seed(1234)
    a = sampler.sample(**kwargs)
    torch.manual_seed(1234)
    b = sampler.sample(**kwargs)
    assert torch.equal(a, b)


@pytest.mark.parametrize("name", ["langevin", "hmc", "flow_sde"])
def test_sampler_generator_does_not_disturb_global_rng(name):
    """An explicit generator must leave the global stream untouched."""
    sampler = make_sampler(name)
    kwargs = dict(dim=DIM, n_samples=BATCH, n_steps=5)
    torch.manual_seed(7)
    baseline = torch.randn(4)
    torch.manual_seed(7)
    sampler.sample(**kwargs, generator=gen(0))
    assert torch.equal(baseline, torch.randn(4))


def make_loss(name):
    if name == "cd":
        model = QuadraticEnergy()
        return ContrastiveDivergence(
            model=model,
            sampler=LangevinDynamics(model=model, step_size=1e-2),
            k_steps=3,
            add_noise_to_real=True,
            noise_scale=0.1,
        )
    if name == "cd_persistent":
        model = QuadraticEnergy()
        return ContrastiveDivergence(
            model=model,
            sampler=LangevinDynamics(model=model, step_size=1e-2),
            k_steps=2,
            persistent=True,
            buffer_size=32,
            new_sample_ratio=0.5,
        )
    if name == "dsm":
        return DenoisingScoreMatching(model=QuadraticEnergy(), noise_scale=0.1)
    if name == "sm_approx":
        return ScoreMatching(model=QuadraticEnergy(), hessian_method="approx")
    if name == "ssm":
        return SlicedScoreMatching(model=QuadraticEnergy(), n_projections=3)
    if name == "eqm":
        return EquilibriumMatchingLoss(model=VelocityField(), energy_type="none")
    if name == "em":
        return EnergyMatchingLoss(model=QuadraticEnergy(), n_langevin_steps=3)
    raise AssertionError(name)


LOSSES = ["cd", "cd_persistent", "dsm", "sm_approx", "ssm", "eqm", "em"]


def build_loss(name):
    """Fresh loss with identical weights: replay buffers and schedulers carry
    state across calls, so reproducibility is only meaningful per instance."""
    torch.manual_seed(0)
    return make_loss(name)


def loss_value(loss_fn, x, generator):
    out = loss_fn(x, generator=generator)
    return out[0] if isinstance(out, tuple) else out


@pytest.mark.parametrize("name", LOSSES)
def test_loss_same_generator_seed_is_reproducible(name):
    x = torch.randn(BATCH, DIM)
    a = loss_value(build_loss(name), x, gen(0))
    b = loss_value(build_loss(name), x, gen(0))
    assert torch.allclose(a, b)


@pytest.mark.parametrize("name", LOSSES)
def test_loss_different_generator_seeds_decorrelate(name):
    x = torch.randn(BATCH, DIM)
    a = loss_value(build_loss(name), x, gen(0))
    b = loss_value(build_loss(name), x, gen(1))
    assert not torch.allclose(a, b)


@pytest.mark.parametrize("name", LOSSES)
def test_loss_generator_none_uses_global_rng(name):
    x = torch.randn(BATCH, DIM)
    fn_a = build_loss(name)
    torch.manual_seed(99)
    a = loss_value(fn_a, x, None)
    fn_b = build_loss(name)
    torch.manual_seed(99)
    b = loss_value(fn_b, x, None)
    assert torch.allclose(a, b)


def test_sinkhorn_coupling_generator_is_reproducible():
    coupling = SinkhornCoupling(reg=0.5, n_iters=10)
    x0 = torch.randn(16, DIM)
    x1 = torch.randn(16, DIM)
    _, a = coupling(x0, x1, generator=gen(0))
    _, b = coupling(x0, x1, generator=gen(0))
    _, c = coupling(x0, x1, generator=gen(1))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_deterministic_coupling_accepts_generator():
    from torchebm.couplings import ExactOTCoupling, IndependentCoupling

    x0 = torch.randn(8, DIM)
    x1 = torch.randn(8, DIM)
    for coupling in (IndependentCoupling(), ExactOTCoupling()):
        _, paired = coupling(x0, x1, generator=gen(0))
        _, again = coupling(x0, x1, generator=gen(1))
        assert torch.equal(paired, again)


def test_replay_buffer_init_honors_generator():
    """Two persistent CD losses seeded alike start from identical buffers."""
    losses = [make_loss("cd_persistent") for _ in range(2)]
    for loss_fn in losses:
        loss_fn.initialize_buffer((DIM,), generator=gen(5))
    assert torch.equal(losses[0].replay_buffer, losses[1].replay_buffer)

    other = make_loss("cd_persistent")
    other.initialize_buffer((DIM,), generator=gen(6))
    assert not torch.equal(losses[0].replay_buffer, other.replay_buffer)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_generator_device_mismatch_raises():
    sampler = LangevinDynamics(
        model=QuadraticEnergy(), step_size=1e-2, device="cuda"
    ).to("cuda")
    with pytest.raises(RuntimeError, match="[Gg]enerator"):
        sampler.sample(dim=DIM, n_samples=BATCH, n_steps=2, generator=gen(0))

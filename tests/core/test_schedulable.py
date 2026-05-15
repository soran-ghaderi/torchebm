"""Tests for the Schedulable mixin and lib-wide scheduler integration."""

import pytest
import torch

from torchebm.core import (
    BaseScheduler,
    ConstantScheduler,
    GaussianModel,
    LinearScheduler,
    Schedulable,
    TorchEBMModule,
    WarmupScheduler,
)
from torchebm.losses import (
    ContrastiveDivergence,
    DenoisingScoreMatching,
    SlicedScoreMatching,
)
from torchebm.samplers import GradientDescentSampler, LangevinDynamics


class _MiniSchedulable(Schedulable, TorchEBMModule):
    """Minimal concrete Schedulable for unit-testing the mixin."""

    pass


def test_register_param_accepts_float():
    m = _MiniSchedulable()
    m._register_param("eta", 0.1)
    assert m.get_scheduled_value("eta") == pytest.approx(0.1)
    assert isinstance(m.schedulers["eta"], ConstantScheduler)


def test_register_param_accepts_scheduler():
    m = _MiniSchedulable()
    sched = LinearScheduler(start_value=1.0, end_value=0.0, n_steps=10)
    m._register_param("eta", sched)
    assert m.schedulers["eta"] is sched
    assert m.get_scheduled_value("eta") == pytest.approx(1.0)


def test_register_param_positive_validation():
    m = _MiniSchedulable()
    with pytest.raises(ValueError, match="eta must be positive"):
        m._register_param("eta", -0.1, positive=True)
    with pytest.raises(ValueError, match="eta must be positive"):
        m._register_param("eta", 0.0, positive=True)
    m._register_param("eta", 0.1, positive=True)
    assert m.get_scheduled_value("eta") == pytest.approx(0.1)


def test_get_scheduled_value_unknown_key_raises():
    m = _MiniSchedulable()
    with pytest.raises(KeyError, match="No scheduler registered"):
        m.get_scheduled_value("missing")


def test_step_and_reset_advance_local_scheduler():
    m = _MiniSchedulable()
    m._register_param("eta", LinearScheduler(start_value=1.0, end_value=0.0, n_steps=4))
    assert m.get_scheduled_value("eta") == pytest.approx(1.0)
    m.step_schedulers()
    assert m.get_scheduled_value("eta") == pytest.approx(0.75)
    m.step_schedulers()
    assert m.get_scheduled_value("eta") == pytest.approx(0.5)
    m.reset_schedulers()
    assert m.get_scheduled_value("eta") == pytest.approx(1.0)


def test_step_schedulers_recurses_into_submodules():
    """A loss owning a sampler should advance both subtree schedulers from one call on the root."""
    energy = GaussianModel(mean=torch.zeros(2), cov=torch.eye(2))
    sampler = LangevinDynamics(
        model=energy,
        step_size=LinearScheduler(start_value=0.1, end_value=0.001, n_steps=10),
        noise_scale=0.5,
    )
    loss = ContrastiveDivergence(model=energy, sampler=sampler, k_steps=2)
    # Register a scheduler directly on the loss (any name works for the recursion test).
    loss.register_scheduler(
        "alpha", LinearScheduler(start_value=0.5, end_value=0.0, n_steps=10)
    )

    assert loss.sampler.get_scheduled_value("step_size") == pytest.approx(0.1)
    assert loss.get_scheduled_value("alpha") == pytest.approx(0.5)

    loss.step_schedulers()

    # Both the loss's own scheduler AND the sampler's step_size scheduler
    # should advance from a single call on the root.
    assert loss.sampler.get_scheduled_value("step_size") == pytest.approx(
        0.1 + (0.001 - 0.1) / 10
    )
    assert loss.get_scheduled_value("alpha") == pytest.approx(0.5 - 0.5 / 10)


def test_dsm_noise_scale_drives_perturbation():
    """DSM with a scheduled noise_scale: the property reflects the scheduled value."""
    energy = GaussianModel(mean=torch.zeros(2), cov=torch.eye(2))
    dsm = DenoisingScoreMatching(
        model=energy,
        noise_scale=LinearScheduler(start_value=1.0, end_value=0.01, n_steps=4),
    )
    assert dsm.noise_scale == pytest.approx(1.0)
    dsm.step_schedulers()
    assert dsm.noise_scale == pytest.approx(0.7525)
    dsm.reset_schedulers()
    assert dsm.noise_scale == pytest.approx(1.0)


def test_property_setter_swaps_scheduler():
    """Assigning a float to a schedulable attribute swaps in a ConstantScheduler."""
    energy = GaussianModel(mean=torch.zeros(2), cov=torch.eye(2))
    ssm = SlicedScoreMatching(model=energy, regularization_strength=0.1)
    assert ssm.regularization_strength == pytest.approx(0.1)
    ssm.regularization_strength = 0.0
    assert ssm.regularization_strength == pytest.approx(0.0)


def test_warmup_scheduler_pure_compute_value():
    """`_compute_value` must not mutate the wrapped main_scheduler."""
    main = LinearScheduler(start_value=1.0, end_value=0.0, n_steps=10)
    warmup = WarmupScheduler(main_scheduler=main, warmup_steps=3, warmup_init_factor=0.1)

    # main was reset by warmup constructor
    assert main.step_count == 0
    assert main.current_value == pytest.approx(1.0)

    # Pre-warmup: _compute_value should not touch main
    warmup.step_count = 1
    warmup._compute_value()
    assert main.step_count == 0
    assert main.current_value == pytest.approx(1.0)


def test_warmup_scheduler_lockstep_drives_main_post_warmup():
    """After warmup, step() advances main_scheduler in lockstep."""
    main = LinearScheduler(start_value=1.0, end_value=0.0, n_steps=10)
    warmup = WarmupScheduler(main_scheduler=main, warmup_steps=2, warmup_init_factor=0.5)

    # Three warmup steps to cover [0, 1, 2 -> end of warmup]; then enter main phase.
    # warmup phase: linear from 0.5 -> 1.0 over 2 steps
    v = warmup.step()  # step 1: progress=1/2 -> 0.75
    assert v == pytest.approx(0.75)
    assert main.step_count == 0  # not yet driven
    v = warmup.step()  # step 2: end of warmup, value=1.0
    assert v == pytest.approx(1.0)
    assert main.step_count == 0
    v = warmup.step()  # step 3: enter main, main.step() to 1, value=main(1)=0.9
    assert v == pytest.approx(0.9)
    assert main.step_count == 1


def test_sample_reset_schedulers_false_persists_state():
    """Successive sample() calls with reset_schedulers=False keep advancing."""
    energy = GaussianModel(mean=torch.zeros(2), cov=torch.eye(2))
    sampler = GradientDescentSampler(
        model=energy,
        step_size=LinearScheduler(start_value=0.1, end_value=0.0, n_steps=20),
    )
    sampler.sample(dim=2, n_steps=5, reset_schedulers=False)
    assert sampler.schedulers["step_size"].step_count == 5
    sampler.sample(dim=2, n_steps=5, reset_schedulers=False)
    assert sampler.schedulers["step_size"].step_count == 10


def test_sample_reset_schedulers_true_resets_each_call():
    """Default behavior: each sample() call starts at step 0."""
    energy = GaussianModel(mean=torch.zeros(2), cov=torch.eye(2))
    sampler = GradientDescentSampler(
        model=energy,
        step_size=LinearScheduler(start_value=0.1, end_value=0.0, n_steps=20),
    )
    sampler.sample(dim=2, n_steps=5)
    assert sampler.schedulers["step_size"].step_count == 5
    sampler.sample(dim=2, n_steps=5)  # default reset=True
    assert sampler.schedulers["step_size"].step_count == 5


def test_sample_thin_reduces_trajectory_length():
    """`thin > 1` keeps every thin-th sample; trajectory length = n_steps // thin."""
    from torchebm.samplers import LangevinDynamics

    energy = GaussianModel(mean=torch.zeros(2), cov=torch.eye(2))
    sampler = LangevinDynamics(model=energy, step_size=0.01, noise_scale=0.1)
    traj = sampler.sample(dim=2, n_samples=4, n_steps=20, thin=5, return_trajectory=True)
    # n_kept = 20 // 5 = 4
    assert traj.shape == (4, 4, 2)


def test_sample_thin_invalid_raises():
    """`thin < 1` raises ValueError."""
    from torchebm.samplers import LangevinDynamics

    energy = GaussianModel(mean=torch.zeros(2), cov=torch.eye(2))
    sampler = LangevinDynamics(model=energy, step_size=0.01, noise_scale=0.1)
    with pytest.raises(ValueError, match="thin must be >= 1"):
        sampler.sample(dim=2, n_steps=10, thin=0)


def test_diagnostics_dict_contract_langevin():
    """Langevin returns Dict[str, Tensor] with documented keys and shapes."""
    from torchebm.samplers import LangevinDynamics

    energy = GaussianModel(mean=torch.zeros(2), cov=torch.eye(2))
    sampler = LangevinDynamics(model=energy, step_size=0.01, noise_scale=0.1)
    n_steps = 10
    out, diag = sampler.sample(
        dim=2, n_samples=4, n_steps=n_steps, return_diagnostics=True
    )
    assert isinstance(diag, dict)
    assert set(diag) == {"mean", "var", "energy"}
    assert diag["mean"].shape == (n_steps, 2)
    assert diag["var"].shape == (n_steps, 2)
    assert diag["energy"].shape == (n_steps,)


def test_diagnostics_dict_contract_hmc():
    """HMC returns Dict[str, Tensor] including acceptance_rate."""
    from torchebm.samplers import HamiltonianMonteCarlo

    energy = GaussianModel(mean=torch.zeros(2), cov=torch.eye(2))
    sampler = HamiltonianMonteCarlo(model=energy, step_size=0.05, n_leapfrog_steps=3)
    n_steps = 5
    out, diag = sampler.sample(
        dim=2, n_samples=4, n_steps=n_steps, return_diagnostics=True
    )
    assert set(diag) == {"mean", "var", "energy", "acceptance_rate"}
    assert diag["acceptance_rate"].shape == (n_steps,)
    assert torch.all(diag["acceptance_rate"] >= 0)
    assert torch.all(diag["acceptance_rate"] <= 1)


def test_diagnostics_thin_matches_kept_length():
    """Diagnostic tensors are sized to n_steps // thin."""
    from torchebm.samplers import LangevinDynamics

    energy = GaussianModel(mean=torch.zeros(2), cov=torch.eye(2))
    sampler = LangevinDynamics(model=energy, step_size=0.01, noise_scale=0.1)
    out, diag = sampler.sample(
        dim=2, n_samples=4, n_steps=20, thin=5, return_diagnostics=True
    )
    assert diag["mean"].shape == (4, 2)
    assert diag["energy"].shape == (4,)

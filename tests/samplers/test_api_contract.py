r"""Cross-sampler API contract guard.

Every sampler must honor the `BaseSampler` contract: the shared `sample()`
signature prefix, the ctor ordering (model -> algorithm -> dtype -> device
-> integrator), tensor-or-(tensor, dict) returns, trajectory/thin shapes,
and int-or-tuple `dim` synthesis. New samplers get a `Case` entry here.
"""

import dataclasses
import inspect
from typing import Callable

import pytest
import torch
import torch.nn as nn

from torchebm.core import GaussianModel
from torchebm.samplers import (
    FlowSampler,
    GradientDescentSampler,
    HamiltonianMonteCarlo,
    LangevinDynamics,
    NesterovSampler,
    RiemannianManifoldHMC,
)

_SAMPLE_PREFIX = (
    "x",
    "dim",
    "n_steps",
    "n_samples",
    "thin",
    "return_trajectory",
    "return_diagnostics",
    "reset_schedulers",
)

_SAMPLE_DEFAULTS = {
    "x": None,
    "dim": None,
    "n_samples": 1,
    "thin": 1,
    "return_trajectory": False,
    "return_diagnostics": False,
    "reset_schedulers": True,
}


class ConstantVelocity(nn.Module):
    def forward(self, x, t, **kwargs):
        return torch.ones_like(x)


def _gaussian(dim=2):
    return GaussianModel(mean=torch.zeros(dim), cov=torch.eye(dim))


def _identity_metric(x):
    dim = x.shape[-1]
    eye = torch.eye(dim, dtype=x.dtype, device=x.device)
    return eye.expand(x.shape[0], dim, dim).contiguous()


@dataclasses.dataclass
class Case:
    name: str
    factory: Callable
    infers_dim: bool = False
    supports_trajectory: bool = True
    tuple_dim: tuple = (2,)
    has_integrator: bool = True
    allows_var_keyword: bool = False


CASES = [
    Case(
        "langevin",
        lambda: LangevinDynamics(_gaussian(), step_size=0.01),
    ),
    Case(
        "hmc",
        lambda: HamiltonianMonteCarlo(_gaussian(), step_size=0.05, n_leapfrog_steps=3),
        infers_dim=True,
        tuple_dim=None,
    ),
    Case(
        "rmhmc",
        lambda: RiemannianManifoldHMC(
            _gaussian(),
            metric_fn=_identity_metric,
            step_size=0.05,
            n_leapfrog_steps=2,
        ),
        infers_dim=True,
        tuple_dim=None,
    ),
    Case(
        "gd",
        lambda: GradientDescentSampler(_gaussian(), step_size=0.05),
        has_integrator=False,
    ),
    Case(
        "nesterov",
        lambda: NesterovSampler(_gaussian(), step_size=0.05),
        has_integrator=False,
    ),
    Case(
        "flow_ode_fixed",
        lambda: FlowSampler(ConstantVelocity(), integrator="euler"),
        tuple_dim=(2, 3),
        allows_var_keyword=True,
    ),
    Case(
        "flow_sde",
        lambda: FlowSampler(
            ConstantVelocity(),
            mode="sde",
            diffusion_form="constant",
            diffusion_norm=0.0,
            last_step=None,
            integrator="euler",
        ),
        tuple_dim=(2, 3),
        allows_var_keyword=True,
    ),
    Case(
        "flow_ode_adaptive",
        lambda: FlowSampler(ConstantVelocity()),
        supports_trajectory=False,
        tuple_dim=(2, 3),
        allows_var_keyword=True,
    ),
]

_IDS = [case.name for case in CASES]


@pytest.fixture(params=CASES, ids=_IDS)
def case(request):
    return request.param


class TestSignatures:

    def test_sample_signature_prefix(self, case):
        sampler_cls = type(case.factory())
        params = list(inspect.signature(sampler_cls.sample).parameters.values())
        names = [p.name for p in params]
        assert names[0] == "self"
        assert tuple(names[1 : 1 + len(_SAMPLE_PREFIX)]) == _SAMPLE_PREFIX

        kinds = {p.name: p.kind for p in params}
        assert not any(
            kind is inspect.Parameter.VAR_POSITIONAL for kind in kinds.values()
        ), f"{sampler_cls.__name__}.sample must not take *args"

        # Conditioning contract: every sampler exposes an explicit keyword-only
        # `model_kwargs` dict param (default None), never bare **kwargs.
        assert "model_kwargs" in kinds, (
            f"{sampler_cls.__name__}.sample must accept model_kwargs"
        )
        assert (
            kinds["model_kwargs"] is inspect.Parameter.KEYWORD_ONLY
        ), f"{sampler_cls.__name__}.sample model_kwargs must be keyword-only"
        model_kwargs_default = {p.name: p.default for p in params}["model_kwargs"]
        assert model_kwargs_default is None

        var_keyword = [
            p.name for p in params if p.kind is inspect.Parameter.VAR_KEYWORD
        ]
        if case.allows_var_keyword:
            # FlowSampler keeps a var-keyword for one deprecation cycle: bare
            # conditioning kwargs still work but warn (see the explicit
            # model_kwargs param above). Renamed to make the legacy status clear.
            assert var_keyword == ["legacy_model_kwargs"]
        else:
            assert (
                not var_keyword
            ), f"{sampler_cls.__name__}.sample must not take **kwargs"

        defaults = {p.name: p.default for p in params}
        for name, expected in _SAMPLE_DEFAULTS.items():
            assert (
                defaults[name] == expected
            ), f"{sampler_cls.__name__}.sample {name} default"
        # n_steps default is sampler-specific but must exist (int or None).
        assert defaults["n_steps"] is None or isinstance(defaults["n_steps"], int)

    def test_ctor_signature(self, case):
        sampler_cls = type(case.factory())
        params = list(inspect.signature(sampler_cls.__init__).parameters.values())
        names = [p.name for p in params]
        assert names[1] == "model"
        assert not any(
            p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for p in params
        ), f"{sampler_cls.__name__}.__init__ must not take *args/**kwargs"

        assert names.index("dtype") < names.index("device")
        if case.has_integrator:
            assert names[-1] == "integrator"
            assert names.index("device") < names.index("integrator")
        else:
            assert "integrator" not in names


class TestSampleContract:

    def test_returns_tensor(self, case):
        sampler = case.factory()
        samples = sampler.sample(n_samples=4, dim=2, n_steps=6)
        assert isinstance(samples, torch.Tensor)
        assert samples.shape == (4, 2)
        assert samples.dtype == sampler.dtype
        assert samples.device == sampler.device

    def test_diagnostics_contract(self, case):
        sampler = case.factory()
        result = sampler.sample(n_samples=4, dim=2, n_steps=6, return_diagnostics=True)
        assert isinstance(result, tuple) and len(result) == 2
        samples, diagnostics = result
        assert isinstance(samples, torch.Tensor)
        assert isinstance(diagnostics, dict)
        n_kept = 6 if case.supports_trajectory else 1
        for key, value in diagnostics.items():
            assert isinstance(value, torch.Tensor), key
            assert value.shape[0] == n_kept, key

    @pytest.mark.parametrize("thin", [1, 2, 3])
    def test_trajectory_and_thin(self, case, thin):
        sampler = case.factory()
        if not case.supports_trajectory:
            with pytest.raises(NotImplementedError):
                sampler.sample(
                    n_samples=4,
                    dim=2,
                    n_steps=6,
                    thin=thin,
                    return_trajectory=True,
                )
            return
        trajectory = sampler.sample(
            n_samples=4, dim=2, n_steps=6, thin=thin, return_trajectory=True
        )
        assert trajectory.shape == (4, 6 // thin, 2)

    def test_thin_below_one_raises(self, case):
        sampler = case.factory()
        with pytest.raises(ValueError, match="thin"):
            sampler.sample(n_samples=2, dim=2, n_steps=4, thin=0)

    def test_x_none_dim_none(self, case):
        sampler = case.factory()
        if case.infers_dim:
            samples = sampler.sample(n_samples=2, n_steps=4)
            assert samples.shape == (2, 2)
        else:
            with pytest.raises(ValueError, match="dim"):
                sampler.sample(n_samples=2, n_steps=4)

    def test_tuple_dim_synthesis(self, case):
        if case.tuple_dim is None:
            pytest.skip("sampler is restricted to flat 2-D states")
        sampler = case.factory()
        samples = sampler.sample(n_samples=3, dim=case.tuple_dim, n_steps=4)
        assert samples.shape == (3, *case.tuple_dim)

    def test_x_passthrough_shape(self, case):
        sampler = case.factory()
        x = torch.zeros(5, 2)
        samples = sampler.sample(x=x, n_steps=4)
        assert samples.shape == (5, 2)

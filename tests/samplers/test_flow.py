import math

import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

from torchebm.samplers.flow import FlowSampler, PredictionType
from torchebm.interpolants import LinearInterpolant


class MockModel(nn.Module):
    def __init__(self, mode="constant", val=0.0):
        super().__init__()
        self.mode = mode
        self.val = val

    def forward(self, x, t, **kwargs):
        if self.mode == "constant":
            return torch.full_like(x, self.val)
        elif self.mode == "linear":
            return x * self.val
        return torch.zeros_like(x)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dtype():
    return torch.float32


class TestFlowSampler:

    def test_initialization(self, device, dtype):
        model = MockModel()
        sampler = FlowSampler(
            model,
            interpolant="linear",
            prediction="velocity",
            device=device,
            dtype=dtype,
        )
        assert sampler.model is model
        assert sampler.mode == "ode"
        assert isinstance(sampler.interpolant, LinearInterpolant)
        assert sampler.prediction_type == PredictionType.VELOCITY
        assert sampler.device == device
        assert sampler.dtype == dtype
        assert sampler.last_step is None
        assert not hasattr(sampler, "sample_ode")
        assert not hasattr(sampler, "sample_sde")

    def test_sde_defaults(self, device, dtype):
        sampler = FlowSampler(MockModel(), mode="sde", device=device, dtype=dtype)
        assert sampler.diffusion_form == "SBDM"
        assert sampler.diffusion_norm == 1.0
        assert sampler.last_step == "Mean"
        assert sampler.last_step_size == 0.04

    def test_get_drift_velocity(self, device, dtype):
        # v(x,t) = 2.0
        model = MockModel(mode="constant", val=2.0).to(device)
        sampler = FlowSampler(
            model,
            interpolant="linear",
            prediction="velocity",
            device=device,
            dtype=dtype,
        )
        drift_fn = sampler._get_drift()

        x = torch.zeros(1, 1, device=device, dtype=dtype)
        t = torch.zeros(1, device=device, dtype=dtype)

        out = drift_fn(x, t)
        assert_close(out, torch.full_like(x, 2.0))

    def test_ode_euler_constant_velocity(self, device, dtype):
        # dx/dt = 1: x(0) = 0 -> x(1) = 1.
        model = MockModel(mode="constant", val=1.0).to(device)
        sampler = FlowSampler(
            model,
            interpolant="linear",
            prediction="velocity",
            sample_eps=0.0,
            device=device,
            dtype=dtype,
            integrator="euler",
        )
        z = torch.zeros(1, 1, device=device, dtype=dtype)
        samples = sampler.sample(x=z, n_steps=10)
        assert_close(samples, torch.full_like(samples, 1.0), atol=1e-5, rtol=1e-5)

    def test_ode_euler_linear_velocity(self, device, dtype):
        # dx/dt = x, x(0) = 1 -> x(1) = e; Euler converges O(dt).
        model = MockModel(mode="linear", val=1.0).to(device)
        sampler = FlowSampler(
            model,
            interpolant="linear",
            prediction="velocity",
            sample_eps=0.0,
            device=device,
            dtype=dtype,
            integrator="euler",
        )
        z = torch.ones(1, 1, device=device, dtype=dtype)
        samples = sampler.sample(x=z, n_steps=1000)
        expected = torch.full_like(samples, math.e)
        assert_close(samples, expected, atol=0.01, rtol=0.01)

    def test_ode_reverse_fixed_step(self, device, dtype):
        # dx/dt = 1 integrated from t=1 to t=0: x -> x - 1.
        model = MockModel(mode="constant", val=1.0).to(device)
        sampler = FlowSampler(
            model,
            interpolant="linear",
            prediction="velocity",
            sample_eps=0.0,
            reverse=True,
            device=device,
            dtype=dtype,
            integrator="euler",
        )
        z = torch.zeros(1, 1, device=device, dtype=dtype)
        samples = sampler.sample(x=z, n_steps=10)
        assert_close(samples, torch.full_like(samples, -1.0))

    def test_ode_reverse_adaptive(self, device, dtype):
        # Same reverse dynamics through the adaptive dopri5 default.
        model = MockModel(mode="constant", val=1.0).to(device)
        sampler = FlowSampler(
            model,
            interpolant="linear",
            prediction="velocity",
            sample_eps=0.0,
            reverse=True,
            device=device,
            dtype=dtype,
        )
        z = torch.zeros(4, 2, device=device, dtype=dtype)
        samples = sampler.sample(x=z, n_steps=10)
        assert_close(samples, torch.full_like(samples, -1.0), atol=1e-4, rtol=1e-4)

    def test_prediction_score_drift_runs(self, device, dtype):
        interpolant = LinearInterpolant()
        model = MockModel(mode="constant", val=1.0).to(device)
        sampler = FlowSampler(
            model,
            interpolant=interpolant,
            prediction="score",
            device=device,
            dtype=dtype,
        )
        x = torch.randn(2, 2, device=device, dtype=dtype)
        drift_fn = sampler._get_drift()
        t_batch = torch.full((x.size(0),), 0.5, device=device, dtype=dtype)
        out = drift_fn(x, t_batch)
        assert out.shape == x.shape

    def test_sde_zero_dynamics_identity(self, device, dtype):
        # Zero velocity and zero diffusion: samples equal the input.
        model = MockModel(mode="constant", val=0.0).to(device)
        sampler = FlowSampler(
            model,
            mode="sde",
            interpolant="linear",
            prediction="velocity",
            diffusion_form="constant",
            diffusion_norm=0.0,
            device=device,
            dtype=dtype,
            integrator="euler",
        )
        z = torch.randn(4, 2, device=device, dtype=dtype)
        samples = sampler.sample(x=z, n_steps=5)
        assert_close(samples, z)

    def test_apply_last_step(self, device, dtype):
        # Mean step: x + drift * step_size.
        model = MockModel(mode="constant", val=1.0).to(device)
        sampler = FlowSampler(model, prediction="velocity", device=device, dtype=dtype)

        x = torch.zeros(1, 1, device=device, dtype=dtype)
        t = torch.ones(1, device=device, dtype=dtype)

        out = sampler._apply_last_step(x, t, sampler._get_drift(), "Mean", 0.1)
        assert_close(out, torch.full_like(x, 0.1))

    def test_check_interval(self):
        model = MockModel()
        # Linear, velocity, ODE -> (0, 1).
        sampler = FlowSampler(
            model, interpolant="linear", prediction="velocity", sample_eps=0.01
        )
        assert sampler._check_interval() == (0.0, 1.0)

        # SDE with SBDM and no last-step correction -> (eps, 1 - eps).
        sampler = FlowSampler(
            model,
            mode="sde",
            interpolant="linear",
            prediction="velocity",
            sample_eps=0.01,
            diffusion_form="SBDM",
            last_step=None,
        )
        t0, t1 = sampler._check_interval()
        assert t0 == 0.01
        assert t1 == 0.99

        # SDE with the default Mean last step -> t1 = 1 - last_step_size.
        sampler = FlowSampler(
            model,
            mode="sde",
            interpolant="linear",
            prediction="velocity",
            sample_eps=0.01,
        )
        t0, t1 = sampler._check_interval()
        assert t0 == 0.01
        assert t1 == pytest.approx(0.96)

        # VP keeps t0 = 0 and clips t1.
        sampler_vp = FlowSampler(
            model, interpolant="vp", prediction="score", sample_eps=0.01
        )
        t0, t1 = sampler_vp._check_interval()
        assert t0 == 0.0
        assert t1 == 0.99

    @pytest.mark.parametrize(
        "diffusion_form",
        ["constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing"],
    )
    def test_diffusion_forms(self, device, dtype, diffusion_form):
        """All supported diffusion forms run without errors."""
        model = MockModel(mode="constant", val=0.0).to(device)
        sampler = FlowSampler(
            model,
            mode="sde",
            interpolant="linear",
            prediction="velocity",
            diffusion_form=diffusion_form,
            diffusion_norm=0.0,
            device=device,
            dtype=dtype,
            integrator="euler",
        )
        z = torch.randn(4, 2, device=device, dtype=dtype)
        samples = sampler.sample(x=z, n_steps=5)
        assert samples.shape == z.shape
        assert torch.isfinite(samples).all()

    def test_invalid_diffusion_form(self, device, dtype):
        """Unknown diffusion form raises at sample time (interpolant check)."""
        model = MockModel(mode="constant", val=0.0).to(device)
        sampler = FlowSampler(
            model,
            mode="sde",
            interpolant="linear",
            diffusion_form="invalid_form",
            device=device,
            dtype=dtype,
            integrator="euler",
        )
        z = torch.randn(4, 2, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="Unknown diffusion form"):
            sampler.sample(x=z, n_steps=5)


class TestFlowConstructorValidation:

    def test_unknown_mode(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            FlowSampler(MockModel(), mode="pde")

    def test_unknown_prediction(self):
        with pytest.raises(ValueError, match="Unknown prediction"):
            FlowSampler(MockModel(), prediction="vector")

    def test_unknown_interpolant(self):
        with pytest.raises(ValueError, match="Unknown interpolant"):
            FlowSampler(MockModel(), interpolant="spline")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"diffusion_form": "SBDM"},
            {"diffusion_norm": 1.0},
            {"last_step": None},
            {"last_step": "Mean"},
            {"last_step_size": 0.04},
        ],
    )
    def test_ode_rejects_sde_only_args(self, kwargs):
        with pytest.raises(ValueError, match="only apply to mode='sde'"):
            FlowSampler(MockModel(), mode="ode", **kwargs)

    def test_sde_rejects_reverse(self):
        with pytest.raises(ValueError, match="reverse"):
            FlowSampler(MockModel(), mode="sde", reverse=True)

    def test_sde_unknown_last_step(self):
        with pytest.raises(ValueError, match="Unknown last_step"):
            FlowSampler(MockModel(), mode="sde", last_step="Midpoint")

    def test_sde_last_step_none_zeroes_size(self):
        sampler = FlowSampler(MockModel(), mode="sde", last_step=None)
        assert sampler.last_step_size == 0.0

    def test_ctor_rejects_wrong_family_instance(self, device, dtype):
        from torchebm.integrators import LeapfrogIntegrator

        with pytest.raises(TypeError, match="BaseRungeKuttaIntegrator"):
            FlowSampler(
                MockModel(),
                device=device,
                dtype=dtype,
                integrator=LeapfrogIntegrator(device=device, dtype=dtype),
            )

    def test_sde_rejects_ode_only_integrator(self, device, dtype):
        from torchebm.integrators import Dopri5Integrator

        with pytest.raises(TypeError, match="BaseSDERungeKuttaIntegrator"):
            FlowSampler(
                MockModel(),
                mode="sde",
                device=device,
                dtype=dtype,
                integrator=Dopri5Integrator(device=device, dtype=dtype),
            )

    def test_sde_rejects_ode_only_integrator_name(self, device, dtype):
        with pytest.raises(TypeError, match="BaseSDERungeKuttaIntegrator"):
            FlowSampler(
                MockModel(),
                mode="sde",
                device=device,
                dtype=dtype,
                integrator="rk4",
            )


class TestFlowSampleContract:
    """FlowSampler honors the full BaseSampler.sample contract."""

    def _euler_sampler(self, device, dtype, **kwargs):
        model = MockModel(mode="constant", val=1.0).to(device)
        return FlowSampler(
            model,
            interpolant="linear",
            prediction="velocity",
            sample_eps=0.0,
            device=device,
            dtype=dtype,
            integrator=kwargs.pop("integrator", "euler"),
            **kwargs,
        )

    def test_trajectory_and_thin(self, device, dtype):
        # dx/dt = 1 from x = 0: state at step k is k/6.
        sampler = self._euler_sampler(device, dtype)
        z = torch.zeros(4, 2, device=device, dtype=dtype)
        trajectory = sampler.sample(x=z, n_steps=6, thin=2, return_trajectory=True)
        assert trajectory.shape == (4, 3, 2)
        expected_t = torch.tensor([2 / 6, 4 / 6, 1.0], device=device, dtype=dtype)
        assert_close(
            trajectory,
            expected_t.view(1, 3, 1).expand(4, 3, 2),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_diagnostics_keys_and_values(self, device, dtype):
        sampler = self._euler_sampler(device, dtype)
        z = torch.zeros(4, 2, device=device, dtype=dtype)
        samples, diagnostics = sampler.sample(
            x=z, n_steps=6, thin=2, return_diagnostics=True
        )
        assert samples.shape == (4, 2)
        assert set(diagnostics) == {"mean", "var", "t"}
        assert diagnostics["mean"].shape == (3, 2)
        assert diagnostics["var"].shape == (3, 2)
        assert diagnostics["t"].shape == (3,)
        assert_close(
            diagnostics["t"],
            torch.tensor([2 / 6, 4 / 6, 1.0], device=device, dtype=dtype),
            atol=1e-6,
            rtol=1e-6,
        )
        # Deterministic dynamics: batch mean equals the trajectory value.
        assert_close(
            diagnostics["mean"][-1],
            torch.ones(2, device=device, dtype=dtype),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_adaptive_rejects_trajectory_and_thin(self, device, dtype):
        sampler = self._euler_sampler(device, dtype, integrator="dopri5")
        z = torch.zeros(2, 2, device=device, dtype=dtype)
        with pytest.raises(NotImplementedError, match="fixed-step"):
            sampler.sample(x=z, n_steps=10, return_trajectory=True)
        with pytest.raises(NotImplementedError, match="fixed-step"):
            sampler.sample(x=z, n_steps=10, thin=2)

    def test_adaptive_diagnostics_single_entry(self, device, dtype):
        sampler = self._euler_sampler(device, dtype, integrator="dopri5")
        z = torch.zeros(4, 2, device=device, dtype=dtype)
        samples, diagnostics = sampler.sample(x=z, n_steps=10, return_diagnostics=True)
        assert diagnostics["mean"].shape == (1, 2)
        assert diagnostics["var"].shape == (1, 2)
        assert diagnostics["t"].shape == (1,)
        assert_close(diagnostics["mean"][0], samples.mean(dim=0), atol=1e-6, rtol=1e-6)

    def test_default_n_steps_per_mode(self, device, dtype):
        ode = self._euler_sampler(device, dtype)
        z = torch.zeros(2, 2, device=device, dtype=dtype)
        _, diagnostics = ode.sample(x=z, return_diagnostics=True)
        assert diagnostics["t"].shape == (50,)

        model = MockModel(mode="constant", val=0.0).to(device)
        sde = FlowSampler(
            model,
            mode="sde",
            interpolant="linear",
            diffusion_form="constant",
            diffusion_norm=0.0,
            last_step=None,
            device=device,
            dtype=dtype,
            integrator="euler",
        )
        _, diagnostics = sde.sample(x=z, return_diagnostics=True)
        assert diagnostics["t"].shape == (250,)

    def test_sde_last_step_updates_trajectory_end(self, device, dtype):
        # dx/dt = 1 with zero diffusion: interval end 0.96, Mean step adds
        # the remaining 0.04, so the returned sample and the recorded end
        # state both reach 1.0.
        model = MockModel(mode="constant", val=1.0).to(device)
        sampler = FlowSampler(
            model,
            mode="sde",
            interpolant="linear",
            prediction="velocity",
            sample_eps=0.0,
            diffusion_form="constant",
            diffusion_norm=0.0,
            device=device,
            dtype=dtype,
            integrator="euler",
        )
        z = torch.zeros(4, 2, device=device, dtype=dtype)
        trajectory, diagnostics = sampler.sample(
            x=z, n_steps=6, return_trajectory=True, return_diagnostics=True
        )
        assert_close(
            trajectory[:, -1],
            torch.ones(4, 2, device=device, dtype=dtype),
            atol=1e-6,
            rtol=1e-6,
        )
        assert_close(
            diagnostics["t"][-1],
            torch.tensor(1.0, device=device, dtype=dtype),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_dim_tuple_synthesis(self, device, dtype):
        sampler = self._euler_sampler(device, dtype)
        samples = sampler.sample(n_samples=3, dim=(2, 4), n_steps=2)
        assert samples.shape == (3, 2, 4)

    def test_x_none_dim_none_raises(self, device, dtype):
        sampler = self._euler_sampler(device, dtype)
        with pytest.raises(ValueError, match="dim must be provided"):
            sampler.sample(n_samples=3, n_steps=2)

    def test_invalid_thin_and_n_steps(self, device, dtype):
        sampler = self._euler_sampler(device, dtype)
        z = torch.zeros(2, 2, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="thin"):
            sampler.sample(x=z, n_steps=4, thin=0)
        with pytest.raises(ValueError, match="n_steps"):
            sampler.sample(x=z, n_steps=0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"mode": "sde"},
            {"shape": (4, 2)},
            {"ode_method": "euler"},
            {"method": "euler"},
            {"atol": 1e-5},
            {"reverse": True},
            {"diffusion_form": "SBDM"},
            {"last_step": "Mean"},
            {"num_steps": 10},
            {"z": None},
        ],
    )
    def test_removed_kwargs_raise(self, device, dtype, kwargs):
        sampler = self._euler_sampler(device, dtype)
        with pytest.raises(TypeError, match="removed from"):
            sampler.sample(n_samples=2, dim=2, n_steps=4, **kwargs)


class TestFlowIntegrators:
    """Constructor integrator= injection across the registry."""

    def _sampler(self, device, dtype, **kwargs):
        model = MockModel(mode="linear", val=1.0).to(device)
        return FlowSampler(
            model,
            interpolant="linear",
            prediction="velocity",
            sample_eps=0.0,
            device=device,
            dtype=dtype,
            **kwargs,
        )

    def test_default_integrator_is_adaptive_dopri5(self, device, dtype):
        # dx/dt = x, x(0) = 1 -> x(1) = e.
        sampler = self._sampler(device, dtype)
        assert sampler.integrator.error_weights is not None
        z = torch.ones(1, 1, device=device, dtype=dtype)
        samples = sampler.sample(x=z, n_steps=50)
        expected = torch.full_like(samples, math.e)
        assert_close(samples, expected, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize(
        "name", ["bosh3", "dopri8", "adaptive_heun", "rk4", "heun"]
    )
    def test_registry_strings_run(self, device, dtype, name):
        sampler = self._sampler(device, dtype, integrator=name)
        z = torch.ones(4, 2, device=device, dtype=dtype)
        samples = sampler.sample(x=z, n_steps=20)
        expected = torch.full_like(samples, math.e)
        assert_close(samples, expected, atol=0.15, rtol=0.05)

    def test_sde_integrator_valid_in_ode_mode(self, device, dtype):
        from torchebm.integrators import EulerMaruyamaIntegrator

        sampler = self._sampler(
            device,
            dtype,
            integrator=EulerMaruyamaIntegrator(device=device, dtype=dtype),
        )
        z = torch.randn(4, 2, device=device, dtype=dtype)
        samples = sampler.sample(x=z, n_steps=10)
        assert torch.isfinite(samples).all()

# `torchebm.samplers`

Sampling algorithms for energy-based models and generative models.

Includes:

- MCMC samplers (Langevin dynamics, HMC) for energy-based models
- Gradient-based optimization samplers for energy minimization
- Flow/diffusion samplers for trained generative models

## `FlowSampler`

Bases: `BaseSampler`

Sampler for flow-based and diffusion generative models.

Supports ODE (probability flow) and SDE (diffusion) sampling with various numerical integration methods including Euler, Heun, and adaptive solvers.

Parameters:

| Name                  | Type                                    | Description                                                                                                        | Default      |
| --------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------ |
| `model`               | `Module`                                | Trained neural network predicting velocity/score/noise.                                                            | *required*   |
| `interpolant`         | `Union[str, BaseInterpolant]`           | Interpolant type ('linear', 'cosine', 'vp') or instance.                                                           | `'linear'`   |
| `prediction`          | `Literal['velocity', 'score', 'noise']` | Model prediction type ('velocity', 'score', or 'noise').                                                           | `'velocity'` |
| `train_eps`           | `float`                                 | Epsilon used during training for time interval stability.                                                          | `0.0`        |
| `sample_eps`          | `float`                                 | Epsilon for sampling time interval.                                                                                | `0.0`        |
| `negate_velocity`     | `bool`                                  | Negate the velocity during sampling. Set True for EqM models which learn (ε - x) direction; velocity is v = -f(x). | `False`      |
| `dtype`               | `dtype`                                 | Data type for computations.                                                                                        | `float32`    |
| `device`              | `Optional[Union[str, device]]`          | Device for computations.                                                                                           | `None`       |
| `use_mixed_precision` | `bool`                                  | Whether to use mixed precision.                                                                                    | `False`      |

Example

```python
from torchebm.samplers import FlowSampler
import torch.nn as nn
import torch

model = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 2))
sampler = FlowSampler(
    model=model,
    interpolant="linear",
    prediction="velocity",
)
z = torch.randn(100, 2)
samples = sampler.sample_ode(z, num_steps=50, method="euler")
```

Source code in `torchebm/samplers/flow.py`

````python
class FlowSampler(BaseSampler):
    r"""
    Sampler for flow-based and diffusion generative models.

    Supports ODE (probability flow) and SDE (diffusion) sampling with various
    numerical integration methods including Euler, Heun, and adaptive solvers.

    Args:
        model: Trained neural network predicting velocity/score/noise.
        interpolant: Interpolant type ('linear', 'cosine', 'vp') or instance.
        prediction: Model prediction type ('velocity', 'score', or 'noise').
        train_eps: Epsilon used during training for time interval stability.
        sample_eps: Epsilon for sampling time interval.
        negate_velocity: Negate the velocity during sampling. Set True for
            EqM models which learn (ε - x) direction; velocity is v = -f(x).
        dtype: Data type for computations.
        device: Device for computations.
        use_mixed_precision: Whether to use mixed precision.

    Example:
        ```python
        from torchebm.samplers import FlowSampler
        import torch.nn as nn
        import torch

        model = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 2))
        sampler = FlowSampler(
            model=model,
            interpolant="linear",
            prediction="velocity",
        )
        z = torch.randn(100, 2)
        samples = sampler.sample_ode(z, num_steps=50, method="euler")
        ```
    """

    def __init__(
        self,
        model: nn.Module,
        interpolant: Union[str, BaseInterpolant] = "linear",
        prediction: Literal["velocity", "score", "noise"] = "velocity",
        train_eps: float = 0.0,
        sample_eps: float = 0.0,
        negate_velocity: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        use_mixed_precision: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            dtype=dtype,
            device=device,
            use_mixed_precision=use_mixed_precision,
        )
        self.train_eps = train_eps
        self.sample_eps = sample_eps
        self.negate_velocity = negate_velocity

        if isinstance(interpolant, str):
            self.interpolant = get_interpolant(interpolant)
        else:
            self.interpolant = interpolant

        prediction_map = {
            "velocity": PredictionType.VELOCITY,
            "score": PredictionType.SCORE,
            "noise": PredictionType.NOISE,
        }
        self.prediction_type = prediction_map[prediction]

        self.interpolant = BaseSampler.safe_to(
            self.interpolant, device=self.device, dtype=self.dtype
        )

    @torch.no_grad()
    def sample(
        self,
        x: Optional[torch.Tensor] = None,
        dim: int = 10,
        n_steps: int = 50,
        n_samples: int = 1,
        thin: int = 1,
        return_trajectory: bool = False,
        return_diagnostics: bool = False,
        *,
        mode: Literal["ode", "sde"] = "ode",
        shape: Optional[Tuple[int, ...]] = None,
        ode_method: str = "dopri5",
        atol: float = 1e-6,
        rtol: float = 1e-3,
        reverse: bool = False,
        sde_method: str = "euler",
        diffusion_form: str = "SBDM",
        diffusion_norm: float = 1.0,
        last_step: Optional[str] = "Mean",
        last_step_size: float = 0.04,
        **model_kwargs,
    ) -> torch.Tensor:
        r"""
        Unified sampling entrypoint for flow/diffusion models.

        This method exists for API compatibility with `BaseSampler`. For full control,
        prefer calling `sample_ode` or `sample_sde` directly.
        """
        if thin != 1:
            raise ValueError("thin is not supported for FlowSampler")
        if return_trajectory or return_diagnostics:
            raise ValueError("FlowSampler does not support trajectories/diagnostics")

        if x is None:
            if shape is not None:
                z = torch.randn(*shape, device=self.device, dtype=self.dtype)
            else:
                z = torch.randn(n_samples, dim, device=self.device, dtype=self.dtype)
        else:
            z = x.to(device=self.device, dtype=self.dtype)

        if mode == "ode":
            return self.sample_ode(
                z=z,
                num_steps=n_steps,
                method=ode_method,
                atol=atol,
                rtol=rtol,
                reverse=reverse,
                **model_kwargs,
            )
        if mode == "sde":
            return self.sample_sde(
                z=z,
                num_steps=n_steps,
                method=sde_method,
                diffusion_form=diffusion_form,
                diffusion_norm=diffusion_norm,
                last_step=last_step,
                last_step_size=last_step_size,
                **model_kwargs,
            )
        raise ValueError(f"Unknown mode: {mode}")

    def _get_drift(self) -> Callable:
        r"""Get drift function for probability flow ODE."""

        def velocity_drift(x, t, **model_kwargs):
            v = self.model(x, t, **model_kwargs)
            return -v if self.negate_velocity else v

        def score_drift(x, t, **model_kwargs):
            drift_mean, drift_var = self.interpolant.compute_drift(x, t)
            model_output = self.model(x, t, **model_kwargs)
            return -drift_mean + drift_var * model_output

        def noise_drift(x, t, **model_kwargs):
            drift_mean, drift_var = self.interpolant.compute_drift(x, t)
            t_expanded = expand_t_like_x(t, x)
            sigma_t, _ = self.interpolant.compute_sigma_t(t_expanded)
            model_output = self.model(x, t, **model_kwargs)
            score = model_output / (-sigma_t + 1e-8)
            return -drift_mean + drift_var * score

        drifts = {
            PredictionType.VELOCITY: velocity_drift,
            PredictionType.SCORE: score_drift,
            PredictionType.NOISE: noise_drift,
        }
        return drifts[self.prediction_type]

    def _get_score(self) -> Callable:
        r"""Get score function from model output."""

        def velocity_score(x, t, **model_kwargs):
            velocity = self.model(x, t, **model_kwargs)
            return self.interpolant.velocity_to_score(velocity, x, t)

        def score_score(x, t, **model_kwargs):
            return self.model(x, t, **model_kwargs)

        def noise_score(x, t, **model_kwargs):
            t_expanded = expand_t_like_x(t, x)
            sigma_t, _ = self.interpolant.compute_sigma_t(t_expanded)
            return self.model(x, t, **model_kwargs) / (-sigma_t + 1e-8)

        scores = {
            PredictionType.VELOCITY: velocity_score,
            PredictionType.SCORE: score_score,
            PredictionType.NOISE: noise_score,
        }
        return scores[self.prediction_type]

    def _check_interval(
        self,
        sde: bool = False,
        reverse: bool = False,
        last_step_size: float = 0.0,
        diffusion_form: str = "SBDM",
    ) -> Tuple[float, float]:
        r"""Compute time interval for sampling."""
        t0 = 0.0
        t1 = 1.0
        eps = self.sample_eps

        is_vp = isinstance(self.interpolant, VariancePreservingInterpolant)
        is_linear_or_cosine = isinstance(
            self.interpolant, (LinearInterpolant, CosineInterpolant)
        )

        if is_vp:
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        elif is_linear_or_cosine and (
            self.prediction_type != PredictionType.VELOCITY or sde
        ):
            t0 = (
                eps
                if (diffusion_form == "SBDM" and sde)
                or self.prediction_type != PredictionType.VELOCITY
                else 0
            )
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    @torch.no_grad()
    def sample_ode(
        self,
        z: torch.Tensor,
        num_steps: int = 50,
        method: str = "dopri5",
        atol: float = 1e-6,
        rtol: float = 1e-3,
        reverse: bool = False,
        **model_kwargs,
    ) -> torch.Tensor:
        r"""
        Sample using probability flow ODE.

        Args:
            z: Initial noise tensor of shape (batch_size, ...).
            num_steps: Number of discretization steps (for fixed-step methods).
            method: ODE solver ('euler', 'heun', 'dopri5', 'dopri8').
            atol: Absolute tolerance for adaptive solvers.
            rtol: Relative tolerance for adaptive solvers.
            reverse: If True, sample from data to noise.
            **model_kwargs: Additional arguments passed to the model.

        Returns:
            Generated samples tensor.
        """
        z = z.to(device=self.device, dtype=self.dtype)
        drift_fn = self._get_drift()

        t0, t1 = self._check_interval(sde=False, reverse=reverse)
        # num_steps is the number of integration steps, so we need num_steps+1 time points
        t = torch.linspace(t0, t1, num_steps + 1, device=self.device, dtype=self.dtype)

        if reverse:

            def wrapped_drift(x, t_val, **kwargs):
                return drift_fn(x, torch.ones_like(t_val) * (1 - t_val), **kwargs)

        else:
            wrapped_drift = drift_fn

        def ode_fn(t_val, x):
            t_batch = (
                torch.ones(x.size(0), device=self.device, dtype=self.dtype) * t_val
            )
            return wrapped_drift(x, t_batch, **model_kwargs)

        def fixed_step_drift(x, t_batch):
            return wrapped_drift(x, t_batch, **model_kwargs)

        if method in ["dopri5", "dopri8", "bosh3", "adaptive_heun"]:
            if not HAS_TORCHDIFFEQ:
                raise ImportError("torchdiffeq required for adaptive solvers")
            samples = odeint(ode_fn, z, t, method=method, atol=atol, rtol=rtol)
            return samples[-1]
        if method == "euler":
            integrator = EulerMaruyamaIntegrator(device=self.device, dtype=self.dtype)
            return integrator.integrate(
                state={"x": z},
                step_size=t[1] - t[0],
                n_steps=num_steps,
                drift=fixed_step_drift,
                t=t,
            )["x"]
        if method == "heun":
            integrator = HeunIntegrator(device=self.device, dtype=self.dtype)
            return integrator.integrate(
                state={"x": z},
                step_size=t[1] - t[0],
                n_steps=num_steps,
                drift=fixed_step_drift,
                t=t,
            )["x"]
        raise ValueError(f"Unknown ODE method: {method}")

    @torch.no_grad()
    def sample_sde(
        self,
        z: torch.Tensor,
        num_steps: int = 250,
        method: str = "euler",
        diffusion_form: str = "SBDM",
        diffusion_norm: float = 1.0,
        last_step: Optional[str] = "Mean",
        last_step_size: float = 0.04,
        **model_kwargs,
    ) -> torch.Tensor:
        r"""
        Sample using reverse-time SDE.

        Args:
            z: Initial noise tensor of shape (batch_size, ...).
            num_steps: Number of discretization steps.
            method: SDE solver ('euler', 'heun').
            diffusion_form: Form of diffusion coefficient. Choices:
                - 'constant': Constant diffusion
                - 'SBDM': Score-based diffusion matching (default)
                - 'sigma': Proportional to noise schedule
                - 'linear': Linear decay
                - 'decreasing': Faster decay towards t=1
                - 'increasing-decreasing': Peak at midpoint
            diffusion_norm: Scaling factor for diffusion.
            last_step: Type of last step ('Mean', 'Tweedie', 'Euler', or None).
            last_step_size: Size of the last step.
            **model_kwargs: Additional arguments passed to the model.

        Returns:
            Generated samples tensor.
        """
        z = z.to(device=self.device, dtype=self.dtype)

        if last_step is None:
            last_step_size = 0.0

        t0, t1 = self._check_interval(
            sde=True, last_step_size=last_step_size, diffusion_form=diffusion_form
        )
        # t needs num_steps+1 points to perform num_steps integration steps
        t = torch.linspace(t0, t1, num_steps + 1, device=self.device, dtype=self.dtype)

        drift_fn = self._get_drift()
        score_fn = self._get_score()

        def diffusion_fn(x, t_val):
            return self.interpolant.compute_diffusion(
                x, t_val, form=diffusion_form, norm=diffusion_norm
            )

        def sde_drift(x, t_val, **kwargs):
            diffusion = diffusion_fn(x, t_val)
            return drift_fn(x, t_val, **kwargs) + diffusion * score_fn(
                x, t_val, **kwargs
            )

        def fixed_sde_drift(x, t_val):
            return sde_drift(x, t_val, **model_kwargs)

        if method == "euler":
            integrator = EulerMaruyamaIntegrator(device=self.device, dtype=self.dtype)
            x = integrator.integrate(
                state={"x": z},
                step_size=t[1] - t[0],
                n_steps=num_steps,
                drift=fixed_sde_drift,
                diffusion=diffusion_fn,
                t=t,
            )["x"]
        elif method == "heun":
            integrator = HeunIntegrator(device=self.device, dtype=self.dtype)
            x = integrator.integrate(
                state={"x": z},
                step_size=t[1] - t[0],
                n_steps=num_steps,
                drift=fixed_sde_drift,
                diffusion=diffusion_fn,
                t=t,
            )["x"]
        else:
            raise ValueError(f"Unknown SDE method: {method}")

        # Apply last step
        if last_step is not None:
            t_final = torch.ones(x.size(0), device=self.device, dtype=self.dtype) * t1
            x = self._apply_last_step(
                x, t_final, sde_drift, last_step, last_step_size, **model_kwargs
            )

        return x

    def _apply_last_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        sde_drift: Callable,
        last_step: str,
        last_step_size: float,
        **model_kwargs,
    ) -> torch.Tensor:
        r"""Apply final denoising step."""
        if last_step == "Mean":
            return x + sde_drift(x, t, **model_kwargs) * last_step_size
        elif last_step == "Euler":
            drift_fn = self._get_drift()
            return x + drift_fn(x, t, **model_kwargs) * last_step_size
        elif last_step == "Tweedie":
            t_expanded = expand_t_like_x(t, x)
            alpha, _ = self.interpolant.compute_alpha_t(t_expanded)
            sigma, _ = self.interpolant.compute_sigma_t(t_expanded)
            score = self._get_score()(x, t, **model_kwargs)
            return x / alpha + (sigma**2) / alpha * score
        else:
            return x

    def prior_logp(self, z: torch.Tensor) -> torch.Tensor:
        r"""Compute log probability under standard Gaussian prior."""
        shape = torch.tensor(z.size())
        N = torch.prod(shape[1:])
        return (
            -N / 2.0 * np.log(2 * np.pi)
            - torch.sum(z**2, dim=tuple(range(1, z.ndim))) / 2.0
        )
````

### `prior_logp(z)`

Compute log probability under standard Gaussian prior.

Source code in `torchebm/samplers/flow.py`

```python
def prior_logp(self, z: torch.Tensor) -> torch.Tensor:
    r"""Compute log probability under standard Gaussian prior."""
    shape = torch.tensor(z.size())
    N = torch.prod(shape[1:])
    return (
        -N / 2.0 * np.log(2 * np.pi)
        - torch.sum(z**2, dim=tuple(range(1, z.ndim))) / 2.0
    )
```

### `sample(x=None, dim=10, n_steps=50, n_samples=1, thin=1, return_trajectory=False, return_diagnostics=False, *, mode='ode', shape=None, ode_method='dopri5', atol=1e-06, rtol=0.001, reverse=False, sde_method='euler', diffusion_form='SBDM', diffusion_norm=1.0, last_step='Mean', last_step_size=0.04, **model_kwargs)`

Unified sampling entrypoint for flow/diffusion models.

This method exists for API compatibility with `BaseSampler`. For full control, prefer calling `sample_ode` or `sample_sde` directly.

Source code in `torchebm/samplers/flow.py`

```python
@torch.no_grad()
def sample(
    self,
    x: Optional[torch.Tensor] = None,
    dim: int = 10,
    n_steps: int = 50,
    n_samples: int = 1,
    thin: int = 1,
    return_trajectory: bool = False,
    return_diagnostics: bool = False,
    *,
    mode: Literal["ode", "sde"] = "ode",
    shape: Optional[Tuple[int, ...]] = None,
    ode_method: str = "dopri5",
    atol: float = 1e-6,
    rtol: float = 1e-3,
    reverse: bool = False,
    sde_method: str = "euler",
    diffusion_form: str = "SBDM",
    diffusion_norm: float = 1.0,
    last_step: Optional[str] = "Mean",
    last_step_size: float = 0.04,
    **model_kwargs,
) -> torch.Tensor:
    r"""
    Unified sampling entrypoint for flow/diffusion models.

    This method exists for API compatibility with `BaseSampler`. For full control,
    prefer calling `sample_ode` or `sample_sde` directly.
    """
    if thin != 1:
        raise ValueError("thin is not supported for FlowSampler")
    if return_trajectory or return_diagnostics:
        raise ValueError("FlowSampler does not support trajectories/diagnostics")

    if x is None:
        if shape is not None:
            z = torch.randn(*shape, device=self.device, dtype=self.dtype)
        else:
            z = torch.randn(n_samples, dim, device=self.device, dtype=self.dtype)
    else:
        z = x.to(device=self.device, dtype=self.dtype)

    if mode == "ode":
        return self.sample_ode(
            z=z,
            num_steps=n_steps,
            method=ode_method,
            atol=atol,
            rtol=rtol,
            reverse=reverse,
            **model_kwargs,
        )
    if mode == "sde":
        return self.sample_sde(
            z=z,
            num_steps=n_steps,
            method=sde_method,
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
            last_step=last_step,
            last_step_size=last_step_size,
            **model_kwargs,
        )
    raise ValueError(f"Unknown mode: {mode}")
```

### `sample_ode(z, num_steps=50, method='dopri5', atol=1e-06, rtol=0.001, reverse=False, **model_kwargs)`

Sample using probability flow ODE.

Parameters:

| Name             | Type     | Description                                              | Default    |
| ---------------- | -------- | -------------------------------------------------------- | ---------- |
| `z`              | `Tensor` | Initial noise tensor of shape (batch_size, ...).         | *required* |
| `num_steps`      | `int`    | Number of discretization steps (for fixed-step methods). | `50`       |
| `method`         | `str`    | ODE solver ('euler', 'heun', 'dopri5', 'dopri8').        | `'dopri5'` |
| `atol`           | `float`  | Absolute tolerance for adaptive solvers.                 | `1e-06`    |
| `rtol`           | `float`  | Relative tolerance for adaptive solvers.                 | `0.001`    |
| `reverse`        | `bool`   | If True, sample from data to noise.                      | `False`    |
| `**model_kwargs` |          | Additional arguments passed to the model.                | `{}`       |

Returns:

| Type     | Description               |
| -------- | ------------------------- |
| `Tensor` | Generated samples tensor. |

Source code in `torchebm/samplers/flow.py`

```python
@torch.no_grad()
def sample_ode(
    self,
    z: torch.Tensor,
    num_steps: int = 50,
    method: str = "dopri5",
    atol: float = 1e-6,
    rtol: float = 1e-3,
    reverse: bool = False,
    **model_kwargs,
) -> torch.Tensor:
    r"""
    Sample using probability flow ODE.

    Args:
        z: Initial noise tensor of shape (batch_size, ...).
        num_steps: Number of discretization steps (for fixed-step methods).
        method: ODE solver ('euler', 'heun', 'dopri5', 'dopri8').
        atol: Absolute tolerance for adaptive solvers.
        rtol: Relative tolerance for adaptive solvers.
        reverse: If True, sample from data to noise.
        **model_kwargs: Additional arguments passed to the model.

    Returns:
        Generated samples tensor.
    """
    z = z.to(device=self.device, dtype=self.dtype)
    drift_fn = self._get_drift()

    t0, t1 = self._check_interval(sde=False, reverse=reverse)
    # num_steps is the number of integration steps, so we need num_steps+1 time points
    t = torch.linspace(t0, t1, num_steps + 1, device=self.device, dtype=self.dtype)

    if reverse:

        def wrapped_drift(x, t_val, **kwargs):
            return drift_fn(x, torch.ones_like(t_val) * (1 - t_val), **kwargs)

    else:
        wrapped_drift = drift_fn

    def ode_fn(t_val, x):
        t_batch = (
            torch.ones(x.size(0), device=self.device, dtype=self.dtype) * t_val
        )
        return wrapped_drift(x, t_batch, **model_kwargs)

    def fixed_step_drift(x, t_batch):
        return wrapped_drift(x, t_batch, **model_kwargs)

    if method in ["dopri5", "dopri8", "bosh3", "adaptive_heun"]:
        if not HAS_TORCHDIFFEQ:
            raise ImportError("torchdiffeq required for adaptive solvers")
        samples = odeint(ode_fn, z, t, method=method, atol=atol, rtol=rtol)
        return samples[-1]
    if method == "euler":
        integrator = EulerMaruyamaIntegrator(device=self.device, dtype=self.dtype)
        return integrator.integrate(
            state={"x": z},
            step_size=t[1] - t[0],
            n_steps=num_steps,
            drift=fixed_step_drift,
            t=t,
        )["x"]
    if method == "heun":
        integrator = HeunIntegrator(device=self.device, dtype=self.dtype)
        return integrator.integrate(
            state={"x": z},
            step_size=t[1] - t[0],
            n_steps=num_steps,
            drift=fixed_step_drift,
            t=t,
        )["x"]
    raise ValueError(f"Unknown ODE method: {method}")
```

### `sample_sde(z, num_steps=250, method='euler', diffusion_form='SBDM', diffusion_norm=1.0, last_step='Mean', last_step_size=0.04, **model_kwargs)`

Sample using reverse-time SDE.

Parameters:

| Name             | Type            | Description                                                                                                                                                                                                                                                                         | Default    |
| ---------------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `z`              | `Tensor`        | Initial noise tensor of shape (batch_size, ...).                                                                                                                                                                                                                                    | *required* |
| `num_steps`      | `int`           | Number of discretization steps.                                                                                                                                                                                                                                                     | `250`      |
| `method`         | `str`           | SDE solver ('euler', 'heun').                                                                                                                                                                                                                                                       | `'euler'`  |
| `diffusion_form` | `str`           | Form of diffusion coefficient. Choices: - 'constant': Constant diffusion - 'SBDM': Score-based diffusion matching (default) - 'sigma': Proportional to noise schedule - 'linear': Linear decay - 'decreasing': Faster decay towards t=1 - 'increasing-decreasing': Peak at midpoint | `'SBDM'`   |
| `diffusion_norm` | `float`         | Scaling factor for diffusion.                                                                                                                                                                                                                                                       | `1.0`      |
| `last_step`      | `Optional[str]` | Type of last step ('Mean', 'Tweedie', 'Euler', or None).                                                                                                                                                                                                                            | `'Mean'`   |
| `last_step_size` | `float`         | Size of the last step.                                                                                                                                                                                                                                                              | `0.04`     |
| `**model_kwargs` |                 | Additional arguments passed to the model.                                                                                                                                                                                                                                           | `{}`       |

Returns:

| Type     | Description               |
| -------- | ------------------------- |
| `Tensor` | Generated samples tensor. |

Source code in `torchebm/samplers/flow.py`

```python
@torch.no_grad()
def sample_sde(
    self,
    z: torch.Tensor,
    num_steps: int = 250,
    method: str = "euler",
    diffusion_form: str = "SBDM",
    diffusion_norm: float = 1.0,
    last_step: Optional[str] = "Mean",
    last_step_size: float = 0.04,
    **model_kwargs,
) -> torch.Tensor:
    r"""
    Sample using reverse-time SDE.

    Args:
        z: Initial noise tensor of shape (batch_size, ...).
        num_steps: Number of discretization steps.
        method: SDE solver ('euler', 'heun').
        diffusion_form: Form of diffusion coefficient. Choices:
            - 'constant': Constant diffusion
            - 'SBDM': Score-based diffusion matching (default)
            - 'sigma': Proportional to noise schedule
            - 'linear': Linear decay
            - 'decreasing': Faster decay towards t=1
            - 'increasing-decreasing': Peak at midpoint
        diffusion_norm: Scaling factor for diffusion.
        last_step: Type of last step ('Mean', 'Tweedie', 'Euler', or None).
        last_step_size: Size of the last step.
        **model_kwargs: Additional arguments passed to the model.

    Returns:
        Generated samples tensor.
    """
    z = z.to(device=self.device, dtype=self.dtype)

    if last_step is None:
        last_step_size = 0.0

    t0, t1 = self._check_interval(
        sde=True, last_step_size=last_step_size, diffusion_form=diffusion_form
    )
    # t needs num_steps+1 points to perform num_steps integration steps
    t = torch.linspace(t0, t1, num_steps + 1, device=self.device, dtype=self.dtype)

    drift_fn = self._get_drift()
    score_fn = self._get_score()

    def diffusion_fn(x, t_val):
        return self.interpolant.compute_diffusion(
            x, t_val, form=diffusion_form, norm=diffusion_norm
        )

    def sde_drift(x, t_val, **kwargs):
        diffusion = diffusion_fn(x, t_val)
        return drift_fn(x, t_val, **kwargs) + diffusion * score_fn(
            x, t_val, **kwargs
        )

    def fixed_sde_drift(x, t_val):
        return sde_drift(x, t_val, **model_kwargs)

    if method == "euler":
        integrator = EulerMaruyamaIntegrator(device=self.device, dtype=self.dtype)
        x = integrator.integrate(
            state={"x": z},
            step_size=t[1] - t[0],
            n_steps=num_steps,
            drift=fixed_sde_drift,
            diffusion=diffusion_fn,
            t=t,
        )["x"]
    elif method == "heun":
        integrator = HeunIntegrator(device=self.device, dtype=self.dtype)
        x = integrator.integrate(
            state={"x": z},
            step_size=t[1] - t[0],
            n_steps=num_steps,
            drift=fixed_sde_drift,
            diffusion=diffusion_fn,
            t=t,
        )["x"]
    else:
        raise ValueError(f"Unknown SDE method: {method}")

    # Apply last step
    if last_step is not None:
        t_final = torch.ones(x.size(0), device=self.device, dtype=self.dtype) * t1
        x = self._apply_last_step(
            x, t_final, sde_drift, last_step, last_step_size, **model_kwargs
        )

    return x
```

## `GradientDescentSampler`

Bases: `BaseSampler`

Gradient descent sampler for energy-based models.

Generates samples by iteratively minimizing the energy function:

[ x\_{k+1} = x_k - \\eta \\nabla_x E(x_k) ]

This is a deterministic optimization-based sampler that finds low-energy configurations by following the negative gradient of the energy function.

Parameters:

| Name                  | Type                           | Description                                | Default    |
| --------------------- | ------------------------------ | ------------------------------------------ | ---------- |
| `model`               | `BaseModel`                    | Energy-based model with gradient() method. | *required* |
| `step_size`           | `Union[float, BaseScheduler]`  | Step size (\\eta) or scheduler.            | `0.001`    |
| `dtype`               | `dtype`                        | Data type for computations.                | `float32`  |
| `device`              | `Optional[Union[str, device]]` | Device for computations.                   | `None`     |
| `use_mixed_precision` | `bool`                         | Whether to use mixed precision.            | `False`    |

Example

```python
from torchebm.samplers import GradientDescentSampler
from torchebm.core import DoubleWellEnergy
import torch

energy = DoubleWellEnergy()
sampler = GradientDescentSampler(energy, step_size=0.1)
samples = sampler.sample(n_samples=100, dim=2, n_steps=500)
```

Source code in `torchebm/samplers/gradient_descent.py`

````python
class GradientDescentSampler(BaseSampler):
    r"""
    Gradient descent sampler for energy-based models.

    Generates samples by iteratively minimizing the energy function:

    \[
    x_{k+1} = x_k - \eta \nabla_x E(x_k)
    \]

    This is a deterministic optimization-based sampler that finds low-energy
    configurations by following the negative gradient of the energy function.

    Args:
        model: Energy-based model with `gradient()` method.
        step_size: Step size \(\eta\) or scheduler.
        dtype: Data type for computations.
        device: Device for computations.
        use_mixed_precision: Whether to use mixed precision.

    Example:
        ```python
        from torchebm.samplers import GradientDescentSampler
        from torchebm.core import DoubleWellEnergy
        import torch

        energy = DoubleWellEnergy()
        sampler = GradientDescentSampler(energy, step_size=0.1)
        samples = sampler.sample(n_samples=100, dim=2, n_steps=500)
        ```
    """

    def __init__(
        self,
        model: BaseModel,
        step_size: Union[float, BaseScheduler] = 1e-3,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        use_mixed_precision: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            dtype=dtype,
            device=device,
            use_mixed_precision=use_mixed_precision,
        )
        if isinstance(step_size, BaseScheduler):
            self.register_scheduler("step_size", step_size)
        else:
            if step_size <= 0:
                raise ValueError("step_size must be positive")
            self.register_scheduler("step_size", ConstantScheduler(step_size))

    @torch.no_grad()
    def sample(
        self,
        x: Optional[torch.Tensor] = None,
        dim: int = 10,
        n_steps: int = 100,
        n_samples: int = 1,
        thin: int = 1,
        return_trajectory: bool = False,
        return_diagnostics: bool = False,
        *args,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[dict]]]:
        r"""
        Generate samples via gradient descent optimization.

        Args:
            x: Initial state. If None, samples from N(0, I).
            dim: Dimension of state space (used if x is None).
            n_steps: Number of gradient descent steps.
            n_samples: Number of parallel chains/samples.
            thin: Thinning factor (not currently supported).
            return_trajectory: Whether to return full trajectory.
            return_diagnostics: Whether to return diagnostics.

        Returns:
            Final samples or (samples, diagnostics) if return_diagnostics=True.
        """
        self.reset_schedulers()

        if x is None:
            x = torch.randn(n_samples, dim, device=self.device, dtype=self.dtype)
        else:
            x = x.to(device=self.device, dtype=self.dtype)

        diagnostics = self._setup_diagnostics() if return_diagnostics else None
        trajectory = [x.clone()] if return_trajectory else None

        with self.autocast_context():
            for _ in range(n_steps):
                self.step_schedulers()
                eta = self.get_scheduled_value("step_size")
                grad = self.model.gradient(x)
                x = x - eta * grad

                if return_trajectory:
                    trajectory.append(x.clone())

        if return_diagnostics:
            return (
                torch.stack(trajectory, dim=1) if return_trajectory else x,
                [diagnostics],
            )
        return torch.stack(trajectory, dim=1) if return_trajectory else x
````

### `sample(x=None, dim=10, n_steps=100, n_samples=1, thin=1, return_trajectory=False, return_diagnostics=False, *args, **kwargs)`

Generate samples via gradient descent optimization.

Parameters:

| Name                 | Type               | Description                                   | Default |
| -------------------- | ------------------ | --------------------------------------------- | ------- |
| `x`                  | `Optional[Tensor]` | Initial state. If None, samples from N(0, I). | `None`  |
| `dim`                | `int`              | Dimension of state space (used if x is None). | `10`    |
| `n_steps`            | `int`              | Number of gradient descent steps.             | `100`   |
| `n_samples`          | `int`              | Number of parallel chains/samples.            | `1`     |
| `thin`               | `int`              | Thinning factor (not currently supported).    | `1`     |
| `return_trajectory`  | `bool`             | Whether to return full trajectory.            | `False` |
| `return_diagnostics` | `bool`             | Whether to return diagnostics.                | `False` |

Returns:

| Type                                       | Description                                                         |
| ------------------------------------------ | ------------------------------------------------------------------- |
| `Union[Tensor, Tuple[Tensor, List[dict]]]` | Final samples or (samples, diagnostics) if return_diagnostics=True. |

Source code in `torchebm/samplers/gradient_descent.py`

```python
@torch.no_grad()
def sample(
    self,
    x: Optional[torch.Tensor] = None,
    dim: int = 10,
    n_steps: int = 100,
    n_samples: int = 1,
    thin: int = 1,
    return_trajectory: bool = False,
    return_diagnostics: bool = False,
    *args,
    **kwargs,
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[dict]]]:
    r"""
    Generate samples via gradient descent optimization.

    Args:
        x: Initial state. If None, samples from N(0, I).
        dim: Dimension of state space (used if x is None).
        n_steps: Number of gradient descent steps.
        n_samples: Number of parallel chains/samples.
        thin: Thinning factor (not currently supported).
        return_trajectory: Whether to return full trajectory.
        return_diagnostics: Whether to return diagnostics.

    Returns:
        Final samples or (samples, diagnostics) if return_diagnostics=True.
    """
    self.reset_schedulers()

    if x is None:
        x = torch.randn(n_samples, dim, device=self.device, dtype=self.dtype)
    else:
        x = x.to(device=self.device, dtype=self.dtype)

    diagnostics = self._setup_diagnostics() if return_diagnostics else None
    trajectory = [x.clone()] if return_trajectory else None

    with self.autocast_context():
        for _ in range(n_steps):
            self.step_schedulers()
            eta = self.get_scheduled_value("step_size")
            grad = self.model.gradient(x)
            x = x - eta * grad

            if return_trajectory:
                trajectory.append(x.clone())

    if return_diagnostics:
        return (
            torch.stack(trajectory, dim=1) if return_trajectory else x,
            [diagnostics],
        )
    return torch.stack(trajectory, dim=1) if return_trajectory else x
```

## `HamiltonianMonteCarlo`

Bases: `BaseSampler`

Hamiltonian Monte Carlo sampler.

Uses Hamiltonian dynamics with Metropolis-Hastings acceptance to sample from the target distribution defined by the energy model.

Parameters:

| Name               | Type                             | Description                              | Default    |
| ------------------ | -------------------------------- | ---------------------------------------- | ---------- |
| `model`            | `BaseModel`                      | Energy-based model to sample from.       | *required* |
| `step_size`        | `Union[float, BaseScheduler]`    | Step size for leapfrog integration.      | `0.001`    |
| `n_leapfrog_steps` | `int`                            | Number of leapfrog steps per trajectory. | `10`       |
| `mass`             | `Optional[Union[float, Tensor]]` | Mass matrix (scalar or tensor).          | `None`     |
| `dtype`            | `dtype`                          | Data type for computations.              | `float32`  |
| `device`           | `Optional[Union[str, device]]`   | Device for computations.                 | `None`     |

Example

```python
from torchebm.samplers import HamiltonianMonteCarlo
from torchebm.core import DoubleWellEnergy
import torch

energy = DoubleWellEnergy()
sampler = HamiltonianMonteCarlo(
    energy, step_size=0.1, n_leapfrog_steps=10
)
samples = sampler.sample(n_samples=100, dim=2, n_steps=500)
```

Source code in `torchebm/samplers/hmc.py`

````python
class HamiltonianMonteCarlo(BaseSampler):
    r"""
    Hamiltonian Monte Carlo sampler.

    Uses Hamiltonian dynamics with Metropolis-Hastings acceptance to sample
    from the target distribution defined by the energy model.

    Args:
        model: Energy-based model to sample from.
        step_size: Step size for leapfrog integration.
        n_leapfrog_steps: Number of leapfrog steps per trajectory.
        mass: Mass matrix (scalar or tensor).
        dtype: Data type for computations.
        device: Device for computations.

    Example:
        ```python
        from torchebm.samplers import HamiltonianMonteCarlo
        from torchebm.core import DoubleWellEnergy
        import torch

        energy = DoubleWellEnergy()
        sampler = HamiltonianMonteCarlo(
            energy, step_size=0.1, n_leapfrog_steps=10
        )
        samples = sampler.sample(n_samples=100, dim=2, n_steps=500)
        ```
    """

    def __init__(
        self,
        model: BaseModel,
        step_size: Union[float, BaseScheduler] = 1e-3,
        n_leapfrog_steps: int = 10,
        mass: Optional[Union[float, torch.Tensor]] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(model=model, dtype=dtype, device=device)
        if isinstance(step_size, BaseScheduler):
            self.register_scheduler("step_size", step_size)
        else:
            if step_size <= 0:
                raise ValueError("step_size must be positive")
            self.register_scheduler("step_size", ConstantScheduler(step_size))

        if n_leapfrog_steps <= 0:
            raise ValueError("n_leapfrog_steps must be positive")

        self.n_leapfrog_steps = n_leapfrog_steps
        self.mass = (
            mass.to(self.device)
            if (mass is not None and not isinstance(mass, float))
            else mass
        )
        self.integrator = LeapfrogIntegrator(device=self.device, dtype=self.dtype)

    def _initialize_momentum(self, shape: torch.Size) -> torch.Tensor:
        """
        Initializes momentum variables from a Gaussian distribution.

        The momentum is sampled from \(\mathcal{N}(0, M)\), where `M` is the mass matrix.

        Args:
            shape (torch.Size): The shape of the momentum tensor to generate.

        Returns:
            torch.Tensor: The initialized momentum tensor.
        """
        p = torch.randn(shape, dtype=self.dtype, device=self.device)

        if self.mass is not None:
            # Apply mass matrix (equivalent to sampling from N(0, M))
            if isinstance(self.mass, float):
                p = p * torch.sqrt(
                    torch.tensor(self.mass, dtype=self.dtype, device=self.device)
                )
            else:
                mass_sqrt = torch.sqrt(self.mass)
                p = p * mass_sqrt.view(*([1] * (len(shape) - 1)), -1).expand_as(p)
        return p

    def _compute_kinetic_energy(self, p: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the kinetic energy of the momentum.

        The kinetic energy is \(K(p) = \frac{1}{2} p^T M^{-1} p\).

        Args:
            p (torch.Tensor): The momentum tensor.

        Returns:
            torch.Tensor: The kinetic energy for each sample in the batch.
        """
        if self.mass is None:
            return 0.5 * torch.sum(p**2, dim=-1)
        elif isinstance(self.mass, float):
            return 0.5 * torch.sum(p**2, dim=-1) / self.mass
        else:
            return 0.5 * torch.sum(
                p**2 / self.mass.view(*([1] * (len(p.shape) - 1)), -1), dim=-1
            )

    @torch.no_grad()
    def sample(
        self,
        x: Optional[torch.Tensor] = None,
        dim: Optional[int] = None,
        n_steps: int = 100,
        n_samples: int = 1,
        thin: int = 1,
        return_trajectory: bool = False,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.reset_schedulers()

        if x is None:
            if dim is None:
                if hasattr(self.model, "mean") and isinstance(
                    self.model.mean, torch.Tensor
                ):
                    dim = self.model.mean.shape[0]
                else:
                    raise ValueError(
                        "dim must be provided when x is None and cannot be inferred from model"
                    )
            x = torch.randn(n_samples, dim, dtype=self.dtype, device=self.device)
        else:
            x = x.to(device=self.device, dtype=self.dtype)

        dim = x.shape[1]
        batch_size = x.shape[0]

        if return_trajectory:
            trajectory = torch.empty(
                (batch_size, n_steps, dim),
                dtype=self.dtype,
                device=self.device,
                requires_grad=False,
            )

        if return_diagnostics:
            diagnostics = self._setup_diagnostics(dim, n_steps, n_samples=batch_size)

        with self.autocast_context():
            for i in range(n_steps):
                self.step_schedulers()

                current_momentum = self._initialize_momentum(x.shape)

                momentum_direction = (
                    torch.randint(0, 2, (batch_size, 1), device=self.device) * 2 - 1
                )  # -1/+1 -> for sign flipping
                current_momentum = current_momentum * momentum_direction

                current_energy = torch.clamp(self.model(x), min=-1e10, max=1e10)
                current_kinetic = torch.clamp(
                    self._compute_kinetic_energy(current_momentum), min=0, max=1e10
                )

                current_hamiltonian = current_energy + current_kinetic

                state = {"x": x, "p": current_momentum}
                drift = lambda x_, t_: -self.model.gradient(x_)
                proposed = self.integrator.integrate(
                    state,
                    step_size=self.get_scheduled_value("step_size"),
                    n_steps=self.n_leapfrog_steps,
                    mass=self.mass,
                    drift=drift,
                    safe=True,
                )
                proposed_position, proposed_momentum = proposed["x"], proposed["p"]

                proposed_energy = torch.clamp(
                    self.model(proposed_position), min=-1e10, max=1e10
                )
                proposed_kinetic = torch.clamp(
                    self._compute_kinetic_energy(proposed_momentum), min=0, max=1e10
                )

                proposed_hamiltonian = proposed_energy + proposed_kinetic

                hamiltonian_diff = current_hamiltonian - proposed_hamiltonian
                hamiltonian_diff = torch.clamp(hamiltonian_diff, max=50, min=-50)

                acceptance_prob = torch.minimum(
                    torch.ones(batch_size, device=self.device),
                    torch.exp(hamiltonian_diff),
                )

                random_uniform = torch.rand(batch_size, device=self.device)
                accepted = random_uniform < acceptance_prob
                accepted_mask = accepted.float().view(-1, *([1] * (len(x.shape) - 1)))

                x = accepted_mask * proposed_position + (1.0 - accepted_mask) * x

                if return_trajectory:
                    trajectory[:, i, :] = x

                if return_diagnostics:
                    mean_x = x.mean(dim=0, keepdim=True)
                    var_x = torch.clamp(
                        x.var(dim=0, unbiased=False, keepdim=True),
                        min=1e-10,
                        max=1e10,
                    )
                    energy = torch.clamp(self.model(x), min=-1e10, max=1e10)
                    acceptance_rate = accepted.float().mean()

                    diagnostics[i, 0, :, :] = mean_x.expand(batch_size, dim)
                    diagnostics[i, 1, :, :] = var_x.expand(batch_size, dim)
                    diagnostics[i, 2, :, :] = energy.view(-1, 1).expand(-1, dim)
                    diagnostics[i, 3, :, :] = torch.full(
                        (batch_size, dim),
                        acceptance_rate,
                        dtype=self.dtype,
                        device=self.device,
                    )

        if return_trajectory:
            if return_diagnostics:
                return trajectory.to(dtype=self.dtype), diagnostics.to(dtype=self.dtype)
            return trajectory.to(dtype=self.dtype)

        if return_diagnostics:
            return x.to(dtype=self.dtype), diagnostics.to(dtype=self.dtype)

        return x.to(dtype=self.dtype)

    def _setup_diagnostics(
        self, dim: int, n_steps: int, n_samples: int = None
    ) -> torch.Tensor:
        r"""
        Initializes a tensor to store diagnostics during sampling.

        Args:
            dim (int): The dimensionality of the state space.
            n_steps (int): The number of sampling steps.
            n_samples (Optional[int]): The number of parallel chains.

        Returns:
            torch.Tensor: An empty tensor for storing diagnostics.
        """
        if n_samples is not None:
            return torch.empty(
                (n_steps, 4, n_samples, dim), device=self.device, dtype=self.dtype
            )
        else:
            return torch.empty((n_steps, 4, dim), device=self.device, dtype=self.dtype)
````

## `LangevinDynamics`

Bases: `BaseSampler`

Langevin Dynamics sampler.

Update rule:

[ x\_{t+1} = x_t - \\eta \\nabla_x U(x_t) + \\sqrt{2\\eta} \\epsilon_t ]

Parameters:

| Name          | Type                           | Description                          | Default    |
| ------------- | ------------------------------ | ------------------------------------ | ---------- |
| `model`       | `BaseModel`                    | Energy-based model to sample from.   | *required* |
| `step_size`   | `Union[float, BaseScheduler]`  | Step size for gradient descent.      | `0.001`    |
| `noise_scale` | `Union[float, BaseScheduler]`  | Scale of Gaussian noise injection.   | `1.0`      |
| `decay`       | `float`                        | Damping coefficient (not supported). | `0.0`      |
| `dtype`       | `dtype`                        | Data type for computations.          | `float32`  |
| `device`      | `Optional[Union[str, device]]` | Device for computations.             | `None`     |

Example

```python
from torchebm.samplers import LangevinDynamics
from torchebm.core import DoubleWellEnergy
import torch

energy = DoubleWellEnergy()
sampler = LangevinDynamics(energy, step_size=0.01, noise_scale=1.0)
samples = sampler.sample(n_samples=100, dim=2, n_steps=500)
```

Source code in `torchebm/samplers/langevin_dynamics.py`

````python
class LangevinDynamics(BaseSampler):
    r"""
    Langevin Dynamics sampler.

    Update rule:

    \[
    x_{t+1} = x_t - \eta \nabla_x U(x_t) + \sqrt{2\eta} \epsilon_t
    \]

    Args:
        model: Energy-based model to sample from.
        step_size: Step size for gradient descent.
        noise_scale: Scale of Gaussian noise injection.
        decay: Damping coefficient (not supported).
        dtype: Data type for computations.
        device: Device for computations.

    Example:
        ```python
        from torchebm.samplers import LangevinDynamics
        from torchebm.core import DoubleWellEnergy
        import torch

        energy = DoubleWellEnergy()
        sampler = LangevinDynamics(energy, step_size=0.01, noise_scale=1.0)
        samples = sampler.sample(n_samples=100, dim=2, n_steps=500)
        ```
    """

    def __init__(
        self,
        model: BaseModel,
        step_size: Union[float, BaseScheduler] = 1e-3,
        noise_scale: Union[float, BaseScheduler] = 1.0,
        decay: float = 0.0,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(model=model, dtype=dtype, device=device)

        if isinstance(step_size, BaseScheduler):
            self.register_scheduler("step_size", step_size)
        else:
            if step_size <= 0:
                raise ValueError("step_size must be positive")
            self.register_scheduler("step_size", ConstantScheduler(step_size))

        if isinstance(noise_scale, BaseScheduler):
            self.register_scheduler("noise_scale", noise_scale)
        else:
            if noise_scale <= 0:
                raise ValueError("noise_scale must be positive")
            self.register_scheduler("noise_scale", ConstantScheduler(noise_scale))

        self.decay = decay
        self.integrator = EulerMaruyamaIntegrator(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def sample(
        self,
        x: Optional[torch.Tensor] = None,
        dim: int = 10,
        n_steps: int = 100,
        n_samples: int = 1,
        thin: int = 1,
        return_trajectory: bool = False,
        return_diagnostics: bool = False,
        *args,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[dict]]]:
        """
        Generates samples using Langevin dynamics.

        Args:
            x (Optional[torch.Tensor]): The initial state to start sampling from. If `None`,
                a random state is created.
            dim (int): The dimension of the state space (if `x` is not provided).
            n_steps (int): The number of MCMC steps to perform.
            n_samples (int): The number of parallel chains/samples to generate.
            thin (int): The thinning factor (not currently supported).
            return_trajectory (bool): Whether to return the full sample trajectory.
            return_diagnostics (bool): Whether to return sampling diagnostics.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, List[dict]]]:
                - The final samples.
                - If `return_trajectory` is `True`, the full trajectory.
                - If `return_diagnostics` is `True`, a tuple of samples and diagnostics.
        """

        self.reset_schedulers()

        if x is None:
            x = torch.randn(n_samples, dim, dtype=self.dtype, device=self.device)
        else:
            x = x.to(device=self.device, dtype=self.dtype)
            dim = x.shape[-1]
            n_samples = x.shape[0]

        if return_trajectory:
            trajectory = torch.empty(
                (n_samples, n_steps, dim), dtype=self.dtype, device=self.device
            )

        if return_diagnostics:
            diagnostics = self._setup_diagnostics(dim, n_steps, n_samples=n_samples)

        drift = lambda x_, t_: -self.model.gradient(x_)
        with self.autocast_context():
            for i in range(n_steps):
                self.step_schedulers()
                state = {"x": x}
                x = self.integrator.step(
                    state=state,
                    step_size=self.get_scheduled_value("step_size"),
                    noise_scale=self.get_scheduled_value("noise_scale"),
                    drift=drift,
                )["x"]

                if return_trajectory:
                    trajectory[:, i, :] = x

                if return_diagnostics:
                    if n_samples > 1:
                        mean_x = x.mean(dim=0, keepdim=True)
                        var_x = torch.clamp(
                            x.var(dim=0, unbiased=False, keepdim=True),
                            min=1e-10,
                            max=1e10,
                        )
                    else:
                        mean_x = x
                        var_x = torch.zeros_like(x)
                    energy = self.model(x)
                    diagnostics[i, 0, :, :] = (
                        mean_x if n_samples > 1 else mean_x.unsqueeze(0)
                    )
                    diagnostics[i, 1, :, :] = (
                        var_x if n_samples > 1 else var_x.unsqueeze(0)
                    )
                    diagnostics[i, 2, :, :] = energy.view(-1, 1).expand(n_samples, dim)

        if return_trajectory:
            if return_diagnostics:
                return trajectory, diagnostics
            return trajectory
        if return_diagnostics:
            return x, diagnostics
        return x

    def _setup_diagnostics(
        self, dim: int, n_steps: int, n_samples: int = None
    ) -> torch.Tensor:
        if n_samples is not None:
            return torch.empty(
                (n_steps, 3, n_samples, dim), device=self.device, dtype=self.dtype
            )
        return torch.empty((n_steps, 3, dim), device=self.device, dtype=self.dtype)
````

### `sample(x=None, dim=10, n_steps=100, n_samples=1, thin=1, return_trajectory=False, return_diagnostics=False, *args, **kwargs)`

Generates samples using Langevin dynamics.

Parameters:

| Name                 | Type               | Description                                                                   | Default |
| -------------------- | ------------------ | ----------------------------------------------------------------------------- | ------- |
| `x`                  | `Optional[Tensor]` | The initial state to start sampling from. If None, a random state is created. | `None`  |
| `dim`                | `int`              | The dimension of the state space (if x is not provided).                      | `10`    |
| `n_steps`            | `int`              | The number of MCMC steps to perform.                                          | `100`   |
| `n_samples`          | `int`              | The number of parallel chains/samples to generate.                            | `1`     |
| `thin`               | `int`              | The thinning factor (not currently supported).                                | `1`     |
| `return_trajectory`  | `bool`             | Whether to return the full sample trajectory.                                 | `False` |
| `return_diagnostics` | `bool`             | Whether to return sampling diagnostics.                                       | `False` |

Returns:

| Type                                       | Description                                                                                                                                                                                              |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Union[Tensor, Tuple[Tensor, List[dict]]]` | Union\[torch.Tensor, Tuple\[torch.Tensor, List[dict]\]\]: - The final samples. - If return_trajectory is True, the full trajectory. - If return_diagnostics is True, a tuple of samples and diagnostics. |

Source code in `torchebm/samplers/langevin_dynamics.py`

```python
@torch.no_grad()
def sample(
    self,
    x: Optional[torch.Tensor] = None,
    dim: int = 10,
    n_steps: int = 100,
    n_samples: int = 1,
    thin: int = 1,
    return_trajectory: bool = False,
    return_diagnostics: bool = False,
    *args,
    **kwargs,
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[dict]]]:
    """
    Generates samples using Langevin dynamics.

    Args:
        x (Optional[torch.Tensor]): The initial state to start sampling from. If `None`,
            a random state is created.
        dim (int): The dimension of the state space (if `x` is not provided).
        n_steps (int): The number of MCMC steps to perform.
        n_samples (int): The number of parallel chains/samples to generate.
        thin (int): The thinning factor (not currently supported).
        return_trajectory (bool): Whether to return the full sample trajectory.
        return_diagnostics (bool): Whether to return sampling diagnostics.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, List[dict]]]:
            - The final samples.
            - If `return_trajectory` is `True`, the full trajectory.
            - If `return_diagnostics` is `True`, a tuple of samples and diagnostics.
    """

    self.reset_schedulers()

    if x is None:
        x = torch.randn(n_samples, dim, dtype=self.dtype, device=self.device)
    else:
        x = x.to(device=self.device, dtype=self.dtype)
        dim = x.shape[-1]
        n_samples = x.shape[0]

    if return_trajectory:
        trajectory = torch.empty(
            (n_samples, n_steps, dim), dtype=self.dtype, device=self.device
        )

    if return_diagnostics:
        diagnostics = self._setup_diagnostics(dim, n_steps, n_samples=n_samples)

    drift = lambda x_, t_: -self.model.gradient(x_)
    with self.autocast_context():
        for i in range(n_steps):
            self.step_schedulers()
            state = {"x": x}
            x = self.integrator.step(
                state=state,
                step_size=self.get_scheduled_value("step_size"),
                noise_scale=self.get_scheduled_value("noise_scale"),
                drift=drift,
            )["x"]

            if return_trajectory:
                trajectory[:, i, :] = x

            if return_diagnostics:
                if n_samples > 1:
                    mean_x = x.mean(dim=0, keepdim=True)
                    var_x = torch.clamp(
                        x.var(dim=0, unbiased=False, keepdim=True),
                        min=1e-10,
                        max=1e10,
                    )
                else:
                    mean_x = x
                    var_x = torch.zeros_like(x)
                energy = self.model(x)
                diagnostics[i, 0, :, :] = (
                    mean_x if n_samples > 1 else mean_x.unsqueeze(0)
                )
                diagnostics[i, 1, :, :] = (
                    var_x if n_samples > 1 else var_x.unsqueeze(0)
                )
                diagnostics[i, 2, :, :] = energy.view(-1, 1).expand(n_samples, dim)

    if return_trajectory:
        if return_diagnostics:
            return trajectory, diagnostics
        return trajectory
    if return_diagnostics:
        return x, diagnostics
    return x
```

## `NesterovSampler`

Bases: `BaseSampler`

Nesterov accelerated gradient sampler for energy-based models.

Uses Nesterov momentum to accelerate convergence to low-energy states:

[ v\_{k+1} = \\mu v_k - \\eta \\nabla_x E(x_k + \\mu v_k) ]

[ x\_{k+1} = x_k + v\_{k+1} ]

where (\\mu) is the momentum coefficient and (\\eta) is the step size.

Parameters:

| Name                  | Type                           | Description                                | Default    |
| --------------------- | ------------------------------ | ------------------------------------------ | ---------- |
| `model`               | `BaseModel`                    | Energy-based model with gradient() method. | *required* |
| `step_size`           | `Union[float, BaseScheduler]`  | Step size (\\eta) or scheduler.            | `0.001`    |
| `momentum`            | `float`                        | Momentum coefficient (\\mu \\in \[0, 1)).  | `0.9`      |
| `dtype`               | `dtype`                        | Data type for computations.                | `float32`  |
| `device`              | `Optional[Union[str, device]]` | Device for computations.                   | `None`     |
| `use_mixed_precision` | `bool`                         | Whether to use mixed precision.            | `False`    |

Example

```python
from torchebm.samplers import NesterovSampler
from torchebm.core import DoubleWellEnergy
import torch

energy = DoubleWellEnergy()
sampler = NesterovSampler(energy, step_size=0.1, momentum=0.9)
samples = sampler.sample(n_samples=100, dim=2, n_steps=500)
```

Source code in `torchebm/samplers/gradient_descent.py`

````python
class NesterovSampler(BaseSampler):
    r"""
    Nesterov accelerated gradient sampler for energy-based models.

    Uses Nesterov momentum to accelerate convergence to low-energy states:

    \[
    v_{k+1} = \mu v_k - \eta \nabla_x E(x_k + \mu v_k)
    \]

    \[
    x_{k+1} = x_k + v_{k+1}
    \]

    where \(\mu\) is the momentum coefficient and \(\eta\) is the step size.

    Args:
        model: Energy-based model with `gradient()` method.
        step_size: Step size \(\eta\) or scheduler.
        momentum: Momentum coefficient \(\mu \in [0, 1)\).
        dtype: Data type for computations.
        device: Device for computations.
        use_mixed_precision: Whether to use mixed precision.

    Example:
        ```python
        from torchebm.samplers import NesterovSampler
        from torchebm.core import DoubleWellEnergy
        import torch

        energy = DoubleWellEnergy()
        sampler = NesterovSampler(energy, step_size=0.1, momentum=0.9)
        samples = sampler.sample(n_samples=100, dim=2, n_steps=500)
        ```
    """

    def __init__(
        self,
        model: BaseModel,
        step_size: Union[float, BaseScheduler] = 1e-3,
        momentum: float = 0.9,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        use_mixed_precision: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            dtype=dtype,
            device=device,
            use_mixed_precision=use_mixed_precision,
        )
        if not (0 <= momentum < 1):
            raise ValueError("momentum must be in [0, 1)")
        self.momentum = momentum

        if isinstance(step_size, BaseScheduler):
            self.register_scheduler("step_size", step_size)
        else:
            if step_size <= 0:
                raise ValueError("step_size must be positive")
            self.register_scheduler("step_size", ConstantScheduler(step_size))

    @torch.no_grad()
    def sample(
        self,
        x: Optional[torch.Tensor] = None,
        dim: int = 10,
        n_steps: int = 100,
        n_samples: int = 1,
        thin: int = 1,
        return_trajectory: bool = False,
        return_diagnostics: bool = False,
        *args,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[dict]]]:
        r"""
        Generate samples via Nesterov accelerated gradient descent.

        Args:
            x: Initial state. If None, samples from N(0, I).
            dim: Dimension of state space (used if x is None).
            n_steps: Number of optimization steps.
            n_samples: Number of parallel chains/samples.
            thin: Thinning factor (not currently supported).
            return_trajectory: Whether to return full trajectory.
            return_diagnostics: Whether to return diagnostics.

        Returns:
            Final samples or (samples, diagnostics) if return_diagnostics=True.
        """
        self.reset_schedulers()

        if x is None:
            x = torch.randn(n_samples, dim, device=self.device, dtype=self.dtype)
        else:
            x = x.to(device=self.device, dtype=self.dtype)

        v = torch.zeros_like(x)
        diagnostics = self._setup_diagnostics() if return_diagnostics else None
        trajectory = [x.clone()] if return_trajectory else None

        mu = self.momentum
        with self.autocast_context():
            for _ in range(n_steps):
                self.step_schedulers()
                eta = self.get_scheduled_value("step_size")
                lookahead = x + mu * v
                grad = self.model.gradient(lookahead)
                v = mu * v - eta * grad
                x = x + v

                if return_trajectory:
                    trajectory.append(x.clone())

        if return_diagnostics:
            return (
                torch.stack(trajectory, dim=1) if return_trajectory else x,
                [diagnostics],
            )
        return torch.stack(trajectory, dim=1) if return_trajectory else x
````

### `sample(x=None, dim=10, n_steps=100, n_samples=1, thin=1, return_trajectory=False, return_diagnostics=False, *args, **kwargs)`

Generate samples via Nesterov accelerated gradient descent.

Parameters:

| Name                 | Type               | Description                                   | Default |
| -------------------- | ------------------ | --------------------------------------------- | ------- |
| `x`                  | `Optional[Tensor]` | Initial state. If None, samples from N(0, I). | `None`  |
| `dim`                | `int`              | Dimension of state space (used if x is None). | `10`    |
| `n_steps`            | `int`              | Number of optimization steps.                 | `100`   |
| `n_samples`          | `int`              | Number of parallel chains/samples.            | `1`     |
| `thin`               | `int`              | Thinning factor (not currently supported).    | `1`     |
| `return_trajectory`  | `bool`             | Whether to return full trajectory.            | `False` |
| `return_diagnostics` | `bool`             | Whether to return diagnostics.                | `False` |

Returns:

| Type                                       | Description                                                         |
| ------------------------------------------ | ------------------------------------------------------------------- |
| `Union[Tensor, Tuple[Tensor, List[dict]]]` | Final samples or (samples, diagnostics) if return_diagnostics=True. |

Source code in `torchebm/samplers/gradient_descent.py`

```python
@torch.no_grad()
def sample(
    self,
    x: Optional[torch.Tensor] = None,
    dim: int = 10,
    n_steps: int = 100,
    n_samples: int = 1,
    thin: int = 1,
    return_trajectory: bool = False,
    return_diagnostics: bool = False,
    *args,
    **kwargs,
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[dict]]]:
    r"""
    Generate samples via Nesterov accelerated gradient descent.

    Args:
        x: Initial state. If None, samples from N(0, I).
        dim: Dimension of state space (used if x is None).
        n_steps: Number of optimization steps.
        n_samples: Number of parallel chains/samples.
        thin: Thinning factor (not currently supported).
        return_trajectory: Whether to return full trajectory.
        return_diagnostics: Whether to return diagnostics.

    Returns:
        Final samples or (samples, diagnostics) if return_diagnostics=True.
    """
    self.reset_schedulers()

    if x is None:
        x = torch.randn(n_samples, dim, device=self.device, dtype=self.dtype)
    else:
        x = x.to(device=self.device, dtype=self.dtype)

    v = torch.zeros_like(x)
    diagnostics = self._setup_diagnostics() if return_diagnostics else None
    trajectory = [x.clone()] if return_trajectory else None

    mu = self.momentum
    with self.autocast_context():
        for _ in range(n_steps):
            self.step_schedulers()
            eta = self.get_scheduled_value("step_size")
            lookahead = x + mu * v
            grad = self.model.gradient(lookahead)
            v = mu * v - eta * grad
            x = x + v

            if return_trajectory:
                trajectory.append(x.clone())

    if return_diagnostics:
        return (
            torch.stack(trajectory, dim=1) if return_trajectory else x,
            [diagnostics],
        )
    return torch.stack(trajectory, dim=1) if return_trajectory else x
```

## `PredictionType`

Bases: `Enum`

Model prediction type for generative models.

Source code in `torchebm/samplers/flow.py`

```python
class PredictionType(enum.Enum):
    r"""Model prediction type for generative models."""

    NOISE = enum.auto()
    SCORE = enum.auto()
    VELOCITY = enum.auto()
```

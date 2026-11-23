r"""Mixed-field sampler: transport to a threshold, then refinement."""

from typing import Any, Dict, Optional, Union

import torch
from torch import nn

from torchebm.core import BaseSampler
from torchebm.core.base_integrator import BaseRungeKuttaIntegrator
from torchebm.integrators.integrator_utils import resolve_integrator


class MixedFieldSampler(BaseSampler):
    r"""
    Sampler for mixed-field matching models (Ghaderi et al., 2026, Algorithm 2).

    A field \(f_\theta(z, \tau)\) trained with `MixedFieldMatchingLoss` is a
    time-conditioned transport field for \(t < a\) and one time-independent
    refinement field at \(\tau = a\). Starting from noise at \(t = 0\),
    transport integrates

    \[
    \frac{dz}{dt} = \frac{1}{\lambda} f_\theta(z, t), \qquad 0 \le t \le a,
    \]

    in ``n_steps`` steps of the integrator, and refinement iterates

    \[
    z_{n+1} = z_n + \eta\, f_\theta(z_n, a), \qquad \eta = \frac{r (1 - a)}{\lambda},
    \]

    ``n_refinement_steps`` times. The step \(\eta\) contracts the residual to
    the data by \(1 - r\) per step for the ideal refinement field, at every
    threshold; the paper uses \(r = 1/2\). The model is called as
    ``model(x, t, **model_kwargs)``, so `ClassifierFreeGuidance` wraps it
    unchanged.

    Args:
        ct_threshold: Threshold \(a\) the model was trained with, strictly
            inside (0, 1).
        ct_multiplier: Target multiplier \(\lambda\) the model was trained
            with.
        refinement_rate: Refinement rate \(r\); stable for \(0 < r < 2\).
        dtype: Data type of the time grid.
        device: Device for computations.
        integrator: Transport integrator, a registry name (``"euler"``
            (default), ``"heun"``, ...) or a `BaseRungeKuttaIntegrator`.

    Example:
        MFM-XL/2 on ImageNet 256 latents, with 230 Euler transport and 20
        refinement steps and guidance 1.5 on the first three latent channels
        (App. C):

        ```python
        from torchebm.models import ClassifierFreeGuidance
        from torchebm.samplers import MixedFieldSampler

        guided = ClassifierFreeGuidance(
            model, guidance_scale=1.5, null_condition=1000, guide_channels=3
        )
        sampler = MixedFieldSampler()
        samples = sampler(
            guided, torch.randn(64, 4, 32, 32), n_steps=230, n_refinement_steps=20,
            model_kwargs={"y": y},
        )
        ```
    """

    prediction_type = "velocity"
    timestep_conditioned = True

    def __init__(
        self,
        ct_threshold: float = 0.8,
        ct_multiplier: float = 4.0,
        refinement_rate: float = 0.5,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        integrator: Union[str, BaseRungeKuttaIntegrator, None] = None,
    ):
        super().__init__(dtype=dtype, device=device)
        if not 0.0 < ct_threshold < 1.0:
            raise ValueError(f"ct_threshold must be in (0, 1), got {ct_threshold}")
        self.ct_threshold = float(ct_threshold)
        self.ct_multiplier = float(ct_multiplier)
        self.refinement_rate = float(refinement_rate)
        self.refinement_step_size = (
            self.refinement_rate * (1.0 - self.ct_threshold) / self.ct_multiplier
        )
        self.integrator = resolve_integrator(
            integrator,
            default="euler",
            family=BaseRungeKuttaIntegrator,
            owner="MixedFieldSampler",
            device=self.device,
            dtype=self.dtype,
        )

    @torch.no_grad()
    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        *,
        n_steps: int = 230,
        n_refinement_steps: int = 20,
        generator: Optional[torch.Generator] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        r"""Transport `x` from noise to the threshold, then refine it.

        Args:
            model: Field called as ``model(x, t, **model_kwargs)``.
            x: Noise at \(t = 0\), shape ``(batch_size, *sample_shape)``.
            n_steps: Number of transport steps on the uniform grid over
                \([0, a]\).
            n_refinement_steps: Number of refinement steps at \(\tau = a\).
            generator: Unused: sampling is deterministic.
            model_kwargs: Conditioning arguments forwarded to the model.
            return_trajectory: If True, return the state after every step,
                shape ``(batch_size, n_steps + n_refinement_steps, *sample_shape)``.

        Returns:
            The final state, or the trajectory.

        Raises:
            ValueError: If `n_steps` is not positive.
        """
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        field = self._model_fn(model, model_kwargs)

        def velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return field(x, t) / self.ct_multiplier

        grid = torch.linspace(
            0.0, self.ct_threshold, n_steps + 1, device=x.device, dtype=self.dtype
        )
        self.reset_schedulers(model)
        trajectory = None
        if return_trajectory:
            trajectory = x.new_empty(
                (x.shape[0], n_steps + n_refinement_steps, *x.shape[1:])
            )
        for i in range(n_steps):
            x = self.integrator.step(
                state={"x": x},
                step_size=grid[i + 1] - grid[i],
                drift=velocity,
                t=grid[i].expand(x.shape[0]),
            )["x"]
            self.step_schedulers(model)
            if trajectory is not None:
                trajectory[:, i] = x
        for i in range(n_steps, n_steps + n_refinement_steps):
            x = torch.add(x, field(x, grid[-1]), alpha=self.refinement_step_size)
            self.step_schedulers(model)
            if trajectory is not None:
                trajectory[:, i] = x
        return x if trajectory is None else trajectory

    def sample(self, *args, **kwargs):
        r"""Not supported: call ``sampler(model, x, n_steps=...)``."""
        raise NotImplementedError(
            "MixedFieldSampler is called as sampler(model, x, n_steps=...)"
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(ct_threshold={self.ct_threshold}, "
            f"ct_multiplier={self.ct_multiplier}, "
            f"refinement_rate={self.refinement_rate}, "
            f"integrator={type(self.integrator).__name__})"
        )


__all__ = ["MixedFieldSampler"]

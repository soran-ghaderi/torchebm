r"""Flow-map sampler for one-step and few-step generation."""

from typing import Any, Callable, Dict, Literal, Optional, Union

import torch
from torch import nn

from torchebm.core import BaseInterpolant, BaseSampler
from torchebm.interpolants.interpolant_utils import resolve_interpolant

_PREDICTION_TYPES = ("flow_map", "average_velocity")


class FlowMapSampler(BaseSampler):
    r"""
    Sampler for flow-map models, for one-step and few-step generation.

    A flow map \(F(x_s, s, t)\) jumps from time \(s\) to time \(t\) of the
    probability-flow ODE in one model call, with noise at \(t = 0\) and data
    at \(t = 1\). On a time grid \(0 = t_0 < t_1 < \dots < t_N = 1\) the
    sampler composes the jumps

    \[
    x_{t_{i+1}} = F(x_{t_i}, t_i, t_{i+1}),
    \]

    so ``n_steps=1`` is one-step generation. A model that outputs the average
    velocity \(u(x_s, s, t)\) (MeanFlow, Geng et al., 2025) gives
    \(F(x_s, s, t) = x_s + (t - s)\, u(x_s, s, t)\).

    With ``stochastic=True`` the sampler runs multistep consistency sampling
    (Song et al., 2023, Algorithm 1): every step jumps to data,
    \(\hat x_1 = F(x_{t_i}, t_i, 1)\), and the next step starts from the
    interpolant path at the next grid time,
    \(x_{t_{i+1}} = \alpha(t_{i+1})\, \hat x_1 + \sigma(t_{i+1})\, \epsilon\)
    with \(\epsilon \sim \mathcal{N}(0, I)\).

    The model is called as ``model(x_s, t, s=s, **model_kwargs)``, the
    two-time call of `DiT` and `TimeConditionedMLP`.

    Args:
        prediction_type: Model output: ``"flow_map"`` (default),
            \(F(x_s, s, t)\), or ``"average_velocity"``, \(u(x_s, s, t)\).
        stochastic: If True, multistep consistency sampling; otherwise the
            jumps are composed deterministically.
        interpolant: Interpolant name ('linear', 'cosine', 'vp') or instance
            that adds the noise back when ``stochastic=True``; the one the
            model was trained with.
        dtype: Data type of the time grid.
        device: Device for computations.

    Example:
        One-step sampling of MeanFlow (Geng et al., 2025) on MeanFlow-XL/2, a
        class-conditional `DiT` with ``prediction_type="average_velocity"``
        on ImageNet 256 latents:

        ```python
        from torchebm.samplers import FlowMapSampler

        sampler = FlowMapSampler(prediction_type="average_velocity")
        samples = sampler(model, torch.randn(64, 4, 32, 32), n_steps=1, model_kwargs={"y": y})
        ```
    """

    timestep_conditioned = True

    def __init__(
        self,
        prediction_type: Literal["flow_map", "average_velocity"] = "flow_map",
        stochastic: bool = False,
        interpolant: Union[str, BaseInterpolant] = "linear",
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__(dtype=dtype, device=device)
        if prediction_type not in _PREDICTION_TYPES:
            raise ValueError(
                f"prediction_type must be one of {_PREDICTION_TYPES}, got {prediction_type!r}"
            )
        self.prediction_type = prediction_type
        self.stochastic = stochastic
        self.interpolant = resolve_interpolant(
            interpolant, default="linear", owner="FlowMapSampler"
        )

    @torch.no_grad()
    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        *,
        n_steps: int = 1,
        t: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        r"""Generate from `x` in `n_steps` jumps of the flow map of `model`.

        Args:
            model: Two-time network called as
                ``model(x_s, t, s=s, **model_kwargs)``; its output is named by
                `prediction_type`.
            x: Initial state at time ``t[0]`` (noise for the default grid).
            n_steps: Number of jumps on the uniform grid from 0 to 1; ignored
                when `t` is given.
            t: Custom time grid, a 1-D tensor of at least two increasing
                times; one jump per consecutive pair, or with
                ``stochastic=True`` one jump to ``t[-1]`` from each time but
                the last.
            generator: RNG for the noise of ``stochastic=True``; the global
                RNG when ``None``.
            model_kwargs: Conditioning arguments forwarded to the model.
            return_trajectory: If True, return the output of every jump,
                shape ``(batch_size, n_steps, *sample_shape)``.

        Returns:
            The final state, or the trajectory.

        Raises:
            ValueError: If `n_steps` is not positive, `t` is not a 1-D tensor
                of at least two entries, or `model.prediction_type` differs
                from `prediction_type`.
        """
        flow_map = self._flow_map(model, model_kwargs)
        grid = self._time_grid(n_steps, t, x)
        n_steps = grid.shape[0] - 1
        self.reset_schedulers(model)
        trajectory = None
        if return_trajectory:
            trajectory = x.new_empty((x.shape[0], n_steps, *x.shape[1:]))
        for i in range(n_steps):
            if self.stochastic and i > 0:
                noise = torch.randn(
                    x.shape, generator=generator, device=x.device, dtype=x.dtype
                )
                x_t, _ = self.interpolant.interpolate(
                    noise, x, grid[i].expand(x.shape[0])
                )
                x = x_t.to(x.dtype)
            x = flow_map(x, grid[i], grid[-1] if self.stochastic else grid[i + 1])
            self.step_schedulers(model)
            if trajectory is not None:
                trajectory[:, i] = x
        return x if trajectory is None else trajectory

    def _flow_map(
        self, model: nn.Module, model_kwargs: Optional[Dict[str, Any]]
    ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        r"""Closure ``(x_s, s, t) -> F(x_s, s, t)`` for 0-d times ``s`` and ``t``."""
        produced = getattr(model, "prediction_type", None)
        if produced is not None and produced != self.prediction_type:
            raise ValueError(
                f"model outputs {produced!r} but FlowMapSampler consumes "
                f"{self.prediction_type!r}; set prediction_type on the model or the sampler"
            )
        kwargs = model_kwargs or {}

        def call(x: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return model(x, t.expand(x.shape[0]), s=s.expand(x.shape[0]), **kwargs)

        if self.prediction_type == "flow_map":
            return call

        def flow_map(x: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return x + (t - s) * call(x, s, t)

        return flow_map

    def _time_grid(
        self, n_steps: int, t: Optional[torch.Tensor], x: torch.Tensor
    ) -> torch.Tensor:
        r"""Times of the jumps, shape ``(n_steps + 1,)``."""
        if t is not None:
            if t.ndim != 1 or t.shape[0] < 2:
                raise ValueError("t must be a 1-D tensor with at least two entries")
            return t.to(device=x.device, dtype=self.dtype)
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        return torch.linspace(0.0, 1.0, n_steps + 1, device=x.device, dtype=self.dtype)

    def sample(self, *args, **kwargs):
        r"""Not supported: call ``sampler(model, x, n_steps=...)``."""
        raise NotImplementedError(
            "FlowMapSampler is called as sampler(model, x, n_steps=...)"
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(prediction_type={self.prediction_type!r}, "
            f"stochastic={self.stochastic}, "
            f"interpolant={type(self.interpolant).__name__})"
        )


__all__ = ["FlowMapSampler"]

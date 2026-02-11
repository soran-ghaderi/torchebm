from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import nn

from torchebm.core import DeviceMixin, BaseModel


class BaseIntegrator(DeviceMixin, nn.Module, ABC):
    """
    Abstract integrator that advances a sampler state according to dynamics.

    The integrator operates on a generic state dict to remain reusable across
    samplers (e.g., Langevin uses only position `x`, HMC uses position `x` and
    momentum `p`).

    Methods follow PyTorch conventions and respect `device`/`dtype` from
    `DeviceMixin`.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        *args,
        **kwargs,
    ):
        super().__init__(device=device, dtype=dtype, *args, **kwargs)

    @abstractmethod
    def step(
        self,
        state: Dict[str, torch.Tensor],
        model: Optional[BaseModel],
        step_size: torch.Tensor,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Advance the dynamical state by one integrator application.

        Args:
            state: Mapping containing required tensors (e.g., {'x': ..., 'p': ...}).
            model: Energy-based model providing `forward` and `gradient`.
            step_size: Step size for the integration.
            *args: Additional positional arguments specific to the integrator.
            **kwargs: Additional keyword arguments specific to the integrator.

        Returns:
            Updated state dict with the same keys as the input `state`.
        """
        raise NotImplementedError

    @abstractmethod
    def integrate(
        self,
        state: Dict[str, torch.Tensor],
        model: Optional[BaseModel],
        step_size: torch.Tensor,
        n_steps: int,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Advance the dynamical state by `n_steps` integrator applications.

        Args:
            state: Mapping containing required tensors (e.g., {'x': ..., 'p': ...}).
            model: Energy-based model providing `forward` and `gradient`.
            step_size: Step size for the integration.
            n_steps: The number of integration steps to perform.
            *args: Additional positional arguments specific to the integrator.
            **kwargs: Additional keyword arguments specific to the integrator.

        Returns:
            Updated state dict with the same keys as the input `state`.
        """
        raise NotImplementedError


class   BaseRungeKuttaIntegrator(BaseIntegrator):
    r"""Abstract base class for explicit Runge-Kutta integrators for ODEs/SDEs.

    Subclasses define a Butcher tableau via the abstract properties
    :attr:`tableau_a`, :attr:`tableau_b`, and :attr:`tableau_c` and
    automatically inherit generic stepping and integration logic.

    For an \(s\)-stage explicit RK method the deterministic update reads

    \[
    k_i = f\!\bigl(x + h \sum_{j=0}^{i-1} a_{ij}\,k_j,\;
                    t + c_i\,h\bigr),
    \quad i = 0,\ldots,s{-}1
    \]

    \[
    x_{n+1} = x_n + h \sum_{i=0}^{s-1} b_i\,k_i
    \]

    When a diffusion coefficient \(D\) is provided the SDE extension
    \(\sqrt{2D}\,\Delta W\) is added after the deterministic update.

    Args:
        device: Device for computations.
        dtype: Data type for computations.

    Example:
        ```python
        from torchebm.core import BaseRungeKuttaIntegrator
        import torch

        class MidpointIntegrator(BaseRungeKuttaIntegrator):
            @property
            def tableau_a(self):
                return ((), (0.5,))

            @property
            def tableau_b(self):
                return (0.0, 1.0)

            @property
            def tableau_c(self):
                return (0.0, 0.5)

        integrator = MidpointIntegrator()
        state = {"x": torch.randn(100, 2)}
        drift = lambda x, t: -x
        result = integrator.step(state, model=None, step_size=0.01, drift=drift)
        ```
    """

    # butcher tableau — must be defined by every subclass

    @property
    @abstractmethod
    def tableau_a(self) -> Tuple[Tuple[float, ...], ...]:
        r"""Lower-triangular RK matrix \(a_{ij}\).

        ``tableau_a[i]`` contains coefficients \(a_{i0}, \ldots, a_{i,i-1}\).
        The first row is the empty tuple ``()``.
        """

    @property
    @abstractmethod
    def tableau_b(self) -> Tuple[float, ...]:
        r"""Weights \(b_i\) used to combine stages into the final update."""

    @property
    @abstractmethod
    def tableau_c(self) -> Tuple[float, ...]:
        r"""Nodes \(c_i\) — time-fraction offsets for each stage evaluation."""

    @property
    def n_stages(self) -> int:
        """Number of stages *s* in the method."""
        return len(self.tableau_c)

    # helpers
    @staticmethod
    def _resolve_drift(
        model: Optional[BaseModel],
        drift: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Return a concrete drift callable, falling back to the model gradient."""
        if drift is not None:
            return drift
        if model is None:
            raise ValueError(
                "Either `model` must be provided or `drift` must be set."
            )
        return lambda x_, t_: -model.gradient(x_)

    @staticmethod
    def _resolve_diffusion(
        diffusion: Optional[torch.Tensor],
        noise_scale: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Return a diffusion tensor from explicit value or ``noise_scale``."""
        if diffusion is not None:
            return diffusion
        if noise_scale is not None:
            if not torch.is_tensor(noise_scale):
                noise_scale = torch.tensor(noise_scale, device=device, dtype=dtype)
            return noise_scale ** 2
        return None

    def _evaluate_stages(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        step_size: torch.Tensor,
        drift_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> List[torch.Tensor]:
        """Evaluate all RK stages and return the list ``[k_0, ..., k_{s-1}]``."""
        a = self.tableau_a
        c = self.tableau_c
        k: List[torch.Tensor] = []
        for i in range(self.n_stages):
            if i == 0:
                x_stage = x
            else:
                dx = torch.zeros_like(x)
                for j in range(i):
                    if a[i][j] != 0:
                        dx = dx + a[i][j] * k[j]
                x_stage = x + step_size * dx
            t_stage = t + c[i] * step_size
            k.append(drift_fn(x_stage, t_stage))
        return k

    def _combine_stages(
        self,
        x: torch.Tensor,
        step_size: torch.Tensor,
        k: List[torch.Tensor],
    ) -> torch.Tensor:
        r"""Combine RK stages into the deterministic update \(x + h \sum b_i k_i\)."""
        b = self.tableau_b
        dx = torch.zeros_like(x)
        for i in range(len(b)):
            if b[i] != 0:
                dx = dx + b[i] * k[i]
        return x + step_size * dx

    def step(
        self,
        state: Dict[str, torch.Tensor],
        model: Optional[BaseModel],
        step_size: torch.Tensor,
        *,
        drift: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        diffusion: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        noise_scale: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = state["x"]
        if not torch.is_tensor(step_size):
            step_size = torch.tensor(step_size, device=x.device, dtype=x.dtype)
        if t is None:
            t = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

        drift_fn = self._resolve_drift(model, drift)
        diffusion_val = self._resolve_diffusion(
            diffusion, noise_scale, x.device, x.dtype
        )

        k = self._evaluate_stages(x, t, step_size, drift_fn)
        x_new = self._combine_stages(x, step_size, k)

        if diffusion_val is not None:
            if noise is None:
                noise = torch.randn_like(x, device=self.device, dtype=self.dtype)
            dw = noise * torch.sqrt(step_size)
            x_new = x_new + torch.sqrt(2.0 * diffusion_val) * dw

        return {"x": x_new}

    def integrate(
        self,
        state: Dict[str, torch.Tensor],
        model: Optional[BaseModel],
        step_size: torch.Tensor,
        n_steps: int,
        *,
        drift: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        diffusion: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        noise_scale: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if t is None:
            if not torch.is_tensor(step_size):
                step_size = torch.tensor(
                    step_size, device=state["x"].device, dtype=state["x"].dtype
                )
            t = (
                torch.arange(
                    n_steps + 1,
                    device=state["x"].device,
                    dtype=state["x"].dtype,
                )
                * step_size
            )
        if t.ndim != 1 or t.numel() < 2:
            raise ValueError("t must be a 1D tensor with length >= 2")

        x = state["x"]
        for i in range(t.numel() - 1):
            dt = t[i + 1] - t[i]
            t_batch = t[i].expand(x.size(0))
            diffusion_t = (
                diffusion(x, t_batch) if diffusion is not None else None
            )
            x = self.step(
                state={"x": x},
                model=model,
                step_size=dt,
                drift=drift,
                diffusion=diffusion_t,
                noise_scale=noise_scale,
                t=t_batch,
            )["x"]

        return {"x": x}

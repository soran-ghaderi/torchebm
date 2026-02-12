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


class BaseRungeKuttaIntegrator(BaseIntegrator):
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

    **Adaptive step-size control** is available automatically for subclasses
    that define :attr:`error_weights` and :attr:`order`.  When
    ``adaptive=True`` is passed to :meth:`integrate` (or left as ``None``
    for auto-detection), the integrator uses an embedded error pair to
    control the step size.

    Args:
        device: Device for computations.
        dtype: Data type for computations.
        atol: Absolute tolerance for adaptive stepping.
        rtol: Relative tolerance for adaptive stepping.
        max_steps: Maximum number of steps (accepted + rejected) before
            raising ``RuntimeError``.
        safety: Safety factor for step-size adjustment (< 1).
        min_factor: Minimum step-size shrink factor.
        max_factor: Maximum step-size growth factor.

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

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        *,
        atol: float = 1e-6,
        rtol: float = 1e-3,
        max_steps: int = 10_000,
        safety: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 10.0,
    ):
        super().__init__(device=device, dtype=dtype)
        self.atol = atol
        self.rtol = rtol
        self.max_steps = max_steps
        self.safety = safety
        self.min_factor = min_factor
        self.max_factor = max_factor

    # butcher tableau, must be defined by subclasses

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

    # adaptive properties, override in embedded-pair subclasses

    @property
    def error_weights(self) -> Optional[Tuple[float, ...]]:
        r"""Error estimation weights \(e_i = b_i - \hat{b}_i\).

        Return ``None`` (the default) for methods without an embedded pair.
        For FSAL methods the tuple has ``n_stages + 1`` entries; for
        non-FSAL methods it has ``n_stages`` entries.
        """
        return None

    @property
    def order(self) -> Optional[int]:
        r"""Order *p* of the higher-order solution.

        Used in the step-size control exponent \(-1/p\).  Return ``None``
        (the default) for methods without adaptive support.
        """
        return None

    @property
    def fsal(self) -> bool:
        """Whether the method has the First Same As Last property.

        When ``True`` the integrator evaluates one extra stage at the
        accepted solution and reuses it as the first stage of the next
        step, saving one drift evaluation per accepted step.
        """
        return False

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
        k0: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        r"""Evaluate all RK stages and return the list ``[k_0, ..., k_{s-1}]``.

        Args:
            x: Current position tensor.
            t: Current time tensor (batch,).
            step_size: Step size \(h\).
            drift_fn: Drift callable ``f(x, t)``.
            k0: Optional pre-computed first stage.  When provided the first
                drift evaluation is skipped (used by FSAL methods to reuse
                the last stage of the previous step).
        """
        a = self.tableau_a
        c = self.tableau_c
        k: List[torch.Tensor] = []
        for i in range(self.n_stages):
            if i == 0 and k0 is not None:
                k.append(k0)
                continue
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


    def _adaptive_integrate(
        self,
        x: torch.Tensor,
        drift_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        t_start: float,
        t_end: float,
        h: float,
    ) -> torch.Tensor:
        """Core adaptive integration loop from *t_start* to *t_end*."""
        t_current = t_start
        e = self.error_weights
        p = self.order
        is_fsal = self.fsal

        k1_cached: Optional[torch.Tensor] = None
        if is_fsal:
            t_batch = torch.full(
                (x.size(0),), t_current, device=x.device, dtype=x.dtype
            )
            k1_cached = drift_fn(x, t_batch)

        for _ in range(self.max_steps):
            if t_current >= t_end - 1e-12 * max(abs(t_end), 1.0):
                break

            h = min(h, t_end - t_current)
            h_t = torch.tensor(h, device=x.device, dtype=x.dtype)
            t_batch = torch.full(
                (x.size(0),), t_current, device=x.device, dtype=x.dtype
            )

            k = self._evaluate_stages(x, t_batch, h_t, drift_fn, k0=k1_cached)
            y_new = self._combine_stages(x, h_t, k)

            # Error estimation
            if is_fsal:
                k_fsal = drift_fn(y_new, t_batch + h_t)
                k_err = k + [k_fsal]
            else:
                k_err = k

            err_vec = torch.zeros_like(x)
            for i in range(len(e)):
                if e[i] != 0:
                    err_vec = err_vec + e[i] * k_err[i]
            err_vec = h_t * err_vec

            scale = self.atol + self.rtol * torch.max(x.abs(), y_new.abs())
            err_ratio = torch.sqrt(
                torch.mean((err_vec / scale) ** 2)
            ).item()

            if err_ratio <= 1.0:
                x = y_new
                t_current += h
                k1_cached = k_fsal if is_fsal else None

            if err_ratio == 0.0:
                factor = self.max_factor
            else:
                factor = min(
                    self.max_factor,
                    max(
                        self.min_factor,
                        self.safety * err_ratio ** (-1.0 / p),
                    ),
                )
            h = h * factor
        else:
            raise RuntimeError(
                f"{type(self).__name__}: maximum number of steps "
                f"({self.max_steps}) exceeded."
            )

        return x

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
        """Advance the state by one step according to the RK update rule.
        
        Args:
            state: Mapping containing required tensors (e.g., {'x': ...}).
            model: Energy-based model providing `forward` and `gradient`.
            step_size: Step size for the integration.
            drift: Explicit drift callable `f(x, t)`.  Falls back to
                `-model.gradient(x)` when `None`.
            diffusion: Time-dependent diffusion tensor `D(x, t)` for the SDE noise term.
            noise: Pre-sampled noise tensor for the SDE term.  When `None`,
                standard normal noise is generated internally.
            noise_scale: Scalar whose square is used as `D` when `diffusion` is not given.
            t: Current time tensor (batch,).  Required if `drift` or `diffusion` is time-dependent.

        Returns:
            Updated state dict with the same keys as the input `state`.

        !!!note:
            The `step` method implements a single RK update and is used by the `integrate` method to perform multiple steps.  The `integrate` method also handles adaptive step-size control when `adaptive=True` and the integrator supports it.
        """
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
        adaptive: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        r"""Integrate the state over a time interval.

        Args:
            state: Mapping with key ``"x"`` holding the position tensor.
            model: Energy-based model whose negative gradient defines the
                drift.  Ignored when ``drift`` is provided.
            step_size: Uniform step size (fixed mode) or initial step size
                (adaptive mode).
            n_steps: Number of integration steps (fixed mode) or, together
                with ``step_size``, defines the integration interval when
                ``t`` is ``None``.
            drift: Explicit drift callable ``f(x, t)``.  Falls back to
                ``-model.gradient(x)`` when ``None``.
            diffusion: Time-dependent diffusion callable ``D(x, t)`` for
                the SDE noise term.
            noise_scale: Scalar whose square is used as \(D\) when
                ``diffusion`` is not given.
            t: 1-D time grid.  Built from ``step_size`` when ``None``.
                In adaptive mode only ``t[0]`` and ``t[-1]`` are used.
            adaptive: ``True`` for adaptive step-size control, ``False``
                for fixed-step.  When ``None`` (default) adaptive mode
                is used automatically if :attr:`error_weights` is defined.

        Returns:
            Updated state dict ``{"x": x_final}``.
        """
        if adaptive is None:
            adaptive = self.error_weights is not None

        # fixed-step path
        if not adaptive:
            if n_steps <= 0:
                raise ValueError("n_steps must be positive")
            if t is None:
                if not torch.is_tensor(step_size):
                    step_size = torch.tensor(
                        step_size,
                        device=state["x"].device,
                        dtype=state["x"].dtype,
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

        # adaptive path
        if self.error_weights is None or self.order is None:
            raise ValueError(
                f"{type(self).__name__} does not define error_weights/order "
                f"and cannot be used with adaptive=True."
            )
        if diffusion is not None or noise_scale is not None:
            raise ValueError(
                "Adaptive stepping is only supported for ODEs. "
                "Pass adaptive=False for SDE integration."
            )

        x = state["x"]
        drift_fn = self._resolve_drift(model, drift)

        if not torch.is_tensor(step_size):
            step_size = torch.tensor(
                step_size, device=x.device, dtype=x.dtype
            )

        if t is not None:
            if t.ndim != 1 or t.numel() < 2:
                raise ValueError("t must be a 1D tensor with length >= 2")
            t_start = t[0].item()
            t_end = t[-1].item()
        else:
            t_start = 0.0
            t_end = float(n_steps) * step_size.item()

        h = min(step_size.item(), t_end - t_start)
        x = self._adaptive_integrate(x, drift_fn, t_start, t_end, h)
        return {"x": x}

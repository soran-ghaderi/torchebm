r"""Symplectic leapfrog (Störmer-Verlet) integrator for Hamiltonian dynamics."""

from typing import Callable, Dict, Optional, Tuple, Union

import torch

from torchebm.core import BaseSymplecticIntegrator


class LeapfrogIntegrator(BaseSymplecticIntegrator):
    r"""
    Symplectic leapfrog (Störmer–Verlet) integrator for Hamiltonian dynamics.

    Update rule:

    \[
    p_{t+1/2} = p_t - \frac{\epsilon}{2} \nabla_x U(x_t)
    \]

    \[
    x_{t+1} = x_t + \epsilon p_{t+1/2}
    \]

    \[
    p_{t+1} = p_{t+1/2} - \frac{\epsilon}{2} \nabla_x U(x_{t+1})
    \]

    Args:
        device: Device for computations.
        dtype: Data type for computations.

    Example:
        ```python
        from torchebm.integrators import LeapfrogIntegrator
        import torch

        energy_fn = ...  # an energy model with .gradient()
        integrator = LeapfrogIntegrator()
        state = {"x": torch.randn(100, 2), "p": torch.randn(100, 2)}
        drift = lambda x, t: -energy_fn.gradient(x)
        result = integrator.integrate(state, step_size=0.01, n_steps=10, drift=drift)
        ```
    """

    separable = True

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self._mass_view_ndim: Optional[int] = None
        self._mass_view_shape: Optional[tuple] = None

    def _broadcast_mass(self, mass: torch.Tensor, ndim: int) -> torch.Tensor:
        r"""Reshape ``mass`` for broadcasting against an ``ndim``-D state tensor."""
        if self._mass_view_ndim != ndim:
            self._mass_view_shape = (1,) * (ndim - 1) + (-1,)
            self._mass_view_ndim = ndim
        return mass.view(self._mass_view_shape)

    def step(
        self,
        state: Dict[str, torch.Tensor],
        step_size: torch.Tensor = None,
        mass: Optional[Union[float, torch.Tensor]] = None,
        *,
        drift: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        safe: bool = False,
    ) -> Dict[str, torch.Tensor]:
        r"""Advance one leapfrog step.

        Args:
            state: Current Hamiltonian state with keys `"x"` and `"p"`.
            step_size: Integration step size.
            mass: Optional mass term. Can be a scalar float or tensor.
            drift: Drift/force callable with signature `(x, t) -> force`.
            safe: If `True`, clamps force magnitudes and replaces NaNs by zeros.

        Returns:
            Updated state dictionary with keys `"x"` and `"p"`.
        """
        x, p, step_size, t = self._unpack_state(state, step_size)
        drift_fn = self._resolve_drift(drift)

        force = drift_fn(x, t)
        if safe:
            self._safe_clamp_(force)

        p_half = p + 0.5 * step_size * force

        if mass is None:
            x_new = x + step_size * p_half
        else:
            if isinstance(mass, float):
                safe_mass = max(mass, 1e-10)
                x_new = x + step_size * p_half / safe_mass
            else:
                safe_mass = torch.clamp(mass, min=1e-10)
                x_new = x + step_size * p_half / self._broadcast_mass(
                    safe_mass, x.ndim
                )

        force_new = drift_fn(x_new, t)
        if safe:
            self._safe_clamp_(force_new)
        p_new = p_half + 0.5 * step_size * force_new

        if safe:
            self._sanitize_state_(x_new, p_new)
        return {"x": x_new, "p": p_new}

    def integrate(
        self,
        state: Dict[str, torch.Tensor],
        step_size: torch.Tensor = None,
        n_steps: int = None,
        mass: Optional[Union[float, torch.Tensor]] = None,
        *,
        drift: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        safe: bool = False,
        inference_mode: bool = False,
    ) -> Dict[str, torch.Tensor]:
        r"""Integrate Hamiltonian dynamics for multiple leapfrog steps.

        Args:
            state: Initial Hamiltonian state with keys `"x"` and `"p"`.
            step_size: Integration step size.
            n_steps: Number of leapfrog steps to apply. Must be positive.
            mass: Optional mass term. Can be a scalar float or tensor.
            drift: Drift/force callable with signature `(x, t) -> force`.
            safe: If `True`, clamps force magnitudes and replaces NaNs by zeros.
            inference_mode: If `True`, runs integration under
                `torch.inference_mode()`.

        Returns:
            Final state dictionary with keys `"x"` and `"p"`.

        Raises:
            ValueError: If `n_steps <= 0`.
        """
        self._validate_n_steps(n_steps)

        if inference_mode:
            with torch.inference_mode():
                return self.integrate(
                    state, step_size=step_size,
                    n_steps=n_steps, mass=mass,
                    drift=drift, safe=safe,
                )

        drift_fn = self._resolve_drift(drift)
        x, p, step_size, t = self._unpack_state(state, step_size)

        for _ in range(n_steps):
            force = drift_fn(x, t)
            if safe:
                self._safe_clamp_(force)

            p_half = p + 0.5 * step_size * force

            if mass is None:
                x = x + step_size * p_half
            else:
                if isinstance(mass, float):
                    safe_mass = max(mass, 1e-10)
                    x = x + step_size * p_half / safe_mass
                else:
                    safe_mass = torch.clamp(mass, min=1e-10)
                    x = x + step_size * p_half / self._broadcast_mass(
                        safe_mass, x.ndim
                    )

            force_new = drift_fn(x, t)
            if safe:
                self._safe_clamp_(force_new)
            p = p_half + 0.5 * step_size * force_new

            if safe:
                self._sanitize_state_(x, p)

        return {"x": x, "p": p}


HamiltonField = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]


class GeneralisedLeapfrogIntegrator(BaseSymplecticIntegrator):
    r"""
    Generalised leapfrog (Störmer–Verlet) integrator for non-separable
    Hamiltonian dynamics.

    Whereas `LeapfrogIntegrator` assumes a separable Hamiltonian
    \(H(x, p) = U(x) + K(p)\) — so that \(\partial H/\partial x\) depends
    only on \(x\) and \(\partial H/\partial p\) only on \(p\) — this
    integrator handles the general case where both partial derivatives may
    depend on \((x, p)\). This is the setting of Riemann-manifold HMC
    (Girolami & Calderhead, 2011), where the kinetic term carries a
    position-dependent metric \(G(x)\).

    **Relation to `LeapfrogIntegrator`.** The `force` callable is the
    non-separable generalisation of `LeapfrogIntegrator`'s `drift`. For a
    separable Hamiltonian, `force(x, p, t)` reduces to `drift(x, t)`
    (\(= -\nabla_x U\)) and `velocity(x, p, t)` reduces to \(p\), and this
    scheme recovers the standard leapfrog update exactly. The Hamiltonian is
    assumed autonomous, so the time argument `t` is passed to the callables
    only for signature uniformity and is held at zero.

    Hamilton's equations are written directly:

    \[
    \dot{x} = \frac{\partial H}{\partial p}(x, p, t) \equiv \text{velocity},
    \qquad
    \dot{p} = -\frac{\partial H}{\partial x}(x, p, t) \equiv \text{force}.
    \]

    Update rule (a symmetric, time-reversible composition):

    \[
    p_{t+1/2} = p_t + \tfrac{\epsilon}{2}\,
        \text{force}\!\left(x_t,\, p_{t+1/2},\, t\right)
        \quad\text{(implicit in } p_{t+1/2}\text{)}
    \]

    \[
    x_{t+1} = x_t + \tfrac{\epsilon}{2}\Bigl[
        \text{velocity}\!\left(x_t,\, p_{t+1/2},\, t\right) +
        \text{velocity}\!\left(x_{t+1},\, p_{t+1/2},\, t\right)\Bigr]
        \quad\text{(implicit in } x_{t+1}\text{)}
    \]

    \[
    p_{t+1} = p_{t+1/2} + \tfrac{\epsilon}{2}\,
        \text{force}\!\left(x_{t+1},\, p_{t+1/2},\, t\right).
    \]

    The two implicit stages are solved by Picard (fixed-point) iteration.
    For a separable Hamiltonian both stages converge in a single iteration
    and the scheme reduces to the standard leapfrog.

    Args:
        device: Device for computations.
        dtype: Data type for computations.
        solver_max_iter: Total Picard iterations per implicit stage
            (warm start + refinements). The default of ``8`` matches the
            implicit-RK solver in `BaseRungeKuttaIntegrator`.
        solver_tol: RMS residual threshold for early termination. Only
            consulted when ``solver_check_every > 0``.
        solver_check_every: When positive, check the residual every
            ``n`` iterations and exit early once below ``solver_tol``.
            Each check incurs one CPU–GPU sync; leave at ``0`` for the
            fastest fixed-iteration path.

    Example:
        ```python
        from torchebm.integrators import GeneralisedLeapfrogIntegrator
        import torch

        # Toy non-separable Hamiltonian with a 1-D position-dependent
        # metric M(x) = 1 + x^2 and potential U(x) = 0.5 * x^2.
        def force(x, p, t):
            dU_dx = x
            inv_M = 1.0 / (1.0 + x ** 2)
            dKinv_dx = -2.0 * x * inv_M ** 2
            dlogdet_dx = 2.0 * x * inv_M
            return -(dU_dx + 0.5 * p ** 2 * dKinv_dx + 0.5 * dlogdet_dx)

        def velocity(x, p, t):
            return p / (1.0 + x ** 2)

        integrator = GeneralisedLeapfrogIntegrator()
        state = {"x": torch.randn(100, 1), "p": torch.randn(100, 1)}
        result = integrator.integrate(
            state, step_size=0.01, n_steps=20,
            force=force, velocity=velocity,
        )
        ```
    """

    separable = False

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        *,
        solver_max_iter: int = 8,
        solver_tol: float = 1e-6,
        solver_check_every: int = 0,
    ):
        super().__init__(device=device, dtype=dtype)
        if solver_max_iter < 1:
            raise ValueError("solver_max_iter must be >= 1")
        self.solver_max_iter = solver_max_iter
        self.solver_tol = solver_tol
        self.solver_check_every = solver_check_every

    @staticmethod
    def _resolve_hamilton_fields(
        force: Optional[HamiltonField],
        velocity: Optional[HamiltonField],
    ) -> Tuple[HamiltonField, HamiltonField]:
        r"""Validate and return ``(force, velocity)`` callables.

        Raises:
            ValueError: If either callable is ``None``.
        """
        if force is None or velocity is None:
            raise ValueError(
                "Both `force` (=-∂H/∂x) and `velocity` (=∂H/∂p) must be "
                "provided as callables (x, p, t) -> Tensor."
            )
        return force, velocity

    def _picard(
        self,
        init: torch.Tensor,
        update: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        r"""Picard iteration \(y_{k+1} = \text{update}(y_k)\) starting from ``init``.

        Runs ``solver_max_iter`` total ``update`` calls (the first acts as
        a warm start from ``init``). When ``solver_check_every > 0``, exits
        early once the RMS residual falls below ``solver_tol`` — at the
        cost of one CPU sync per check.
        """
        y = update(init)
        if self.solver_check_every <= 0:
            for _ in range(self.solver_max_iter - 1):
                y = update(y)
            return y
        for it in range(1, self.solver_max_iter):
            y_next = update(y)
            if it % self.solver_check_every == 0:
                resid = (y_next - y).square().mean().sqrt()
                y = y_next
                # Inherent host sync for the data-dependent convergence check;
                # bounded to one every `solver_check_every` iterations (explicit
                # integrators avoid it). See the GPU-first contract in the docs.
                if resid.item() < self.solver_tol:
                    break
            else:
                y = y_next
        return y

    def step(
        self,
        state: Dict[str, torch.Tensor],
        step_size: Optional[Union[float, torch.Tensor]] = None,
        *,
        force: Optional[HamiltonField] = None,
        velocity: Optional[HamiltonField] = None,
        safe: bool = False,
    ) -> Dict[str, torch.Tensor]:
        r"""Advance one generalised-leapfrog step.

        Args:
            state: Current Hamiltonian state with keys ``"x"`` and ``"p"``.
            step_size: Integration step size.
            force: Callable ``(x, p, t) -> -∂H/∂x``.
            velocity: Callable ``(x, p, t) -> ∂H/∂p``.
            safe: If ``True``, clamps intermediate values and replaces
                NaNs in the final state by zeros.

        Returns:
            Updated state dictionary with keys ``"x"`` and ``"p"``.
        """
        x, p, step_size, t = self._unpack_state(state, step_size)
        force_fn, velocity_fn = self._resolve_hamilton_fields(force, velocity)

        half = 0.5 * step_size

        # Stage 1: implicit half-step momentum at the current position.
        #     p_half = p + (ε/2) * force(x, p_half, t)
        p_half = self._picard(
            init=p,
            update=lambda ph: torch.addcmul(p, half, force_fn(x, ph, t)),
        )
        if safe:
            self._safe_clamp_(p_half)

        # Stage 2: implicit position update (trapezoidal in velocity).
        #     x_new = x + (ε/2) * (velocity(x, p_half, t)
        #                          + velocity(x_new, p_half, t))
        v_at_x = velocity_fn(x, p_half, t)
        base_x = torch.addcmul(x, half, v_at_x)
        x_new = self._picard(
            init=x,
            update=lambda xn: torch.addcmul(
                base_x, half, velocity_fn(xn, p_half, t)
            ),
        )
        if safe:
            self._safe_clamp_(x_new)

        # Stage 3: explicit half-step momentum at the new position.
        force_new = force_fn(x_new, p_half, t)
        if safe:
            # Out-of-place on purpose: `force_new` may alias a tensor owned
            # by the caller's `force_fn`.
            force_new = force_new.clamp(
                min=-self._SAFE_CLAMP, max=self._SAFE_CLAMP
            )
        p_new = torch.addcmul(p_half, half, force_new)

        if safe:
            self._sanitize_state_(x_new, p_new)

        return {"x": x_new, "p": p_new}

    def integrate(
        self,
        state: Dict[str, torch.Tensor],
        step_size: Optional[Union[float, torch.Tensor]] = None,
        n_steps: Optional[int] = None,
        *,
        force: Optional[HamiltonField] = None,
        velocity: Optional[HamiltonField] = None,
        safe: bool = False,
        inference_mode: bool = False,
    ) -> Dict[str, torch.Tensor]:
        r"""Integrate non-separable Hamiltonian dynamics for ``n_steps``.

        Args:
            state: Initial Hamiltonian state with keys ``"x"`` and ``"p"``.
            step_size: Integration step size.
            n_steps: Number of generalised-leapfrog steps. Must be positive.
            force: Callable ``(x, p, t) -> -∂H/∂x``.
            velocity: Callable ``(x, p, t) -> ∂H/∂p``.
            safe: If ``True``, clamps intermediate values and replaces
                NaNs by zeros after each step.
            inference_mode: If ``True``, runs integration under
                ``torch.inference_mode()``.

        Returns:
            Final state dictionary with keys ``"x"`` and ``"p"``.

        Raises:
            ValueError: If ``n_steps <= 0`` or the required callables are
                missing.
        """
        self._validate_n_steps(n_steps)

        if inference_mode:
            with torch.inference_mode():
                return self.integrate(
                    state, step_size=step_size, n_steps=n_steps,
                    force=force, velocity=velocity, safe=safe,
                )

        force_fn, velocity_fn = self._resolve_hamilton_fields(force, velocity)
        x, p, step_size, t = self._unpack_state(state, step_size)

        half = 0.5 * step_size

        for _ in range(n_steps):
            p_half = self._picard(
                init=p,
                update=lambda ph: torch.addcmul(p, half, force_fn(x, ph, t)),
            )
            if safe:
                self._safe_clamp_(p_half)

            v_at_x = velocity_fn(x, p_half, t)
            base_x = torch.addcmul(x, half, v_at_x)
            x_new = self._picard(
                init=x,
                update=lambda xn: torch.addcmul(
                    base_x, half, velocity_fn(xn, p_half, t)
                ),
            )
            if safe:
                self._safe_clamp_(x_new)

            force_new = force_fn(x_new, p_half, t)
            if safe:
                force_new = force_new.clamp(
                    min=-self._SAFE_CLAMP, max=self._SAFE_CLAMP
                )
            p_new = torch.addcmul(p_half, half, force_new)

            if safe:
                self._sanitize_state_(x_new, p_new)

            x, p = x_new, p_new

        return {"x": x, "p": p}

r"""Hamiltonian Monte Carlo Sampler Module and Riemannian Manifold Hamiltonian Monte Carlo sampler."""

import math
import warnings
from typing import Callable, Dict, Optional, Tuple, Union

import torch

from torchebm.core import (
    BaseModel,
    BaseSampler,
    BaseScheduler,
    BaseSymplecticIntegrator,
)
from torchebm.integrators import GeneralisedLeapfrogIntegrator
from torchebm.integrators.integrator_utils import resolve_integrator


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
        integrator: Separable symplectic integrator used for trajectories.
            `None` (default) uses `LeapfrogIntegrator`; a registry name
            (e.g. `"leapfrog"`) constructs that integrator with defaults;
            a separable `BaseSymplecticIntegrator` instance is used as-is
            and must match the sampler's device/dtype.

    Example:
        ```python
        from torchebm.samplers import HamiltonianMonteCarlo
        from torchebm.core import DoubleWellModel
        import torch

        energy = DoubleWellModel()
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
        integrator: Union[str, BaseSymplecticIntegrator, None] = None,
    ):
        super().__init__(model=model, dtype=dtype, device=device)
        self._register_param("step_size", step_size, positive=True)

        if n_leapfrog_steps <= 0:
            raise ValueError("n_leapfrog_steps must be positive")

        self.n_leapfrog_steps = n_leapfrog_steps
        self.mass = (
            mass.to(self.device)
            if (mass is not None and not isinstance(mass, float))
            else mass
        )
        integ = resolve_integrator(
            integrator,
            default="leapfrog",
            family=BaseSymplecticIntegrator,
            owner="HamiltonianMonteCarlo",
            device=self.device,
            dtype=self.dtype,
        )
        if not integ.separable:
            raise TypeError(
                "HamiltonianMonteCarlo requires a separable symplectic "
                "integrator (drift/mass contract); got non-separable "
                f"{type(integ).__name__}. Use RiemannianManifoldHMC for "
                "non-separable Hamiltonians."
            )
        self.integrator = integ

    def _initialize_momentum(self, shape: torch.Size) -> torch.Tensor:
        r"""
        Initializes momentum variables from a Gaussian distribution.

        The momentum is sampled from \(\mathcal{N}(0, M)\), where `M` is the mass matrix.
        Reuses a cached buffer when shape/dtype/device match to avoid per-step allocation.

        Args:
            shape (torch.Size): The shape of the momentum tensor to generate.

        Returns:
            torch.Tensor: The initialized momentum tensor.
        """
        buf = getattr(self, "_momentum_buf", None)
        if (
            buf is None
            or buf.shape != shape
            or buf.dtype != self.dtype
            or buf.device != self.device
        ):
            buf = torch.empty(shape, dtype=self.dtype, device=self.device)
            self._momentum_buf = buf
        p = buf.normal_()

        if self.mass is not None:
            # Apply mass matrix (equivalent to sampling from N(0, M))
            if isinstance(self.mass, float):
                if getattr(self, "_mass_sqrt_float", None) is None:
                    self._mass_sqrt_float = math.sqrt(self.mass)
                p.mul_(self._mass_sqrt_float)
            else:
                ndim = p.ndim
                cached = getattr(self, "_mass_sqrt_view", None)
                if cached is None or cached[0] != ndim:
                    mass_sqrt = torch.sqrt(self.mass)
                    view_shape = (1,) * (ndim - 1) + (-1,)
                    self._mass_sqrt_view = (ndim, mass_sqrt.view(view_shape))
                    cached = self._mass_sqrt_view
                p.mul_(cached[1])
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
            return 0.5 * torch.sum(p.square(), dim=-1)
        elif isinstance(self.mass, float):
            return 0.5 * torch.sum(p.square(), dim=-1) / self.mass
        else:
            ndim = p.ndim
            cached = getattr(self, "_mass_kin_view", None)
            if cached is None or cached[0] != ndim:
                view_shape = (1,) * (ndim - 1) + (-1,)
                self._mass_kin_view = (ndim, self.mass.view(view_shape))
                cached = self._mass_kin_view
            return 0.5 * torch.sum(p.square() / cached[1], dim=-1)

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
        reset_schedulers: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        r"""Generate samples via Hamiltonian Monte Carlo.

        Args:
            x: Initial state. If `None`, samples from `N(0, I)`.
            dim: State-space dimension (used when `x is None`); inferred from
                `model.mean` when available.
            n_steps: Number of MH proposals.
            n_samples: Number of parallel chains.
            thin: Keep every `thin`-th sample (final length `n_steps // thin`).
            return_trajectory: If True, return the full kept trajectory.
            return_diagnostics: If True, also return a dict with keys
                ``"mean"`` (`[n_kept, dim]`), ``"var"`` (`[n_kept, dim]`),
                ``"energy"`` (`[n_kept]`), and ``"acceptance_rate"`` (`[n_kept]`).
            reset_schedulers: If True (default), reset registered schedulers.

        Raises:
            ValueError: If `thin < 1`.
        """
        if thin < 1:
            raise ValueError("thin must be >= 1")
        if reset_schedulers:
            self.reset_schedulers()

        if x is None and dim is None:
            if hasattr(self.model, "mean") and isinstance(
                self.model.mean, torch.Tensor
            ):
                dim = self.model.mean.shape[0]
            else:
                raise ValueError(
                    "dim must be provided when x is None and cannot be inferred from model"
                )
        x = self._init_state(x, dim, n_samples)

        dim = x.shape[1]
        batch_size = x.shape[0]
        n_kept = n_steps // thin

        if return_trajectory:
            trajectory = torch.empty(
                (batch_size, n_kept, dim),
                dtype=self.dtype,
                device=self.device,
                requires_grad=False,
            )

        diagnostics: Optional[Dict[str, torch.Tensor]] = None
        if return_diagnostics:
            diagnostics = {
                "mean": torch.empty(n_kept, dim, dtype=self.dtype, device=self.device),
                "var": torch.empty(n_kept, dim, dtype=self.dtype, device=self.device),
                "energy": torch.empty(n_kept, dtype=self.dtype, device=self.device),
                "acceptance_rate": torch.empty(
                    n_kept, dtype=self.dtype, device=self.device
                ),
            }

        keep_idx = 0
        with self.autocast_context():
            for i in range(n_steps):
                current_momentum = self._initialize_momentum(x.shape)

                current_energy = self.model(x).clamp_(min=-1e10, max=1e10)
                current_kinetic = self._compute_kinetic_energy(
                    current_momentum
                ).clamp_(min=0.0, max=1e10)

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

                proposed_energy = self.model(proposed_position).clamp_(
                    min=-1e10, max=1e10
                )
                proposed_kinetic = self._compute_kinetic_energy(
                    proposed_momentum
                ).clamp_(min=0.0, max=1e10)

                proposed_hamiltonian = proposed_energy + proposed_kinetic

                hamiltonian_diff = (current_hamiltonian - proposed_hamiltonian).clamp_(
                    min=-50.0, max=50.0
                )

                # acceptance_prob = min(1, exp(diff)); fused via in-place clamp on
                # the freshly allocated `exp` result (no `ones` tensor allocation).
                acceptance_prob = torch.exp(hamiltonian_diff).clamp_(max=1.0)

                random_uniform = torch.rand(batch_size, device=self.device)
                accepted = random_uniform < acceptance_prob
                # `torch.where` on the broadcast accept mask: single fused kernel,
                # no `mask*proposed + (1-mask)*x` quartet of temporaries.
                accept_view = accepted.view(-1, *([1] * (x.ndim - 1)))
                x = torch.where(accept_view, proposed_position, x)

                if (i + 1) % thin == 0:
                    if return_trajectory:
                        trajectory[:, keep_idx, :] = x
                    if return_diagnostics:
                        diagnostics["mean"][keep_idx] = x.mean(dim=0)
                        diagnostics["var"][keep_idx] = (
                            x.var(dim=0, unbiased=False).clamp_(min=1e-10, max=1e10)
                            if batch_size > 1
                            else torch.zeros(dim, dtype=self.dtype, device=self.device)
                        )
                        diagnostics["energy"][keep_idx] = (
                            self.model(x).clamp_(min=-1e10, max=1e10).mean()
                        )
                        diagnostics["acceptance_rate"][keep_idx] = accepted.float().mean()
                    keep_idx += 1

                self.step_schedulers()

        output = trajectory if return_trajectory else x
        return (output, diagnostics) if return_diagnostics else output


class RiemannianManifoldHMC(BaseSampler):
    r"""
    Riemannian Manifold Hamiltonian Monte Carlo (RMHMC).

    Hamiltonian Monte Carlo on a Riemannian manifold equipped with a
    position-dependent metric tensor \(G(x)\) (Girolami & Calderhead, 2011).
    Whereas standard HMC uses a constant mass matrix and a separable
    Hamiltonian \(H(x, p) = U(x) + \tfrac{1}{2} p^T M^{-1} p\), RMHMC adapts
    the local geometry to the target via

    \[
    H(x, p) = U(x) + \tfrac{1}{2} p^T G(x)^{-1} p
              + \tfrac{1}{2} \log\!\left|G(x)\right|.
    \]

    The induced Hamilton's equations are non-separable, so trajectories
    are simulated with a `GeneralisedLeapfrogIntegrator` whose two
    implicit stages are solved by Picard iteration. The integrator is
    volume-preserving and reversible, which keeps the Metropolis–Hastings
    acceptance correction valid.

    Args:
        model: Energy-based model defining the potential \(U(x)\).
        metric_fn: Callable ``x -> G(x)`` returning a symmetric positive-
            definite tensor of shape ``(batch, dim, dim)``. The metric is
            evaluated under `torch.enable_grad` when computing the
            force, so ``metric_fn`` must be differentiable w.r.t. ``x``.
        step_size: Step size for the generalised-leapfrog integrator.
        n_leapfrog_steps: Number of integrator steps per MH proposal.
        solver_max_iter: Deprecated. Construct a
            `GeneralisedLeapfrogIntegrator` with the desired solver
            settings and pass it as ``integrator`` instead.
        solver_tol: Deprecated. See ``solver_max_iter``.
        solver_check_every: Deprecated. See ``solver_max_iter``.
        dtype: Data type for computations.
        device: Device for computations.
        integrator: Non-separable symplectic integrator used for
            trajectories. `None` (default) uses
            `GeneralisedLeapfrogIntegrator`; a registry name (e.g.
            `"generalised_leapfrog"`) constructs that integrator with
            defaults; a non-separable `BaseSymplecticIntegrator` instance
            is used as-is and must match the sampler's device/dtype.

    Example:
        ```python
        from torchebm.core import GaussianModel
        from torchebm.samplers import RiemannianManifoldHMC
        import torch

        dim = 2
        model = GaussianModel(mean=torch.zeros(dim), cov=torch.eye(dim))

        # Identity metric — reduces to plain HMC.
        def metric_fn(x):
            eye = torch.eye(dim, dtype=x.dtype, device=x.device)
            return eye.expand(x.shape[0], dim, dim).contiguous()

        sampler = RiemannianManifoldHMC(
            model, metric_fn=metric_fn,
            step_size=0.1, n_leapfrog_steps=10,
        )
        samples = sampler.sample(n_samples=200, dim=dim, n_steps=500)
        ```
    """

    def __init__(
        self,
        model: BaseModel,
        metric_fn: Callable[[torch.Tensor], torch.Tensor],
        step_size: Union[float, BaseScheduler] = 1e-3,
        n_leapfrog_steps: int = 10,
        solver_max_iter: Optional[int] = None,
        solver_tol: Optional[float] = None,
        solver_check_every: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        integrator: Union[str, BaseSymplecticIntegrator, None] = None,
    ):
        super().__init__(model=model, dtype=dtype, device=device)
        if not callable(metric_fn):
            raise TypeError("metric_fn must be callable: x -> G(x)")
        if n_leapfrog_steps <= 0:
            raise ValueError("n_leapfrog_steps must be positive")

        self._register_param("step_size", step_size, positive=True)
        self.metric_fn = metric_fn
        self.n_leapfrog_steps = n_leapfrog_steps

        solver_kwargs_given = not (
            solver_max_iter is None
            and solver_tol is None
            and solver_check_every is None
        )
        if solver_kwargs_given:
            if integrator is not None:
                raise ValueError(
                    "Pass either integrator= or the deprecated solver_* "
                    "arguments, not both. Set the solver options on the "
                    "GeneralisedLeapfrogIntegrator instance instead."
                )
            warnings.warn(
                "solver_max_iter/solver_tol/solver_check_every on "
                "RiemannianManifoldHMC are deprecated; construct "
                "GeneralisedLeapfrogIntegrator(solver_max_iter=..., ...) "
                "and pass it as integrator=.",
                DeprecationWarning,
                stacklevel=2,
            )
        if solver_kwargs_given:
            # Deprecated path (integrator= excluded above): thread the old
            # solver options into a directly constructed integrator.
            integ = GeneralisedLeapfrogIntegrator(
                device=self.device,
                dtype=self.dtype,
                solver_max_iter=(
                    solver_max_iter if solver_max_iter is not None else 8
                ),
                solver_tol=solver_tol if solver_tol is not None else 1e-6,
                solver_check_every=(
                    solver_check_every if solver_check_every is not None else 0
                ),
            )
        else:
            integ = resolve_integrator(
                integrator,
                default="generalised_leapfrog",
                family=BaseSymplecticIntegrator,
                owner="RiemannianManifoldHMC",
                device=self.device,
                dtype=self.dtype,
            )
        if integ.separable:
            raise TypeError(
                "RiemannianManifoldHMC requires a non-separable "
                "symplectic integrator (force/velocity contract); got "
                f"separable {type(integ).__name__}. Use "
                "HamiltonianMonteCarlo for separable Hamiltonians."
            )
        self.integrator = integ

    @staticmethod
    def _cholesky(G: torch.Tensor) -> torch.Tensor:
        r"""Batched lower-triangular Cholesky factor of an SPD metric tensor."""
        return torch.linalg.cholesky(G)

    def _solve_metric(self, L: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        r"""Return \(G^{-1} p\) given the Cholesky factor ``L`` of \(G\)."""
        return torch.cholesky_solve(p.unsqueeze(-1), L).squeeze(-1)

    @staticmethod
    def _logdet_from_chol(L: torch.Tensor) -> torch.Tensor:
        r"""Return \(\log\!\left|G\right| = 2 \sum_i \log L_{ii}\)."""
        return 2.0 * torch.diagonal(L, dim1=-2, dim2=-1).log().sum(dim=-1)

    def _metric_chol(self, x: torch.Tensor) -> torch.Tensor:
        r"""Cholesky factor of \(G(x)\), memoised for the most recent position.

        The metric and its factorisation are evaluated repeatedly at the same
        position within one sampling step (momentum draw, current kinetic
        energy, the integrator's first velocity evaluation). A single-entry
        cache keyed on tensor identity and version returns the cached factor
        for those hits while staying correct when ``x`` changes. The factor is
        computed under `torch.no_grad`, so a graph-carrying tensor is never
        cached; the autograd force path in `_force` deliberately bypasses this
        helper.
        """
        cache = getattr(self, "_chol_cache", None)
        if cache is not None and cache[0] is x and cache[1] == x._version:
            return cache[2]
        with torch.no_grad():
            L = self._cholesky(self.metric_fn(x))
        self._chol_cache = (x, x._version, L)
        return L

    def _kinetic_energy(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        *,
        L: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Compute \(K(x, p) = \tfrac{1}{2} p^T G(x)^{-1} p + \tfrac{1}{2} \log\!\left|G(x)\right|\).

        ``L`` is the Cholesky factor of \(G(x)\); when ``None`` it is computed
        here (the autograd path in `_force` relies on this to build the graph).
        """
        if L is None:
            L = self._cholesky(self.metric_fn(x))
        G_inv_p = self._solve_metric(L, p)
        quad = 0.5 * (p * G_inv_p).sum(dim=-1)
        return quad + 0.5 * self._logdet_from_chol(L)

    def _initialize_momentum(
        self,
        x: torch.Tensor,
        *,
        L: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Sample \(p \sim \mathcal{N}(0, G(x))\) via \(p = L z,\; z \sim \mathcal{N}(0, I)\).

        Reuses a cached standard-normal buffer to avoid a per-step allocation.
        """
        if L is None:
            L = self._metric_chol(x)
        buf = getattr(self, "_z_buf", None)
        if (
            buf is None
            or buf.shape != x.shape
            or buf.dtype != x.dtype
            or buf.device != x.device
        ):
            buf = torch.empty_like(x)
            self._z_buf = buf
        z = buf.normal_()
        return torch.einsum("bij,bj->bi", L, z)

    def _velocity(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        r"""\(\dot{x} = \partial H/\partial p = G(x)^{-1} p\)."""
        return self._solve_metric(self._metric_chol(x), p)

    def _force(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        r"""\(\dot{p} = -\partial H/\partial x\) computed by autograd through
        ``model`` and ``metric_fn``.

        Detaches inputs before enabling grad so the force can be called from
        inside the outer ``torch.no_grad`` sampling loop without polluting the
        autograd graph.
        """
        p_detached = p.detach()
        with torch.enable_grad():
            x_grad = x.detach().requires_grad_(True)
            U = self.model(x_grad)
            K = self._kinetic_energy(x_grad, p_detached)
            H = (U + K).sum()
            (dH_dx,) = torch.autograd.grad(H, x_grad)
        return -dH_dx

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
        reset_schedulers: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        r"""Generate samples via Riemannian-manifold HMC.

        Args:
            x: Initial state ``(n_samples, dim)``. If ``None``, samples from
                \(\mathcal{N}(0, I)\).
            dim: State-space dimension (used when ``x is None``); inferred
                from ``model.mean`` when available.
            n_steps: Number of MH proposals.
            n_samples: Number of parallel chains.
            thin: Keep every ``thin``-th sample.
            return_trajectory: If True, return the full kept trajectory.
            return_diagnostics: If True, also return a dict with keys
                ``"mean"``, ``"var"``, ``"energy"``, and ``"acceptance_rate"``.
            reset_schedulers: If True, reset registered schedulers.

        Raises:
            ValueError: If ``thin < 1`` or ``x`` is not 2-D.
        """
        if thin < 1:
            raise ValueError("thin must be >= 1")
        if reset_schedulers:
            self.reset_schedulers()

        if x is None and dim is None:
            if hasattr(self.model, "mean") and isinstance(
                self.model.mean, torch.Tensor
            ):
                dim = self.model.mean.shape[0]
            else:
                raise ValueError(
                    "dim must be provided when x is None and cannot be "
                    "inferred from model"
                )
        x = self._init_state(x, dim, n_samples)

        if x.ndim != 2:
            raise ValueError(
                f"RMHMC currently expects 2-D state tensors (batch, dim); "
                f"got x.ndim={x.ndim}."
            )

        batch_size, dim = x.shape
        n_kept = n_steps // thin

        trajectory: Optional[torch.Tensor] = None
        if return_trajectory:
            trajectory = torch.empty(
                (batch_size, n_kept, dim),
                dtype=self.dtype,
                device=self.device,
            )

        diagnostics: Optional[Dict[str, torch.Tensor]] = None
        if return_diagnostics:
            diagnostics = {
                "mean": torch.empty(n_kept, dim, dtype=self.dtype, device=self.device),
                "var": torch.empty(n_kept, dim, dtype=self.dtype, device=self.device),
                "energy": torch.empty(n_kept, dtype=self.dtype, device=self.device),
                "acceptance_rate": torch.empty(
                    n_kept, dtype=self.dtype, device=self.device
                ),
            }

        keep_idx = 0
        with self.autocast_context():
            for i in range(n_steps):
                L = self._metric_chol(x)
                p = self._initialize_momentum(x, L=L)

                current_U = self.model(x).clamp_(min=-1e10, max=1e10)
                current_K = self._kinetic_energy(x, p, L=L).clamp_(
                    min=-1e10, max=1e10
                )
                current_H = current_U + current_K

                proposed = self.integrator.integrate(
                    {"x": x, "p": p},
                    step_size=self.get_scheduled_value("step_size"),
                    n_steps=self.n_leapfrog_steps,
                    force=self._force,
                    velocity=self._velocity,
                    safe=True,
                )
                x_prop, p_prop = proposed["x"], proposed["p"]

                proposed_U = self.model(x_prop).clamp_(min=-1e10, max=1e10)
                proposed_K = self._kinetic_energy(
                    x_prop, p_prop, L=self._metric_chol(x_prop)
                ).clamp_(min=-1e10, max=1e10)
                proposed_H = proposed_U + proposed_K

                # exp(min(0, H_current - H_proposed)) = min(1, exp(diff))
                hamiltonian_diff = (current_H - proposed_H).clamp_(min=-50.0, max=50.0)
                acceptance_prob = torch.exp(hamiltonian_diff).clamp_(max=1.0)

                # Reject NaN/Inf proposals outright (otherwise the Cholesky at
                # the next iteration's momentum draw blows up).
                finite_proposal = torch.isfinite(x_prop).all(dim=-1) & torch.isfinite(
                    proposed_H
                )
                acceptance_prob = torch.where(
                    finite_proposal, acceptance_prob, torch.zeros_like(acceptance_prob)
                )

                accepted = torch.rand(batch_size, device=self.device) < acceptance_prob
                accept_view = accepted.view(-1, *([1] * (x.ndim - 1)))
                x = torch.where(accept_view, x_prop, x)

                if (i + 1) % thin == 0:
                    if return_trajectory:
                        trajectory[:, keep_idx, :] = x
                    if return_diagnostics:
                        diagnostics["mean"][keep_idx] = x.mean(dim=0)
                        diagnostics["var"][keep_idx] = (
                            x.var(dim=0, unbiased=False).clamp_(min=1e-10, max=1e10)
                            if batch_size > 1
                            else torch.zeros(
                                dim, dtype=self.dtype, device=self.device
                            )
                        )
                        diagnostics["energy"][keep_idx] = (
                            self.model(x).clamp_(min=-1e10, max=1e10).mean()
                        )
                        diagnostics["acceptance_rate"][keep_idx] = (
                            accepted.float().mean()
                        )
                    keep_idx += 1

                self.step_schedulers()

        output = trajectory if return_trajectory else x
        return (output, diagnostics) if return_diagnostics else output

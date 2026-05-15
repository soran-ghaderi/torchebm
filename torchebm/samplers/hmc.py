r"""Hamiltonian Monte Carlo Sampler Module."""

import math
from typing import Dict, Optional, Tuple, Union

import torch

from torchebm.core import (
    BaseModel,
    BaseSampler,
    BaseScheduler,
)
from torchebm.integrators import LeapfrogIntegrator


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
        *args,
        **kwargs,
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
        self.integrator = LeapfrogIntegrator(device=self.device, dtype=self.dtype)

    def _initialize_momentum(self, shape: torch.Size) -> torch.Tensor:
        """
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

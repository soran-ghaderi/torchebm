r"""Hamiltonian Monte Carlo Sampler Module."""

import math
from typing import Optional, Union, Tuple

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
                current_momentum = self._initialize_momentum(x.shape)

                momentum_direction = (
                    torch.randint(0, 2, (batch_size, 1), device=self.device) * 2 - 1
                )  # -1/+1 -> for sign flipping
                current_momentum = current_momentum * momentum_direction

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
                proposed_energy = self.model(proposed_position).clamp_(
                    min=-1e10, max=1e10
                )
                proposed_kinetic = self._compute_kinetic_energy(
                    proposed_momentum
                ).clamp_(min=0.0, max=1e10)
                proposed_kinetic = self._compute_kinetic_energy(
                    proposed_momentum
                ).clamp_(min=0.0, max=1e10)

                proposed_hamiltonian = proposed_energy + proposed_kinetic

                hamiltonian_diff = (current_hamiltonian - proposed_hamiltonian).clamp_(
                    min=-50.0, max=50.0
                )
                hamiltonian_diff = (current_hamiltonian - proposed_hamiltonian).clamp_(
                    min=-50.0, max=50.0
                )

                # acceptance_prob = min(1, exp(diff)); fused via in-place clamp on
                # the freshly allocated `exp` result (no `ones` tensor allocation).
                acceptance_prob = torch.exp(hamiltonian_diff).clamp_(max=1.0)
                # acceptance_prob = min(1, exp(diff)); fused via in-place clamp on
                # the freshly allocated `exp` result (no `ones` tensor allocation).
                acceptance_prob = torch.exp(hamiltonian_diff).clamp_(max=1.0)

                random_uniform = torch.rand(batch_size, device=self.device)
                accepted = random_uniform < acceptance_prob
                # `torch.where` on the broadcast accept mask: single fused kernel,
                # no `mask*proposed + (1-mask)*x` quartet of temporaries.
                accept_view = accepted.view(-1, *([1] * (x.ndim - 1)))
                x = torch.where(accept_view, proposed_position, x)
                # `torch.where` on the broadcast accept mask: single fused kernel,
                # no `mask*proposed + (1-mask)*x` quartet of temporaries.
                accept_view = accepted.view(-1, *([1] * (x.ndim - 1)))
                x = torch.where(accept_view, proposed_position, x)

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

                    mean_exp = mean_x.expand(batch_size, dim)
                    diagnostics[i, 0, :, :] = mean_exp
                    mean_exp = mean_x.expand(batch_size, dim)
                    diagnostics[i, 0, :, :] = mean_exp
                    diagnostics[i, 1, :, :] = var_x.expand(batch_size, dim)
                    diagnostics[i, 2, :, :] = energy.view(-1, 1).expand(-1, dim)
                    diagnostics[i, 3, :, :] = acceptance_rate

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

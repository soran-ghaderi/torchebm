r"""Langevin Dynamics Sampler Module."""

from typing import Dict, Optional, Tuple, Union

import torch

from torchebm.core import (
    BaseModel,
    BaseSampler,
    BaseScheduler,
)
from torchebm.integrators import EulerMaruyamaIntegrator


class LangevinDynamics(BaseSampler):
    r"""
    Langevin Dynamics sampler.

    Update rule:

    \[
    x_{t+1} = x_t - \eta \nabla_x U(x_t) + \sqrt{2\eta} \epsilon_t
    \]

    Args:
        model: Energy-based model to sample from.
        step_size: Step size for gradient descent. Float or `BaseScheduler`.
        noise_scale: Scale of Gaussian noise injection. Float or `BaseScheduler`.
        decay: Damping coefficient (not supported).
        dtype: Data type for computations.
        device: Device for computations.

    Example:
        ```python
        from torchebm.samplers import LangevinDynamics
        from torchebm.core import DoubleWellModel
        import torch

        energy = DoubleWellModel()
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

        self._register_param("step_size", step_size, positive=True)
        self._register_param("noise_scale", noise_scale, positive=True)

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
        reset_schedulers: bool = True,
        *args,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        r"""Generate samples via Langevin dynamics.

        Args:
            x: Initial state. If `None`, samples from `N(0, I)`.
            dim: State-space dimension (used when `x is None`).
            n_steps: Number of MCMC steps to perform.
            n_samples: Number of parallel chains to generate.
            thin: Keep every `thin`-th sample. Final stored length is
                `n_steps // thin`. Must be `>= 1`.
            return_trajectory: If True, return the full kept trajectory of shape
                `[n_samples, n_steps // thin, dim]`.
            return_diagnostics: If True, also return a dict with keys
                ``"mean"`` (`[n_kept, dim]`), ``"var"`` (`[n_kept, dim]`), and
                ``"energy"`` (`[n_kept]`).
            reset_schedulers: If True (default), reset registered schedulers.

        Returns:
            Sample tensor (or trajectory if `return_trajectory=True`),
            optionally paired with the diagnostics dict.

        Raises:
            ValueError: If `thin < 1`.
        """
        if thin < 1:
            raise ValueError("thin must be >= 1")
        if reset_schedulers:
            self.reset_schedulers()

        if x is None:
            x = torch.randn(n_samples, dim, dtype=self.dtype, device=self.device)
        else:
            x = x.to(device=self.device, dtype=self.dtype)
            dim = x.shape[-1]
            n_samples = x.shape[0]

        n_kept = n_steps // thin

        if return_trajectory:
            trajectory = torch.empty(
                (n_samples, n_kept, dim), dtype=self.dtype, device=self.device
            )

        diagnostics: Optional[Dict[str, torch.Tensor]] = None
        if return_diagnostics:
            diagnostics = {
                "mean": torch.empty(n_kept, dim, dtype=self.dtype, device=self.device),
                "var": torch.empty(n_kept, dim, dtype=self.dtype, device=self.device),
                "energy": torch.empty(n_kept, dtype=self.dtype, device=self.device),
            }

        drift = lambda x_, t_: -self.model.gradient(x_)
        keep_idx = 0
        with self.autocast_context():
            for i in range(n_steps):
                state = {"x": x}
                x = self.integrator.step(
                    state=state,
                    step_size=self.get_scheduled_value("step_size"),
                    noise_scale=self.get_scheduled_value("noise_scale"),
                    drift=drift,
                )["x"]
                self.step_schedulers()

                if (i + 1) % thin == 0:
                    if return_trajectory:
                        trajectory[:, keep_idx, :] = x
                    if return_diagnostics:
                        if n_samples > 1:
                            diagnostics["mean"][keep_idx] = x.mean(dim=0)
                            diagnostics["var"][keep_idx] = x.var(
                                dim=0, unbiased=False
                            ).clamp_(min=1e-10, max=1e10)
                        else:
                            diagnostics["mean"][keep_idx] = x.squeeze(0)
                            diagnostics["var"][keep_idx].zero_()
                        diagnostics["energy"][keep_idx] = self.model(x).mean()
                    keep_idx += 1

        output = trajectory if return_trajectory else x
        return (output, diagnostics) if return_diagnostics else output

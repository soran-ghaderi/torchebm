r"""Contrastive Divergence Loss Module."""

from typing import Union, Callable, Tuple, Any, Optional, Dict

import torch
from torch import nn
import math
from abc import abstractmethod
import warnings

from torchebm.core import BaseContrastiveDivergence


class ContrastiveDivergence(BaseContrastiveDivergence):
    r"""
    Standard Contrastive Divergence (CD-k) loss.

    CD approximates the log-likelihood gradient by running an MCMC sampler
    for `k_steps` to generate negative samples.

    Args:
        model: The energy-based model to train.
        sampler: The MCMC sampler for generating negative samples.
        k_steps: The number of MCMC steps (k in CD-k).
        persistent: If True, uses Persistent CD with a replay buffer.
        buffer_size: Size of the replay buffer for PCD.
        init_steps: Number of MCMC steps to warm up the buffer.
        new_sample_ratio: Fraction of new random samples for PCD chains.
        energy_reg_weight: Weight for energy regularization term.
        use_temperature_annealing: Whether to use temperature annealing.
        min_temp: Minimum temperature for annealing.
        max_temp: Maximum temperature for annealing.
        temp_decay: Decay rate for temperature annealing.
        dtype: Data type for computations.
        device: Device for computations.

    Example:
        ```python
        from torchebm.losses import ContrastiveDivergence
        from torchebm.samplers import LangevinDynamics
        from torchebm.core import DoubleWellEnergy

        energy = DoubleWellEnergy()
        sampler = LangevinDynamics(energy, step_size=0.01)
        cd_loss = ContrastiveDivergence(model=energy, sampler=sampler, k_steps=10)
        x = torch.randn(32, 2)
        loss, neg_samples = cd_loss(x)
        ```
    """

    def __init__(
        self,
        model,
        sampler,
        k_steps=10,
        persistent=False,
        buffer_size=10000,
        init_steps=100,
        new_sample_ratio=0.05,
        energy_reg_weight=0.001,
        use_temperature_annealing=False,
        min_temp=0.01,
        max_temp=2.0,
        temp_decay=0.999,
        dtype=torch.float32,
        device=torch.device("cpu"),
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            sampler=sampler,
            k_steps=k_steps,
            persistent=persistent,
            buffer_size=buffer_size,
            new_sample_ratio=new_sample_ratio,
            init_steps=init_steps,
            dtype=dtype,
            device=device,
            *args,
            **kwargs,
        )
        # Additional parameters for improved stability
        self.energy_reg_weight = energy_reg_weight
        self.use_temperature_annealing = use_temperature_annealing
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.temp_decay = temp_decay
        self.current_temp = max_temp

        # Register temperature as buffer for persistence
        self.register_buffer(
            "temperature", torch.tensor(max_temp, dtype=self.dtype, device=self.device)
        )

    def forward(
        self, x: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the Contrastive Divergence loss and generates negative samples.

        Args:
            x (torch.Tensor): A batch of real data samples (positive samples).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - The scalar CD loss value.
                - The generated negative samples.
        """

        batch_size = x.shape[0]
        data_shape = x.shape[1:]

        # Update temperature if annealing is enabled
        if self.use_temperature_annealing and self.training:
            self.current_temp = max(self.min_temp, self.current_temp * self.temp_decay)
            self.temperature[...] = self.current_temp  # Use ellipsis instead of index

            # If sampler has a temperature parameter, update it
            if hasattr(self.sampler, "temperature"):
                self.sampler.temperature = self.current_temp
            elif hasattr(self.sampler, "noise_scale"):
                # For samplers like Langevin, adjust noise scale based on temperature
                original_noise = getattr(self.sampler, "_original_noise_scale", None)
                if original_noise is None:
                    setattr(
                        self.sampler, "_original_noise_scale", self.sampler.noise_scale
                    )
                    original_noise = self.sampler.noise_scale

                self.sampler.noise_scale = original_noise * math.sqrt(self.current_temp)

        # Get starting points for chains (either from buffer or data)
        start_points = self.get_start_points(x)

        # Run MCMC chains to get negative samples
        pred_samples = self.sampler.sample(
            x=start_points,
            n_steps=self.k_steps,
        )

        # Update persistent buffer if using PCD
        if self.persistent:
            with torch.no_grad():
                self.update_buffer(pred_samples.detach())

        # Add energy regularization to kwargs for compute_loss
        kwargs["energy_reg_weight"] = kwargs.get(
            "energy_reg_weight", self.energy_reg_weight
        )

        # Compute contrastive divergence loss
        loss = self.compute_loss(x, pred_samples, *args, **kwargs)

        return loss, pred_samples

    def compute_loss(
        self, x: torch.Tensor, pred_x: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        Computes the Contrastive Divergence loss from positive and negative samples.

        The loss is the difference between the mean energy of positive samples
        and the mean energy of negative samples.

        Args:
            x (torch.Tensor): Real data samples (positive samples).
            pred_x (torch.Tensor): Generated negative samples.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The scalar loss value.
        """
        # Ensure inputs are on the correct device and dtype
        x = x.to(self.device, self.dtype)
        pred_x = pred_x.to(self.device, self.dtype)

        # Compute energy of real and generated samples
        with torch.set_grad_enabled(True):
            # Add small noise to real data for stability (optional)
            if kwargs.get("add_noise_to_real", False):
                noise_scale = kwargs.get("noise_scale", 1e-4)
                x_noisy = x + noise_scale * torch.randn_like(x)
                x_energy = self.model(x_noisy)
            else:
                x_energy = self.model(x)

            pred_x_energy = self.model(pred_x)

        # Compute mean energies with improved numerical stability
        mean_x_energy = torch.mean(x_energy)
        mean_pred_energy = torch.mean(pred_x_energy)

        # Basic contrastive divergence loss: E[data] - E[model]
        loss = mean_x_energy - mean_pred_energy

        # Optional: Regularization to prevent energies from becoming too large
        # This helps with stability especially in the early phases of training
        energy_reg_weight = kwargs.get("energy_reg_weight", 0.001)
        if energy_reg_weight > 0:
            energy_reg = energy_reg_weight * (
                torch.mean(x_energy**2) + torch.mean(pred_x_energy**2)
            )
            loss = loss + energy_reg

        # Prevent extremely large gradients with a safety check
        if torch.isnan(loss) or torch.isinf(loss):
            warnings.warn(
                f"NaN or Inf detected in CD loss. x_energy: {mean_x_energy}, pred_energy: {mean_pred_energy}",
                RuntimeWarning,
            )
            # Return a small positive constant instead of NaN/Inf to prevent training collapse
            return torch.tensor(0.1, device=self.device, dtype=self.dtype)

        return loss


class PersistentContrastiveDivergence(BaseContrastiveDivergence):
    def __init__(self, buffer_size=100):
        super().__init__(k_steps=1)
        self.buffer = None  # Persistent chain state
        self.buffer_size = buffer_size

    # def sample(self, energy_model, x_pos):
    #     if self.buffer is None or len(self.buffer) != self.buffer_size:
    #         # Initialize buffer with random noise
    #         self.buffer = torch.randn(self.buffer_size, *x_pos.batch_shape[1:],
    #                                   device=x_pos.device)
    #
    #     # Update buffer with Gibbs steps
    #     for _ in range(self.k_steps):
    #         self.buffer = energy_model.gibbs_step(self.buffer)
    #
    #     # Return a subset of the buffer as negative samples
    #     idx = torch.randint(0, self.buffer_size, (x_pos.batch_shape[0],))
    #     return self.buffer[idx]


class ParallelTemperingCD(BaseContrastiveDivergence):
    def __init__(self, temps=[1.0, 0.5], k=5):
        super().__init__(k)
        self.temps = temps  # List of temperatures

    # def sample(self, energy_model, x_pos):
    #     chains = [x_pos.detach().clone() for _ in self.temps]
    #     for _ in range(self.k_steps):
    #         # Run Gibbs steps at each temperature
    #         for i, temp in enumerate(self.temps):
    #             chains[i] = energy_model.gibbs_step(chains[i], temp=temp)
    #
    #         # Swap states between chains (temperature exchange)
    #         swap_idx = torch.randint(0, len(self.temps) - 1, (1,))
    #         chains[swap_idx], chains[swap_idx + 1] = chains[swap_idx + 1], chains[swap_idx]
    #
    #     return chains[0]  # Return samples from the highest-temperature chain

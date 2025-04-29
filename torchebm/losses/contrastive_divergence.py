r"""
Contrastive Divergence Loss Module.

This module provides implementations of Contrastive Divergence (CD) and its variants for training energy-based models (EBMs). 
Contrastive Divergence is a computationally efficient approximation to the maximum likelihood estimation that avoids the need 
for complete MCMC sampling from the model distribution.

!!! success "Key Features"
    - Standard Contrastive Divergence (CD-k)
    - Persistent Contrastive Divergence (PCD)
    - Parallel Tempering Contrastive Divergence (PTCD)
    - Support for different MCMC samplers

---

## Module Components

Classes:
    ContrastiveDivergence: Standard CD-k implementation.
    PersistentContrastiveDivergence: Implementation with persistent Markov chains.
    ParallelTemperingCD: Implementation with parallel chains at different temperatures.

---

## Usage Example

!!! example "Basic ContrastiveDivergence Usage"
    ```python
    from torchebm.losses import ContrastiveDivergence
    from torchebm.samplers import LangevinDynamics
    from torchebm.energy_functions import MLPEnergyFunction
    import torch

    # Define the energy function
    energy_fn = MLPEnergyFunction(input_dim=2, hidden_dim=64)

    # Set up the sampler
    sampler = LangevinDynamics(
        energy_function=energy_fn,
        step_size=0.1,
        noise_scale=0.01
    )

    # Create the CD loss
    cd_loss = ContrastiveDivergence(
        energy_function=energy_fn,
        sampler=sampler,
        k_steps=10,
        persistent=False
    )

    # In the training loop:
    data_batch = torch.randn(32, 2)  # Real data samples
    loss, negative_samples = cd_loss(data_batch)
    loss.backward()
    ```

---

## Mathematical Foundations

!!! info "Contrastive Divergence Principles"
    Contrastive Divergence approximates the gradient of the log-likelihood:

    $$
    \nabla_\theta \log p_\theta(x) = -\nabla_\theta E_\theta(x) + \mathbb{E}_{p_\theta(x')} [\nabla_\theta E_\theta(x')]
    $$

    by replacing the expectation under the model distribution with samples obtained after \( k \) steps of MCMC starting from the data:

    $$
    \nabla_\theta \log p_\theta(x) \approx -\nabla_\theta E_\theta(x) + \nabla_\theta E_\theta(x_k)
    $$

    where \( x_k \) is obtained after running \( k \) steps of MCMC starting from \( x \).

!!! question "Why Contrastive Divergence Works"
    - **Computational Efficiency**: Requires only a few MCMC steps rather than running chains to convergence.
    - **Stability**: Starting chains from data points ensures the negative samples are in high-density regions.
    - **Effective Learning**: Despite theoretical limitations, works well in practice for many energy-based models.

### Variants

!!! note "Persistent Contrastive Divergence (PCD)"
    PCD improves upon standard CD by maintaining a persistent set of Markov chains between parameter updates. Instead of 
    restarting chains from the data, it continues chains from the previous iterations:

    1. Initialize a set of persistent chains (often with random noise)
    2. For each training batch:
       a. Update the persistent chains with k steps of MCMC
       b. Use these updated chains for the negative samples
       c. Keep the updated state for the next batch

    PCD can explore the energy landscape more thoroughly, especially for complex distributions.

!!! note "Parallel Tempering CD"
    Parallel Tempering CD uses multiple chains at different temperatures to improve exploration:

    1. Maintain chains at different temperatures \( T_1 < T_2 < ... < T_n \)
    2. For each chain, perform MCMC steps using the energy function \( E(x)/T_i \)
    3. Occasionally swap states between adjacent temperature chains
    4. Use samples from the chain with \( T_1 = 1 \) as negative samples

    This helps overcome energy barriers and explore multimodal distributions.

---

## Practical Considerations

!!! warning "Tuning Parameters"
    - **k_steps**: More steps improves quality of negative samples but increases computational cost.
    - **persistent**: Setting to True enables PCD, which often improves learning for complex distributions.
    - **sampler parameters**: The quality of CD depends heavily on the underlying MCMC sampler parameters.

!!! question "How to Diagnose Issues?"
    Watch for these signs of problematic training:

    - Exploding or vanishing gradients
    - Increasing loss values over time
    - Negative samples that don't resemble the data distribution
    - Energy function collapsing (assigning same energy to all points)

!!! warning "Common Pitfalls"
    - **Too Few MCMC Steps**: Can lead to biased gradients and poor convergence
    - **Improper Initialization**: For PCD, poor initial chain states may hinder learning
    - **Unbalanced Energy**: If negative samples have much higher energy than positive samples, learning may be ineffective

---

## Useful Insights

!!! abstract "Why CD May Outperform MLE"
    In some cases, CD might actually lead to better models than exact maximum likelihood:

    - Prevents overfitting to noise in the data
    - Focuses the model capacity on distinguishing data from nearby non-data regions
    - May result in more useful representations for downstream tasks

???+ info "Further Reading"
    - Hinton, G. E. (2002). "Training products of experts by minimizing contrastive divergence."
    - Tieleman, T. (2008). "Training restricted Boltzmann machines using approximations to the likelihood gradient."
    - Desjardins, G., et al. (2010). "Tempered Markov chain Monte Carlo for training of restricted Boltzmann machines."
"""

from typing import Union, Callable, Tuple, Any, Optional, Dict

import torch
from torch import nn
import math
from abc import abstractmethod
import warnings

from torchebm.core import BaseContrastiveDivergence


class ContrastiveDivergence(BaseContrastiveDivergence):
    r"""
    Implementation of the standard Contrastive Divergence (CD-k) algorithm.

    Contrastive Divergence approximates the gradient of the log-likelihood by comparing
    the energy between real data samples and samples generated after k steps of MCMC
    initialized from the data samples (or from noise in persistent mode).

    The CD loss is defined as:

    $$\mathcal{L}_{CD} = \mathbb{E}_{p_{data}}[E_\theta(x)] - \mathbb{E}_{p_k}[E_\theta(x')]$$

    where:

    - $E_\theta(x)$ is the energy function with parameters $\theta$
    - $p_{data}$ is the data distribution
    - $p_k$ is the distribution after $k$ steps of MCMC

    !!! note "Algorithm Overview"
        1. For non-persistent CD:
           a. Start MCMC chains from real data samples
           b. Run MCMC for k steps to generate negative samples
           c. Compute gradient comparing real and negative samples

        2. For persistent CD:
           a. Maintain a set of persistent chains between updates
           b. Continue chains from previous state for k steps
           c. Update the persistent state for next iteration

    Args:
        energy_function (BaseEnergyFunction): Energy function to train
        sampler (BaseSampler): MCMC sampler for generating negative samples
        k_steps (int): Number of MCMC steps (k in CD-k)
        persistent (bool): Whether to use persistent Contrastive Divergence
        buffer_size (int, optional): Size of buffer for PCD. Defaults to 10000.
        init_steps (int, optional): Number of initial MCMC steps to warm up buffer. Defaults to 100.
        new_sample_ratio (float, optional): Fraction of new random samples to introduce. Defaults to 0.05.
        energy_reg_weight (float, optional): Weight for energy regularization. Defaults to 0.001.
        use_temperature_annealing (bool, optional): Whether to use temperature annealing for sampler. Defaults to False.
        min_temp (float, optional): Minimum temperature for annealing. Defaults to 0.01.
        max_temp (float, optional): Maximum temperature for annealing. Defaults to 2.0.
        temp_decay (float, optional): Decay rate for temperature annealing. Defaults to 0.999.
        dtype (torch.dtype): Data type for computations
        device (torch.device): Device to run computations on

    !!! example "Basic Usage"
        ```python
        # Setup energy function, sampler and CD loss
        energy_fn = MLPEnergyFunction(input_dim=2, hidden_dim=64)
        sampler = LangevinDynamics(energy_fn, step_size=0.1)
        cd_loss = ContrastiveDivergence(
            energy_function=energy_fn,
            sampler=sampler,
            k_steps=10,
            persistent=False
        )

        # In training loop
        optimizer = torch.optim.Adam(energy_fn.parameters(), lr=0.001)

        for batch in dataloader:
            optimizer.zero_grad()
            loss, _ = cd_loss(batch)
            loss.backward()
            optimizer.step()
        ```

    !!! tip "Persistent vs Standard CD"
        - Standard CD (`persistent=False`) is more stable but can struggle with complex distributions
        - Persistent CD (`persistent=True`) can explore better but may require careful initialization
    """

    def __init__(
        self,
        energy_function,
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
        device=None,
        **kwargs
    ):
        super().__init__(
            energy_function=energy_function,
            sampler=sampler,
            k_steps=k_steps,
            persistent=persistent,
            buffer_size=buffer_size,
            new_sample_ratio=new_sample_ratio,
            init_steps=init_steps,
            dtype=dtype,
            device=device,
            **kwargs
        )
        # Additional parameters for improved stability
        self.energy_reg_weight = energy_reg_weight
        self.use_temperature_annealing = use_temperature_annealing
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.temp_decay = temp_decay
        self.current_temp = max_temp
        
        # Register temperature as buffer for persistence
        self.register_buffer("temperature", torch.tensor(max_temp, dtype=self.dtype))
        
    def forward(
        self, x: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the Contrastive Divergence loss and generate negative samples.

        This method implements the energy_functions CD algorithm by:

        1. Initializing MCMC chains (either from data or persistent state)
        2. Running the sampler for k_steps to generate negative samples
        3. Computing the CD loss using the energy difference

        Args:
            x (torch.Tensor): Batch of real data samples (positive samples)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - loss: The CD loss value (scalar)
                - pred_samples: Generated negative samples

        !!! note "Shape Information"
            - Input `x`: (batch_size, feature_dimensions)
            - Output loss: scalar
            - Output pred_samples: (batch_size, feature_dimensions)
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
                    setattr(self.sampler, "_original_noise_scale", self.sampler.noise_scale)
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
        kwargs['energy_reg_weight'] = kwargs.get('energy_reg_weight', self.energy_reg_weight)
        
        # Compute contrastive divergence loss
        loss = self.compute_loss(x, pred_samples, *args, **kwargs)

        return loss, pred_samples

    def compute_loss(
        self, x: torch.Tensor, pred_x: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        Compute the Contrastive Divergence loss given positive and negative samples.

        The CD loss is defined as the difference between the average energy of positive samples
        (from the data distribution) and the average energy of negative samples (from the model).

        Args:
            x (torch.Tensor): Real data samples (positive samples)
            pred_x (torch.Tensor): Generated negative samples
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Scalar loss value

        !!! warning "Gradient Direction"
            Note that this implementation returns `E(x) - E(x')`, so during optimization
            we *minimize* this value. This is different from some formulations that
            maximize `E(x') - E(x)`.
        """
        # Ensure inputs are on the correct device and dtype
        x = x.to(self.device, self.dtype)
        pred_x = pred_x.to(self.device, self.dtype)

        # Compute energy of real and generated samples
        with torch.set_grad_enabled(True):
            # Add small noise to real data for stability (optional)
            if kwargs.get('add_noise_to_real', False):
                noise_scale = kwargs.get('noise_scale', 1e-4)
                x_noisy = x + noise_scale * torch.randn_like(x)
                x_energy = self.energy_function(x_noisy)
            else:
                x_energy = self.energy_function(x)
                
            pred_x_energy = self.energy_function(pred_x)

        # Compute mean energies with improved numerical stability
        mean_x_energy = torch.mean(x_energy)
        mean_pred_energy = torch.mean(pred_x_energy)
        
        # Basic contrastive divergence loss: E[data] - E[model]
        loss = mean_x_energy - mean_pred_energy
        
        # Optional: Regularization to prevent energies from becoming too large
        # This helps with stability especially in the early phases of training
        energy_reg_weight = kwargs.get('energy_reg_weight', 0.001)
        if energy_reg_weight > 0:
            energy_reg = energy_reg_weight * (torch.mean(x_energy**2) + torch.mean(pred_x_energy**2))
            loss = loss + energy_reg
        
        # Prevent extremely large gradients with a safety check
        if torch.isnan(loss) or torch.isinf(loss):
            warnings.warn(
                f"NaN or Inf detected in CD loss. x_energy: {mean_x_energy}, pred_energy: {mean_pred_energy}",
                RuntimeWarning
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

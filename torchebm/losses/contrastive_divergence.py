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

        if self.persistent and (
            not hasattr(self, "buffer_initialized") or not self.buffer_initialized
        ):
            self.initialize_buffer(x.shape)

        start_points = self.get_negative_samples(batch_size, data_shape)

        # generate negative samples
        pred_samples = self.sampler.sample(
            x=start_points,
            n_steps=self.k_steps,
            # n_samples=batch_size,
        ).detach()

        if self.persistent:
            self.update_buffer(pred_samples.detach())

        loss = self.compute_loss(x, pred_samples, *args, **kwargs)

        return loss, pred_samples.detach()

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

        x_energy = self.energy_function(x)
        pred_x_energy = self.energy_function(pred_x)

        # Contrastive Divergence loss: E[data] - E[model]
        loss = torch.mean(x_energy - pred_x_energy)

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

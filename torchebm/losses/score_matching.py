r"""
Score Matching Loss Functions

This module implements various Score Matching techniques for training energy-based models (EBMs). 
Score Matching provides a way to train EBMs without requiring MCMC sampling, which can be 
computationally expensive and potentially unstable.

## Overview

Score Matching methods directly estimate the score function \( \nabla_x \log p(x) \), which is the 
gradient of the log probability density with respect to the input data. For energy-based models where 
\( p(x) \propto \exp(-E(x)) \), this score function equals \( -\nabla_x E(x) \).

The key advantage of Score Matching is that it avoids the sampling problem inherent in methods like 
Maximum Likelihood Estimation, which requires computing the partition function or using MCMC sampling.

## Implemented Methods

### Original Score Matching (Hyvärinen, 2005)

Original Score Matching minimizes the expected squared distance between the model's score and the data's score:

\[
J(\theta) = \frac{1}{2} \mathbb{E}_{p_{\text{data}}} \left[ \| \nabla_x E_\theta(x) \|^2 \right] + 
\mathbb{E}_{p_{\text{data}}} \left[ \text{tr}(\nabla_x^2 E_\theta(x)) \right]
\]

where:
- \( E_\theta(x) \) is the energy function with parameters \( \theta \)
- \( \nabla_x E_\theta(x) \) is the gradient of the energy with respect to the input
- \( \nabla_x^2 E_\theta(x) \) is the Hessian of the energy function
- \( \text{tr}(\cdot) \) denotes the trace operation

The computational challenge in this approach is computing the trace of the Hessian, which can be 
expensive for high-dimensional data.

### Denoising Score Matching (Vincent, 2011)

Denoising Score Matching (DSM) avoids computing the Hessian trace by working with noise-perturbed data.
We perturb the data with noise \( \varepsilon \sim \mathcal{N}(0, \sigma^2 I) \) and train the model to 
predict the score of the perturbed data distribution:

\[
J_{\text{DSM}}(\theta) = \frac{1}{2} \mathbb{E}_{p_{\text{data}}(x)} \mathbb{E}_{\varepsilon} 
\left[ \left\| \nabla_x E_\theta(x + \varepsilon) + \frac{\varepsilon}{\sigma^2} \right\|^2 \right]
\]

The key insight is that the score of the noise-perturbed data can be approximated using the noise itself.

### Sliced Score Matching (Song et al., 2019)

Sliced Score Matching (SSM) is a computationally efficient variant that uses random projections to 
estimate the score matching objective without computing the full Hessian:

\[
J_{\text{SSM}}(\theta) = \frac{1}{2} \mathbb{E}_{p_{\text{data}}(x)} \mathbb{E}_{p(v)} 
\left[ (v^T \nabla_x E_\theta(x))^2 \right] - \mathbb{E}_{p_{\text{data}}(x)} \mathbb{E}_{p(v)} 
\left[ v^T \nabla_x^2 E_\theta(x) v \right]
\]

where \( v \) is a random projection vector, typically sampled from a Rademacher or Gaussian distribution.
This approach significantly reduces computation, especially for high-dimensional data, as it avoids 
computing the full Hessian trace.

## References

- Hyvärinen, A. (2005). Estimation of non-normalized statistical models by score matching. 
  Journal of Machine Learning Research, 6, 695-709.
- Vincent, P. (2011). A connection between score matching and denoising autoencoders. 
  Neural Computation, 23(7), 1661-1674.
- Song, Y., Garg, S., Shi, J., & Ermon, S. (2019). Sliced score matching: A scalable approach to 
  density and score estimation. Uncertainty in Artificial Intelligence.
"""

import torch
import warnings
from typing import Optional, Union, Dict, Tuple, Any, Callable

from torchebm.core import BaseEnergyFunction
from torchebm.core.base_loss import BaseScoreMatching


class ScoreMatching(BaseScoreMatching):
    r"""
    Implementation of the original Score Matching method by Hyvärinen (2005).

    Score Matching trains energy-based models by making the gradient of the model's
    energy function (score) match the gradient of the data's log density. This method
    avoids the need for MCMC sampling that is typically required in contrastive divergence.

    ## Mathematical Formulation

    The score matching objective minimizes:

    \[
    J(\theta) = \frac{1}{2} \mathbb{E}_{p_{\text{data}}} \left[ \| \nabla_x E_\theta(x) \|^2 \right] +
    \mathbb{E}_{p_{\text{data}}} \left[ \text{tr}(\nabla_x^2 E_\theta(x)) \right]
    \]

    where:
    - \( E_\theta(x) \) is the energy function with parameters \( \theta \)
    - \( \nabla_x E_\theta(x) \) is the score function (gradient of energy w.r.t. input)
    - \( \nabla_x^2 E_\theta(x) \) is the Hessian of the energy function
    - \( \text{tr}(\cdot) \) denotes the trace operator

    ## Implementation Details

    This implementation provides three different methods for computing the Hessian trace:

    1. **exact**: Computes the full Hessian diagonal elements directly. Most accurate but
       computationally expensive for high-dimensional data.

    2. **hutchinson**: Uses Hutchinson's trace estimator, which approximates the trace using
       random projections: \( \text{tr}(H) \approx \mathbb{E}_v[v^T H v] \) where v is typically
       sampled from a Rademacher distribution. More efficient for high dimensions.

    3. **approx**: Uses a finite-difference approximation of the Hessian trace which can be
       more numerically stable in some cases.

    Args:
        energy_function (BaseEnergyFunction): Energy function to train
        hessian_method (str): Method to compute Hessian trace. One of {"exact", "hutchinson", "approx"}
        regularization_strength (float): Coefficient for regularization terms
        hutchinson_samples (int): Number of random vectors for Hutchinson's trace estimator
        custom_regularization (Optional[Callable]): Optional function for custom regularization
        use_mixed_precision (bool): Whether to use mixed precision training
        dtype (torch.dtype): Data type for computations
        device (Optional[Union[str, torch.device]]): Device for computations

    References:
        Hyvärinen, A. (2005). Estimation of non-normalized statistical models by score matching.
        Journal of Machine Learning Research, 6, 695-709.
    """

    def __init__(
        self,
        energy_function: BaseEnergyFunction,
        hessian_method: str = "exact",
        regularization_strength: float = 0.0,
        hutchinson_samples: int = 1,
        custom_regularization: Optional[Callable] = None,
        use_mixed_precision: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            energy_function=energy_function,
            regularization_strength=regularization_strength,
            use_autograd=True,
            hutchinson_samples=hutchinson_samples,
            custom_regularization=custom_regularization,
            use_mixed_precision=use_mixed_precision,
            dtype=dtype,
            device=device,
            *args,
            **kwargs,
        )

        self.hessian_method = hessian_method

        # Validate hessian_method
        valid_methods = ["exact", "hutchinson", "approx"]
        if self.hessian_method not in valid_methods:
            warnings.warn(
                f"Invalid hessian_method '{self.hessian_method}'. "
                f"Using 'exact' instead. Valid options are: {valid_methods}",
                UserWarning,
            )
            self.hessian_method = "exact"

        # For mixed precision, hutchinson is more stable
        if self.use_mixed_precision and self.hessian_method == "exact":
            warnings.warn(
                "Using 'exact' Hessian method with mixed precision may be unstable. "
                "Consider using 'hutchinson' method for better numerical stability.",
                UserWarning,
            )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Compute the score matching loss for a batch of data.

        This method first calculates the loss using the specified Hessian computation method,
        then adds regularization if needed.

        !!! note
            The input tensor is automatically converted to the device and dtype specified
            during initialization.

        Args:
            x (torch.Tensor): Input data tensor of shape (batch_size, *data_dims)
            *args: Additional positional arguments passed to compute_loss
            **kwargs: Additional keyword arguments passed to compute_loss

        Returns:
            torch.Tensor: The score matching loss (scalar)

        Examples:
            >>> energy_fn = MLPEnergyFunction(dim=2, hidden_dim=32)
            >>> loss_fn = ScoreMatching(energy_fn, hessian_method="hutchinson")
            >>> x = torch.randn(128, 2)  # 128 samples of 2D data
            >>> loss = loss_fn(x)  # Compute the score matching loss
            >>> loss.backward()  # Backpropagate the loss
        """
        # Ensure x is on the correct device and dtype
        x = x.to(device=self.device, dtype=self.dtype)

        # Compute the loss using the specified method
        loss = self.compute_loss(x, *args, **kwargs)

        # Add regularization if needed
        if self.regularization_strength > 0 or self.custom_regularization is not None:
            loss = self.add_regularization(loss, x)

        return loss

    def compute_loss(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Compute the score matching loss using the specified Hessian computation method.

        This method selects between different implementations of the score matching loss
        based on the `hessian_method` attribute.

        !!! note
            For high-dimensional data, the "hutchinson" method is recommended for better
            computational efficiency.

        !!! warning
            The "exact" method requires computing the full Hessian diagonal, which can be
            computationally expensive for high-dimensional data.

        Args:
            x (torch.Tensor): Input data tensor of shape (batch_size, *data_dims)
            *args: Additional arguments passed to the specific method implementation
            **kwargs: Additional keyword arguments passed to the specific method implementation

        Returns:
            torch.Tensor: The score matching loss (scalar)

        Examples:
            >>> # Different Hessian methods can be chosen at initialization:
            >>> # Exact method (computationally expensive but accurate)
            >>> loss_fn_exact = ScoreMatching(energy_fn, hessian_method="exact")
            >>>
            >>> # Hutchinson method (more efficient for high dimensions)
            >>> loss_fn_hutch = ScoreMatching(
            ...     energy_fn,
            ...     hessian_method="hutchinson",
            ...     hutchinson_samples=10
            ... )
            >>>
            >>> # Approximation method (using finite differences)
            >>> loss_fn_approx = ScoreMatching(energy_fn, hessian_method="approx")
        """
        # Handle different Hessian computation methods
        if self.hessian_method == "exact":
            return self._exact_score_matching(x)
        elif self.hessian_method == "hutchinson":
            return self._hutchinson_score_matching(x)
        elif self.hessian_method == "approx":
            return self._approx_score_matching(x)
        else:
            # This should never happen due to validation in __init__
            return self._exact_score_matching(x)

    def _exact_score_matching(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Compute score matching loss using exact Hessian trace computation.

        This computes the score matching objective:

        \[
        \mathcal{L}(\theta) = \frac{1}{2} \mathbb{E}_{p_{\text{data}}} \left[ \| \nabla_x E_\theta(x) \|^2 \right] +
        \mathbb{E}_{p_{\text{data}}} \left[ \text{tr}(\nabla_x^2 E_\theta(x)) \right]
        \]

        where the trace of the Hessian is computed exactly by calculating each diagonal element.

        !!! warning
            This method computes the full Hessian diagonal elements and can be very
            computationally expensive for high-dimensional data. Consider using the
            Hutchinson estimator via `hessian_method="hutchinson"` for high dimensions.

        !!! note
            For each dimension \( i \), this computes \( \frac{\partial^2 E}{\partial x_i^2} \),
            requiring a separate backward pass, making it \( O(d^2) \) in computational complexity
            where \( d \) is the data dimension.

        Args:
            x (torch.Tensor): Input data tensor of shape (batch_size, *data_dims)

        Returns:
            torch.Tensor: The score matching loss (scalar)
        """
        batch_size = x.shape[0]
        data_dim = x.numel() // batch_size

        # Clone and detach x to avoid modifying the original tensor
        x_detached = x.detach().clone()
        x_detached.requires_grad_(True)

        # Compute first term: 1/2 * ||∇E(x)||²
        # Use mixed precision if enabled
        if self.use_mixed_precision and self.autocast_available:
            from torch.cuda.amp import autocast

            with autocast():
                # Compute the score (gradient of energy w.r.t. input)
                score = self.compute_score(x_detached)
                score_square_term = (
                    0.5 * torch.sum(score**2, dim=list(range(1, len(x.shape)))).mean()
                )
        else:
            # Compute the score (gradient of energy w.r.t. input)
            score = self.compute_score(x_detached)
            score_square_term = (
                0.5 * torch.sum(score**2, dim=list(range(1, len(x.shape)))).mean()
            )

        # Compute second term: tr(∇²E(x))
        # Create vector of second derivatives for each dimension
        hessian_trace = 0

        # Iterate over each dimension to compute diagonal elements of Hessian
        for i in range(data_dim):
            # Reshape x for easier indexing if needed
            x_flat = x_detached.view(batch_size, -1)

            # Compute ∂E/∂x_i
            if self.use_mixed_precision and self.autocast_available:
                from torch.cuda.amp import autocast

                with autocast():
                    # Get energy
                    energy = self.energy_function(x_detached)

                    # Compute first derivative
                    grad_i = torch.autograd.grad(
                        energy.sum(),
                        x_detached,
                        create_graph=True,
                    )[0].view(batch_size, -1)[:, i]
            else:
                # Get energy
                energy = self.energy_function(x_detached)

                # Compute first derivative
                grad_i = torch.autograd.grad(
                    energy.sum(),
                    x_detached,
                    create_graph=True,
                )[0].view(batch_size, -1)[:, i]

            # Compute second derivative ∂²E/∂x_i²
            if self.use_mixed_precision and self.autocast_available:
                from torch.cuda.amp import autocast

                with autocast():
                    grad_grad_i = torch.autograd.grad(
                        grad_i.sum(),
                        x_detached,
                        create_graph=True,
                    )[0].view(batch_size, -1)[:, i]
            else:
                grad_grad_i = torch.autograd.grad(
                    grad_i.sum(),
                    x_detached,
                    create_graph=True,
                )[0].view(batch_size, -1)[:, i]

            # Add to trace
            hessian_trace += grad_grad_i.mean()

        # Combine terms for full loss
        loss = score_square_term + hessian_trace

        return loss

    def _hutchinson_score_matching(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Compute score matching loss using Hutchinson's trace estimator.

        This uses random projections to estimate the trace of the Hessian:

        \[
        \text{tr}(H) \approx \mathbb{E}_v[v^T H v]
        \]

        where \( v \) is a random vector, typically sampled from a Rademacher distribution.

        The complete objective becomes:

        \[
        \mathcal{L}(\theta) = \frac{1}{2} \mathbb{E}_{p_{\text{data}}} \left[ \| \nabla_x E_\theta(x) \|^2 \right] +
        \mathbb{E}_{p_{\text{data}}} \mathbb{E}_v \left[ v^T \nabla_x^2 E_\theta(x) v \right]
        \]

        !!! tip
            This method is more computationally efficient than exact Hessian computation,
            especially for high-dimensional data, as it scales as \( O(d) \) rather than
            \( O(d^2) \) where \( d \) is the data dimension.

        !!! note
            The number of random projections (`hutchinson_samples`) controls the variance
            of the estimator. More samples give a better approximation but require more
            computation.

        Args:
            x (torch.Tensor): Input data tensor of shape (batch_size, *data_dims)

        Returns:
            torch.Tensor: The score matching loss (scalar)

        Examples:
            >>> # The Hutchinson method can be used with multiple samples:
            >>> loss_fn = ScoreMatching(
            ...     energy_fn,
            ...     hessian_method="hutchinson",
            ...     hutchinson_samples=5  # Use 5 random projections
            ... )
            >>> loss = loss_fn(data)
        """
        batch_size = x.shape[0]

        # Clone and detach x to avoid modifying the original tensor
        x_detached = x.detach().clone()
        x_detached.requires_grad_(True)

        # Compute first term: 1/2 * ||∇E(x)||²
        score = self.compute_score(x_detached)
        # Add small epsilon for numerical stability
        score_square_term = (
            0.5 * torch.sum(score**2, dim=list(range(1, len(x.shape)))).mean()
        )

        # Compute second term using Hutchinson's estimator
        hessian_trace = 0
        for _ in range(self.hutchinson_samples):
            # Generate random vectors (typically Rademacher or Gaussian)
            # Rademacher (+1/-1) is often preferred for lower variance
            v = torch.randint(0, 2, x_detached.shape, device=self.device) * 2 - 1
            v = v.to(dtype=self.dtype)

            # Compute Jacobian-vector product
            if self.use_mixed_precision and self.autocast_available:
                from torch.cuda.amp import autocast

                with autocast():
                    # Get energy - must sum to scalar for autograd.grad
                    energy = self.energy_function(x_detached)

                    # We need to provide grad_outputs since energy is not a scalar
                    grad_outputs = torch.ones_like(
                        energy, device=self.device, dtype=self.dtype
                    )
                    Jv = (
                        torch.autograd.grad(
                            energy,
                            x_detached,
                            grad_outputs=grad_outputs,
                            create_graph=True,
                        )[0]
                        * v
                    )

                    # Compute v^T H v using another Jacobian-vector product
                    Jv_sum = torch.sum(Jv)  # Sum to scalar
                    vHv = (
                        torch.autograd.grad(
                            Jv_sum,
                            x_detached,
                            retain_graph=True,
                        )[0]
                        * v
                    )
            else:
                # Get energy
                energy = self.energy_function(x_detached)

                # We need to provide grad_outputs since energy is not a scalar
                grad_outputs = torch.ones_like(
                    energy, device=self.device, dtype=self.dtype
                )
                Jv = (
                    torch.autograd.grad(
                        energy,
                        x_detached,
                        grad_outputs=grad_outputs,
                        create_graph=True,
                    )[0]
                    * v
                )

                # Compute v^T H v using another Jacobian-vector product
                Jv_sum = torch.sum(Jv)  # Sum to scalar
                vHv = (
                    torch.autograd.grad(
                        Jv_sum,
                        x_detached,
                        retain_graph=True,
                    )[0]
                    * v
                )

            # Add to trace estimate
            hessian_trace += vHv.sum() / self.hutchinson_samples

        # Average over batch
        hessian_trace = hessian_trace / batch_size

        # Apply magnitude scaling for better training dynamics
        # The Hessian trace is typically much larger than the score term, so scale it down
        hessian_scaling = 0.1
        hessian_trace = hessian_scaling * hessian_trace

        # Combine terms for full loss
        loss = score_square_term + hessian_trace

        # Clip loss to prevent instability
        if torch.isnan(loss) or torch.isinf(loss) or loss > 1e6:
            warnings.warn(
                f"Score matching loss is unstable: {loss.item()}. Clipping to a reasonable value.",
                UserWarning,
            )
            loss = torch.clamp(loss, -1e6, 1e6)

        return loss

    def _approx_score_matching(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Compute score matching loss using a more efficient finite-difference approximation.

        This method combines the exact computation of the score term with a more
        efficient approximation of the Hessian trace using finite differences:

        \[
        \text{tr}(\nabla_x^2 E_\theta(x)) \approx \frac{1}{\epsilon^2 d} \mathbb{E}_{\delta \sim \mathcal{N}(0, \epsilon^2 I)}
        \left[ (\nabla_x E_\theta(x + \delta) - \nabla_x E_\theta(x))^T \delta \right]
        \]

        where \( \epsilon \) is a small constant and \( d \) is the data dimension.

        !!! note
            This approximation requires only two score computations regardless of the
            data dimensionality, making it more efficient than the exact method for
            high-dimensional data.

        !!! warning
            The approximation quality depends on the choice of \( \epsilon \). Too small
            values may lead to numerical instability, while too large values may give
            inaccurate estimates.

        Args:
            x (torch.Tensor): Input data tensor of shape (batch_size, *data_dims)

        Returns:
            torch.Tensor: The score matching loss (scalar)
        """
        batch_size = x.shape[0]
        data_dim = x.numel() // batch_size

        # Clone and detach x to avoid modifying the original tensor
        x_detached = x.detach().clone()
        x_detached.requires_grad_(True)

        # Compute first term: 1/2 * ||∇E(x)||²
        score = self.compute_score(x_detached)
        score_square_term = (
            0.5 * torch.sum(score**2, dim=list(range(1, len(x.shape)))).mean()
        )

        # Compute an efficient approximation for the Hessian trace
        # Add small noise to input for finite difference approximation
        epsilon = 1e-5
        x_noise = x_detached + epsilon * torch.randn_like(x_detached)

        # Compute score at original and perturbed points
        score_x = self.compute_score(x_detached)
        score_x_noise = self.compute_score(x_noise)

        # Approximate Hessian trace using differential quotient
        hessian_trace = torch.sum(
            (score_x_noise - score_x) * (x_noise - x_detached),
            dim=list(range(1, len(x.shape))),
        ).mean() / (epsilon**2 * data_dim)

        # Combine terms for full loss
        loss = score_square_term + hessian_trace

        return loss

r"""
Score Matching Loss Module.

This module provides implementations of various Score Matching techniques for training energy-based models (EBMs).
Score Matching offers a powerful alternative to Contrastive Divergence by directly estimating the score function
without requiring MCMC sampling, making it more computationally efficient and stable in many cases.

!!! success "Key Features"
    - Original Score Matching (Hyvärinen, 2005)
    - Denoising Score Matching (Vincent, 2011)
    - Sliced Score Matching (Song et al., 2019)
    - Support for different Hessian computation methods
    - Mixed precision training support

---

## Module Components

Classes:
    ScoreMatching: Original score matching with exact or approximate Hessian computation
    DenosingScoreMatching: Denoising variant that avoids Hessian computation
    SlicedScoreMatching: Efficient variant using random projections

---

## Usage Example

!!! example "Basic Score Matching Usage"
    ```python
    from torchebm.losses import ScoreMatching
    from torchebm.energy_functions import MLPEnergyFunction
    import torch

    # Define the energy function
    energy_fn = MLPEnergyFunction(input_dim=2, hidden_dim=64)

    # Create the score matching loss
    sm_loss = ScoreMatching(
        energy_function=energy_fn,
        hessian_method="hutchinson",  # More efficient for high dimensions
        hutchinson_samples=5
    )

    # In the training loop:
    data_batch = torch.randn(32, 2)  # Real data samples
    loss = sm_loss(data_batch)
    loss.backward()
    ```

---

## Mathematical Foundations

!!! info "Score Matching Principles"
    Score Matching minimizes the expected squared distance between the model's score and the data's score:

    $$
    J(\theta) = \frac{1}{2} \mathbb{E}_{p_{\text{data}}} \left[ \| \nabla_x E_\theta(x) \|^2 \right] + 
    \mathbb{E}_{p_{\text{data}}} \left[ \text{tr}(\nabla_x^2 E_\theta(x)) \right]
    $$

    where:
    - \( E_\theta(x) \) is the energy function with parameters \( \theta \)
    - \( \nabla_x E_\theta(x) \) is the score function (gradient of energy w.r.t. input)
    - \( \nabla_x^2 E_\theta(x) \) is the Hessian of the energy function

!!! question "Why Score Matching Works"
    - **No MCMC Required**: Directly estimates the score function without sampling
    - **Computational Efficiency**: Avoids the need for expensive MCMC chains
    - **Stability**: More stable training dynamics compared to CD
    - **Theoretical Guarantees**: Consistent estimator under mild conditions

### Variants

!!! note "Denoising Score Matching (DSM)"
    DSM avoids computing the Hessian trace by working with noise-perturbed data:

    $$
    J_{\text{DSM}}(\theta) = \frac{1}{2} \mathbb{E}_{p_{\text{data}}(x)} \mathbb{E}_{\varepsilon} 
    \left[ \left\| \nabla_x E_\theta(x + \varepsilon) + \frac{\varepsilon}{\sigma^2} \right\|^2 \right]
    $$

    where \( \varepsilon \sim \mathcal{N}(0, \sigma^2 I) \) is the added noise.

    !!! tip "Noise Scale Selection"
        - Smaller \( \sigma \): Better for fine details but may be unstable
        - Larger \( \sigma \): More stable but may lose fine structure
        - Annealing \( \sigma \) during training can help

!!! note "Sliced Score Matching (SSM)"
    SSM uses random projections to estimate the score matching objective:

    $$
    J_{\text{SSM}}(\theta) = \frac{1}{2} \mathbb{E}_{p_{\text{data}}(x)} \mathbb{E}_{p(v)} 
    \left[ (v^T \nabla_x E_\theta(x))^2 \right] - \mathbb{E}_{p_{\text{data}}(x)} \mathbb{E}_{p(v)} 
    \left[ v^T \nabla_x^2 E_\theta(x) v \right]
    $$

    where \( v \) is a random projection vector.

    !!! tip "Projection Types"
        - **Rademacher**: Values in \(\{-1, +1\}\), often lower variance
        - **Gaussian**: Standard normal distribution, more general

---

## Practical Considerations

!!! warning "Hessian Computation Methods"
    - **exact**: Computes full Hessian diagonal, accurate but expensive
    - **hutchinson**: Uses random projections, efficient for high dimensions
    - **approx**: Uses finite differences, numerically stable

!!! question "How to Choose the Right Method?"
    Consider these factors when selecting a score matching variant:

    - **Data Dimension**: For high dimensions, prefer SSM or DSM
    - **Computational Resources**: Exact Hessian requires more memory
    - **Training Stability**: DSM often more stable than original SM
    - **Accuracy Requirements**: Exact method most accurate but slowest

!!! warning "Common Pitfalls"
    - **Numerical Instability**: Hessian computation can be unstable
    - **Gradient Explosion**: Score terms can grow very large
    - **Memory Usage**: Exact Hessian requires \( O(d^2) \) memory
    - **Noise Scale**: Poor choice of \( \sigma \) in DSM can hurt performance

---

## Useful Insights

!!! abstract "When to Use Score Matching"
    Score Matching is particularly effective when:

    - Training high-dimensional energy-based models
    - Working with continuous data distributions
    - Computational efficiency is important
    - MCMC sampling is unstable or expensive

???+ info "Further Reading"
    - Hyvärinen, A. (2005). "Estimation of non-normalized statistical models by score matching."
    - Vincent, P. (2011). "A connection between score matching and denoising autoencoders."
    - Song, Y., et al. (2019). "Sliced score matching: A scalable approach to density and score estimation."

---

## Other Examples

!!! example "Denoising Score Matching with Annealing"
    ```python
    from torchebm.losses import DenosingScoreMatching
    from torchebm.core import LinearScheduler

    # Create noise scale scheduler
    noise_scheduler = LinearScheduler(
        start_value=0.1,
        end_value=0.01,
        n_steps=1000
    )

    # Create DSM loss with dynamic noise scale
    dsm_loss = DenosingScoreMatching(
        energy_function=energy_fn,
        noise_scale=noise_scheduler
    )
    ```

!!! example "Sliced Score Matching with Multiple Projections"
    ```python
    from torchebm.losses import SlicedScoreMatching

    # Create SSM loss with multiple projections
    ssm_loss = SlicedScoreMatching(
        energy_function=energy_fn,
        n_projections=10,
        projection_type="rademacher"
    )
    ```

!!! example "Mixed Precision Training"
    ```python
    # Enable mixed precision for better performance
    sm_loss = ScoreMatching(
        energy_function=energy_fn,
        hessian_method="hutchinson",
        use_mixed_precision=True
    )
    ```
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

    !!! success "Key Advantages"
        - No MCMC sampling required
        - Direct estimation of score function
        - More stable training dynamics
        - Consistent estimator under mild conditions

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

    !!! tip "Computational Considerations"
        The computational cost varies significantly with the choice of Hessian computation method:
        - Exact method: \( O(d^2) \) for d-dimensional data
        - Hutchinson method: \( O(d) \) with variance depending on number of samples
        - Approximation method: \( O(d) \) but may be less accurate

    ## Implementation Details

    This implementation provides three different methods for computing the Hessian trace:

    1. **exact**: Computes the full Hessian diagonal elements directly. Most accurate but
       computationally expensive for high-dimensional data.

    2. **hutchinson**: Uses Hutchinson's trace estimator, which approximates the trace using
       random projections: \( \text{tr}(H) \approx \mathbb{E}_v[v^T H v] \) where v is typically
       sampled from a Rademacher distribution. More efficient for high dimensions.

    3. **approx**: Uses a finite-difference approximation of the Hessian trace which can be
       more numerically stable in some cases.

    !!! warning "Numerical Stability"
        - The exact method can be unstable with mixed precision training
        - Large values in the Hessian can cause numerical issues
        - Gradient clipping is applied automatically to prevent instability

    !!! example "Basic Usage"
        ```python
        # Create a simple energy function
        energy_fn = MLPEnergyFunction(input_dim=2, hidden_dim=64)

        # Initialize score matching with Hutchinson estimator
        sm_loss = ScoreMatching(
            energy_function=energy_fn,
            hessian_method="hutchinson",
            hutchinson_samples=5
        )

        # Training loop
        optimizer = torch.optim.Adam(energy_fn.parameters(), lr=1e-3)

        for batch in dataloader:
            optimizer.zero_grad()
            loss = sm_loss(batch)
            loss.backward()
            optimizer.step()
        ```

    !!! example "Advanced Configuration"
        ```python
        # With mixed precision training
        sm_loss = ScoreMatching(
            energy_function=energy_fn,
            hessian_method="hutchinson",
            use_mixed_precision=True
        )

        # With custom regularization
        def l2_regularization(energy_fn, x):
            return torch.mean(energy_fn(x)**2)

        sm_loss = ScoreMatching(
            energy_function=energy_fn,
            regularization_strength=0.1,
            custom_regularization=l2_regularization
        )
        ```

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


class DenosingScoreMatching(BaseScoreMatching):
    r"""
    Implementation of Denoising Score Matching (DSM) by Vincent (2011).

    DSM is a variant of score matching that avoids computing the trace of the Hessian
    by instead matching the score function to the score of noise-perturbed data. This makes
    it more computationally efficient and numerically stable than the original score matching.

    !!! success "Key Advantages"
        - No Hessian computation required
        - More stable than original score matching
        - Computationally efficient
        - Works well with high-dimensional data

    ## Mathematical Formulation

    For data \( x \), we add noise \( \varepsilon \sim \mathcal{N}(0, \sigma^2 I) \) to get
    perturbed data \( \tilde{x} = x + \varepsilon \). The DSM objective is:

    \[
    J_{\text{DSM}}(\theta) = \frac{1}{2} \mathbb{E}_{p_{\text{data}}(x)} \mathbb{E}_{\varepsilon}
    \left[ \left\| \nabla_{\tilde{x}} E_\theta(\tilde{x}) + \frac{\varepsilon}{\sigma^2} \right\|^2 \right]
    \]

    The key insight is that the score of the noise-perturbed data distribution can be related to
    the original data distribution and the noise model:

    \[
    \nabla_{\tilde{x}} \log p(\tilde{x}) \approx \frac{x - \tilde{x}}{\sigma^2} = -\frac{\varepsilon}{\sigma^2}
    \]

    This allows us to train the model without computing Hessians, using only first-order gradients.

    !!! tip "Noise Scale Selection"
        The choice of noise scale \( \sigma \) is crucial:
        - Small \( \sigma \): Better for fine details but may be unstable
        - Large \( \sigma \): More stable but may lose fine structure
        - Annealing \( \sigma \) during training can help balance these trade-offs

    ## Practical Considerations

    - The noise scale \( \sigma \) is a critical hyperparameter that affects the training dynamics
    - Smaller noise scales focus on fine details of the data distribution
    - Larger noise scales help with stability but may lose some detailed structure
    - Annealing the noise scale during training can sometimes improve results

    !!! warning "Common Issues"
        - Too small noise scale can lead to numerical instability
        - Too large noise scale can cause loss of fine details
        - Noise scale should be tuned based on data characteristics

    !!! example "Basic Usage"
        ```python
        # Create energy function
        energy_fn = MLPEnergyFunction(input_dim=2, hidden_dim=64)

        # Initialize DSM with default noise scale
        dsm_loss = DenosingScoreMatching(
            energy_function=energy_fn,
            noise_scale=0.01
        )

        # Training loop
        optimizer = torch.optim.Adam(energy_fn.parameters(), lr=1e-3)

        for batch in dataloader:
            optimizer.zero_grad()
            loss = dsm_loss(batch)
            loss.backward()
            optimizer.step()
        ```

    !!! example "Advanced Configuration"
        ```python
        # With noise scale annealing
        from torchebm.core import LinearScheduler

        noise_scheduler = LinearScheduler(
            start_value=0.1,
            end_value=0.01,
            n_steps=1000
        )

        dsm_loss = DenosingScoreMatching(
            energy_function=energy_fn,
            noise_scale=noise_scheduler
        )

        # With mixed precision training
        dsm_loss = DenosingScoreMatching(
            energy_function=energy_fn,
            noise_scale=0.01,
            use_mixed_precision=True
        )
        ```

    Args:
        energy_function (BaseEnergyFunction): Energy function to train
        noise_scale (float): Scale of Gaussian noise for data perturbation
        regularization_strength (float): Coefficient for regularization terms
        custom_regularization (Optional[Callable]): Optional function for custom regularization
        use_mixed_precision (bool): Whether to use mixed precision training
        dtype (torch.dtype): Data type for computations
        device (Optional[Union[str, torch.device]]): Device for computations

    References:
        Vincent, P. (2011). A connection between score matching and denoising autoencoders.
        Neural Computation, 23(7), 1661-1674.
    """

    def __init__(
        self,
        energy_function: BaseEnergyFunction,
        noise_scale: float = 0.01,
        regularization_strength: float = 0.0,
        custom_regularization: Optional[Callable] = None,
        use_mixed_precision: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            energy_function=energy_function,
            noise_scale=noise_scale,
            regularization_strength=regularization_strength,
            use_autograd=True,
            custom_regularization=custom_regularization,
            use_mixed_precision=use_mixed_precision,
            dtype=dtype,
            device=device,
            *args,
            **kwargs,
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Compute the denoising score matching loss for a batch of data.

        This method first computes the denoising score matching loss using perturbed data,
        then adds regularization if needed.

        !!! note
            The input tensor is automatically converted to the device and dtype specified
            during initialization.

        Args:
            x (torch.Tensor): Input data tensor of shape (batch_size, *data_dims)
            *args: Additional positional arguments passed to compute_loss
            **kwargs: Additional keyword arguments passed to compute_loss

        Returns:
            torch.Tensor: The denoising score matching loss (scalar)

        Examples:
            >>> energy_fn = MLPEnergyFunction(dim=2, hidden_dim=32)
            >>> loss_fn = DenosingScoreMatching(
            ...     energy_fn,
            ...     noise_scale=0.01  # Controls the noise level added to data
            ... )
            >>> x = torch.randn(128, 2)  # 128 samples of 2D data
            >>> loss = loss_fn(x)  # Compute the DSM loss
            >>> loss.backward()  # Backpropagate the loss
        """
        # Ensure x is on the correct device and dtype
        x = x.to(device=self.device, dtype=self.dtype)

        # Compute the loss
        loss = self.compute_loss(x, *args, **kwargs)

        # Add regularization if needed
        if self.regularization_strength > 0 or self.custom_regularization is not None:
            loss = self.add_regularization(loss, x)

        return loss

    def compute_loss(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Compute the denoising score matching loss.

        DSM adds noise \( \varepsilon \sim \mathcal{N}(0, \sigma^2 I) \) to the data and
        trains the score network to predict \( -\varepsilon/\sigma^2 \):

        \[
        \mathcal{L}_{\text{DSM}}(\theta) = \frac{1}{2} \mathbb{E}_{p_{\text{data}}(x)} \mathbb{E}_{\varepsilon}
        \left[ \left\| \nabla_{\tilde{x}} E_\theta(\tilde{x}) + \frac{\varepsilon}{\sigma^2} \right\|^2 \right]
        \]

        where \( \tilde{x} = x + \varepsilon \) is the perturbed data point.

        !!! note
            The noise scale \( \sigma \) is a critical hyperparameter that affects the learning dynamics.

        !!! tip
            - Smaller noise scales focus on fine details of the data distribution
            - Larger noise scales help with stability but may lose some detailed structure

        Args:
            x (torch.Tensor): Input data tensor of shape (batch_size, *data_dims)
            *args: Additional arguments (not used)
            **kwargs: Additional keyword arguments (not used)

        Returns:
            torch.Tensor: The denoising score matching loss (scalar)

        Examples:
            >>> # Creating loss functions with different noise scales:
            >>> # Small noise for capturing fine details
            >>> fine_dsm = DenosingScoreMatching(energy_fn, noise_scale=0.01)
            >>>
            >>> # Larger noise for stability
            >>> stable_dsm = DenosingScoreMatching(energy_fn, noise_scale=0.1)
            >>>
            >>> # Computing loss
            >>> x = torch.randn(32, 2)  # 32 samples of 2D data
            >>> loss = fine_dsm(x)
        """
        # Perturb the data with noise
        x_perturbed, noise = self.perturb_data(x)

        # Compute score at the perturbed point
        score = self.compute_score(x_perturbed)

        # Target score is -noise/sigma²
        target_score = -noise / (self.noise_scale**2)

        # Compute loss as mean squared error between score and target
        loss = (
            0.5
            * torch.sum(
                (score - target_score) ** 2, dim=list(range(1, len(x.shape)))
            ).mean()
        )

        return loss


class SlicedScoreMatching(BaseScoreMatching):
    r"""
    Implementation of Sliced Score Matching (SSM) by Song et al. (2019).

    SSM is a computationally efficient variant of score matching that uses random projections
    to estimate the score matching objective. It avoids computing the full Hessian matrix and
    instead uses random projections to estimate the trace of the Hessian.

    !!! success "Key Advantages"
        - Significantly more efficient than exact score matching
        - Scales well to high dimensions
        - No need for MCMC sampling
        - Works with any energy function architecture

    ## Mathematical Formulation

    For data \( x \) and random projection vectors \( v \), the SSM objective is:

    \[
    J_{\text{SSM}}(\theta) = \mathbb{E}_{p_{\text{data}}(x)} \mathbb{E}_{v \sim \mathcal{N}(0, I)}
    \left[ v^T \nabla_x^2 E_\theta(x) v + \frac{1}{2} (v^T \nabla_x E_\theta(x))^2 \right]
    \]

    The key insight is that we can estimate the trace of the Hessian using random projections:

    \[
    \text{tr}(\nabla_x^2 E_\theta(x)) \approx \mathbb{E}_{v \sim \mathcal{N}(0, I)} [v^T \nabla_x^2 E_\theta(x) v]
    \]

    This allows us to compute the score matching objective using only first-order gradients
    and Hessian-vector products, which are much more efficient than computing the full Hessian.

    !!! tip "Projection Selection"
        The choice of projection type and number of projections affects the accuracy:
        - Gaussian projections: Most common, works well in practice
        - Rademacher projections: Binary values, can be more efficient
        - More projections: Better accuracy but higher computational cost
        - Fewer projections: Faster but may be less accurate

    ## Practical Considerations

    - The number of projections \( n_{\text{projections}} \) is a key hyperparameter
    - More projections lead to better accuracy but higher computational cost
    - The projection type (Gaussian or Rademacher) can affect performance
    - SSM is particularly useful for high-dimensional data where exact score matching is infeasible

    !!! warning "Common Issues"
        - Too few projections can lead to high variance in the gradient estimates
        - Projection type should be chosen based on the data characteristics
        - May require more iterations than exact score matching for convergence

    !!! example "Basic Usage"
        ```python
        # Create energy function
        energy_fn = MLPEnergyFunction(input_dim=2, hidden_dim=64)

        # Initialize SSM with default parameters
        ssm_loss = SlicedScoreMatching(
            energy_function=energy_fn,
            n_projections=10,
            projection_type="gaussian"
        )

        # Training loop
        optimizer = torch.optim.Adam(energy_fn.parameters(), lr=1e-3)

        for batch in dataloader:
            optimizer.zero_grad()
            loss = ssm_loss(batch)
            loss.backward()
            optimizer.step()
        ```

    !!! example "Advanced Configuration"
        ```python
        # With Rademacher projections
        ssm_loss = SlicedScoreMatching(
            energy_function=energy_fn,
            n_projections=20,
            projection_type="rademacher"
        )

        # With mixed precision training
        ssm_loss = SlicedScoreMatching(
            energy_function=energy_fn,
            n_projections=10,
            projection_type="gaussian",
            use_mixed_precision=True
        )

        # With custom regularization
        def custom_reg(energy_fn, x):
            return torch.mean(energy_fn(x)**2)

        ssm_loss = SlicedScoreMatching(
            energy_function=energy_fn,
            n_projections=10,
            projection_type="gaussian",
            custom_regularization=custom_reg
        )
        ```

    Args:
        energy_function (BaseEnergyFunction): Energy function to train
        n_projections (int): Number of random projections to use
        projection_type (str): Type of random projections ("gaussian" or "rademacher")
        regularization_strength (float): Coefficient for regularization terms
        custom_regularization (Optional[Callable]): Optional function for custom regularization
        use_mixed_precision (bool): Whether to use mixed precision training
        dtype (torch.dtype): Data type for computations
        device (Optional[Union[str, torch.device]]): Device for computations

    References:
        Song, Y., Garg, S., Shi, J., & Ermon, S. (2019). Sliced score matching: A scalable approach
        to density and score estimation. In Uncertainty in Artificial Intelligence (pp. 574-584).
    """

    def __init__(
        self,
        energy_function: BaseEnergyFunction,
        n_projections: int = 5,
        projection_type: str = "rademacher",
        regularization_strength: float = 0.0,
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
            custom_regularization=custom_regularization,
            use_mixed_precision=use_mixed_precision,
            dtype=dtype,
            device=device,
            *args,
            **kwargs,
        )

        self.n_projections = n_projections
        self.projection_type = projection_type

        # Validate projection_type
        valid_types = ["rademacher", "gaussian"]
        if self.projection_type not in valid_types:
            warnings.warn(
                f"Invalid projection_type '{self.projection_type}'. "
                f"Using 'rademacher' instead. Valid options are: {valid_types}",
                UserWarning,
            )
            self.projection_type = "rademacher"

    def _get_random_projections(self, shape: torch.Size) -> torch.Tensor:
        r"""
        Generate random vectors for projection-based score matching.

        This function samples vectors from either a Rademacher or Gaussian distribution
        based on the `projection_type` parameter.

        !!! note
            Rademacher distributions (values in \(\{-1, +1\}\)) often have lower variance
            in the trace estimator compared to Gaussian distributions.

        Args:
            shape (torch.Size): Shape of vectors to generate

        Returns:
            torch.Tensor: Random projection vectors of the specified shape

        Examples:
            >>> loss_fn = SlicedScoreMatching(energy_fn)
            >>> # Internally used to generate projection vectors:
            >>> v = loss_fn._get_random_projections((32, 2))  # 32 samples of dim 2
            >>> # v will be of shape (32, 2) with values in {-1, +1} (Rademacher)
        """
        if self.projection_type == "rademacher":
            # Rademacher (+1/-1) distribution
            return (torch.randint(0, 2, shape, device=self.device) * 2 - 1).to(
                dtype=self.dtype
            )
        else:  # "gaussian"
            # Standard normal distribution
            return torch.randn(shape, device=self.device, dtype=self.dtype)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Compute the sliced score matching loss for a batch of data.

        This method first calculates the sliced score matching loss using random
        projections, then adds regularization if needed.

        !!! note
            The input tensor is automatically converted to the device and dtype specified
            during initialization.

        Args:
            x (torch.Tensor): Input data tensor of shape (batch_size, *data_dims)
            *args: Additional positional arguments passed to compute_loss
            **kwargs: Additional keyword arguments passed to compute_loss

        Returns:
            torch.Tensor: The sliced score matching loss (scalar)

        Examples:
            >>> energy_fn = MLPEnergyFunction(dim=2, hidden_dim=32)
            >>> loss_fn = SlicedScoreMatching(
            ...     energy_fn,
            ...     n_projections=10,  # Number of random projections to use
            ...     projection_type="rademacher"  # Type of random vectors
            ... )
            >>> x = torch.randn(128, 2)  # 128 samples of 2D data
            >>> loss = loss_fn(x)  # Compute the SSM loss
            >>> loss.backward()  # Backpropagate the loss
        """
        # Ensure x is on the correct device and dtype
        x = x.to(device=self.device, dtype=self.dtype)

        # Compute the loss
        loss = self.compute_loss(x, *args, **kwargs)

        # Add regularization if needed
        if self.regularization_strength > 0 or self.custom_regularization is not None:
            loss = self.add_regularization(loss, x)

        return loss

    def compute_loss(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Compute the sliced score matching loss using random projections.

        Sliced Score Matching avoids computing the full Hessian trace by using random
        projections. The objective is:

        \[
        \mathcal{L}_{\text{SSM}}(\theta) = \mathbb{E}_{p_{\text{data}}(x)} \mathbb{E}_{v \sim \mathcal{N}(0, I)}
        \left[ v^T \nabla_x^2 E_\theta(x) v + \frac{1}{2} (v^T \nabla_x E_\theta(x))^2 \right]
        \]

        where \( v \) is a random projection vector sampled from a Rademacher or Gaussian distribution.

        !!! tip
            This method is computationally efficient for high-dimensional data, with complexity
            \( O(d) \) in the data dimension rather than \( O(d^2) \) for exact score matching.

        !!! note
            The number of projections (`n_projections`) controls the trade-off between
            computational cost and estimation accuracy. More projections give better
            approximation but require more computation.

        Args:
            x (torch.Tensor): Input data tensor of shape (batch_size, *data_dims)
            *args: Additional arguments (not used)
            **kwargs: Additional keyword arguments (not used)

        Returns:
            torch.Tensor: The sliced score matching loss (scalar)

        Examples:
            >>> # Different ways to configure sliced score matching:
            >>>
            >>> # With Rademacher projections (default)
            >>> ssm_rade = SlicedScoreMatching(
            ...     energy_fn,
            ...     n_projections=5,
            ...     projection_type="rademacher"
            ... )
            >>>
            >>> # With Gaussian projections
            >>> ssm_gauss = SlicedScoreMatching(
            ...     energy_fn,
            ...     n_projections=5,
            ...     projection_type="gaussian"
            ... )
            >>>
            >>> # More projections for better accuracy
            >>> ssm_accurate = SlicedScoreMatching(
            ...     energy_fn,
            ...     n_projections=20
            ... )
            >>>
            >>> loss = ssm_rade(data)
        """
        batch_size = x.shape[0]

        # Clone and detach x to avoid modifying the original tensor
        x_detached = x.detach().clone()
        x_detached.requires_grad_(True)

        # Initialize loss terms
        total_loss = 0.0

        for _ in range(self.n_projections):
            # Generate random projection vectors
            v = self._get_random_projections(x_detached.shape)

            if self.use_mixed_precision and self.autocast_available:
                from torch.cuda.amp import autocast

                with autocast():
                    # Get energy
                    energy = self.energy_function(x_detached)

                    # First term: v^T ∇E
                    # Need grad_outputs for batched energy outputs
                    grad_outputs = torch.ones_like(
                        energy, device=self.device, dtype=self.dtype
                    )
                    score = torch.autograd.grad(
                        energy,
                        x_detached,
                        grad_outputs=grad_outputs,
                        create_graph=True,
                    )[0]

                    v_score = torch.sum(v * score, dim=list(range(1, len(x.shape))))
                    term1 = 0.5 * torch.mean(v_score**2)

                    # Second term: v^T ∇²E v
                    # Sum v_score to scalar for autograd
                    v_score_sum = v_score.sum()
                    grad_v_score = torch.autograd.grad(
                        v_score_sum,
                        x_detached,
                        create_graph=True,
                    )[0]

                    term2 = torch.mean(
                        torch.sum(v * grad_v_score, dim=list(range(1, len(x.shape))))
                    )
            else:
                # Get energy
                energy = self.energy_function(x_detached)

                # First term: v^T ∇E
                # Need grad_outputs for batched energy outputs
                grad_outputs = torch.ones_like(
                    energy, device=self.device, dtype=self.dtype
                )
                score = torch.autograd.grad(
                    energy,
                    x_detached,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                )[0]

                v_score = torch.sum(v * score, dim=list(range(1, len(x.shape))))
                term1 = 0.5 * torch.mean(v_score**2)

                # Second term: v^T ∇²E v
                # Sum v_score to scalar for autograd
                v_score_sum = v_score.sum()
                grad_v_score = torch.autograd.grad(
                    v_score_sum,
                    x_detached,
                    create_graph=True,
                )[0]

                term2 = torch.mean(
                    torch.sum(v * grad_v_score, dim=list(range(1, len(x.shape))))
                )

            # Apply scaling to balance terms (second term can dominate)
            scaling_factor = 0.1
            # Add to total loss
            total_loss += term1 - scaling_factor * term2

        # Average over number of projections
        loss = total_loss / self.n_projections

        # Clip to reasonable values for stability
        if torch.isnan(loss) or torch.isinf(loss) or loss > 1e6:
            warnings.warn(
                f"Sliced Score Matching loss is unstable: {loss.item()}. Clipping to a reasonable value.",
                UserWarning,
            )
            loss = torch.clamp(loss, -1e6, 1e6)

        return loss

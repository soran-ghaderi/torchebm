---
title: BaseLoss Functions Implementation
description: Detailed implementation details of TorchEBM's loss functions
icon: material/function-variant
---

# BaseLoss Functions Implementation

!!! abstract "Implementation Details"
    This guide provides detailed information about the implementation of loss functions in TorchEBM, including mathematical foundations, code structure, and optimization techniques.

## Mathematical Foundation

Energy-based models can be trained using various loss functions, each with different properties. The primary goal is to shape the energy landscape such that observed data has low energy while other regions have high energy.

## Base BaseLoss Implementation

The `BaseLoss` base class provides the foundation for all loss functions:

```python
from abc import ABC, abstractmethod
import torch
from typing import Optional, Dict, Any, Tuple

from torchebm.core import BaseEnergyFunction


class BaseLoss(ABC):
    """Base class for all loss functions."""

    def __init__(self, energy_function: BaseEnergyFunction):
        """Initialize loss with an energy function.
        
        Args:
            energy_function: The energy function to train
        """
        self.energy_function = energy_function

    @abstractmethod
    def __call__(
            self,
            pos_samples: torch.Tensor,
            neg_samples: torch.Tensor,
            **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the loss.
        
        Args:
            pos_samples: Positive samples from the data distribution
            neg_samples: Negative samples from the model distribution
            **kwargs: Additional loss-specific parameters
            
        Returns:
            Tuple of (loss value, dictionary of metrics)
        """
        pass
```

## Maximum Likelihood Estimation (MLE)

### Mathematical Background

The MLE approach aims to maximize the log-likelihood of the data under the model:

$$\mathcal{L}_{\text{MLE}} = \mathbb{E}_{p_{\text{data}}(x)}[E(x)] - \mathbb{E}_{p_{\text{model}}(x)}[E(x)]$$

### Implementation

```python
import torch
from typing import Dict, Tuple

from torchebm.core import BaseEnergyFunction
from torchebm.losses.base import BaseLoss


class MLELoss(BaseLoss):
    """Maximum Likelihood Estimation loss."""

    def __init__(
            self,
            energy_function: BaseEnergyFunction,
            alpha: float = 1.0,
            regularization: Optional[str] = None,
            reg_strength: float = 0.0
    ):
        """Initialize MLE loss.
        
        Args:
            energy_function: Energy function to train
            alpha: Weight for the negative phase
            regularization: Type of regularization ('l1', 'l2', or None)
            reg_strength: Strength of regularization
        """
        super().__init__(energy_function)
        self.alpha = alpha
        self.regularization = regularization
        self.reg_strength = reg_strength

    def __call__(
            self,
            pos_samples: torch.Tensor,
            neg_samples: torch.Tensor,
            **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the MLE loss.
        
        Args:
            pos_samples: Positive samples from the data distribution
            neg_samples: Negative samples from the model distribution
            
        Returns:
            Tuple of (loss value, dictionary of metrics)
        """
        # Compute energies
        pos_energy = self.energy_function(pos_samples)
        neg_energy = self.energy_function(neg_samples)

        # Compute loss components
        pos_term = pos_energy.mean()
        neg_term = neg_energy.mean()

        # Full loss
        loss = pos_term - self.alpha * neg_term

        # Add regularization if specified
        reg_loss = torch.tensor(0.0, device=pos_energy.device)
        if self.regularization is not None and self.reg_strength > 0:
            if self.regularization == 'l2':
                for param in self.energy_function.parameters():
                    reg_loss += torch.sum(param ** 2)
            elif self.regularization == 'l1':
                for param in self.energy_function.parameters():
                    reg_loss += torch.sum(torch.abs(param))

            loss = loss + self.reg_strength * reg_loss

        # Metrics to track
        metrics = {
            'pos_energy': pos_term.detach(),
            'neg_energy': neg_term.detach(),
            'energy_gap': (neg_term - pos_term).detach(),
            'loss': loss.detach(),
            'reg_loss': reg_loss.detach()
        }

        return loss, metrics
```

## Contrastive Divergence (CD)

### Mathematical Background

Contrastive Divergence is a variant of MLE that uses a specific sampling scheme where negative samples are obtained by starting from positive samples and running MCMC for a few steps:

$$\mathcal{L}_{\text{CD}} = \mathbb{E}_{p_{\text{data}}(x)}[E(x)] - \mathbb{E}_{p_{K}(x|x_{\text{data}})}[E(x)]$$

where $p_{K}(x|x_{\text{data}})$ is the distribution after $K$ steps of MCMC starting from data samples.

### Implementation

```python
import torch
from typing import Dict, Tuple, Optional

from torchebm.core import BaseEnergyFunction
from torchebm.samplers import Sampler, LangevinDynamics
from torchebm.losses.base import BaseLoss


class ContrastiveDivergenceLoss(BaseLoss):
    """Contrastive Divergence loss."""

    def __init__(
            self,
            energy_function: BaseEnergyFunction,
            sampler: Optional[Sampler] = None,
            n_steps: int = 10,
            alpha: float = 1.0
    ):
        """Initialize CD loss.
        
        Args:
            energy_function: Energy function to train
            sampler: Sampler for generating negative samples
            n_steps: Number of sampling steps for negative samples
            alpha: Weight for the negative phase
        """
        super().__init__(energy_function)
        self.sampler = sampler or LangevinDynamics(energy_function)
        self.n_steps = n_steps
        self.alpha = alpha

    def __call__(
            self,
            pos_samples: torch.Tensor,
            neg_samples: Optional[torch.Tensor] = None,
            **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the CD loss.
        
        Args:
            pos_samples: Positive samples from the data distribution
            neg_samples: Optional negative samples (if None, will be generated)
            
        Returns:
            Tuple of (loss value, dictionary of metrics)
        """
        # Generate negative samples if not provided
        if neg_samples is None:
            with torch.no_grad():
                neg_samples = self.sampler.sample_chain(
                    pos_samples.shape[1],
                    self.n_steps,
                    n_samples=pos_samples.shape[0],
                    initial_samples=pos_samples.detach()
                )

        # Compute energies
        pos_energy = self.energy_function(pos_samples)
        neg_energy = self.energy_function(neg_samples)

        # Compute loss components
        pos_term = pos_energy.mean()
        neg_term = neg_energy.mean()

        # Full loss
        loss = pos_term - self.alpha * neg_term

        # Metrics to track
        metrics = {
            'pos_energy': pos_term.detach(),
            'neg_energy': neg_term.detach(),
            'energy_gap': (neg_term - pos_term).detach(),
            'loss': loss.detach()
        }

        return loss, metrics
```

## Noise Contrastive Estimation (NCE)

### Mathematical Background

NCE treats the problem as a binary classification task, distinguishing between data samples and noise samples:

$$\mathcal{L}_{\text{NCE}} = -\mathbb{E}_{p_{\text{data}}(x)}[\log \sigma(f_\theta(x))] - \mathbb{E}_{p_{\text{noise}}(x)}[\log (1 - \sigma(f_\theta(x)))]$$

where $f_\theta(x) = -E(x) - \log Z$ and $\sigma$ is the sigmoid function.

### Implementation

```python
import torch
import torch.nn.functional as F
from typing import Dict, Tuple

from torchebm.core import BaseEnergyFunction
from torchebm.losses.base import BaseLoss


class NCELoss(BaseLoss):
    """Noise Contrastive Estimation loss."""

    def __init__(
            self,
            energy_function: BaseEnergyFunction,
            log_partition: float = 0.0,
            learn_partition: bool = True
    ):
        """Initialize NCE loss.
        
        Args:
            energy_function: Energy function to train
            log_partition: Initial value of log partition function
            learn_partition: Whether to learn the partition function
        """
        super().__init__(energy_function)
        if learn_partition:
            self.log_z = torch.nn.Parameter(torch.tensor([log_partition], dtype=torch.float32))
        else:
            self.register_buffer('log_z', torch.tensor([log_partition], dtype=torch.float32))
        self.learn_partition = learn_partition

    def __call__(
            self,
            pos_samples: torch.Tensor,
            neg_samples: torch.Tensor,
            **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the NCE loss.
        
        Args:
            pos_samples: Positive samples from the data distribution
            neg_samples: Negative samples from noise distribution
            
        Returns:
            Tuple of (loss value, dictionary of metrics)
        """
        # Compute energies
        pos_energy = self.energy_function(pos_samples)
        neg_energy = self.energy_function(neg_samples)

        # Compute logits
        pos_logits = -pos_energy - self.log_z
        neg_logits = -neg_energy - self.log_z

        # Binary classification loss
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits,
            torch.ones_like(pos_logits)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits,
            torch.zeros_like(neg_logits)
        )

        # Full loss
        loss = pos_loss + neg_loss

        # Metrics to track
        metrics = {
            'pos_loss': pos_loss.detach(),
            'neg_loss': neg_loss.detach(),
            'loss': loss.detach(),
            'log_z': self.log_z.detach(),
            'pos_energy': pos_energy.mean().detach(),
            'neg_energy': neg_energy.mean().detach()
        }

        return loss, metrics
```

## Score Matching

### Mathematical Background

Score Matching minimizes the difference between the model's score function (gradient of log-probability) and the data score:

$$\mathcal{L}_{\text{SM}} = \frac{1}{2}\mathbb{E}_{p_{\text{data}}(x)}\left[\left\|\nabla_x \log p_{\text{data}}(x) - \nabla_x \log p_{\text{model}}(x)\right\|^2\right]$$

This can be simplified to:

$$\mathcal{L}_{\text{SM}} = \mathbb{E}_{p_{\text{data}}(x)}\left[\text{tr}(\nabla_x^2 E(x)) + \frac{1}{2}\|\nabla_x E(x)\|^2\right]$$

### Implementation

```python
import torch
from typing import Dict, Tuple

from torchebm.core import BaseEnergyFunction
from torchebm.losses.base import BaseLoss


class ScoreMatchingLoss(BaseLoss):
    """Score Matching loss."""

    def __init__(
            self,
            energy_function: BaseEnergyFunction,
            implicit: bool = True
    ):
        """Initialize Score Matching loss.
        
        Args:
            energy_function: Energy function to train
            implicit: Whether to use implicit score matching
        """
        super().__init__(energy_function)
        self.implicit = implicit

    def _compute_explicit_score_matching(self, x: torch.Tensor) -> torch.Tensor:
        """Compute explicit score matching loss.
        
        This requires computing both the score and the Hessian trace.
        
        Args:
            x: Input samples of shape (n_samples, dim)
            
        Returns:
            BaseLoss value
        """
        x.requires_grad_(True)

        # Compute energy
        energy = self.energy_function(x)

        # Compute score (first derivatives)
        score = torch.autograd.grad(
            energy.sum(), x, create_graph=True
        )[0]

        # Compute trace of Hessian (second derivatives)
        trace = 0.0
        for i in range(x.shape[1]):
            grad_score_i = torch.autograd.grad(
                score[:, i].sum(), x, create_graph=True
            )[0]
            trace += grad_score_i[:, i]

        # Compute squared norm of score
        score_norm = torch.sum(score ** 2, dim=1)

        # Full loss
        loss = trace + 0.5 * score_norm

        return loss.mean()

    def _compute_implicit_score_matching(self, x: torch.Tensor) -> torch.Tensor:
        """Compute implicit score matching loss.
        
        This avoids computing the Hessian trace.
        
        Args:
            x: Input samples of shape (n_samples, dim)
            
        Returns:
            BaseLoss value
        """
        # Add noise to inputs
        x_noise = x + torch.randn_like(x) * 0.01
        x_noise.requires_grad_(True)

        # Compute energy and its gradient
        energy = self.energy_function(x_noise)
        score = torch.autograd.grad(
            energy.sum(), x_noise, create_graph=True
        )[0]

        # Compute loss as squared difference between gradient and vector field
        vector_field = (x_noise - x) / (0.01 ** 2)
        loss = 0.5 * torch.sum((score + vector_field) ** 2, dim=1)

        return loss.mean()

    def __call__(
            self,
            pos_samples: torch.Tensor,
            neg_samples: torch.Tensor = None,
            **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the Score Matching loss.
        
        Args:
            pos_samples: Positive samples from the data distribution
            neg_samples: Not used in Score Matching
            
        Returns:
            Tuple of (loss value, dictionary of metrics)
        """
        # Compute loss based on method
        if self.implicit:
            loss = self._compute_implicit_score_matching(pos_samples)
        else:
            loss = self._compute_explicit_score_matching(pos_samples)

        # Metrics to track
        metrics = {
            'loss': loss.detach()
        }

        return loss, metrics
```

## Denoising Score Matching

### Mathematical Background

Denoising Score Matching is a variant of score matching that adds noise to the data and tries to predict the score of the noisy distribution:

$$\mathcal{L}_{\text{DSM}} = \mathbb{E}_{p_{\text{data}}(x)}\mathbb{E}_{q_\sigma(\tilde{x}|x)}\left[\left\|\nabla_{\tilde{x}} \log q_\sigma(\tilde{x}|x) - \nabla_{\tilde{x}} \log p_{\text{model}}(\tilde{x})\right\|^2\right]$$

where $q_\sigma(\tilde{x}|x) = \mathcal{N}(\tilde{x}|x, \sigma^2\mathbf{I})$.

### Implementation

```python
import torch
from typing import Dict, Tuple, Union, List

from torchebm.core import BaseEnergyFunction
from torchebm.losses.base import BaseLoss


class DenoisingScoreMatchingLoss(BaseLoss):
    """Denoising Score Matching loss."""

    def __init__(
            self,
            energy_function: BaseEnergyFunction,
            sigma: Union[float, List[float]] = 0.01
    ):
        """Initialize DSM loss.
        
        Args:
            energy_function: Energy function to train
            sigma: Noise level(s) for denoising
        """
        super().__init__(energy_function)
        if isinstance(sigma, (int, float)):
            self.sigma = [float(sigma)]
        else:
            self.sigma = sigma

    def __call__(
            self,
            pos_samples: torch.Tensor,
            neg_samples: torch.Tensor = None,
            **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the DSM loss.
        
        Args:
            pos_samples: Positive samples from the data distribution
            neg_samples: Not used in DSM
            
        Returns:
            Tuple of (loss value, dictionary of metrics)
        """
        total_loss = 0.0
        metrics = {}

        for i, sigma in enumerate(self.sigma):
            # Add noise to inputs
            noise = torch.randn_like(pos_samples) * sigma
            x_noisy = pos_samples + noise

            # Compute score of model
            x_noisy.requires_grad_(True)
            energy = self.energy_function(x_noisy)
            score_model = torch.autograd.grad(
                energy.sum(), x_noisy, create_graph=True
            )[0]

            # Target score (gradient of log density of noise model)
            # For Gaussian noise, this is -(x_noisy - pos_samples) / sigma^2
            score_target = -noise / (sigma ** 2)

            # Compute loss
            loss_sigma = 0.5 * torch.sum((score_model + score_target) ** 2, dim=1).mean()
            total_loss += loss_sigma

            metrics[f'loss_sigma_{sigma}'] = loss_sigma.detach()

        # Average loss over all noise levels
        avg_loss = total_loss / len(self.sigma)
        metrics['loss'] = avg_loss.detach()

        return avg_loss, metrics
```

## SlicedScoreMatching

```python
import torch
from typing import Dict, Tuple

from torchebm.core import BaseEnergyFunction
from torchebm.losses.base import BaseLoss


class SlicedScoreMatchingLoss(BaseLoss):
    """Sliced Score Matching loss."""

    def __init__(
            self,
            energy_function: BaseEnergyFunction,
            n_projections: int = 1
    ):
        """Initialize SSM loss.
        
        Args:
            energy_function: Energy function to train
            n_projections: Number of random projections
        """
        super().__init__(energy_function)
        self.n_projections = n_projections

    def __call__(
            self,
            pos_samples: torch.Tensor,
            neg_samples: torch.Tensor = None,
            **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the SSM loss.
        
        Args:
            pos_samples: Positive samples from the data distribution
            neg_samples: Not used in SSM
            
        Returns:
            Tuple of (loss value, dictionary of metrics)
        """
        x = pos_samples.detach().requires_grad_(True)

        # Compute energy
        energy = self.energy_function(x)

        # Compute score (first derivatives)
        score = torch.autograd.grad(
            energy.sum(), x, create_graph=True
        )[0]

        total_loss = 0.0
        for _ in range(self.n_projections):
            # Generate random vectors
            v = torch.randn_like(x)
            v = v / torch.norm(v, p=2, dim=1, keepdim=True)

            # Compute directional derivative
            Jv = torch.sum(score * v, dim=1)

            # Compute second directional derivative
            J2v = torch.autograd.grad(
                Jv.sum(), x, create_graph=True
            )[0]

            # Compute sliced score matching loss terms
            loss_1 = torch.sum(J2v * v, dim=1)
            loss_2 = 0.5 * torch.sum(score ** 2, dim=1)

            # Full loss
            loss = loss_1 + loss_2
            total_loss += loss.mean()

        # Average loss over projections
        avg_loss = total_loss / self.n_projections

        # Metrics to track
        metrics = {
            'loss': avg_loss.detach()
        }

        return avg_loss, metrics
```

## Performance Optimizations

For computationally intensive loss functions like Score Matching, we can use vectorized operations and CUDA optimizations:

```python
def batched_hessian_trace(energy_function, x, batch_size=16):
    """Compute the trace of the Hessian in batches to save memory."""
    x.requires_grad_(True)
    trace = torch.zeros(x.size(0), device=x.device)
    
    # Compute energy and score
    energy = energy_function(x)
    score = torch.autograd.grad(
        energy.sum(), x, create_graph=True
    )[0]
    
    # Compute trace of Hessian in batches
    for i in range(0, x.size(1), batch_size):
        end_i = min(i + batch_size, x.size(1))
        sub_dims = list(range(i, end_i))
        
        for j in sub_dims:
            # Compute diagonal elements of Hessian
            grad_score_j = torch.autograd.grad(
                score[:, j].sum(), x, create_graph=True
            )[0]
            trace += grad_score_j[:, j]
    
    return trace
```

## Factory Methods

Factory methods simplify loss creation:

```python
def create_loss(
    loss_type: str,
    energy_function: BaseEnergyFunction,
    **kwargs
) -> BaseLoss:
    """Create a loss function instance.
    
    Args:
        loss_type: Type of loss function
        energy_function: Energy function to train
        **kwargs: BaseLoss-specific parameters
        
    Returns:
        BaseLoss instance
    """
    if loss_type.lower() == 'mle':
        return MLELoss(energy_function, **kwargs)
    elif loss_type.lower() == 'cd':
        return ContrastiveDivergenceLoss(energy_function, **kwargs)
    elif loss_type.lower() == 'nce':
        return NCELoss(energy_function, **kwargs)
    elif loss_type.lower() == 'sm':
        return ScoreMatchingLoss(energy_function, **kwargs)
    elif loss_type.lower() == 'dsm':
        return DenoisingScoreMatchingLoss(energy_function, **kwargs)
    elif loss_type.lower() == 'ssm':
        return SlicedScoreMatchingLoss(energy_function, **kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
```

## Testing BaseLoss Functions

For testing loss implementations:

```python
def validate_loss_gradients(
    loss_fn: BaseLoss,
    dim: int = 2,
    n_samples: int = 10,
    seed: int = 42
) -> bool:
    """Validate that loss function produces valid gradients.
    
    Args:
        loss_fn: BaseLoss function to test
        dim: Dimensionality of test samples
        n_samples: Number of test samples
        seed: Random seed
        
    Returns:
        True if validation passes, False otherwise
    """
    torch.manual_seed(seed)
    
    # Generate test samples
    pos_samples = torch.randn(n_samples, dim)
    neg_samples = torch.randn(n_samples, dim)
    
    # Ensure parameters require grad
    for param in loss_fn.energy_function.parameters():
        param.requires_grad_(True)
    
    # Compute loss
    loss, _ = loss_fn(pos_samples, neg_samples)
    
    # Check if loss is scalar
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        print(f"BaseLoss is not a scalar: {loss}")
        return False
    
    # Check if loss produces gradients
    try:
        loss.backward()
        has_grad = all(p.grad is not None for p in loss_fn.energy_function.parameters())
        if not has_grad:
            print("Some parameters did not receive gradients")
            return False
    except Exception as e:
        print(f"Error during backward pass: {e}")
        return False
    
    return True
```

## Best Practices for Custom BaseLoss Functions

When implementing custom loss functions, follow these best practices:

<div class="grid" markdown>

<div markdown>
### Do

* Subclass the `BaseLoss` base class
* Return both the loss and metrics dictionary
* Validate inputs
* Use autograd for derivatives
* Consider numerical stability
</div>

<div markdown>
### Don't

* Modify input tensors in-place
* Compute unnecessary gradients
* Forget to detach metrics
* Mix device types
* Ignore potential NaN values
</div>

</div>

!!! example "Custom BaseLoss Example"
    ```python
    class CustomLoss(BaseLoss):
        """Custom loss example."""
        
        def __init__(self, energy_function, alpha=1.0, beta=0.5):
            super().__init__(energy_function)
            self.alpha = alpha
            self.beta = beta
            
        def __call__(self, pos_samples, neg_samples, **kwargs):
            # Compute energies
            pos_energy = self.energy_function(pos_samples)
            neg_energy = self.energy_function(neg_samples)
            
            # Custom loss logic
            loss = (pos_energy.mean() - self.alpha * neg_energy.mean()) + \
                   self.beta * torch.abs(pos_energy.mean() - neg_energy.mean())
            
            # Return loss and metrics
            metrics = {
                'pos_energy': pos_energy.mean().detach(),
                'neg_energy': neg_energy.mean().detach(),
                'loss': loss.detach()
            }
            
            return loss, metrics
    ```

## Resources

<div class="grid cards" markdown>

-   :material-function:{ .lg .middle } __Core Components__

    ---

    Learn about core components and their interactions.

    [:octicons-arrow-right-24: Core Components](core_components.md)

-   :material-code-json:{ .lg .middle } __Energy Functions__

    ---

    Explore energy function implementation details.

    [:octicons-arrow-right-24: Energy Functions](implementation_energy.md)

-   :material-function-variant:{ .lg .middle } __Samplers__

    ---

    Understand sampler implementation details.

    [:octicons-arrow-right-24: Implementation Samplers](implementation_samplers.md)

</div> 
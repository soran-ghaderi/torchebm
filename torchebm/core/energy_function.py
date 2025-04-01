import math
from abc import ABC, abstractmethod
import torch
from torch import nn
from typing import Optional


class EnergyFunction(nn.Module, ABC):
    """
    Abstract base class for energy functions (Potential Energy U(x)).

    This class serves as a standard interface for defining energy functions used
    within the torchebm library. It is compatible with both pre-defined analytical
    functions (like Gaussian, DoubleWell) and trainable neural network models.
    It represents the potential energy U(x), often related to a probability
    distribution p(x) by U(x) = -log p(x) + constant.

    Core Requirements for Subclasses:
    1.  Implement the `forward(x)` method to compute the scalar energy per sample.
    2.  Optionally, override the `gradient(x)` method if an efficient analytical
        gradient is available. Otherwise, the default implementation using
        `torch.autograd` will be used.

    Inheriting from `torch.nn.Module` ensures that:
    - Subclasses can contain trainable parameters (`nn.Parameter`).
    - Standard PyTorch methods like `.to(device)`, `.parameters()`, `.state_dict()`,
      and integration with `torch.optim` work as expected.
    """

    def __init__(self):
        """Initializes the EnergyFunction base class."""
        super().__init__()
        # Optional: store device, though nn.Module handles parameter/buffer device placement
        self._device: Optional[torch.device] = None

    @property
    def device(self) -> Optional[torch.device]:
        """Returns the device associated with the module's parameters/buffers (if any)."""
        try:
            # Attempt to infer device from the first parameter/buffer found
            return next(self.parameters()).device
        except StopIteration:
            try:
                return next(self.buffers()).device
            except StopIteration:
                # If no parameters or buffers, return the explicitly set _device or None
                return self._device

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the scalar energy value for each input sample.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_dims).
                               It's recommended that subclasses handle moving `x`
                               to the correct device if necessary, although callers
                               should ideally provide `x` on the correct device.

        Returns:
            torch.Tensor: Tensor of scalar energy values with shape (batch_size,).
                          Lower values typically indicate higher probability density.
        """
        pass

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the gradient of the energy function with respect to the input x (∇_x U(x)).

        This default implementation uses automatic differentiation based on the
        `forward` method. Subclasses should override this method if a more
        efficient or numerically stable analytical gradient is available.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_dims).

        Returns:
            torch.Tensor: Gradient tensor of the same shape as x.
        """
        # Ensure x is on the same device as the model if possible
        if self.device and x.device != self.device:
             x = x.to(self.device)

        # Detach x from any previous computation graph and ensure it requires gradients
        x_req_grad = x.detach().requires_grad_(True)
        energy = self.forward(x_req_grad)

        # Validate energy shape - should be one scalar per batch item
        if energy.shape != (x.shape[0],):
             raise ValueError(f"EnergyFunction forward() output expected shape ({x.shape[0]},), but got {energy.shape}.")

        if not x_req_grad.grad_fn:
             # If x_req_grad was not used in computation graph leading to energy
              raise RuntimeError("Cannot compute gradient: input `x` is not part of the computation graph for the energy.")


        # Compute gradient using autograd
        gradient = torch.autograd.grad(
            outputs=energy,
            inputs=x_req_grad,
            grad_outputs=torch.ones_like(energy, device=energy.device), # Ensure grad_outputs is on the same device
            create_graph=True,  # Allows for higher-order derivatives if needed later
            retain_graph=True, # Allows calling gradient multiple times if needed within a larger scope
                               # Consider setting based on context if performance is critical
        )[0]

        if gradient is None:
            # This should theoretically not happen if checks above pass, but good to have.
            raise RuntimeError("Gradient computation failed unexpectedly. Check the forward pass implementation.")

        return gradient

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Alias for the forward method for standard PyTorch module usage."""
        # Note: nn.Module.__call__ has hooks; calling forward directly bypasses them.
        # It's generally better to call the module instance: energy_fn(x)
        return super().__call__(x, *args, **kwargs) # Use nn.Module's __call__

    # Override the base nn.Module `to` method to also store the device hint
    def to(self, *args, **kwargs):
        """Moves and/or casts the parameters and buffers."""
        new_self = super().to(*args, **kwargs)
        # Try to update the internal device hint after moving
        try:
            # Get device from args/kwargs (handling different ways .to can be called)
            device = None
            if args:
                if isinstance(args[0], torch.device):
                    device = args[0]
                elif isinstance(args[0], str):
                    device = torch.device(args[0])
            if 'device' in kwargs:
                device = kwargs['device']

            if device:
                 new_self._device = device
        except Exception:
             # Ignore potential errors in parsing .to args, rely on parameter/buffer device
             pass
        return new_self


class DoubleWellEnergy(EnergyFunction):
    """
    Energy function for a double well potential.

    Args:
        barrier_height (float): Height of the barrier between the wells.
    """

    def __init__(self, barrier_height: float = 2.0):
        super().__init__()
        self.barrier_height = barrier_height

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.barrier_height * (x.pow(2) - 1).pow(2).sum(dim=-1)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        return 4 * self.barrier_height * x * (x.pow(2) - 1)


class GaussianEnergy(EnergyFunction):
    """
    Energy function for a Gaussian distribution. U(x) = 0.5 * (x-μ)ᵀ Σ⁻¹ (x-μ).

    Args:
        mean (torch.Tensor): Mean vector (μ) of the Gaussian distribution.
        cov (torch.Tensor): Covariance matrix (Σ) of the Gaussian distribution.
    """

    def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
        super().__init__()
        # Validate shapes
        if mean.ndim != 1:
            raise ValueError("Mean must be a 1D tensor.")
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("Covariance must be a 2D square matrix.")
        if mean.shape[0] != cov.shape[0]:
            raise ValueError("Mean vector dimension must match covariance matrix dimension.")

        # Register mean and covariance inverse as buffers.
        # Buffers are part of the module's state (`state_dict`) and moved by `.to()`,
        # but are not considered parameters by optimizers.
        self.register_buffer('mean', mean)
        try:
            cov_inv = torch.inverse(cov)
            self.register_buffer('cov_inv', cov_inv)
        except RuntimeError as e:
            raise ValueError(f"Failed to invert covariance matrix: {e}. Ensure it is invertible.") from e
        # Optional: Store original covariance as buffer if needed elsewhere
        # self.register_buffer('cov', cov)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the Gaussian energy: 0.5 * (x-μ)ᵀ Σ⁻¹ (x-μ)."""
        # Ensure x is compatible shape (batch_size, dim)
        if x.ndim == 1: # Handle single sample case
             x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[1] != self.mean.shape[0]:
             raise ValueError(f"Input x expected shape (batch_size, {self.mean.shape[0]}), but got {x.shape}")

        # mean and cov_inv are automatically on the correct device via register_buffer
        delta = x - self.mean
        # Using einsum for batched matrix-vector product: (B,i) * (i,j) * (B,j) -> (B,)
        energy = 0.5 * torch.einsum("bi,ij,bj->b", delta, self.cov_inv, delta)
        return energy

    # Override gradient for efficiency (analytical gradient)
    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the analytical gradient: Σ⁻¹ (x-μ)."""
         # Ensure x is compatible shape (batch_size, dim)
        if x.ndim == 1: # Handle single sample case
             x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[1] != self.mean.shape[0]:
             raise ValueError(f"Input x expected shape (batch_size, {self.mean.shape[0]}), but got {x.shape}")

        # mean and cov_inv are automatically on the correct device
        delta = x - self.mean
        # Using einsum for batched matrix-vector product: (i,j) * (B,j) -> (B,i)
        grad = torch.einsum("ij,bj->bi", self.cov_inv, delta)
        # Squeeze if input was single sample
        if grad.shape[0] == 1 and x.ndim == 1:
             grad = grad.squeeze(0)
        return grad

    # No custom `to` method needed - nn.Module handles buffers.


class HarmonicEnergy(EnergyFunction):
    """
    Energy function for a harmonic oscillator.

    Args:
        k (float): Spring constant.
    """

    def __init__(self, k: float = 1.0):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.k * x.pow(2).sum(dim=-1)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        return self.k * x


class RosenbrockEnergy(EnergyFunction):
    """
    Energy function for the Rosenbrock function.

    Args:
        a (float): Parameter `a` of the Rosenbrock function.
        b (float): Parameter `b` of the Rosenbrock function.
    """

    def __init__(self, a: float = 1.0, b: float = 100.0):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.a - x[..., 0]) ** 2 + self.b * (x[..., 1] - x[..., 0] ** 2) ** 2

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        grad = torch.zeros_like(x)
        grad[..., 0] = -2 * (self.a - x[..., 0]) - 4 * self.b * x[..., 0] * (
            x[..., 1] - x[..., 0] ** 2
        )
        grad[..., 1] = 2 * self.b * (x[..., 1] - x[..., 0] ** 2)
        return grad


class AckleyEnergy(EnergyFunction):
    """
    Energy function for the Ackley function.

    Args:
        a (float): Parameter `a` of the Ackley function.
        b (float): Parameter `b` of the Ackley function.
        c (float): Parameter `c` of the Ackley function.
    """

    def __init__(self, a: float = 20.0, b: float = 0.2, c: float = 2 * math.pi):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        sum1 = torch.sum(x**2, dim=-1)
        sum2 = torch.sum(torch.cos(self.c * x), dim=-1)
        term1 = -self.a * torch.exp(-self.b * torch.sqrt(sum1 / n))
        term2 = -torch.exp(sum2 / n)
        return term1 + term2 + self.a + math.e

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        sum1 = torch.sum(x**2, dim=-1, keepdim=True)
        sum2 = torch.sum(torch.cos(self.c * x), dim=-1, keepdim=True)
        term1 = (
            self.a
            * self.b
            * torch.exp(-self.b * torch.sqrt(sum1 / n))
            * (x / torch.sqrt(sum1 / n))
        )
        term2 = torch.exp(sum2 / n) * torch.sin(self.c * x) * self.c / n
        return term1 + term2


class RastriginEnergy(EnergyFunction):
    """
    Energy function for the Rastrigin function.

    Args:
        a (float): Parameter `a` of the Rastrigin function.
    """

    def __init__(self, a: float = 10.0):
        super().__init__()
        self.a = a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        return self.a * n + torch.sum(
            x**2 - self.a * torch.cos(2 * math.pi * x), dim=-1
        )

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * x + 2 * math.pi * self.a * torch.sin(2 * math.pi * x)

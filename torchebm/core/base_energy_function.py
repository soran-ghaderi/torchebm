import math
from abc import ABC, abstractmethod
import torch
from torch import nn
from typing import Optional


class BaseEnergyFunction(nn.Module, ABC):
    """
    Abstract base class for energy functions (Potential Energy E(x)).

    This class serves as a standard interface for defining energy functions used
    within the torchebm library. It is compatible with both pre-defined analytical
    functions (like Gaussian, DoubleWell) and trainable neural network models.
    It represents the potential energy E(x), often related to a probability
    distribution p(x) by E(x) = -log p(x) + constant.

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
        """Initializes the BaseEnergyFunction base class."""
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
        Computes the gradient of the energy function with respect to the input x (∇_x E(x)).

        This default implementation uses automatic differentiation based on the
        `forward` method. Subclasses should override this method if a more
        efficient or numerically stable analytical gradient is available.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_dims).

        Returns:
            torch.Tensor: Gradient tensor of the same shape as x.
        """
        # Store original dtype and device
        original_dtype = x.dtype
        device = x.device

        # Ensure x is on the correct device (if specified by the model)
        if self.device and device != self.device:
            x = x.to(self.device)
            device = self.device  # Update device if x was moved

        with torch.enable_grad():
            # Detach, convert to float32, and enable gradient tracking
            x_for_grad = (
                x.detach().to(dtype=torch.float32, device=device).requires_grad_(True)
            )

            # Perform forward pass with float32 input
            energy = self.forward(x_for_grad)

            # Validate energy shape - should be one scalar per batch item
            if energy.shape != (x_for_grad.shape[0],):
                raise ValueError(
                    f"BaseEnergyFunction forward() output expected shape ({x_for_grad.shape[0]},), but got {energy.shape}."
                )

            # Check grad_fn on the float32 energy
            if not energy.grad_fn:
                raise RuntimeError(
                    "Cannot compute gradient: `forward` method did not use the input `x` (as float32) in a differentiable way."
                )

            # Compute gradient using autograd w.r.t. the float32 input
            gradient_float32 = torch.autograd.grad(
                outputs=energy,
                inputs=x_for_grad,  # Compute gradient w.r.t the float32 version
                grad_outputs=torch.ones_like(energy, device=energy.device),
                create_graph=False,  # Set to False for standard gradient computation
                retain_graph=None,  # Usually not needed when create_graph=False, let PyTorch decide
            )[0]

        if gradient_float32 is None:
            # This should theoretically not happen if checks above pass, but good to have.
            raise RuntimeError(
                "Gradient computation failed unexpectedly. Check the forward pass implementation."
            )

        # Cast gradient back to the original dtype before returning
        gradient = gradient_float32.to(original_dtype)

        return gradient.detach()

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Alias for the forward method for standard PyTorch module usage."""
        # Note: nn.Module.__call__ has hooks; calling forward directly bypasses them.
        # It's generally better to call the module instance: energy_fn(x)
        return super().__call__(x, *args, **kwargs)  # Use nn.Module's __call__

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
            if "device" in kwargs:
                device = kwargs["device"]

            if device:
                new_self._device = device
        except Exception:
            # Ignore potential errors in parsing .to args, rely on parameter/buffer device
            pass
        return new_self


class DoubleWellEnergy(BaseEnergyFunction):
    """
    Energy function for a double well potential. E(x) = h * Σ((x²-1)²) where h is the barrier height.

    This energy function creates a bimodal distribution with two modes at x = +1 and x = -1
    (in each dimension), separated by a barrier of height h at x = 0.

    Args:
        barrier_height (float): Height of the barrier between the wells.
    """

    def __init__(self, barrier_height: float = 2.0):
        super().__init__()
        self.barrier_height = barrier_height

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the double well energy: h * Σ((x²-1)²)."""
        # Ensure x is compatible shape
        if x.ndim == 1:  # Handle single sample case
            x = x.unsqueeze(0)

        return self.barrier_height * (x.pow(2) - 1).pow(2).sum(dim=-1)

    # Override gradient for efficiency (analytical gradient)
    # def gradient(self, x: torch.Tensor) -> torch.Tensor:
    #     """Computes the analytical gradient: 4h * x * (x²-1)."""
    #     # Ensure x is compatible shape
    #     if x.ndim == 1:  # Handle single sample case
    #         x = x.unsqueeze(0)
    #
    #     return 4 * self.barrier_height * x * (x.pow(2) - 1)


class GaussianEnergy(BaseEnergyFunction):
    """
    Energy function for a Gaussian distribution. E(x) = 0.5 * (x-μ)ᵀ Σ⁻¹ (x-μ).

    Args:
        mean (torch.Tensor): Mean vector (μ) of the Gaussian distribution.
        cov (torch.Tensor): Covariance matrix (Σ) of the Gaussian distribution.
    """

    def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
        super().__init__()
        if mean.ndim != 1:
            raise ValueError("Mean must be a 1D tensor.")
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("Covariance must be a 2D square matrix.")
        if mean.shape[0] != cov.shape[0]:
            raise ValueError(
                "Mean vector dimension must match covariance matrix dimension."
            )

        # Register mean and covariance inverse as buffers.
        # Buffers are part of the module's state (`state_dict`) and moved by `.to()`,
        # but are not considered parameters by optimizers.
        self.register_buffer("mean", mean)
        try:
            cov_inv = torch.inverse(cov)
            self.register_buffer("cov_inv", cov_inv)
        except RuntimeError as e:
            raise ValueError(
                f"Failed to invert covariance matrix: {e}. Ensure it is invertible."
            ) from e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the Gaussian energy: 0.5 * (x-μ)ᵀ Σ⁻¹ (x-μ)."""
        # Ensure x is compatible shape (batch_size, dim)
        if x.ndim == 1:  # Handle single sample case
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"Input x expected shape (batch_size, {self.mean.shape[0]}), but got {x.shape}"
            )

        # Get mean and cov_inv on the same device as x
        # We don't change the dtype because gradient() already converted x to float32
        mean = self.mean.to(device=x.device)
        cov_inv = self.cov_inv.to(device=x.device)

        # Compute centered vectors
        # Important: use x directly without detaching or converting to maintain grad tracking
        delta = x - mean

        # Calculate energy
        # Use batch matrix multiplication for better numerical stability
        # We use einsum which maintains gradients through operations
        energy = 0.5 * torch.einsum("bi,ij,bj->b", delta, cov_inv, delta)

        return energy

    # Override gradient for efficiency (analytical gradient)
    # def gradient(self, x: torch.Tensor) -> torch.Tensor:
    #     """Computes the analytical gradient: Σ⁻¹ (x-μ)."""
    #     # Ensure x is compatible shape (batch_size, dim)
    #     if x.ndim == 1:  # Handle single sample case
    #         x = x.unsqueeze(0)
    #     if x.ndim != 2 or x.shape[1] != self.mean.shape[0]:
    #         raise ValueError(
    #             f"Input x expected shape (batch_size, {self.mean.shape[0]}), but got {x.shape}"
    #         )
    #
    #     # mean and cov_inv are automatically on the correct device
    #     delta = x - self.mean
    #     # Using einsum for batched matrix-vector product: (i,j) * (B,j) -> (B,i)
    #     grad = torch.einsum("ij,bj->bi", self.cov_inv, delta)
    #     # Squeeze if input was single sample
    #     if grad.shape[0] == 1 and x.ndim == 1:
    #         grad = grad.squeeze(0)
    #     return grad

    # No custom `to` method needed - nn.Module handles buffers.


class HarmonicEnergy(BaseEnergyFunction):
    """
    Energy function for a harmonic oscillator. E(x) = 0.5 * n_steps * Σ(x²).

    This energy function represents a quadratic potential centered at the origin,
    equivalent to a Gaussian distribution with zero mean and variance proportional to 1/n_steps.

    Args:
        k (float): Spring constant.
    """

    def __init__(self, k: float = 1.0):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the harmonic oscillator energy: 0.5 * n_steps * Σ(x²)."""
        # Ensure x is compatible shape
        if x.ndim == 1:  # Handle single sample case
            x = x.unsqueeze(0)

        return 0.5 * self.k * x.pow(2).sum(dim=-1)

    # Override gradient for efficiency (analytical gradient)
    # def gradient(self, x: torch.Tensor) -> torch.Tensor:
    #     """Computes the analytical gradient: n_steps * x."""
    #     # Ensure x is compatible shape
    #     if x.ndim == 1:  # Handle single sample case
    #         x = x.unsqueeze(0)
    #
    #     return self.n_steps * x


class RosenbrockEnergy(BaseEnergyFunction):
    """
    Energy function for the Rosenbrock function. E(x) = (a-x₁)² + b·(x₂-x₁²)².

    This energy function creates a challenging valley-shaped distribution with the
    global minimum at (a, a²). It's commonly used as a benchmark for optimization algorithms
    due to its curved, narrow valley which is difficult to traverse.

    Args:
        a (float): Parameter `a` of the Rosenbrock function.
        b (float): Parameter `b` of the Rosenbrock function.
    """

    def __init__(self, a: float = 1.0, b: float = 100.0):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the Rosenbrock energy: (a-x₁)² + b·(x₂-x₁²)²."""
        # Ensure x is compatible shape
        if x.ndim == 1:  # Handle single sample case
            x = x.unsqueeze(0)
        # Validate dimensions - Rosenbrock requires at least 2 dimensions
        if x.shape[-1] < 2:
            raise ValueError(
                f"Rosenbrock energy function requires at least 2 dimensions, got {x.shape[-1]}"
            )

        return (self.a - x[..., 0]) ** 2 + self.b * (x[..., 1] - x[..., 0] ** 2) ** 2

    # Override gradient for efficiency (analytical gradient)
    # def gradient(self, x: torch.Tensor) -> torch.Tensor:
    #     """Computes the analytical gradient for the Rosenbrock function."""
    #     # Ensure x is compatible shape
    #     if x.ndim == 1:  # Handle single sample case
    #         x = x.unsqueeze(0)
    #     # Validate dimensions
    #     if x.shape[-1] < 2:
    #         raise ValueError(
    #             f"Rosenbrock energy function requires at least 2 dimensions, got {x.shape[-1]}"
    #         )
    #
    #     grad = torch.zeros_like(x)
    #     grad[..., 0] = -2 * (self.a - x[..., 0]) - 4 * self.b * x[..., 0] * (
    #         x[..., 1] - x[..., 0] ** 2
    #     )
    #     grad[..., 1] = 2 * self.b * (x[..., 1] - x[..., 0] ** 2)
    #     return grad


class AckleyEnergy(BaseEnergyFunction):
    """
    Energy function for the Ackley function.

    The Ackley energy is defined as:

    $$E(x) = -a \cdot \exp\left(-b \cdot \sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2}\right) - \exp\left(\frac{1}{n}\sum_{i=1}^{n} \cos(c \cdot x_i)\right) + a + e$$

    This function has a global minimum at the origin surrounded by many local minima,
    creating a challenging optimization landscape that tests an algorithm's ability to
    escape local optima.

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
        """
        Computes the Ackley energy.

        $$E(x) = -a \cdot \exp\left(-b \cdot \sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2}\right) - \exp\left(\frac{1}{n}\sum_{i=1}^{n} \cos(c \cdot x_i)\right) + a + e$$
        """
        # Ensure x is compatible shape
        if x.ndim == 1:  # Handle single sample case
            x = x.unsqueeze(0)

        n = x.shape[-1]
        sum1 = torch.sum(x**2, dim=-1)
        sum2 = torch.sum(torch.cos(self.c * x), dim=-1)
        term1 = -self.a * torch.exp(-self.b * torch.sqrt(sum1 / n))
        term2 = -torch.exp(sum2 / n)
        return term1 + term2 + self.a + math.e

    # Override gradient for efficiency (analytical gradient)
    # def gradient(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Computes the analytical gradient of the Ackley function.
    #
    #     The gradient components are:
    #
    #     $$\nabla E(x)_i = \frac{a \cdot b \cdot x_i}{\sqrt{n \cdot \sum_{j=1}^{n} x_j^2}} \cdot \exp\left(-b \cdot \sqrt{\frac{1}{n}\sum_{j=1}^{n} x_j^2}\right) + \frac{c}{n} \cdot \sin(c \cdot x_i) \cdot \exp\left(\frac{1}{n}\sum_{j=1}^{n} \cos(c \cdot x_j)\right)$$
    #     """
    #     # Ensure x is compatible shape
    #     if x.ndim == 1:  # Handle single sample case
    #         x = x.unsqueeze(0)
    #
    #     n = x.shape[-1]
    #     sum1 = torch.sum(x**2, dim=-1, keepdim=True)
    #     sum2 = torch.sum(torch.cos(self.c * x), dim=-1, keepdim=True)
    #     term1 = (
    #         self.a
    #         * self.b
    #         * torch.exp(-self.b * torch.sqrt(sum1 / n))
    #         * (x / torch.sqrt(sum1 / n))
    #     )
    #     term2 = torch.exp(sum2 / n) * torch.sin(self.c * x) * self.c / n
    #     return term1 + term2


class RastriginEnergy(BaseEnergyFunction):
    """
    Energy function for the Rastrigin function.

    The Rastrigin energy is defined as:

    $$E(x) = an + \sum_{i=1}^{n} [x_i^2 - a \cos(2\pi x_i)]$$

    This function is characterized by a large number of local minima arranged in a
    regular lattice, with a global minimum at the origin. It's a classic test for
    optimization algorithms due to its highly multimodal nature.

    Args:
        a (float): Parameter `a` of the Rastrigin function.
    """

    def __init__(self, a: float = 10.0):
        super().__init__()
        self.a = a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the Rastrigin energy.

        $$E(x) = an + \sum_{i=1}^{n} [x_i^2 - a \cos(2\pi x_i)]$$
        """
        # Ensure x is compatible shape
        if x.ndim == 1:  # Handle single sample case
            x = x.unsqueeze(0)

        n = x.shape[-1]
        return self.a * n + torch.sum(
            x**2 - self.a * torch.cos(2 * math.pi * x), dim=-1
        )

    # Override gradient for efficiency (analytical gradient)
    # def gradient(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Computes the analytical gradient of the Rastrigin function.
    #
    #     $$\nabla E(x)_i = 2x_i + 2\pi a \sin(2\pi x_i)$$
    #     """
    #     # Ensure x is compatible shape
    #     if x.ndim == 1:  # Handle single sample case
    #         x = x.unsqueeze(0)
    #
    #     return 2 * x + 2 * math.pi * self.a * torch.sin(2 * math.pi * x)

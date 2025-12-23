import math
import warnings
from abc import ABC, abstractmethod
import torch
from torch import nn
from typing import Optional, Union

from torchebm.core import DeviceMixin


class BaseModel(DeviceMixin, nn.Module, ABC):
    r"""
    Abstract base class for energy-based models (EBMs).

    This class provides a unified interface for defining EBMs, which represent
    the unnormalized negative log-likelihood of a probability distribution.
    It supports both analytical models and trainable neural networks.

    Subclasses must implement the `forward(x)` method and can optionally
    override the `gradient(x)` method for analytical gradients.
    """
    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        use_mixed_precision: bool = False,
        *args,
        **kwargs,
    ):
        """Initializes the BaseModel base class."""
        super().__init__(*args, **kwargs)
        self.dtype = dtype
        self.setup_mixed_precision(use_mixed_precision)

    # @property
    # def device(self) -> torch.device:
    #     """Returns the device associated with the module's parameters/buffers (if any)."""
    #     try:
    #         return next(self.parameters()).device
    #     except StopIteration:
    #         try:
    #             return next(self.buffers()).device
    #         except StopIteration:
    #             return self._device

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the scalar energy value for each input sample.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_dims).

        Returns:
            torch.Tensor: Tensor of scalar energy values with shape (batch_size,).
        """
        pass

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the gradient of the energy function with respect to the input, \(\nabla_x E(x)\).

        This default implementation uses `torch.autograd`. Subclasses can override it
        for analytical gradients.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_dims).

        Returns:
            torch.Tensor: Gradient tensor of the same shape as `x`.
        """

        original_dtype = x.dtype
        device = x.device

        if self.device and device != self.device:
            x = x.to(self.device)
            device = self.device

        with torch.enable_grad():  # todo: consider removing conversion to fp32 and uncessary device change
            x_for_grad = (
                x.detach().to(dtype=torch.float32, device=device).requires_grad_(True)
            )

            with self.autocast_context():
                energy = self.forward(x_for_grad)

            if energy.shape != (x_for_grad.shape[0],):
                raise ValueError(
                    f"BaseModel forward() output expected shape ({x_for_grad.shape[0]},), but got {energy.shape}."
                )

            if not energy.grad_fn:
                raise RuntimeError(
                    "Cannot compute gradient: `forward` method did not use the input `x` (as float32) in a differentiable way."
                )

            gradient_float32 = torch.autograd.grad(
                outputs=energy,
                inputs=x_for_grad,
                grad_outputs=torch.ones_like(energy, device=energy.device),
                create_graph=False,  # false for standard grad computation
                retain_graph=None,  # since create_graph=False, let PyTorch decide
            )[0]

        if gradient_float32 is None:  # for triple checking!
            raise RuntimeError(
                "Gradient computation failed unexpectedly. Check the forward pass implementation."
            )

        gradient = gradient_float32.to(original_dtype)

        return gradient.detach()

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Alias for the forward method for standard PyTorch module usage."""
        if (x.device != self.device) or (x.dtype != self.dtype):
            x = x.to(device=self.device, dtype=self.dtype)

        with self.autocast_context():
            return super().__call__(x, *args, **kwargs)


class DoubleWellModel(BaseModel):
    r"""
    Energy-based model for a double-well potential.

    Args:
        barrier_height (float): The height of the energy barrier between the wells.
        b (float): The position of the wells (default is 1.0, creating wells at ±1).
    """
    def __init__(self, barrier_height: float = 2.0, b: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.barrier_height = barrier_height
        self.b = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Computes the double well energy: \(h \sum_{i=1}^{n} (x_i^2 - b^2)^2\)."""
        if x.ndim == 1:
            x = x.unsqueeze(0)

        return self.barrier_height * (x.pow(2) - self.b**2).pow(2).sum(dim=-1)


class GaussianModel(BaseModel):
    r"""
    Energy-based model for a Gaussian distribution.

    Args:
        mean (torch.Tensor): The mean vector (μ) of the Gaussian distribution.
        cov (torch.Tensor): The covariance matrix (Σ) of the Gaussian distribution.
    """
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if mean.ndim != 1:
            raise ValueError("Mean must be a 1D tensor.")
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("Covariance must be a 2D square matrix.")
        if mean.shape[0] != cov.shape[0]:
            raise ValueError(
                "Mean vector dimension must match covariance matrix dimension."
            )

        self.register_buffer("mean", mean.to(dtype=self.dtype, device=self.device))
        try:
            cov_inv = torch.inverse(cov)
            self.register_buffer(
                "cov_inv", cov_inv.to(dtype=self.dtype, device=self.device)
            )
        except RuntimeError as e:
            raise ValueError(
                f"Failed to invert covariance matrix: {e}. Ensure it is invertible."
            ) from e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Computes the Gaussian energy: \(E(x) = \frac{1}{2} (x - \mu)^{\top} \Sigma^{-1} (x - \mu)\)."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"Input x expected batch_shape (batch_size, {self.mean.shape[0]}), but got {x.shape}"
            )

        x = x.to(dtype=self.dtype, device=self.device)
        # mean = self.mean.to(device=x.device)
        cov_inv = self.cov_inv.to(dtype=self.dtype, device=x.device)

        delta = (
            x - self.mean
        )  # avoid detaching or converting x to maintain grad tracking
        # energy = 0.5 * torch.einsum("bi,ij,bj->b", delta, cov_inv, delta)

        if delta.shape[0] > 1:
            delta_expanded = delta.unsqueeze(-1)  # (batch_size, dim, 1)
            cov_inv_expanded = cov_inv.unsqueeze(0).expand(
                delta.shape[0], -1, -1
            )  # (batch_size, dim, dim)

            temp = torch.bmm(cov_inv_expanded, delta_expanded)  # (batch_size, dim, 1)
            energy = 0.5 * torch.bmm(delta.unsqueeze(1), temp).squeeze(-1).squeeze(-1)
        else:
            energy = 0.5 * torch.sum(delta * torch.matmul(delta, cov_inv), dim=-1)

        return energy


class HarmonicModel(BaseModel):
    r"""
    Energy-based model for a harmonic oscillator.

    Args:
        k (float): The spring constant.
    """
    def __init__(self, k: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Computes the harmonic oscillator energy: \(\frac{1}{2} k \sum_{i=1}^{n} x_i^{2}\)."""
        if x.ndim == 1:
            x = x.unsqueeze(0)

        return 0.5 * self.k * x.pow(2).sum(dim=-1)


class RosenbrockModel(BaseModel):
    r"""
    Energy-based model for the Rosenbrock function.

    Args:
        a (float): The `a` parameter of the Rosenbrock function.
        b (float): The `b` parameter of the Rosenbrock function.
    """
    def __init__(self, a: float = 1.0, b: float = 100.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = a
        self.b = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Computes the Rosenbrock energy: \(\sum_{i=1}^{n-1} \left[ b(x_{i+1} - x_i^2)^2 + (a - x_i)^2 \right]\)."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.shape[-1] < 2:
            raise ValueError(
                f"Rosenbrock energy function requires at least 2 dimensions, got {x.shape[-1]}"
            )

        # return (self.a - x[..., 0]) ** 2 + self.b * (x[..., 1] - x[..., 0] ** 2) ** 2
        # return sum(
        #     self.b * (x[..., i + 1] - x[..., i] ** 2) ** 2 + (self.a - x[i]) ** 2
        #     for i in range(len(x) - 1)
        # )

        x_i = x[:, :-1]
        x_ip1 = x[:, 1:]
        term1 = (self.a - x_i).pow(2)
        term2 = self.b * (x_ip1 - x_i.pow(2)).pow(2)
        return (term1 + term2).sum(dim=-1)


class AckleyModel(BaseModel):
    r"""
    Energy-based model for the Ackley function.

    Args:
        a (float): The `a` parameter of the Ackley function.
        b (float): The `b` parameter of the Ackley function.
        c (float): The `c` parameter of the Ackley function.
    """
    def __init__(
        self, a: float = 20.0, b: float = 0.2, c: float = 2 * math.pi, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.a = a
        self.b = b
        self.c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Computes the Ackley energy."""
        if x.ndim == 1:
            x = x.unsqueeze(0)

        n = x.shape[-1]
        sum1 = torch.sum(x**2, dim=-1)
        sum2 = torch.sum(torch.cos(self.c * x), dim=-1)
        term1 = -self.a * torch.exp(-self.b * torch.sqrt(sum1 / n))
        term2 = -torch.exp(sum2 / n)
        return term1 + term2 + self.a + math.e


class RastriginModel(BaseModel):
    r"""
    Energy-based model for the Rastrigin function.

    Args:
        a (float): The `a` parameter of the Rastrigin function.
    """
    def __init__(self, a: float = 10.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Computes the Rastrigin energy."""
        if x.ndim == 1:
            x = x.unsqueeze(0)

        n = x.shape[-1]
        return self.a * n + torch.sum(
            x**2 - self.a * torch.cos(2 * math.pi * x), dim=-1
        )


# Deprecated Classes -> will be removed in the next two pypi releases starting from v0.4.0 

class BaseEnergyFunction(BaseModel):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "`BaseEnergyFunction` is deprecated and will be removed in a future version. "
            "Please use `BaseModel` instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)

class DoubleWellEnergy(DoubleWellModel):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "`DoubleWellEnergy` is deprecated and will be removed in a future version. "
            "Please use `DoubleWellModel` instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


class GaussianEnergy(GaussianModel):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "`GaussianEnergy` is deprecated and will be removed in a future version. "
            "Please use `GaussianModel` instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


class HarmonicEnergy(HarmonicModel):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "`HarmonicEnergy` is deprecated and will be removed in a future version. "
            "Please use `HarmonicModel` instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


class RosenbrockEnergy(RosenbrockModel):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "`RosenbrockEnergy` is deprecated and will be removed in a future version. "
            "Please use `RosenbrockModel` instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


class AckleyEnergy(AckleyModel):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "`AckleyEnergy` is deprecated and will be removed in a future version. "
            "Please use `AckleyModel` instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


class RastriginEnergy(RastriginModel):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "`RastriginEnergy` is deprecated and will be removed in a future version. "
            "Please use `RastriginModel` instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)

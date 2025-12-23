r"""Score Matching Loss Module"""

import torch
import warnings
from typing import Optional, Union, Dict, Tuple, Any, Callable

from torchebm.core import BaseModel
from torchebm.core.base_loss import BaseScoreMatching


class ScoreMatching(BaseScoreMatching):
    r"""
    Original Score Matching loss from Hyvärinen (2005).

    Trains an energy-based model by matching the model's score function
    \(\nabla_x \log p_\theta(x)\) to the data's score. Avoids MCMC sampling
    but requires computing the trace of the Hessian.

    Args:
        model: The energy-based model to train.
        hessian_method: Method for Hessian trace ('exact' or 'approx').
        regularization_strength: Coefficient for regularization.
        custom_regularization: A custom regularization function.
        use_mixed_precision: Whether to use mixed-precision training.
        dtype: Data type for computations.
        device: Device for computations.

    Example:
        ```python
        from torchebm.losses import ScoreMatching
        from torchebm.core import DoubleWellEnergy
        import torch

        energy = DoubleWellEnergy()
        loss_fn = ScoreMatching(model=energy, hessian_method="exact")
        x = torch.randn(32, 2)
        loss = loss_fn(x)
        ```
    """

    def __init__(
        self,
        model: BaseModel,
        hessian_method: str = "exact",
        regularization_strength: float = 0.0,
        custom_regularization: Optional[Callable] = None,
        use_mixed_precision: bool = False,
        is_training=True,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            regularization_strength=regularization_strength,
            use_autograd=True,
            custom_regularization=custom_regularization,
            use_mixed_precision=use_mixed_precision,
            dtype=dtype,
            device=device,
            *args,
            **kwargs,
        )

        self.hessian_method = hessian_method
        self.training = is_training
        valid_methods = ["exact", "approx"]
        if self.hessian_method not in valid_methods:
            warnings.warn(
                f"Invalid hessian_method '{self.hessian_method}'. "
                f"Using 'exact' instead. Valid options are: {valid_methods}",
                UserWarning,
            )
            self.hessian_method = "exact"

        if self.use_mixed_precision and self.hessian_method == "exact":
            warnings.warn(
                "Using 'exact' Hessian method with mixed precision may be unstable. "
                "Consider using SlicedScoreMatching for better numerical stability.",
                UserWarning,
            )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Computes the score matching loss for a batch of data.

        Args:
            x (torch.Tensor): Input data tensor of shape `(batch_size, *data_dims)`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The scalar score matching loss.
        """
        if (x.device != self.device) or (x.dtype != self.dtype):
            x = x.to(device=self.device, dtype=self.dtype)

        with self.autocast_context():
            loss = self.compute_loss(x, *args, **kwargs)

        if self.regularization_strength > 0 or self.custom_regularization is not None:
            loss = self.add_regularization(loss, x)

        return loss

    def compute_loss(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Computes the score matching loss using the specified Hessian computation method.

        Args:
            x (torch.Tensor): Input data tensor of shape `(batch_size, *data_dims)`.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The scalar score matching loss.
        """

        if self.hessian_method == "approx":
            return self._approx_score_matching(x)
        else:
            return self._exact_score_matching(x)

    def _exact_score_matching(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Computes score matching loss with an exact Hessian trace.

        This method is computationally expensive for high-dimensional data.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: The score matching loss.
        """
        batch_size = x.shape[0]
        feature_dim = x.numel() // batch_size

        x_leaf = x.detach().clone()
        x_leaf.requires_grad_(True)

        energy = self.model(x_leaf)
        logp_sum = (-energy).sum()
        grad1 = torch.autograd.grad(
            logp_sum, x_leaf, create_graph=True, retain_graph=True
        )[0]

        grad1_flat = grad1.view(batch_size, -1)
        term1 = 0.5 * grad1_flat.pow(2).sum(dim=1)

        laplacian = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
        for i in range(feature_dim):
            comp_sum = grad1_flat[:, i].sum()
            grad2_full = torch.autograd.grad(
                comp_sum,
                x_leaf,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if grad2_full is None:
                grad2_comp = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
            else:
                grad2_comp = grad2_full.view(batch_size, -1)[:, i]
            laplacian += grad2_comp

        loss_per_sample = term1 + laplacian
        return loss_per_sample.mean()

    def _approx_score_matching(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Computes score matching loss using a finite-difference approximation for the Hessian trace.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: The score matching loss.
        """

        batch_size = x.shape[0]
        data_dim = x.numel() // batch_size

        x_detached = x.detach().clone()
        x_detached.requires_grad_(True)

        score = self.compute_score(x_detached)
        score_square_term = (
            0.5 * torch.sum(score**2, dim=list(range(1, len(x.shape)))).mean()
        )

        epsilon = 1e-5
        x_noise = x_detached + epsilon * torch.randn_like(x_detached)

        score_x = self.compute_score(x_detached)
        score_x_noise = self.compute_score(x_noise)

        hessian_trace = torch.sum(
            (score_x_noise - score_x) * (x_noise - x_detached),
            dim=list(range(1, len(x.shape))),
        ).mean() / (epsilon**2 * data_dim)

        loss = score_square_term - hessian_trace

        return loss

    def _hutchinson_score_matching(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        DEPRECATED: Use SlicedScoreMatching for efficient trace estimation.

        This method has been deprecated in favor of SlicedScoreMatching which provides
        a more efficient and theoretically sound implementation of Hutchinson's estimator.
        """
        warnings.warn(
            "ScoreMatching._hutchinson_score_matching is deprecated. "
            "Use SlicedScoreMatching for efficient trace estimation instead.",
            DeprecationWarning,
        )
        return self._exact_score_matching(x)


class DenoisingScoreMatching(BaseScoreMatching):
    r"""
    Denoising Score Matching (DSM) from Vincent (2011).

    Avoids computing the Hessian trace by matching the score of noise-perturbed
    data. More computationally efficient and often more stable than standard
    Score Matching.

    Args:
        model: The energy-based model to train.
        noise_scale: Standard deviation of Gaussian noise to add.
        regularization_strength: Coefficient for regularization.
        custom_regularization: A custom regularization function.
        use_mixed_precision: Whether to use mixed-precision training.
        dtype: Data type for computations.
        device: Device for computations.

    Example:
        ```python
        from torchebm.losses import DenoisingScoreMatching
        from torchebm.core import DoubleWellEnergy
        import torch

        energy = DoubleWellEnergy()
        loss_fn = DenoisingScoreMatching(model=energy, noise_scale=0.1)
        x = torch.randn(32, 2)
        loss = loss_fn(x)
        ```
    """

    def __init__(
        self,
        model: BaseModel,
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
            model=model,
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
        Computes the denoising score matching loss for a batch of data.

        Args:
            x (torch.Tensor): Input data tensor of shape `(batch_size, *data_dims)`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The scalar denoising score matching loss.
        """
        if (x.device != self.device) or (x.dtype != self.dtype):
            x = x.to(device=self.device, dtype=self.dtype)

        with self.autocast_context():
            loss = self.compute_loss(x, *args, **kwargs)

        if self.regularization_strength > 0 or self.custom_regularization is not None:
            loss = self.add_regularization(loss, x)

        return loss

    def compute_loss(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Computes the denoising score matching loss.

        Args:
            x (torch.Tensor): Input data tensor of shape `(batch_size, *data_dims)`.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The scalar denoising score matching loss.
        """
        x_perturbed, noise = self.perturb_data(x)

        score = self.compute_score(x_perturbed)

        target_score = -noise / (self.noise_scale**2)

        loss = (
            0.5
            * torch.sum(
                (score - target_score) ** 2, dim=list(range(1, len(x.shape)))
            ).mean()
        )

        return loss


class SlicedScoreMatching(BaseScoreMatching):
    r"""
    Sliced Score Matching (SSM) from Song et al. (2019).

    A scalable variant that uses random projections to efficiently approximate
    the score matching objective, avoiding expensive Hessian trace computation.

    Args:
        model: The energy-based model to train.
        n_projections: Number of random projections to use.
        projection_type: Type of projections ('rademacher', 'sphere', 'gaussian').
        regularization_strength: Coefficient for regularization.
        custom_regularization: A custom regularization function.
        use_mixed_precision: Whether to use mixed-precision training.
        dtype: Data type for computations.
        device: Device for computations.

    Example:
        ```python
        from torchebm.losses import SlicedScoreMatching
        from torchebm.core import DoubleWellEnergy
        import torch

        energy = DoubleWellEnergy()
        loss_fn = SlicedScoreMatching(model=energy, n_projections=5)
        x = torch.randn(32, 2)
        loss = loss_fn(x)
        ```
    """

    def __init__(
        self,
        model: BaseModel,
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
            model=model,
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
        valid_types = ["rademacher", "sphere", "gaussian"]
        if self.projection_type not in valid_types:
            warnings.warn(
                f"Invalid projection_type '{self.projection_type}'. "
                f"Using 'rademacher' instead. Valid options are: {valid_types}",
                UserWarning,
            )
            self.projection_type = "rademacher"

    def _get_random_projections(self, shape: torch.Size) -> torch.Tensor:
        r"""
        Generates random vectors for projections.

        Args:
            shape (torch.Size): The shape of the vectors to generate.

        Returns:
            torch.Tensor: A tensor of random projection vectors.
        """
        vectors = torch.randn_like(shape)
        if self.projection_type == "rademacher":
            return vectors.sign()
        elif self.projection_type == "sphere":
            return (
                vectors
                / torch.norm(vectors, dim=-1, keepdim=True)
                * torch.sqrt(vectors.shape[-1])
            )
        else:
            return vectors

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Computes the sliced score matching loss for a batch of data.

        Args:
            x (torch.Tensor): Input data tensor of shape `(batch_size, *data_dims)`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The scalar sliced score matching loss.
        """
        if (x.device != self.device) or (x.dtype != self.dtype):
            x = x.to(device=self.device, dtype=self.dtype)

        with self.autocast_context():
            loss = self.compute_loss(x, *args, **kwargs)

        if self.regularization_strength > 0 or self.custom_regularization is not None:
            loss = self.add_regularization(loss, x)

        return loss

    def compute_loss(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Computes the sliced score matching loss using random projections.

        Args:
            x (torch.Tensor): Input data tensor of shape `(batch_size, *data_dims)`.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The scalar sliced score matching loss.
        """

        dup_x = (
            x.unsqueeze(0)
            .expand(self.n_projections, *x.shape)
            .contiguous()
            .view(-1, *x.shape[1:])
        ).requires_grad_(
            True
        )  # final shape: (n_particles * batch_size, d). tracing the shape: (batch_size, d) -> (1, batch_size, d)
        # -> (n_particles, batch_size, d) -> (n_particles, batch_size, d) -> (n_particles * batch_size, d)

        n_vectors = self._get_random_projections(dup_x)

        logp = (-self.model(dup_x)).sum()
        grad1 = torch.autograd.grad(logp, dup_x, create_graph=True)[0]
        v_score = torch.sum(grad1 * n_vectors, dim=-1)
        term1 = 0.5 * (v_score**2)

        grad_v = torch.autograd.grad(v_score.sum(), dup_x, create_graph=True)[0]
        term2 = torch.sum(n_vectors * grad_v, dim=-1)

        term1 = term1.view(self.n_projections, -1).mean(dim=0)
        term2 = term2.view(self.n_projections, -1).mean(dim=0)

        loss = term2 + term1

        return loss.mean()

    
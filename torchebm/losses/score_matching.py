r"""Score Matching Loss Module"""

import math
import warnings
from typing import Optional, Union, Dict, Tuple, Any, Callable

import torch
from torch import nn
from torch.func import grad as func_grad, vmap, jacrev


from torchebm.core import BaseModel, BaseScoreMatching
# from torchebm.core.base_loss import BaseScoreMatching


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
        dtype: Data type for computations.
        device: Device for computations.

    Example:
        ```python
        from torchebm.losses import ScoreMatching
        from torchebm.core import DoubleWellModel
        import torch

        energy = DoubleWellModel()
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
        use_autograd: bool = True,
        functional_model: Optional[nn.Module] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            regularization_strength=regularization_strength,
            use_autograd=use_autograd,
            custom_regularization=custom_regularization,
            functional_model=functional_model,
            dtype=dtype,
            device=device,
            *args,
            **kwargs,
        )

        self.hessian_method = hessian_method
        valid_methods = ["exact", "approx"]
        if self.hessian_method not in valid_methods:
            warnings.warn(
                f"Invalid hessian_method '{self.hessian_method}'. "
                f"Using 'exact' instead. Valid options are: {valid_methods}",
                UserWarning,
            )
            self.hessian_method = "exact"

    def forward(
        self,
        x: torch.Tensor,
        *args,
        model_kwargs: Optional[dict] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Computes the score matching loss for a batch of data.

        Args:
            x (torch.Tensor): Input data tensor of shape `(batch_size, *data_dims)`.
            *args: Additional positional arguments.
            model_kwargs: Conditioning arguments (e.g. class labels) forwarded to
                the model. Supported for ``hessian_method="approx"``; the exact
                Hessian path raises if conditioning is passed (see below).
            **kwargs: Deprecated bare model kwargs.

        Returns:
            torch.Tensor: The scalar score matching loss.
        """
        if (x.device != self.device) or (x.dtype != self.dtype):
            x = x.to(device=self.device, dtype=self.dtype)

        with self.autocast_context():
            loss = self.compute_loss(
                x, *args, model_kwargs=model_kwargs, generator=generator, **kwargs
            )

        if self.regularization_strength > 0 or self.custom_regularization is not None:
            mk = self._resolve_model_kwargs(
                model_kwargs, kwargs, warn_key="sm-bare-model-kwargs"
            )
            loss = self.add_regularization(loss, x, model_kwargs=mk)

        return loss

    def compute_loss(
        self,
        x: torch.Tensor,
        *args,
        model_kwargs: Optional[dict] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Computes the score matching loss using the specified Hessian computation method.

        Args:
            x (torch.Tensor): Input data tensor of shape `(batch_size, *data_dims)`.
            *args: Additional arguments.
            model_kwargs: Conditioning arguments forwarded to the model
                (``hessian_method="approx"`` only).
            **kwargs: Deprecated bare model kwargs.

        Returns:
            torch.Tensor: The scalar score matching loss.
        """
        mk = self._resolve_model_kwargs(
            model_kwargs, kwargs, warn_key="sm-bare-model-kwargs"
        )
        if self.hessian_method == "approx":
            return self._approx_score_matching(x, model_kwargs=mk, generator=generator)
        else:
            return self._exact_score_matching(x, model_kwargs=mk)

    def _exact_score_matching(
        self, x: torch.Tensor, model_kwargs: Optional[dict] = None
    ) -> torch.Tensor:
        r"""
        Computes score matching loss with an exact Hessian trace.

        This method is computationally expensive for high-dimensional data.

        Args:
            x (torch.Tensor): Input data tensor.
            model_kwargs: Conditioning is not supported here: the exact Hessian
                trace uses ``vmap`` over single samples, so per-sample
                conditioning cannot be batched. A non-empty mapping raises.

        Returns:
            torch.Tensor: The score matching loss.
        """
        if model_kwargs:
            raise NotImplementedError(
                "Conditional exact score matching is not supported (the vmap "
                "Hessian trace cannot batch per-sample conditioning). Use "
                "hessian_method='approx' or DenoisingScoreMatching for "
                "conditional training."
            )
        if self._functional_state()[2] is not None:
            raise NotImplementedError(
                "Exact score matching computes per-sample Hessians with vmap, "
                "which does not support DTensor parameters. Use "
                "hessian_method='approx' or DenoisingScoreMatching."
            )
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1).detach().requires_grad_(True)

        def score_fn(x_single):
            return func_grad(
                lambda x_i: -self.model(x_i.unsqueeze(0)).squeeze()
            )(x_single)
        
        def laplacian_fn(x_single):
            J = jacrev(score_fn)(x_single)
            return J.diagonal().sum()

        score = vmap(score_fn)(x_flat)
        laplacian = vmap(laplacian_fn)(x_flat)

        term1 = 0.5 * score.square().sum(dim=-1)
        return (term1 + laplacian).mean()

    def _approx_score_matching(
        self,
        x: torch.Tensor,
        model_kwargs: Optional[dict] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        r"""
        Computes score matching loss using a finite-difference approximation for the Hessian trace.

        Args:
            x (torch.Tensor): Input data tensor.
            model_kwargs: Conditioning arguments forwarded to the model.
            generator: RNG for the finite-difference probe noise; the global RNG
                when `None`.

        Returns:
            torch.Tensor: The score matching loss.
        """

        batch_size = x.shape[0]
        data_dim = x.numel() // batch_size

        x_detached = x.detach()
        x_detached.requires_grad_(True)

        score = self.compute_score(x_detached, model_kwargs=model_kwargs)
        score_square_term = (
            0.5 * torch.sum(score.square(), dim=list(range(1, len(x.shape)))).mean()
        )

        epsilon = 1e-5
        x_noise = x_detached + epsilon * torch.randn_like(
            x_detached, generator=generator
        )

        score_x = self.compute_score(x_detached, model_kwargs=model_kwargs)
        score_x_noise = self.compute_score(x_noise, model_kwargs=model_kwargs)

        hessian_trace = torch.sum(
            (score_x_noise - score_x) * (x_noise - x_detached),
            dim=list(range(1, len(x.shape))),
        ).mean() / (epsilon**2 * data_dim)

        loss = score_square_term - hessian_trace

        return loss

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
        dtype: Data type for computations.
        device: Device for computations.

    Example:
        ```python
        from torchebm.losses import DenoisingScoreMatching
        from torchebm.core import DoubleWellModel
        import torch

        energy = DoubleWellModel()
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
        use_autograd: bool = True,
        functional_model: Optional[nn.Module] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            noise_scale=noise_scale,
            regularization_strength=regularization_strength,
            use_autograd=use_autograd,
            custom_regularization=custom_regularization,
            functional_model=functional_model,
            dtype=dtype,
            device=device,
            *args,
            **kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        *args,
        model_kwargs: Optional[dict] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Computes the denoising score matching loss for a batch of data.

        Args:
            x (torch.Tensor): Input data tensor of shape `(batch_size, *data_dims)`.
            *args: Additional positional arguments.
            model_kwargs: Conditioning arguments (e.g. class labels) forwarded to
                the model. ``None`` (default) is the unconditional path.
            **kwargs: Deprecated bare model kwargs (see `model_kwargs`).

        Returns:
            torch.Tensor: The scalar denoising score matching loss.
        """
        if (x.device != self.device) or (x.dtype != self.dtype):
            x = x.to(device=self.device, dtype=self.dtype)

        with self.autocast_context():
            loss = self.compute_loss(
                x, *args, model_kwargs=model_kwargs, generator=generator, **kwargs
            )

        if self.regularization_strength > 0 or self.custom_regularization is not None:
            mk = self._resolve_model_kwargs(
                model_kwargs, kwargs, warn_key="dsm-bare-model-kwargs"
            )
            loss = self.add_regularization(loss, x, model_kwargs=mk)

        return loss

    def compute_loss(
        self,
        x: torch.Tensor,
        *args,
        model_kwargs: Optional[dict] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Computes the denoising score matching loss.

        Args:
            x (torch.Tensor): Input data tensor of shape `(batch_size, *data_dims)`.
            *args: Additional arguments.
            model_kwargs: Conditioning arguments forwarded to the model.
            **kwargs: Deprecated bare model kwargs.

        Returns:
            torch.Tensor: The scalar denoising score matching loss.
        """
        mk = self._resolve_model_kwargs(
            model_kwargs, kwargs, warn_key="dsm-bare-model-kwargs"
        )
        x_perturbed, noise = self.perturb_data(x, generator=generator)

        score = self.compute_score(x_perturbed, model_kwargs=mk)

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
        dtype: Data type for computations.
        device: Device for computations.

    Example:
        ```python
        from torchebm.losses import SlicedScoreMatching
        from torchebm.core import DoubleWellModel
        import torch

        energy = DoubleWellModel()
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
        use_autograd: bool = True,
        functional_model: Optional[nn.Module] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            regularization_strength=regularization_strength,
            use_autograd=use_autograd,
            custom_regularization=custom_regularization,
            functional_model=functional_model,
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

    def _get_random_projections(
        self, shape: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        r"""
        Generates random vectors for projections.

        Args:
            shape (torch.Tensor): Tensor whose shape, dtype and device the
                projection vectors match.
            generator: RNG for the projection draw; the global RNG when `None`.

        Returns:
            torch.Tensor: A tensor of random projection vectors.
        """
        vectors = torch.randn_like(shape, generator=generator)
        if self.projection_type == "rademacher":
            return vectors.sign()
        elif self.projection_type == "sphere":
            return torch.nn.functional.normalize(vectors, dim=-1) * math.sqrt(
                vectors.shape[-1]
            )
        else:
            return vectors

    def forward(
        self,
        x: torch.Tensor,
        *args,
        model_kwargs: Optional[dict] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Computes the sliced score matching loss for a batch of data.

        Args:
            x (torch.Tensor): Input data tensor of shape `(batch_size, *data_dims)`.
            *args: Additional positional arguments.
            model_kwargs: Conditioning is not supported (the projection tiling
                expands the batch, so per-sample conditioning cannot be aligned);
                a non-empty mapping raises in `compute_loss`.
            **kwargs: Deprecated bare model kwargs.

        Returns:
            torch.Tensor: The scalar sliced score matching loss.
        """
        if (x.device != self.device) or (x.dtype != self.dtype):
            x = x.to(device=self.device, dtype=self.dtype)

        with self.autocast_context():
            loss = self.compute_loss(
                x, *args, model_kwargs=model_kwargs, generator=generator, **kwargs
            )

        if self.regularization_strength > 0 or self.custom_regularization is not None:
            loss = self.add_regularization(loss, x)

        return loss

    def compute_loss(
        self,
        x: torch.Tensor,
        *args,
        model_kwargs: Optional[dict] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Computes the sliced score matching loss using random projections.

        Args:
            x (torch.Tensor): Input data tensor of shape `(batch_size, *data_dims)`.
            *args: Additional arguments.
            model_kwargs: Not supported; a non-empty mapping raises.
            **kwargs: Deprecated bare model kwargs.

        Returns:
            torch.Tensor: The scalar sliced score matching loss.
        """
        mk = self._resolve_model_kwargs(
            model_kwargs, kwargs, warn_key="ssm-bare-model-kwargs"
        )
        if mk:
            raise NotImplementedError(
                "Conditional sliced score matching is not supported (random "
                "projections expand the batch, so per-sample conditioning "
                "cannot be aligned). Use DenoisingScoreMatching for conditional "
                "training."
            )

        dup_x = (
            x.unsqueeze(0)
            .expand(self.n_projections, *x.shape)
            .contiguous()
            .view(-1, *x.shape[1:])
        )  # final shape: (n_particles * batch_size, d). tracing the shape: (batch_size, d) -> (1, batch_size, d)
        # -> (n_particles, batch_size, d) -> (n_particles, batch_size, d) -> (n_particles * batch_size, d)

        if not self.use_autograd:
            return self._functional_sliced_loss(dup_x, generator=generator)

        self._require_autograd_safe_params()
        dup_x = dup_x.requires_grad_(True)
        n_vectors = self._get_random_projections(dup_x, generator=generator)

        logp = (-self.model(dup_x)).sum()
        grad1 = torch.autograd.grad(logp, dup_x, create_graph=True)[0]
        v_score = torch.sum(grad1 * n_vectors, dim=-1)
        term1 = 0.5 * v_score.square()

        grad_v = torch.autograd.grad(v_score.sum(), dup_x, create_graph=True)[0]
        term2 = torch.sum(n_vectors * grad_v, dim=-1)

        term1 = term1.view(self.n_projections, -1).mean(dim=0)
        term2 = term2.view(self.n_projections, -1).mean(dim=0)

        loss = term2 + term1

        return loss.mean()

    def _functional_sliced_loss(
        self, dup_x: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        r"""Sliced loss via the functional path (see `BaseScoreMatching`).

        Projections are drawn locally and, on a mesh, wrapped batch-sharded so
        both second-order gradients run through differentiable DTensor
        collectives instead of module hooks.

        Args:
            dup_x (torch.Tensor): Projection-tiled batch of shape
                `(n_projections * batch_size, d)`.
            generator: RNG for the projection draw; the global RNG when `None`.

        Returns:
            torch.Tensor: The scalar sliced score matching loss.
        """
        params, buffers, mesh = self._functional_state()
        n_vectors = self._get_random_projections(dup_x, generator=generator)
        leaf = self._functional_leaf(dup_x, mesh)
        if mesh is not None:
            from torch.distributed.tensor import DTensor, Shard

            n_vectors = DTensor.from_local(
                n_vectors, mesh, [Shard(0)], run_check=False
            )
        energy = self._functional_energy(leaf, params, buffers, mesh)
        grad1 = torch.autograd.grad((-energy).sum(), leaf, create_graph=True)[0]
        v_score = torch.sum(grad1 * n_vectors, dim=-1)
        grad_v = torch.autograd.grad(v_score.sum(), leaf, create_graph=True)[0]
        term2 = torch.sum(n_vectors * grad_v, dim=-1)

        v_score = self._functional_localize(v_score, mesh)
        term2 = self._functional_localize(term2, mesh)
        term1 = (0.5 * v_score.square()).view(self.n_projections, -1).mean(dim=0)
        term2 = term2.view(self.n_projections, -1).mean(dim=0)
        return (term1 + term2).mean()

    
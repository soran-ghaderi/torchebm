r"""Score Matching Loss Module"""

import math
import warnings
from typing import Optional, Union, Dict, Tuple, Any, Callable

import torch
from torch import nn
from torch.func import grad as func_grad, vmap, jacrev


from torchebm._deprecation import deprecated
from torchebm.core import BaseLoss, BaseModel, BaseScoreMatching
from torchebm.losses.functional import (
    denoising_score_matching_loss,
    score_matching_loss,
    sliced_score_matching_loss,
)
# from torchebm.core.base_loss import BaseScoreMatching

_PROJECTION_TYPES = ("rademacher", "sphere", "gaussian")


def _random_projections(
    like: torch.Tensor, projection_type: str, generator: Optional[torch.Generator]
) -> torch.Tensor:
    r"""Projection directions shaped like ``like``.

    ``"rademacher"``: random signs; ``"sphere"``: uniform on the sphere of
    radius \(\sqrt{d}\); ``"gaussian"``: standard normal.
    """
    vectors = torch.randn_like(like, generator=generator)
    if projection_type == "rademacher":
        return vectors.sign()
    if projection_type == "sphere":
        return torch.nn.functional.normalize(vectors, dim=-1) * math.sqrt(
            vectors.shape[-1]
        )
    return vectors


class ScoreMatchingLoss(BaseLoss):
    r"""Score matching loss (Hyvärinen, 2005) on a score and its divergence.

    \[
    L = \tfrac{1}{2}\|s_\theta(x)\|^2 + \nabla_x \cdot s_\theta(x)
    \]

    The divergence is a model-side quantity: compute it with `torch.func`
    (``vmap(jacrev(...))`` for the exact trace) or a Hutchinson estimate.
    See `score_matching_loss`.

    Args:
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.
        dtype: Data type for computations.
        device: Device for computations.

    Example:
        ```python
        loss_fn = ScoreMatchingLoss()
        loss = loss_fn(score, divergence)
        ```
    """

    def forward(self, input: torch.Tensor, divergence: torch.Tensor) -> torch.Tensor:
        r"""Compute the loss from the score and its divergence.

        Args:
            input: Score \(s_\theta(x)\), shape (batch_size, ...).
            divergence: \(\nabla_x \cdot s_\theta(x)\), shape (batch_size,).

        Returns:
            Scalar loss, or shape (batch_size,) for ``reduction="none"``.
        """
        return score_matching_loss(input, divergence, self.reduction)


class DenoisingScoreMatchingLoss(BaseLoss):
    r"""Denoising score matching loss (Vincent, 2011).

    \[
    L = \tfrac{1}{2}\, w\, \|s_\theta(\tilde x) - \nabla_{\tilde x} \log q(\tilde x \mid x)\|^2
    \]

    For \(\tilde x = x + \sigma \epsilon\) the kernel score is
    \(-\epsilon / \sigma\), and ``weight`` \(= \sigma^2\) is the noise-level
    weighting of Song and Ermon (2019). See `denoising_score_matching_loss`.

    Args:
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.
        dtype: Data type for computations.
        device: Device for computations.

    Example:
        ```python
        loss_fn = DenoisingScoreMatchingLoss()
        noise = torch.randn_like(x)
        loss = loss_fn(model(x + sigma * noise), -noise / sigma)
        ```
    """

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Compute the loss from the score at the perturbed input.

        Args:
            input: Score \(s_\theta(\tilde x)\), shape (batch_size, ...).
            target: Score of the perturbation kernel, same shape.
            weight: Per-sample weight, shape (batch_size,).

        Returns:
            Scalar loss, or shape (batch_size,) for ``reduction="none"``.
        """
        return denoising_score_matching_loss(input, target, weight, self.reduction)


class SlicedScoreMatchingLoss(BaseLoss):
    r"""Sliced score matching loss (Song et al., 2019).

    \[
    L = \mathbb{E}_v\left[ v^\top \nabla_x \big(v^\top s_\theta(x)\big)
        + \tfrac{1}{2} \big(v^\top s_\theta(x)\big)^2 \right]
    \]

    Projections are drawn inside the call, as `torch.nn.Dropout` draws its
    mask. The score must be evaluated on inputs tiled ``n_projections`` times
    and keep its graph to them. See `sliced_score_matching_loss`.

    Args:
        n_projections: Projections per sample.
        projection_type: ``"rademacher"`` (default), ``"sphere"``, or
            ``"gaussian"``.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.
        dtype: Data type for computations.
        device: Device for computations.

    Example:
        ```python
        loss_fn = SlicedScoreMatchingLoss(n_projections=1)
        x_tiled = x.repeat(loss_fn.n_projections, 1).requires_grad_(True)
        loss = loss_fn(model(x_tiled), x_tiled)
        ```
    """

    def __init__(
        self,
        n_projections: int = 1,
        projection_type: str = "rademacher",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if n_projections < 1:
            raise ValueError(f"n_projections must be positive, got {n_projections}")
        if projection_type not in _PROJECTION_TYPES:
            raise ValueError(
                f"projection_type must be one of {_PROJECTION_TYPES}, "
                f"got {projection_type!r}"
            )
        self.n_projections = int(n_projections)
        self.projection_type = projection_type

    def forward(
        self,
        input: torch.Tensor,
        x: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        r"""Compute the loss from the score at the tiled inputs.

        Args:
            input: Score at the tiled inputs, shape
                (n_projections * batch_size, ...), differentiable w.r.t. ``x``.
            x: The tiled inputs, same shape, requiring grad.
            generator: RNG for the projections; the global RNG when ``None``.

        Returns:
            Scalar loss, or shape (batch_size,) for ``reduction="none"``.

        Raises:
            ValueError: If the batch dimension is not a multiple of
                ``n_projections``.
        """
        if input.shape[0] % self.n_projections:
            raise ValueError(
                f"batch dimension {input.shape[0]} must be a multiple of "
                f"n_projections={self.n_projections}"
            )
        projections = _random_projections(x, self.projection_type, generator)
        return sliced_score_matching_loss(
            input,
            x,
            projections,
            n_projections=self.n_projections,
            reduction=self.reduction,
        )


@deprecated(
    since="0.8.10",
    deprecated_on="2026-09-29",
    until="0.12.0",
    grace_months=4,
    replacement="ScoreMatchingLoss on the model's score and its divergence",
    message=(
        "ScoreMatching(model) is deprecated: compute the score and its "
        "divergence from the model and use ScoreMatchingLoss()(score, divergence)."
    ),
    removal="ScoreMatching and its tests",
)
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
        y: Optional[torch.Tensor] = None,
        model_kwargs: Optional[dict] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Computes the score matching loss for a batch of data.

        Args:
            x (torch.Tensor): Input data tensor of shape `(batch_size, *data_dims)`.
            *args: Additional positional arguments.
            y: Optional conditioning tensor; shorthand for
                ``model_kwargs={'y': y}`` with the same support constraints.
            model_kwargs: Conditioning arguments (e.g. class labels) forwarded to
                the model. Supported for ``hessian_method="approx"``; the exact
                Hessian path raises if conditioning is passed (see below).
            **kwargs: Deprecated bare model kwargs.

        Returns:
            torch.Tensor: The scalar score matching loss.
        """
        model_kwargs = self._merge_condition(model_kwargs, y)
        if (x.device != self.device) or (x.dtype != self.dtype):
            x = x.to(device=self.device, dtype=self.dtype)

        with self.autocast_context():
            loss = self.compute_loss(
                x, *args, model_kwargs=model_kwargs, generator=generator, **kwargs
            )

        if self.regularization_strength > 0 or self.custom_regularization is not None:
            mk = self._resolve_model_kwargs(model_kwargs, kwargs)
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
        mk = self._resolve_model_kwargs(model_kwargs, kwargs)
        self._check_condition(x, mk)
        mk = self._apply_cfg_dropout(mk, generator)
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

        return score_matching_loss(score, laplacian)

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


@deprecated(
    since="0.8.10",
    deprecated_on="2026-09-29",
    until="0.12.0",
    grace_months=4,
    replacement="DenoisingScoreMatchingLoss on the model's score at the perturbed input",
    message=(
        "DenoisingScoreMatching(model) is deprecated: perturb the data in the "
        "training loop and use DenoisingScoreMatchingLoss()(model(x_noisy), target)."
    ),
    removal="DenoisingScoreMatching, BaseScoreMatching and their tests",
)
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
        y: Optional[torch.Tensor] = None,
        model_kwargs: Optional[dict] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Computes the denoising score matching loss for a batch of data.

        Args:
            x (torch.Tensor): Input data tensor of shape `(batch_size, *data_dims)`.
            *args: Additional positional arguments.
            y: Optional conditioning tensor forwarded to the model; shorthand
                for ``model_kwargs={'y': y}``.
            model_kwargs: Conditioning arguments (e.g. class labels) forwarded to
                the model. ``None`` (default) is the unconditional path.
            **kwargs: Deprecated bare model kwargs (see `model_kwargs`).

        Returns:
            torch.Tensor: The scalar denoising score matching loss.
        """
        model_kwargs = self._merge_condition(model_kwargs, y)
        if (x.device != self.device) or (x.dtype != self.dtype):
            x = x.to(device=self.device, dtype=self.dtype)

        with self.autocast_context():
            loss = self.compute_loss(
                x, *args, model_kwargs=model_kwargs, generator=generator, **kwargs
            )

        if self.regularization_strength > 0 or self.custom_regularization is not None:
            mk = self._resolve_model_kwargs(model_kwargs, kwargs)
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
        mk = self._resolve_model_kwargs(model_kwargs, kwargs)
        self._check_condition(x, mk)
        mk = self._apply_cfg_dropout(mk, generator)
        x_perturbed, noise = self.perturb_data(x, generator=generator)

        score = self.compute_score(x_perturbed, model_kwargs=mk)

        target_score = noise / (self.noise_scale**2)

        return denoising_score_matching_loss(score, target_score)


@deprecated(
    since="0.8.10",
    deprecated_on="2026-09-29",
    until="0.12.0",
    grace_months=4,
    replacement="SlicedScoreMatchingLoss on the model's score at tiled inputs",
    message=(
        "SlicedScoreMatching(model) is deprecated: tile the batch in the "
        "training loop and use SlicedScoreMatchingLoss()(model(x_tiled), x_tiled)."
    ),
    removal="SlicedScoreMatching and its tests",
)
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
        return _random_projections(shape, self.projection_type, generator)

    def forward(
        self,
        x: torch.Tensor,
        *args,
        y: Optional[torch.Tensor] = None,
        model_kwargs: Optional[dict] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Computes the sliced score matching loss for a batch of data.

        Args:
            x (torch.Tensor): Input data tensor of shape `(batch_size, *data_dims)`.
            *args: Additional positional arguments.
            y: Optional conditioning tensor; conditioning is not supported here,
                so any non-None value raises in `compute_loss` (see
                `model_kwargs`).
            model_kwargs: Conditioning is not supported (the projection tiling
                expands the batch, so per-sample conditioning cannot be aligned);
                a non-empty mapping raises in `compute_loss`.
            **kwargs: Deprecated bare model kwargs.

        Returns:
            torch.Tensor: The scalar sliced score matching loss.
        """
        model_kwargs = self._merge_condition(model_kwargs, y)
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
        mk = self._resolve_model_kwargs(model_kwargs, kwargs)
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

        return sliced_score_matching_loss(
            grad1, dup_x, n_vectors, n_projections=self.n_projections
        )

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


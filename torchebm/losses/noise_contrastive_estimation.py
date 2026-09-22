r"""Noise-Contrastive Estimation (NCE) Loss Module.

References:
    - Gutmann, M., & Hyvärinen, A. (2010). Noise-contrastive estimation: A new
      estimation principle for unnormalized statistical models. In Proceedings of
      the Thirteenth International Conference on Artificial Intelligence and
      Statistics (AISTATS 2010), JMLR: W&CP 9, 297-304.
    - Gutmann, M. U., & Hyvärinen, A. (2012). Noise-contrastive estimation of
      unnormalized statistical models, with applications to natural image
      statistics. Journal of Machine Learning Research (JMLR), 13(11), 307-361.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Tuple, Union

import torch
from torch import nn

from torchebm.core import BaseLoss


class GaussianNoise:
    r"""Gaussian (Normal) noise distribution for Noise-Contrastive Estimation.

    Supports scalar, diagonal, or multi-dimensional Gaussian noise.
    Log probabilities are summed over all non-batch dimensions to yield
    per-sample log densities of shape ``(batch_size,)``.

    Args:
        loc: Mean of the distribution. Scalar or tensor.
        scale: Standard deviation of the distribution. Scalar or tensor.
        device: Device for computations.
        dtype: Data type for computations.
    """

    def __init__(
        self,
        loc: Union[float, torch.Tensor] = 0.0,
        scale: Union[float, torch.Tensor] = 1.0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if not isinstance(loc, torch.Tensor):
            self.loc = torch.tensor(loc, dtype=dtype or torch.float32, device=device)
        else:
            self.loc = loc.to(device=device, dtype=dtype)

        if not isinstance(scale, torch.Tensor):
            self.scale = torch.tensor(scale, dtype=dtype or torch.float32, device=device)
        else:
            self.scale = scale.to(device=device, dtype=dtype)

        if (self.scale <= 0).any():
            raise ValueError("scale must be strictly positive")

    @property
    def device(self) -> torch.device:
        return self.loc.device

    @property
    def dtype(self) -> torch.dtype:
        return self.loc.dtype

    def sample(self, sample_shape: Union[torch.Size, tuple, int] = ()) -> torch.Tensor:
        r"""Draw samples from the Gaussian noise distribution.

        Args:
            sample_shape: Shape of the samples to draw.

        Returns:
            Tensor of samples from the distribution.
        """
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        shape = tuple(sample_shape)
        if self.loc.ndim > 0 and shape[-len(self.loc.shape):] == self.loc.shape:
            full_shape = shape
        else:
            full_shape = (*shape, *self.loc.shape)
        eps = torch.randn(full_shape, device=self.loc.device, dtype=self.loc.dtype)
        return self.loc + self.scale * eps

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        r"""Compute log probability density of value under the Gaussian noise.

        Args:
            value: Tensor of shape ``(batch_size, ...)``.

        Returns:
            Tensor of shape ``(batch_size,)`` containing log probabilities.
        """
        var = self.scale**2
        log_scale = torch.log(self.scale)
        lp = -0.5 * (math.log(2.0 * math.pi) + 2.0 * log_scale + ((value - self.loc) ** 2) / var)
        if lp.ndim > 1:
            return lp.flatten(1).sum(dim=1)
        return lp


class GaussianMixtureNoise:
    r"""Gaussian Mixture Model noise distribution for Noise-Contrastive Estimation.

    Mixture of K Gaussians with component weights \(w_k\), means \(\mu_k\),
    and standard deviations \(\sigma_k\).

    Args:
        weights: 1-D tensor of mixture component probabilities of shape ``(K,)``.
        locs: Tensor of component means of shape ``(K, *feature_dims)``.
        scales: Tensor of component standard deviations of shape ``(K, *feature_dims)``.
        device: Device for computations.
        dtype: Data type for computations.
    """

    def __init__(
        self,
        weights: torch.Tensor,
        locs: torch.Tensor,
        scales: torch.Tensor,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        weights = weights.to(device=device, dtype=dtype or torch.float32)
        locs = locs.to(device=device, dtype=dtype or torch.float32)
        scales = scales.to(device=device, dtype=dtype or torch.float32)

        if weights.ndim != 1:
            raise ValueError(f"weights must be 1-D, got shape {weights.shape}")
        if (weights < 0).any():
            raise ValueError("weights must be non-negative")
        if (scales <= 0).any():
            raise ValueError("scales must be strictly positive")
        if locs.shape[0] != weights.shape[0] or scales.shape[0] != weights.shape[0]:
            raise ValueError("weights, locs, and scales must have matching leading dimension K")

        self.weights = weights / weights.sum()
        self.locs = locs
        self.scales = scales

    @property
    def device(self) -> torch.device:
        return self.weights.device

    @property
    def dtype(self) -> torch.dtype:
        return self.weights.dtype

    def sample(self, sample_shape: Union[torch.Size, tuple, int] = ()) -> torch.Tensor:
        r"""Draw samples from the Gaussian mixture distribution.

        Args:
            sample_shape: Shape of the samples to draw (e.g. batch size).

        Returns:
            Tensor of samples drawn from the mixture.
        """
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        shape = tuple(sample_shape)
        n = shape[0] if shape else 1
        comp_indices = torch.multinomial(self.weights, num_samples=n, replacement=True)
        sel_locs = self.locs[comp_indices]
        sel_scales = self.scales[comp_indices]
        eps = torch.randn_like(sel_locs)
        samples = sel_locs + sel_scales * eps
        if len(shape) > 1:
            samples = samples.view(*shape, *self.locs.shape[1:])
        return samples

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        r"""Compute log probability density of value under the Gaussian mixture.

        Uses ``torch.logsumexp`` across mixture components for numerical stability.

        Args:
            value: Tensor of shape ``(batch_size, *feature_dims)``.

        Returns:
            Tensor of shape ``(batch_size,)`` containing log probabilities.
        """
        k = self.weights.shape[0]
        # value: (N, *feature_dims)
        comp_log_probs = []
        for i in range(k):
            loc_i = self.locs[i]
            scale_i = self.scales[i]
            var_i = scale_i**2
            log_scale_i = torch.log(scale_i)
            lp = -0.5 * (
                math.log(2.0 * math.pi)
                + 2.0 * log_scale_i
                + ((value - loc_i) ** 2) / var_i
            )
            if lp.ndim > 1:
                lp = lp.flatten(1).sum(dim=1)
            comp_log_probs.append(torch.log(self.weights[i]) + lp)

        stacked = torch.stack(comp_log_probs, dim=0)  # (K, N)
        return torch.logsumexp(stacked, dim=0)  # (N,)


class NoiseContrastiveEstimation(BaseLoss):
    r"""Noise-Contrastive Estimation (NCE) Loss.

    The classic NCE objective (Gutmann & Hyvärinen, AISTATS 2010; JMLR 2012)
    estimates an unnormalized energy-based model \(E_\theta(x)\) by training a
    binary classifier to distinguish data samples \(x \sim p_d\) from noise
    samples \(y \sim p_n\).

    The model density is parameterized as an unnormalized energy with a
    learnable log-partition parameter \(c\):
    \[
    \log p_m(x; \theta, c) = -E_\theta(x) + c
    \]
    where \(c \approx -\log Z_\theta\).

    The NCE loss minimizes the negative log posterior odds:
    \[
    \mathcal{L}_{\text{NCE}} = \mathbb{E}_{x \sim p_d}[\text{softplus}(-h(x))] + \nu \mathbb{E}_{y \sim p_n}[\text{softplus}(h(y))]
    \]
    where:
    \[
    h(u) = -E_\theta(u) + c - \log \nu - \log p_n(u)
    \]
    and \(\nu = M / N\) is the noise-to-data ratio.

    NCE requires no MCMC sampling during training and simultaneously estimates
    both the energy model parameters and the normalization constant.

    Args:
        model: The energy-based model \(E_\theta(x)\).
        noise_distribution: The known noise distribution \(p_n\) providing
            ``sample()`` and ``log_prob()``. If None, defaults to standard
            Gaussian noise \(\mathcal{N}(0, I)\).
        noise_ratio: Ratio \(\nu = M / N\) of noise samples to data samples.
            Default: 1.0.
        learnable_c: If True (default), \(c\) is a learnable scalar parameter
            representing the negative log-partition function. If False, \(c\) is
            fixed at `initial_c`.
        initial_c: Initial value for the log-partition parameter \(c\). Default: 0.0.
        dtype: Data type for computations. Default: ``torch.float32``.
        device: Device for computations.
        cfg_dropout: Classifier-free guidance dropout probability in [0, 1].
        null_condition: Null condition representation for CFG dropout.
        check_conditioning: Whether to verify conditioning consumption.

    Example:
        ```python
        import torch
        from torchebm.core import HarmonicModel
        from torchebm.losses import GaussianNoise, NoiseContrastiveEstimation

        model = HarmonicModel()
        noise = GaussianNoise(loc=0.0, scale=1.0)
        loss_fn = NoiseContrastiveEstimation(model=model, noise_distribution=noise)

        x = torch.randn(32, 2)
        loss = loss_fn(x)
        ```
    """

    def __init__(
        self,
        model: nn.Module,
        noise_distribution: Optional[Any] = None,
        noise_ratio: float = 1.0,
        learnable_c: bool = True,
        initial_c: float = 0.0,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        cfg_dropout: float = 0.0,
        null_condition: Union[int, float, torch.Tensor, Callable, None] = None,
        check_conditioning: bool = True,
        *args: Any,
        **kwargs: Any,
    ):
        if model is None:
            raise ValueError("model cannot be None")
        if noise_ratio <= 0:
            raise ValueError(f"noise_ratio must be positive, got {noise_ratio}")

        super().__init__(
            dtype=dtype,
            device=device,
            cfg_dropout=cfg_dropout,
            null_condition=null_condition,
            check_conditioning=check_conditioning,
            *args,
            **kwargs,
        )

        self.model = model
        self.noise_ratio = float(noise_ratio)
        self.learnable_c = bool(learnable_c)

        if noise_distribution is None:
            self.noise_distribution = GaussianNoise(loc=0.0, scale=1.0, device=device, dtype=dtype)
        else:
            self.noise_distribution = noise_distribution

        c_tensor = torch.tensor(float(initial_c), dtype=dtype, device=device)
        if self.learnable_c:
            self.c = nn.Parameter(c_tensor)
        else:
            self.register_buffer("c", c_tensor)

    @staticmethod
    def _eval_log_prob(dist: Any, samples: torch.Tensor) -> torch.Tensor:
        lp = dist.log_prob(samples)
        if not torch.isfinite(lp).all():
            raise ValueError("noise distribution log_prob returned non-finite values (NaN or Inf)")
        if lp.ndim > 1:
            lp = lp.flatten(1).sum(dim=1)
        elif lp.ndim == 0:
            lp = lp.expand(samples.shape[0])
        return lp.to(device=samples.device, dtype=samples.dtype)

    def _sample_noise(self, target_shape: tuple[int, ...]) -> torch.Tensor:
        r"""Sample noise matching target_shape (batch_size, *feature_dims)."""
        n_noise = target_shape[0]
        feature_shape = target_shape[1:]

        # Case 1: If distribution has event_shape matching feature_shape (e.g. PyTorch MultivariateNormal)
        if (
            hasattr(self.noise_distribution, "event_shape")
            and tuple(self.noise_distribution.event_shape) == feature_shape
        ):
            try:
                return self.noise_distribution.sample((n_noise,))
            except TypeError:
                return self.noise_distribution.sample(n_noise)

        # Case 2: If distribution has batch_shape + event_shape matching feature_shape
        if (
            hasattr(self.noise_distribution, "batch_shape")
            and hasattr(self.noise_distribution, "event_shape")
            and tuple(self.noise_distribution.batch_shape) + tuple(self.noise_distribution.event_shape) == feature_shape
        ):
            try:
                return self.noise_distribution.sample((n_noise,))
            except TypeError:
                return self.noise_distribution.sample(n_noise)

        # Case 3: Try sampling with full target_shape
        try:
            samples = self.noise_distribution.sample(target_shape)
            if samples.shape == target_shape:
                return samples
            if samples.ndim == 1 + 2 * len(feature_shape) and samples.shape[1:1 + len(feature_shape)] == feature_shape:
                try:
                    s = self.noise_distribution.sample((n_noise,))
                    if s.shape == target_shape:
                        return s
                except Exception:
                    pass
        except (TypeError, ValueError):
            pass

        # Case 4: Try sampling with (n_noise,)
        try:
            samples = self.noise_distribution.sample((n_noise,))
            if samples.shape == target_shape:
                return samples
        except (TypeError, ValueError):
            pass

        # Case 5: Try sampling with integer n_noise
        try:
            samples = self.noise_distribution.sample(n_noise)
            if samples.shape == target_shape:
                return samples
        except (TypeError, ValueError):
            pass

        return self.noise_distribution.sample(target_shape)

    def forward(
        self,
        x: torch.Tensor,
        *args: Any,
        noise_samples: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        model_kwargs: Optional[dict] = None,
        generator: Optional[torch.Generator] = None,
        return_diagnostics: bool = False,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        r"""Compute the Noise-Contrastive Estimation loss for a batch of data.

        Args:
            x: Input data tensor from the target distribution of shape ``(batch_size, ...)``.
            *args: Additional positional arguments.
            noise_samples: Optional pre-sampled noise tensor. If None, samples
                are automatically drawn from ``noise_distribution``.
            y: Optional conditioning tensor; shorthand for ``model_kwargs={'y': y}``.
            model_kwargs: Conditioning keyword arguments forwarded to the energy model.
            generator: Optional torch.Generator for reproducible sampling.
            return_diagnostics: If True, returns a tuple ``(loss, diagnostics_dict)``.
            **kwargs: Deprecated bare model kwargs forwarded to the model.

        Returns:
            Scalar loss tensor (or tuple of loss and diagnostics dictionary).
        """
        if y is not None:
            model_kwargs = {**(model_kwargs or {}), "y": y}
        model_kwargs = self._resolve_model_kwargs(model_kwargs, kwargs)
        self._check_condition(x, model_kwargs)
        model_kwargs = self._apply_cfg_dropout(model_kwargs, generator=generator)

        n_data = x.shape[0]
        if noise_samples is None:
            n_noise = max(1, int(round(n_data * self.noise_ratio)))
            noise_samples = self._sample_noise((n_noise, *x.shape[1:]))
        else:
            n_noise = noise_samples.shape[0]

        noise_samples = noise_samples.to(device=x.device, dtype=x.dtype)

        # Evaluate energy model: unnormalized log density is -E(x)
        e_data = self.model(x, **(model_kwargs or {}))
        e_noise = self.model(noise_samples, **(model_kwargs or {}))

        if e_data.ndim > 1:
            e_data = e_data.flatten(1).sum(dim=1)
        elif e_data.ndim == 0:
            e_data = e_data.unsqueeze(0)
        else:
            e_data = e_data.view(-1)

        if e_noise.ndim > 1:
            e_noise = e_noise.flatten(1).sum(dim=1)
        elif e_noise.ndim == 0:
            e_noise = e_noise.unsqueeze(0)
        else:
            e_noise = e_noise.view(-1)

        # Evaluate noise distribution log-probabilities
        log_pn_data = self._eval_log_prob(self.noise_distribution, x)
        log_pn_noise = self._eval_log_prob(self.noise_distribution, noise_samples)

        log_nu = math.log(self.noise_ratio)

        # Log-odds logits: h(u) = -E(u) + c - log(nu) - log p_n(u)
        h_data = -e_data + self.c - log_nu - log_pn_data
        h_noise = -e_noise + self.c - log_nu - log_pn_noise

        # Numerically stable NCE loss
        loss_data = torch.nn.functional.softplus(-h_data).mean()
        loss_noise = self.noise_ratio * torch.nn.functional.softplus(h_noise).mean()
        total_loss = loss_data + loss_noise

        if return_diagnostics:
            diagnostics = {
                "loss_data": loss_data.detach(),
                "loss_noise": loss_noise.detach(),
                "c": self.c.detach(),
                "h_data_mean": h_data.mean().detach(),
                "h_noise_mean": h_noise.mean().detach(),
            }
            return total_loss, diagnostics

        return total_loss


__all__ = [
    "NoiseContrastiveEstimation",
    "GaussianNoise",
    "GaussianMixtureNoise",
]

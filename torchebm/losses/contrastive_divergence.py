r"""Contrastive Divergence Loss Module."""

from typing import Union, Callable, Tuple, Any, Optional, Dict

import torch
from torch import nn
from abc import abstractmethod

from torchebm.core import BaseContrastiveDivergence, BaseLoss, BaseSampler, BaseModel
from torchebm._deprecation import declare_deprecation, deprecated
from torchebm.losses.functional import contrastive_divergence_loss

# First warned in v0.7.5; window restarted when the deprecation ledger was
# adopted.
_CD_OPTION_KWARGS_DEPRECATION = declare_deprecation(
    module=__name__,
    name="ContrastiveDivergence loss-option call kwargs",
    since="0.8.5",
    deprecated_on="2026-08-31",
    replacement="the energy_reg_weight/add_noise_to_real/noise_scale constructor parameters",
    message=(
        "Passing energy_reg_weight/add_noise_to_real/noise_scale to "
        "ContrastiveDivergence.__call__ is deprecated; set them on the "
        "constructor instead."
    ),
    removal="the _CD_OPTION_KEYS forwarding shim in ContrastiveDivergence.forward",
)


class ContrastiveDivergenceLoss(BaseLoss):
    r"""Contrastive divergence criterion on the energies of data and model samples.

    \[
    L = \mathbb{E}[E(x^+)] - \widetilde{\mathbb{E}}[E(x^-)]
        + \lambda \left(\mathbb{E}[E(x^+)^2] + \mathbb{E}[E(x^-)^2]\right)
    \]

    The negatives \(x^-\) come from a sampler in the training loop, as the
    generator's output does in a GAN step; `torchebm.utils.ReplayBuffer`
    keeps them persistent. See `contrastive_divergence_loss`.

    Args:
        energy_reg_weight: Weight \(\lambda\) of the squared-energy penalty.
        trim_fraction: Fraction of the largest negative energies dropped from
            their mean, in [0, 1) (the robust statistic of Energy Matching).
        clamp: Floor \(-c\) applied to the loss; ``None`` (default) disables it.
        dtype: Data type for computations.
        device: Device for computations.

    Example:
        ```python
        criterion = ContrastiveDivergenceLoss()
        x_neg = sampler.sample(x=buffer.sample(len(x)), n_steps=10)
        buffer.add(x_neg)
        loss = criterion(energy(x), energy(x_neg))
        ```
    """

    def __init__(
        self,
        energy_reg_weight: float = 0.0,
        trim_fraction: float = 0.0,
        clamp: Optional[float] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if energy_reg_weight < 0:
            raise ValueError(
                f"energy_reg_weight must be >= 0, got {energy_reg_weight}"
            )
        if not 0.0 <= trim_fraction < 1.0:
            raise ValueError(f"trim_fraction must be in [0, 1), got {trim_fraction}")
        if clamp is not None and clamp < 0:
            raise ValueError(f"clamp must be >= 0 or None, got {clamp}")
        self.energy_reg_weight = float(energy_reg_weight)
        self.trim_fraction = float(trim_fraction)
        self.clamp = clamp

    def forward(self, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        r"""Compute the loss from the energies of data and model samples.

        Args:
            positive: Energies of data samples, shape (batch_size,).
            negative: Energies of model samples, shape (n,).

        Returns:
            Scalar loss.
        """
        loss = contrastive_divergence_loss(
            positive,
            negative,
            energy_reg_weight=self.energy_reg_weight,
            trim_fraction=self.trim_fraction,
        )
        if self.clamp is not None:
            loss = loss.clamp(min=-self.clamp)
        return loss


@deprecated(
    since="0.8.10",
    deprecated_on="2026-09-29",
    grace=4,
    grace_months=4,
    replacement="ContrastiveDivergenceLoss on the energies of data and sampled negatives",
    message=(
        "ContrastiveDivergence(model, sampler) is deprecated: draw negatives "
        "with a sampler (and ReplayBuffer) in the training loop and use "
        "ContrastiveDivergenceLoss()(energy(x), energy(x_neg)) instead."
    ),
    removal=(
        "ContrastiveDivergence, PersistentContrastiveDivergence, "
        "BaseContrastiveDivergence with its replay buffer, and their tests"
    ),
)
class ContrastiveDivergence(BaseContrastiveDivergence):
    r"""
    Standard Contrastive Divergence (CD-k) loss.

    CD approximates the log-likelihood gradient by running an MCMC sampler
    for `k_steps` to generate negative samples.

    Args:
        model: The energy-based model to train.
        sampler: The MCMC sampler for generating negative samples.
        k_steps: The number of MCMC steps (k in CD-k).
        persistent: If True, uses Persistent CD with a replay buffer.
        buffer_size: Size of the replay buffer for PCD.
        init_steps: Number of MCMC steps to warm up the buffer.
        new_sample_ratio: Fraction of new random samples for PCD chains.
        energy_reg_weight: Weight for energy regularization term.
        add_noise_to_real: Whether to perturb positive samples before
            evaluating their energy.
        noise_scale: Standard deviation of the optional positive-sample
            perturbation.
        dtype: Data type for computations.
        device: Device for computations.

    Example:
        ```python
        from torchebm.losses import ContrastiveDivergence
        from torchebm.samplers import LangevinDynamics
        from torchebm.core import DoubleWellModel

        energy = DoubleWellModel()
        sampler = LangevinDynamics(energy, step_size=0.01)
        cd_loss = ContrastiveDivergence(model=energy, sampler=sampler, k_steps=10)
        x = torch.randn(32, 2)
        loss, neg_samples = cd_loss(x)
        ```
    """

    def __init__(
        self,
        model: BaseModel,
        sampler: BaseSampler,
        k_steps: int = 10,
        persistent: bool = False,
        buffer_size: int = 10000,
        init_steps: int = 100,
        new_sample_ratio: float = 0.05,
        energy_reg_weight: float = 0.001,
        add_noise_to_real: bool = False,
        noise_scale: float = 1e-4,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = torch.device("cpu"),
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            sampler=sampler,
            k_steps=k_steps,
            persistent=persistent,
            buffer_size=buffer_size,
            new_sample_ratio=new_sample_ratio,
            init_steps=init_steps,
            dtype=dtype,
            device=device,
            *args,
            **kwargs,
        )
        self.energy_reg_weight = energy_reg_weight
        self.add_noise_to_real = add_noise_to_real
        self.noise_scale = noise_scale

    _CD_OPTION_KEYS = ("energy_reg_weight", "add_noise_to_real", "noise_scale")

    def forward(
        self,
        x: torch.Tensor,
        *args,
        y: Optional[torch.Tensor] = None,
        model_kwargs: Optional[dict] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the Contrastive Divergence loss and generates negative samples.

        Args:
            x (torch.Tensor): A batch of real data samples (positive samples).
            *args: Additional positional arguments.
            y (Optional[torch.Tensor]): Optional conditioning tensor forwarded
                to the model on both positive and negative paths; shorthand for
                ``model_kwargs={'y': y}``.
            model_kwargs (Optional[dict]): Conditioning arguments (e.g. class
                labels) forwarded to the model on both the positive energy calls
                and the negative-sampling MCMC chains, so the negatives come from
                the same conditional energy as the positives.
            generator: RNG for the chain start points, the MCMC chains and the
                optional real-data noise; the global RNG when ``None``. In
                distributed runs, a per-rank generator keeps the negative chains
                of each rank independent.
            **kwargs: Deprecated. The loss options ``energy_reg_weight``,
                ``add_noise_to_real`` and ``noise_scale`` are now constructor
                parameters; passing them here still works for one release but
                emits a ``DeprecationWarning``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - The scalar CD loss value.
                - The generated negative samples.
        """
        model_kwargs = self._prepare_model_kwargs(
            self._merge_condition(model_kwargs, y)
        )
        self._check_condition(x, model_kwargs)
        model_kwargs = self._apply_cfg_dropout(model_kwargs, generator)
        if any(key in kwargs for key in self._CD_OPTION_KEYS):
            _CD_OPTION_KWARGS_DEPRECATION.warn()

        batch_size = x.shape[0]
        data_shape = x.shape[1:]

        # Get starting points for chains (either from buffer or data)
        start_points = self.get_start_points(x, generator=generator)

        pred_samples = self.sampler.sample(
            x=start_points,
            n_steps=self.k_steps,
            model_kwargs=model_kwargs,
            generator=generator,
        )

        # Update persistent buffer if using PCD
        if self.persistent:
            with torch.no_grad():
                self.update_buffer(pred_samples)

        kwargs.setdefault("energy_reg_weight", self.energy_reg_weight)
        kwargs.setdefault("add_noise_to_real", self.add_noise_to_real)
        kwargs.setdefault("noise_scale", self.noise_scale)

        # Compute contrastive divergence loss
        loss = self.compute_loss(
            x,
            pred_samples,
            *args,
            model_kwargs=model_kwargs,
            generator=generator,
            **kwargs,
        )

        return loss, pred_samples

    def compute_loss(
        self,
        x: torch.Tensor,
        pred_x: torch.Tensor,
        *args,
        model_kwargs: Optional[dict] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes the Contrastive Divergence loss from positive and negative samples.

        The loss is the difference between the mean energy of positive samples
        and the mean energy of negative samples.

        Args:
            x (torch.Tensor): Real data samples (positive samples).
            pred_x (torch.Tensor): Generated negative samples.
            *args: Additional positional arguments.
            generator: RNG for the optional real-data noise; the global RNG
                when `None`.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The scalar loss value.
        """
        # Ensure inputs are on the correct device and dtype
        x = x.to(self.device, self.dtype)
        pred_x = pred_x.to(self.device, self.dtype)

        mk = model_kwargs or {}

        # Compute energy of real and generated samples
        with torch.set_grad_enabled(True):
            # Add small noise to real data for stability (optional)
            if kwargs.get("add_noise_to_real", self.add_noise_to_real):
                noise_scale = kwargs.get("noise_scale", self.noise_scale)
                x_noisy = x + noise_scale * torch.randn_like(x, generator=generator)
                x_energy = self.model(x_noisy, **mk)
            else:
                x_energy = self.model(x, **mk)

            pred_x_energy = self.model(pred_x, **mk)

        loss = contrastive_divergence_loss(
            x_energy,
            pred_x_energy,
            energy_reg_weight=kwargs.get("energy_reg_weight", self.energy_reg_weight),
        )

        # Sync-free NaN/Inf guard: substitute a small constant on-device when
        # the loss is non-finite. Avoids the per-step CPU sync that
        # ``if torch.isnan(loss) or torch.isinf(loss)`` would force.
        fallback = torch.tensor(0.1, device=loss.device, dtype=loss.dtype)
        loss = torch.where(torch.isfinite(loss), loss, fallback)

        return loss


class PersistentContrastiveDivergence(ContrastiveDivergence):
    """Contrastive divergence with replay-buffer persistence enabled."""

    def __init__(self, model, sampler, *args, **kwargs):
        kwargs["persistent"] = True
        super().__init__(model, sampler, *args, **kwargs)

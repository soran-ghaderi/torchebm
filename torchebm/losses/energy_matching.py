r"""Energy Matching (EM) loss.

Implements the Energy Matching objective (Balcerak et al., 2025,
arXiv:2504.10612), which unifies flow matching and energy-based models
through a single time-independent scalar potential \(V_\theta(x)\):

- **Flow term** (\(L_{OT}\)): far from the data, \(-\nabla_x V\) is regressed
  onto minibatch optimal-transport displacements,

    \[
    L_{OT} = \mathbb{E}\left[ w(t)\, \| -\nabla_x V_\theta(x_t) - (x_1 - x_0) \|^2 \right]
    \]

  with \((x_0, x_1)\) OT-coupled, \(x_t\) an interpolant sample with smoothing
  noise \(\sigma\), and \(w(t)\) a time gate that fades the flow supervision
  out near the data.

- **Contrastive term** (\(L_{CD}\)): near the data, the Boltzmann density
  \(\rho(x) \propto e^{-V_\theta(x)/\epsilon_{max}}\) is sharpened with
  contrastive divergence,

    \[
    L_{CD} = \lambda_{CD} \left( \mathbb{E}_{x \sim p_{data}}[V_\theta(x)]
    - \widetilde{\mathbb{E}}_{\tilde x \sim \rho}[V_\theta(\tilde x)] \right)
    \]

  where negatives \(\tilde x\) come from Langevin chains under the piecewise
  temperature \(\epsilon(t)\) (half initialized from noise sweeping
  \(t: 0 \to 1\), half from data held at \(\epsilon_{max}\)),
  \(\widetilde{\mathbb{E}}\) is a one-sided trimmed mean, and the term is
  floored for stability.

Training is two-phase: a warm-up with `lambda_cd=0.0` (pure OT flow
matching, no Langevin), then joint optimization after setting
`loss_fn.lambda_cd` to a positive value.

Key differences from `EquilibriumMatchingLoss` (EqM):

- EM learns a scalar potential \(V(x)\) with signature `model(x) -> (B,)`
  (a `BaseModel`); EqM trains a vector field `model(x, t)`.
- EM adds OT coupling and a contrastive Langevin term; EqM is purely
  regression-based.
- EM samples with `LangevinDynamics` + `TemperatureScheduler` (one SDE
  pass); EqM samples with `FlowSampler`.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Union

import torch

from torchebm.core import (
    BaseCoupling,
    BaseInterpolant,
    BaseLoss,
    BaseModel,
    BaseScheduler,
    ConstantScheduler,
    TemperatureScheduler,
)
from torchebm.couplings import resolve_coupling
from torchebm.interpolants import resolve_interpolant
from torchebm.losses.loss_utils import (
    compute_flow_weight,
    mean_flat,
    trimmed_mean,
)
from torchebm.samplers import LangevinDynamics


class EnergyMatchingLoss(BaseLoss):
    r"""Energy Matching (EM) training loss.

    Trains a time-independent scalar potential \(V_\theta(x)\) so that
    \(-\nabla_x V\) transports noise to data (OT flow matching) and
    \(e^{-V/\epsilon_{max}}\) matches the data distribution near the manifold
    (contrastive divergence). Defaults follow the paper's 2D configuration.

    Args:
        model: Scalar potential \(V(x)\) returning shape (batch_size,).
        sampler: Langevin sampler for negatives. Auto-built with
            `step_size=langevin_dt` when None. The loss owns the sampler's
            `noise_scale` schedule during training.
        coupling: Minibatch coupling: a name ('ot'/'exact_ot', 'sinkhorn',
            'independent') or a `BaseCoupling` instance.
        interpolant: Interpolant name ('linear', 'cosine', 'vp') or instance.
        sigma: Smoothing noise added to interpolated points. Float or
            `BaseScheduler`.
        flow_weight_cutoff: Time-gate onset for the flow term; >= 1.0
            disables gating.
        lambda_cd: Contrastive term weight; 0.0 selects the warm-up phase
            (Langevin chains are skipped entirely). Float or `BaseScheduler`.
        epsilon_max: Plateau temperature of \(\epsilon(t)\).
        tau_star: Transport/diffusion switch time \(\tau^*\) in [0, 1).
        n_langevin_steps: Langevin steps per negative chain.
        langevin_dt: Langevin step size (used for the auto-built sampler).
        noise_fraction: Fraction of negatives initialized from \(N(0, I)\)
            (these sweep \(t: 0 \to 1\)); the rest start at data samples and
            stay at \(\epsilon_{max}\).
        cd_trim_fraction: Fraction of highest-energy negatives dropped by the
            one-sided trimmed mean.
        cd_clamp: Stability floor: the contrastive term is clamped to
            `>= -cd_clamp`. None disables.
        dtype: Data type for computations.
        device: Device for computations.

    Example:
        ```python
        from torchebm.core import TemperatureScheduler
        from torchebm.losses import EnergyMatchingLoss
        from torchebm.samplers import LangevinDynamics

        loss_fn = EnergyMatchingLoss(model=V, lambda_cd=0.0)  # phase 1
        for step in range(2000):
            loss = loss_fn(data_batch())
            opt.zero_grad(); loss.backward(); opt.step()

        loss_fn.lambda_cd = 2.0                                # phase 2
        for step in range(1000):
            loss = loss_fn(data_batch())
            opt.zero_grad(); loss.backward(); opt.step()

        # Generation: one SDE sweep t: 0 -> 1 (dt = 0.01 -> 200 steps)
        temp = TemperatureScheduler(0.15, 0.8, n_steps=200, t_end=1.0)
        gen = LangevinDynamics(model=V, step_size=0.01, noise_scale=temp)
        samples = gen.sample(x=torch.randn(1000, 2), n_steps=200)
        ```

    Sampling with `t_end > 1.0` equilibrates at \(\epsilon_{max}\) and draws
    from the tempered density \(\propto e^{-V/\epsilon_{max}}\) itself; the
    paper's image models do this (T = 3.25 with a small
    \(\epsilon_{max} = 0.01\)). At toy scale prefer the single sweep to 1.0,
    since finitely-trained potentials confine more weakly than the data.

    For diverse conditional generation, wrap the trained potential in
    `torchebm.models.InteractionModel` at sampling time (the paper's
    pairwise repulsion \(W\)).
    """

    def __init__(
        self,
        model: BaseModel,
        sampler: Optional[LangevinDynamics] = None,
        coupling: Union[str, BaseCoupling] = "ot",
        interpolant: Union[str, BaseInterpolant] = "linear",
        sigma: Union[float, BaseScheduler] = 0.1,
        flow_weight_cutoff: float = 0.8,
        lambda_cd: Union[float, BaseScheduler] = 2.0,
        epsilon_max: float = 0.15,
        tau_star: float = 0.8,
        n_langevin_steps: int = 200,
        langevin_dt: float = 0.01,
        noise_fraction: float = 0.5,
        cd_trim_fraction: float = 0.1,
        cd_clamp: Optional[float] = 0.02,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(dtype=dtype, device=device, *args, **kwargs)

        if not 0.0 <= noise_fraction <= 1.0:
            raise ValueError(f"noise_fraction must be in [0, 1], got {noise_fraction}")
        if not 0.0 <= cd_trim_fraction < 1.0:
            raise ValueError(
                f"cd_trim_fraction must be in [0, 1), got {cd_trim_fraction}"
            )
        if cd_clamp is not None and cd_clamp < 0:
            raise ValueError(f"cd_clamp must be >= 0 or None, got {cd_clamp}")
        if langevin_dt <= 0:
            raise ValueError(f"langevin_dt must be positive, got {langevin_dt}")

        self.model = model
        self.sampler = (
            sampler
            if sampler is not None
            else LangevinDynamics(
                model=model,
                step_size=langevin_dt,
                noise_scale=1.0,
                dtype=dtype,
                device=device,
            )
        )
        self.coupling = resolve_coupling(
            coupling, default="ot", owner="EnergyMatchingLoss"
        )
        self.interpolant = resolve_interpolant(
            interpolant, default="linear", owner="EnergyMatchingLoss"
        )
        self._register_param("sigma", sigma)
        self._register_param("lambda_cd", lambda_cd)
        self.flow_weight_cutoff = flow_weight_cutoff
        self.epsilon_max = epsilon_max
        self.tau_star = tau_star
        self.n_langevin_steps = n_langevin_steps
        self.langevin_dt = langevin_dt
        self.noise_fraction = noise_fraction
        self.cd_trim_fraction = cd_trim_fraction
        self.cd_clamp = cd_clamp

        # Temperature profiles for the two negative chains. `sample()` resets
        # registered schedulers on entry, so these are reused across steps.
        self._noise_sweep = TemperatureScheduler(
            epsilon_max=epsilon_max, tau_star=tau_star, n_steps=n_langevin_steps
        )
        self._noise_const = ConstantScheduler(math.sqrt(epsilon_max))

    @property
    def sigma(self) -> float:
        return self.get_scheduled_value("sigma")

    @sigma.setter
    def sigma(self, value: Union[float, BaseScheduler]) -> None:
        self._register_param("sigma", value)

    @property
    def lambda_cd(self) -> float:
        return self.get_scheduled_value("lambda_cd")

    @lambda_cd.setter
    def lambda_cd(self, value: Union[float, BaseScheduler]) -> None:
        self._register_param("lambda_cd", value)

    def forward(
        self,
        x: torch.Tensor,
        *args,
        x0: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""Compute the Energy Matching loss (nn.Module interface).

        Args:
            x: Data samples of shape (batch_size, ...).
            *args: Additional positional arguments.
            x0: Optional source samples of shape (batch_size, ...). Defaults
                to standard Gaussian noise; pass a batch from any source
                distribution for arbitrary source-to-target transport.
            **kwargs: Additional model arguments.

        Returns:
            Scalar loss value.
        """
        if (x.device != self.device) or (x.dtype != self.dtype):
            x = x.to(device=self.device, dtype=self.dtype)

        return self.compute_loss(x, *args, x0=x0, **kwargs)

    def compute_loss(
        self,
        x: torch.Tensor,
        *args,
        x0: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""Compute the Energy Matching loss.

        Args:
            x: Data samples of shape (batch_size, ...).
            *args: Additional positional arguments.
            x0: Optional source samples (see `forward`).
            **kwargs: Additional model arguments passed to the potential.

        Returns:
            Scalar loss value.
        """
        terms = self.training_losses(x, model_kwargs=kwargs, x0=x0)
        return terms["loss"]

    def _sample_negatives(
        self, x1: torch.Tensor, x0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""Generate negatives via temperature-scheduled Langevin chains.

        Source-initialized chains sweep the full transport-to-Boltzmann
        profile \(\epsilon(t), t: 0 \to 1\); data-initialized chains are held
        at \(\epsilon_{max}\). Chains run under `torch.no_grad` (stop-grad).

        Args:
            x1: Data batch of shape (batch_size, ...).
            x0: Optional source batch; source-initialized chains start from
                it (standard Gaussian noise when None).

        Returns:
            Detached negative samples of shape (batch_size, ...).
        """
        batch = x1.shape[0]
        n_noise = int(round(batch * self.noise_fraction))
        negatives = []

        if n_noise > 0:
            if x0 is None:
                init = torch.randn_like(x1[:n_noise])
            else:
                init = x0[torch.randperm(x0.shape[0], device=x0.device)[:n_noise]]
            self.sampler.register_scheduler("noise_scale", self._noise_sweep)
            negatives.append(
                self.sampler.sample(
                    x=init.detach(),
                    n_steps=self.n_langevin_steps,
                )
            )
        if batch - n_noise > 0:
            idx = torch.randperm(batch, device=x1.device)[: batch - n_noise]
            self.sampler.register_scheduler("noise_scale", self._noise_const)
            negatives.append(
                self.sampler.sample(
                    x=x1[idx].detach(),
                    n_steps=self.n_langevin_steps,
                )
            )
        return torch.cat(negatives, dim=0).detach()

    def training_losses(
        self,
        x1: torch.Tensor,
        model_kwargs: Optional[Dict[str, Any]] = None,
        x0: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        r"""Compute training losses with detailed outputs.

        Args:
            x1: Data samples of shape (batch_size, ...).
            model_kwargs: Additional arguments passed to the potential.
            x0: Optional source samples of shape (batch_size, ...); standard
                Gaussian noise when None. Enables arbitrary source-to-target
                transport (the paper's 2D experiment maps 8 Gaussians to
                two moons).

        Returns:
            Dictionary with 'loss' (scalar), 'flow_loss', 'cd_loss', and,
            when the contrastive branch runs, 'cd_value' and 'negatives'.
        """
        if model_kwargs is None:
            model_kwargs = {}

        x1 = x1.to(device=self.device, dtype=self.dtype)
        batch = x1.shape[0]

        # Flow term: regress -grad V onto the OT-coupled displacement.
        if x0 is None:
            x0 = torch.randn_like(x1)
        else:
            x0 = x0.to(device=self.device, dtype=self.dtype)
            if x0.shape != x1.shape:
                raise ValueError(
                    f"x0 shape {tuple(x0.shape)} must match x1 shape {tuple(x1.shape)}"
                )
        coupled = self.coupling(x0, x1)
        x0, x1c = coupled
        t = torch.rand(batch, device=self.device, dtype=self.dtype)
        xt, ut = self.interpolant.interpolate(x0, x1c, t)

        sigma = self.sigma
        if sigma > 0:
            xt = xt + sigma * torch.randn_like(xt)
        xt = xt.detach().requires_grad_(True)

        with self.autocast_context():
            energy = self.model(xt, **model_kwargs)
        grad = torch.autograd.grad(
            energy.sum(), xt, create_graph=self.model.training
        )[0]

        w = compute_flow_weight(t, cutoff=self.flow_weight_cutoff)
        per_pair = w * mean_flat((-grad - ut).square())
        if coupled.weights is not None:
            # Weighted couplings (unbalanced OT) carry per-pair importance
            # weights; uniform weights reduce this exactly to the plain mean.
            flow_loss = (coupled.weights * per_pair).sum() / coupled.weights.sum().clamp_min(
                1e-12
            )
        else:
            flow_loss = per_pair.mean()

        terms: Dict[str, torch.Tensor] = {"flow_loss": flow_loss}

        # Contrastive term: sharpen the Boltzmann density near the data.
        lambda_cd = self.lambda_cd
        if lambda_cd > 0:
            negatives = self._sample_negatives(x1, x0=x0)
            with self.autocast_context():
                pos_energy = self.model(x1, **model_kwargs)
                neg_energy = self.model(negatives, **model_kwargs)
            cd_value = pos_energy.mean() - trimmed_mean(
                neg_energy, self.cd_trim_fraction
            )
            cd_loss = lambda_cd * cd_value
            if self.cd_clamp is not None:
                cd_loss = torch.clamp(cd_loss, min=-self.cd_clamp)
            terms["cd_value"] = cd_value
            terms["negatives"] = negatives
        else:
            cd_loss = flow_loss.new_zeros(())

        terms["cd_loss"] = cd_loss
        terms["loss"] = flow_loss + cd_loss
        return terms

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"coupling={type(self.coupling).__name__}, "
            f"interpolant={type(self.interpolant).__name__}, "
            f"epsilon_max={self.epsilon_max}, "
            f"tau_star={self.tau_star})"
        )


__all__ = ["EnergyMatchingLoss"]

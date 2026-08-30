from __future__ import annotations

from typing import Callable, Literal, Optional, Union

import torch
import torch.nn as nn

from torchebm.core import BaseModel, BaseScheduler, Schedulable
from torchebm.core.base_module import substitute_condition


class ClassifierFreeGuidance(BaseModel):
    r"""Classifier-free guidance in one batched forward.

    Wraps a conditional model and returns

    \[
    \text{out} = u + w \, (c - u)
    \]

    where \(c\) and \(u\) are the wrapped model's outputs under the true and
    null conditioning, evaluated in a single 2x-batch forward (no python
    loop). Works with field models, called as ``model(x, t, y=y)`` by
    `FlowSampler`, and with scalar energies, called as ``model(x, y=y)`` and
    differentiated through the inherited `BaseModel.gradient` by the MCMC and
    gradient-descent samplers.

    \(w = 0\) reproduces the null-conditioned (unconditional) model exactly
    via a single un-doubled forward; \(w = 1\) matches the conditional model.
    ``y=None`` is a plain passthrough. Batch-aligned tensors in ``kwargs``
    are doubled alongside ``x``; a model returning a tuple is unwrapped to
    its first element.

    Args:
        model: Conditional model accepting ``y=``.
        guidance_scale: Guidance weight \(w\).
        null_condition: Null condition for the unconditional half: an int
            (the ``num_classes`` label convention), a tensor broadcast over
            the non-batch dims (e.g. a zero or learned null embedding), or a
            callable ``(y, mask) -> y``.
        guide_channels: If set, apply guidance only to the first
            ``guide_channels`` entries of dim 1 and keep the unconditional
            output for the rest (models whose extra channels, e.g. a learned
            sigma, must not be guided). ``None`` (default) guides everything.
    """

    def __init__(
        self,
        model: nn.Module,
        guidance_scale: float,
        null_condition: Union[int, float, torch.Tensor, Callable],
        guide_channels: Optional[int] = None,
    ):
        super().__init__()
        if null_condition is None:
            raise ValueError("null_condition is required")
        self.model = model
        self.guidance_scale = float(guidance_scale)
        self.null_condition = null_condition
        self.guide_channels = guide_channels

    def _null_y(self, y: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(y.shape[0], dtype=torch.bool, device=y.device)
        return substitute_condition(y, mask, self.null_condition)

    def _call(self, x, t, y, kwargs):
        if t is None:
            out = self.model(x, **kwargs) if y is None else self.model(x, y=y, **kwargs)
        else:
            out = (
                self.model(x, t, **kwargs)
                if y is None
                else self.model(x, t, y=y, **kwargs)
            )
        return out[0] if isinstance(out, tuple) else out

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if y is None:
            return self._call(x, t, None, kwargs)
        w = self.guidance_scale
        if w == 0.0:
            return self._call(x, t, self._null_y(y), kwargs)

        batch = x.shape[0]

        def _double(v):
            if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == batch:
                return torch.cat([v, v])
            return v

        out = self._call(
            torch.cat([x, x]),
            _double(t),
            torch.cat([y, self._null_y(y)]),
            {k: _double(v) for k, v in kwargs.items()},
        )
        cond, uncond = out.chunk(2)
        if self.guide_channels is not None and out.ndim >= 2:
            c = min(self.guide_channels, cond.shape[1])
            guided = uncond[:, :c] + w * (cond[:, :c] - uncond[:, :c])
            if c < cond.shape[1]:
                guided = torch.cat([guided, uncond[:, c:]], dim=1)
            return guided
        return uncond + w * (cond - uncond)


class InteractionModel(Schedulable, BaseModel):
    r"""Potential with pairwise repulsive interaction for diverse sampling.

    Wraps a scalar potential \(V(x)\) with the Energy Matching interaction
    energy (Balcerak et al., 2025, arXiv:2504.10612):

    \[
    E_i = V(x_i) - W_i, \qquad
    W_i = \frac{1}{2} \frac{s}{\sigma_W^2} \sum_{j} \|x_i - x_j\|^2
    \]

    so that \(\sum_i W_i = \tfrac{1}{2} \tfrac{s}{\sigma_W^2}
    \sum_{i \neq j} \|x_i - x_j\|^2\). Because `BaseModel.gradient()`
    differentiates the summed batch energy, sampling with this model yields
    the batch-coupled repulsive Langevin drift of the paper: samples are
    pushed apart, increasing diversity (used for inverse design and
    conditional generation).

    The strength \(s\) is a `Schedulable` parameter. Pass a
    `TemperatureScheduler(..., sqrt=False)` to reproduce the paper's
    \(\epsilon(t)\)-scaled interaction: the wrapper sits inside the sampler's
    module subtree, so `LangevinDynamics.sample()` resets and advances it in
    lockstep with the noise schedule.

    Note:
        The squared distances use the expansion
        \(\sum_j \|x_i - x_j\|^2 = B \|x_i\|^2 + \sum_j \|x_j\|^2
        - 2 x_i \cdot \sum_j x_j\), which is exact, O(batch x dim), and
        differentiable everywhere (`torch.cdist` has a NaN derivative on the
        zero diagonal).

    Note:
        The repulsive drift on each sample scales as
        \(2 s B / \sigma_W^2 \cdot (x_i - \bar{x})\) for batch size \(B\).
        Keep \(2 s B \Delta t / \sigma_W^2 \ll 1\) or the chains expand
        exponentially; for \(s = 0.15\), \(B = 64\), \(\Delta t = 0.01\)
        this means \(\sigma_W \gtrsim 4\).

    Args:
        model: Base potential \(V(x)\) returning shape (batch_size,).
        sigma_w: Interaction bandwidth \(\sigma_W\). Must be positive.
        strength: Interaction strength \(s\), typically the temperature
            \(\epsilon(t)\). Float or `BaseScheduler`.

    Example:
        ```python
        from torchebm.core import TemperatureScheduler
        from torchebm.models import InteractionModel
        from torchebm.samplers import LangevinDynamics

        temp_noise = TemperatureScheduler(0.15, 0.8, n_steps=325, t_end=3.25)
        temp_strength = TemperatureScheduler(
            0.15, 0.8, n_steps=325, t_end=3.25, sqrt=False
        )
        repulsive = InteractionModel(V, sigma_w=4.0, strength=temp_strength)
        sampler = LangevinDynamics(
            model=repulsive, step_size=0.01, noise_scale=temp_noise
        )
        samples = sampler.sample(x=torch.randn(64, 2), n_steps=325)
        ```
    """

    def __init__(
        self,
        model: BaseModel,
        sigma_w: float,
        strength: Union[float, BaseScheduler] = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if sigma_w <= 0:
            raise ValueError(f"sigma_w must be positive, got {sigma_w}")
        self.model = model
        self.sigma_w = float(sigma_w)
        self._register_param("strength", strength)

    @property
    def strength(self) -> float:
        return self.get_scheduled_value("strength")

    @strength.setter
    def strength(self, value: Union[float, BaseScheduler]) -> None:
        self._register_param("strength", value)

    def forward(self, x: torch.Tensor, **model_kwargs) -> torch.Tensor:
        r"""Compute the interacting energy \(V(x_i) - W_i\) per sample.

        Any ``model_kwargs`` (e.g. class labels) are forwarded to the wrapped
        potential, so conditional diverse sampling works: samplers reach this
        through `BaseModel.gradient(x, model_kwargs=...)`.
        """
        batch = x.shape[0]
        flat = x.reshape(batch, -1)
        sq_norms = flat.square().sum(dim=1)
        pair_sq = batch * sq_norms + sq_norms.sum() - 2.0 * flat @ flat.sum(dim=0)
        w = 0.5 * (self.strength / self.sigma_w**2) * pair_sq
        return self.model(x, **model_kwargs) - w

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={type(self.model).__name__}, sigma_w={self.sigma_w})"
        )


class EqMEnergy(BaseModel):
    r"""Scalar-energy adapter for sampling Equilibrium Matching fields.

    Equilibrium Matching trains a vector field \(f(x, t)\)
    (`EquilibriumMatchingLoss`), but the gradient-based samplers
    (`GradientDescentSampler`, `NesterovSampler`, `LangevinDynamics`,
    `HamiltonianMonteCarlo`) and `InteractionModel` consume a scalar
    `BaseModel`. This wrapper turns the field into that scalar energy
    \(g(x)\) so an EqM model samples through any of them with no user code.

    The energy is time-invariant: the field is always evaluated at \(t = 0\).
    The formulas mirror `EquilibriumMatchingLoss` exactly:

    - ``"dot"`` / ``"mean"``: \(g(x) = x \cdot f(x)\)
    - ``"l2"``: \(g(x) = -\tfrac{1}{2}\|f(x)\|^2\)
    - ``"implicit"``: `gradient(x)` returns \(f(x)\) directly (the paper's
      implicit gradient-descent field, matching `energy_type="none"` training);
      `forward` returns the \(x \cdot f\) surrogate for diagnostics / OOD scoring.

    Direction: EqM's field points **data -> noise**, so descending the energy
    (\(x \leftarrow x - \eta\,\nabla g\), or \(-f\) in the implicit case)
    transports **noise -> data** - the same direction as
    ``FlowSampler(negate_velocity=True)``.

    Note:
        `InteractionModel` differentiates a summed scalar energy, so it must
        wrap an explicit mode (``"dot"``/``"mean"``/``"l2"``). Wrapping the
        ``"implicit"`` adapter would differentiate the surrogate rather than use
        the field, giving the wrong drift; use an explicit mode for repulsive /
        diverse sampling.

    Args:
        field: Vector field called as ``field(x, t, **model_kwargs)`` returning
            shape ``(batch_size, *data_shape)`` (a plain tuple ``(out, act)`` is
            unwrapped). Typically the `model` of an `EquilibriumMatchingLoss`.
        energy_type: Scalar-energy formulation, one of ``"dot"``, ``"mean"``,
            ``"l2"``, ``"implicit"``.

    Example:
        ```python
        from torchebm.models import EqMEnergy
        from torchebm.samplers import GradientDescentSampler

        energy = EqMEnergy(field, energy_type="dot")
        sampler = GradientDescentSampler(energy, step_size=0.01)
        samples = sampler.sample(x=torch.randn(512, 2), n_steps=200)
        ```
    """

    def __init__(
        self,
        field: nn.Module,
        energy_type: Literal["dot", "mean", "l2", "implicit"] = "dot",
    ):
        super().__init__()
        valid = {"dot", "mean", "l2", "implicit"}
        if energy_type not in valid:
            raise ValueError(
                f"energy_type must be one of {sorted(valid)}, got {energy_type!r}"
            )
        self.field = field
        self.energy_type = energy_type

    @classmethod
    def from_loss(cls, loss) -> "EqMEnergy":
        r"""Build the adapter matching a loss's ``energy_type``.

        Maps the loss's implicit formulation (``energy_type="none"``) to the
        adapter's ``"implicit"`` mode and passes explicit modes through, so the
        sampled energy always matches what was trained.
        """
        energy_type = "implicit" if loss.energy_type == "none" else loss.energy_type
        return cls(loss.model, energy_type=energy_type)

    def _reduce_dims(self, ndim: int) -> tuple:
        cache = getattr(self, "_reduce_dims_cache", None)
        if cache is None or cache[0] != ndim:
            self._reduce_dims_cache = (ndim, tuple(range(1, ndim)))
        return self._reduce_dims_cache[1]

    def _field(self, x: torch.Tensor, **model_kwargs) -> torch.Tensor:
        r"""Evaluate the time-invariant field f(x, 0), unwrapping activations."""
        t0 = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        out = self.field(x, t0, **model_kwargs)
        if isinstance(out, tuple):
            out = out[0]
        return out

    def forward(self, x: torch.Tensor, **model_kwargs) -> torch.Tensor:
        r"""Scalar energy g(x) of shape (batch_size,)."""
        f = self._field(x, **model_kwargs)
        dims = self._reduce_dims(x.ndim)
        if self.energy_type == "l2":
            return -0.5 * f.square().sum(dim=dims)
        return (x * f).sum(dim=dims)

    def gradient(
        self, x: torch.Tensor, model_kwargs: Optional[dict] = None
    ) -> torch.Tensor:
        r"""Energy gradient consumed by the samplers.

        For explicit modes this is the autograd gradient of `forward` (the true
        \(\nabla g\), including the Jacobian term the ``"dot"`` loss trains). For
        ``"implicit"`` it is the field \(f(x, 0)\) itself (the trained gradient
        field), returned without the training-time \(c(t)\) scaling.
        """
        if self.energy_type == "implicit":
            return self._field(x, **(model_kwargs or {}))
        return super().gradient(x, model_kwargs=model_kwargs)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"field={type(self.field).__name__}, energy_type={self.energy_type!r})"
        )

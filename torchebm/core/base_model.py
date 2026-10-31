import math
from abc import ABC, abstractmethod
import torch
from torch import nn
from typing import Any, Optional, Tuple, Union

from torchebm._deprecation import declare_deprecation
from torchebm.core import TorchEBMModule
from torchebm.core.base_module import (
    _has_dtensor_params,
    _unexpected_init_args_message,
)
from torchebm.models.functional import energy_gradient

try:
    from torch.distributed.tensor import DTensor, Replicate, Shard
except ImportError:
    DTensor = Replicate = Shard = None

_GRADIENT_DEPRECATION = declare_deprecation(
    name="BaseModel.gradient",
    since="0.9.0",
    deprecated_on="2026-09-29",
    until="0.12.0",
    grace_months=4,
    replacement="model(x) with prediction_type='score', which returns -gradient",
    message=(
        "BaseModel.gradient is deprecated: build the model with "
        "prediction_type='score' and call it for the score -grad E, or with "
        "prediction_type=('energy', 'score') for both; a closed-form gradient "
        "belongs in AnalyticModel._score."
    ),
    removal=(
        "BaseModel.gradient and BaseModel._gradient, BaseSampler._model_gradient "
        "and EqMEnergy.gradient"
    ),
)


def _local_shard(value: torch.Tensor, mesh) -> torch.Tensor:
    r"""Local ``Shard(0)`` piece of a DTensor with data-parallel gradient scale.

    The backward of the parameter all-gather sums the per-rank contributions,
    so a local-mean loss would give the cross-rank sum. Scaling the gradient by
    ``1 / world_size`` while keeping the forward value exact makes it the
    global-batch mean, the convention of gradient-averaging data parallelism.
    """
    if value.placements != (Shard(0),):
        value = value.redistribute(mesh, [Shard(0)])
    local = value.to_local()
    scale = 1.0 / mesh.size()
    return local * scale + local.detach() * (1.0 - scale)


def _restore_gradient_placement(param: torch.Tensor) -> None:
    r"""Reduce-scatter a ``Partial`` gradient to the parameter's placements."""
    grad = param.grad
    if tuple(grad.placements) != tuple(param.placements):
        param.grad = grad.redistribute(param.device_mesh, param.placements)


class BaseModel(TorchEBMModule, ABC):
    r"""
    Abstract base class for energy-based models (EBMs).

    This class provides a unified interface for defining EBMs, which represent
    the unnormalized negative log-likelihood of a probability distribution.
    It supports both analytical models and trainable neural networks.

    Subclasses implement `forward(x)` returning the energy; `AnalyticModel` is
    the base for models with a closed-form score.

    Calling the model returns the output selected by `prediction_type`:

    - ``"energy"`` (default): the energy \(E(x)\), shape (batch_size,).
    - ``"score"``: the score \(-\nabla_x E(x)\), shape of `x`.
    - ``("energy", "score")``: both, from a single forward pass.

    Args:
        dtype: Data type for computations.
        device: Device for computations.
        prediction_type: Output of the model call; keyword-only.
        functional_model: Hook-free instance of the same architecture, used by
            `torch.func.functional_call` to train through the score when the
            parameters are FSDP2 (DTensor) shards; its own parameters are never
            used, so build it on the meta device. Held as structure only, not
            registered as a submodule. Keyword-only.
    """

    prediction_type: Union[str, Tuple[str, str]] = "energy"

    #: When True, the autograd score and the deprecated `gradient` compute in
    #: float32 regardless of the input dtype (the historical behavior). Default
    #: False respects the input dtype - byte-identical for float32, no fp64
    #: downcast, no bf16/fp16 round-trip. Set per instance for fp32-precision
    #: gradients under low precision: ``model.force_fp32_gradient = True``.
    force_fp32_gradient: bool = False

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        device=None,
        *args,
        prediction_type: Union[str, Tuple[str, str]] = "energy",
        functional_model: Optional[nn.Module] = None,
        **kwargs,
    ):
        """Initializes the BaseModel base class.

        Raises:
            TypeError: If constructor arguments remain that no class in the
                model's MRO binds; the message lists the supported parameters
                and the installed torchebm version.
            ValueError: If `prediction_type` is not ``"energy"``, ``"score"``,
                or ``("energy", "score")``, or if `functional_model` is not an
                energy model.
        """
        if args or kwargs:
            raise TypeError(
                _unexpected_init_args_message(type(self), args, kwargs, BaseModel)
            )
        if prediction_type not in ("energy", "score", ("energy", "score")):
            raise ValueError(
                "prediction_type must be 'energy', 'score', or ('energy', 'score'), "
                f"got {prediction_type!r}"
            )
        if getattr(functional_model, "prediction_type", "energy") != "energy":
            raise ValueError("functional_model must have prediction_type='energy'")
        super().__init__(dtype=dtype, device=device)
        self.prediction_type = prediction_type
        object.__setattr__(self, "_functional_model", functional_model)
        self._gradient_placement_hooks_registered = False

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

    def __call__(
        self, x: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Return the output selected by `prediction_type`.

        The score is computed by autograd through `forward`. Under grad mode it
        keeps its graph (``create_graph=True``), so a loss on the score trains
        the model; under `torch.no_grad()` it is detached. `force_fp32_gradient`
        applies as in `gradient`.

        Note:
            Training through the score is a second-order backward. Under DDP it
            is exact with the default ``static_graph=False`` when every parameter
            receives a gradient: build the energy head without an output bias,
            which the score does not depend on. Under FSDP2 it runs through
            `functional_model` (see `_sharded_energy_and_score`).

        Args:
            x: Input tensor of shape (batch_size, *input_dims).
            *args: Positional arguments forwarded to `forward`.
            **kwargs: Keyword arguments forwarded to `forward` (e.g. ``y=``).

        Returns:
            The energy, the score, or the tuple ``(energy, score)``.

        Raises:
            RuntimeError: If a score is requested under `torch.inference_mode()`,
                where autograd is unavailable, or under grad mode with FSDP2
                (DTensor) parameters and no `functional_model`.
        """
        if self.prediction_type == "energy":
            return super().__call__(x, *args, **kwargs)
        if torch.is_inference_mode_enabled():
            raise RuntimeError(
                "prediction_type='score' needs autograd; call the model under "
                "torch.no_grad() instead of torch.inference_mode()"
            )
        create_graph = torch.is_grad_enabled()
        if create_graph and _has_dtensor_params(self):
            energy, score = self._sharded_energy_and_score(x, args, kwargs)
        else:
            with torch.enable_grad():
                if not x.requires_grad:
                    x = x.detach().requires_grad_(True)
                inputs = x.float() if self.force_fp32_gradient else x
                energy = super().__call__(inputs, *args, **kwargs)
                score = -energy_gradient(energy, x, create_graph=create_graph)
            if not create_graph:
                energy, score = energy.detach(), score.detach()
        if self.prediction_type == "score":
            return score
        return energy, score

    def _sharded_energy_and_score(
        self, x: torch.Tensor, args: tuple, kwargs: dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Energy and score through differentiable DTensor collectives.

        FSDP2's hooks reshard parameter storage after the first backward, so
        they cannot run the second-order backward that trains through the
        score. The forward runs instead through `torch.func.functional_call` on
        the hook-free `functional_model` with this model's sharded parameters:
        the parameter all-gathers are differentiable DTensor redistributions
        that stay alive until both backwards are done. The local batch is a
        ``Shard(0)`` leaf, so the second-order backward is exact across ranks.

        Returns:
            Local shards of ``(energy, score)`` whose gradients follow the
            global-batch-mean convention of data-parallel training (see
            `_local_shard`). Parameter gradients are redistributed to the
            parameter placements on accumulation, the reduce-scatter FSDP's
            hooks would have run.

        Raises:
            RuntimeError: If `functional_model` was not given.
            NotImplementedError: If the device mesh is not 1-D.
        """
        if self._functional_model is None:
            raise RuntimeError(
                "Training through prediction_type='score' with FSDP-managed "
                "(DTensor) parameters needs functional_model=<hook-free instance "
                "of the same architecture, built on the meta device>."
            )
        params = dict(self.named_parameters())
        mesh = next(iter(params.values())).device_mesh
        if mesh.ndim != 1:
            raise NotImplementedError(
                f"Training through the score supports 1-D device meshes, got {mesh.ndim}-D."
            )
        batch = x.shape[0]

        def on_mesh(value, placement):
            if not torch.is_tensor(value) or isinstance(value, DTensor):
                return value
            return DTensor.from_local(value, mesh, [placement], run_check=False)

        def batch_aligned(value):
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch:
                return on_mesh(value, Shard(0))
            return on_mesh(value, Replicate())

        buffers = {name: on_mesh(b, Replicate()) for name, b in self.named_buffers()}
        leaf = on_mesh(x.detach(), Shard(0)).requires_grad_(True)
        inputs = leaf.float() if self.force_fp32_gradient else leaf
        energy = torch.func.functional_call(
            self._functional_model,
            {**params, **buffers},
            (inputs, *map(batch_aligned, args)),
            kwargs={k: batch_aligned(v) for k, v in kwargs.items()},
        )
        score = -energy_gradient(energy, leaf, create_graph=True)
        if not self._gradient_placement_hooks_registered:
            for param in params.values():
                param.register_post_accumulate_grad_hook(_restore_gradient_placement)
            self._gradient_placement_hooks_registered = True
        return _local_shard(energy, mesh), _local_shard(score, mesh)

    def gradient(
        self, x: torch.Tensor, model_kwargs: Optional[dict] = None
    ) -> torch.Tensor:
        r"""
        Computes the gradient of the energy function with respect to the input, \(\nabla_x E(x)\).

        Deprecated: build the model with ``prediction_type="score"`` and call it
        for the score \(-\nabla_x E(x)\), or with ``("energy", "score")`` for
        both; a closed-form gradient belongs in `AnalyticModel._score`.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_dims).
            model_kwargs (Optional[dict]): Conditioning arguments forwarded to
                `forward(x, **model_kwargs)` (e.g. class labels). ``None`` (the
                default) reproduces the unconditional call exactly, so analytic
                `gradient(self, x)` overrides remain valid. Tensor values should
                already be device-resident (samplers normalize them once at
                `sample()` entry); they are **not** dtype-cast here, unlike `x`.

        Returns:
            torch.Tensor: Gradient tensor of the same shape as `x`.
        """
        _GRADIENT_DEPRECATION.warn()
        return self._gradient(x, model_kwargs)

    def _gradient(
        self, x: torch.Tensor, model_kwargs: Optional[dict] = None
    ) -> torch.Tensor:
        r"""Autograd \(\nabla_x E(x)\) behind `gradient`; the legacy `sample()`
        paths call it directly, so they warn about the sampler only."""

        original_dtype = x.dtype
        device = x.device

        if self.device and device != self.device:
            x = x.to(self.device)
            device = self.device

        grad_dtype = torch.float32 if self.force_fp32_gradient else original_dtype

        with torch.enable_grad():
            x_for_grad = (
                x.detach().to(dtype=grad_dtype, device=device).requires_grad_(True)
            )

            with self.autocast_context():
                energy = self.forward(x_for_grad, **(model_kwargs or {}))

            if energy.shape != (x_for_grad.shape[0],):
                raise ValueError(
                    f"BaseModel forward() output expected shape ({x_for_grad.shape[0]},), but got {energy.shape}."
                )

            if not energy.grad_fn:
                raise RuntimeError(
                    "Cannot compute gradient: `forward` method did not use the "
                    "input `x` in a differentiable way."
                )

            from torchebm.models.functional import energy_gradient

            grad = energy_gradient(energy, x_for_grad)

        if grad is None:
            raise RuntimeError(
                "Gradient computation failed unexpectedly. Check the forward pass implementation."
            )

        gradient = grad.to(original_dtype)

        return gradient.detach()


class AnalyticModel(BaseModel):
    r"""Energy with a closed-form score.

    Subclasses implement `_energy` and `_score`; `forward` returns the output
    selected by `prediction_type` (the energy, the score, or both) without
    autograd, so the call is the plain module call rather than the
    differentiating `BaseModel.__call__`. A 1-D input is a batch of one. This
    is the pattern for potentials with analytic forces:

    ```python
    class Harmonic(AnalyticModel):
        def _energy(self, x):
            return 0.5 * x.square().sum(dim=-1)

        def _score(self, x):
            return -x

    score = Harmonic(prediction_type="score")(x)
    ```
    """

    __call__ = nn.Module.__call__

    @abstractmethod
    def _energy(self, x: torch.Tensor) -> torch.Tensor:
        r"""Energy \(E(x)\) of shape (batch_size,)."""

    @abstractmethod
    def _score(self, x: torch.Tensor) -> torch.Tensor:
        r"""Score \(-\nabla_x E(x)\) of the shape of `x`."""

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Return the energy, the score, or ``(energy, score)`` per `prediction_type`."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if self.prediction_type == "energy":
            return self._energy(x)
        if self.prediction_type == "score":
            return self._score(x)
        return self._energy(x), self._score(x)


class DoubleWellModel(AnalyticModel):
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

    def _energy(self, x: torch.Tensor) -> torch.Tensor:
        r"""Computes the double well energy: \(h \sum_{i=1}^{n} (x_i^2 - b^2)^2\)."""
        return self.barrier_height * (x.pow(2) - self.b**2).pow(2).sum(dim=-1)

    def _score(self, x: torch.Tensor) -> torch.Tensor:
        r"""\(-4 h\, x_i (x_i^2 - b^2)\)."""
        return -4.0 * self.barrier_height * x * (x.pow(2) - self.b**2)


class GaussianModel(AnalyticModel):
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

    def _delta(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"Input x expected batch_shape (batch_size, {self.mean.shape[0]}), but got {x.shape}"
            )
        return x.to(dtype=self.dtype, device=self.device) - self.mean

    def _energy(self, x: torch.Tensor) -> torch.Tensor:
        r"""Computes the Gaussian energy: \(E(x) = \frac{1}{2} (x - \mu)^{\top} \Sigma^{-1} (x - \mu)\)."""
        delta = self._delta(x)
        cov_inv = self.cov_inv.to(dtype=self.dtype, device=delta.device)

        if delta.shape[0] > 1:
            delta_expanded = delta.unsqueeze(-1)
            cov_inv_expanded = cov_inv.unsqueeze(0).expand(delta.shape[0], -1, -1)
            temp = torch.bmm(cov_inv_expanded, delta_expanded)
            energy = 0.5 * torch.bmm(delta.unsqueeze(1), temp).squeeze(-1).squeeze(-1)
        else:
            energy = 0.5 * torch.sum(delta * torch.matmul(delta, cov_inv), dim=-1)

        return energy

    def _score(self, x: torch.Tensor) -> torch.Tensor:
        r"""\(-\Sigma^{-1}(x - \mu)\) for the symmetric covariance."""
        return -torch.matmul(self._delta(x), self.cov_inv)


class HarmonicModel(AnalyticModel):
    r"""
    Energy-based model for a harmonic oscillator.

    Args:
        k (float): The spring constant.
    """
    def __init__(self, k: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def _energy(self, x: torch.Tensor) -> torch.Tensor:
        r"""Computes the harmonic oscillator energy: \(\frac{1}{2} k \sum_{i=1}^{n} x_i^{2}\)."""
        return 0.5 * self.k * x.pow(2).sum(dim=-1)

    def _score(self, x: torch.Tensor) -> torch.Tensor:
        r"""\(-k\, x\)."""
        return -self.k * x


class RosenbrockModel(AnalyticModel):
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
        if x.shape[-1] < 2:
            raise ValueError(
                f"Rosenbrock energy function requires at least 2 dimensions, got {x.shape[-1]}"
            )
        return super().forward(x)

    def _energy(self, x: torch.Tensor) -> torch.Tensor:
        r"""Computes the Rosenbrock energy: \(\sum_{i=1}^{n-1} \left[ b(x_{i+1} - x_i^2)^2 + (a - x_i)^2 \right]\)."""
        x_i = x[:, :-1]
        x_ip1 = x[:, 1:]
        term1 = (self.a - x_i).pow(2)
        term2 = self.b * (x_ip1 - x_i.pow(2)).pow(2)
        return (term1 + term2).sum(dim=-1)

    def _score(self, x: torch.Tensor) -> torch.Tensor:
        r"""\(2(a - x_i) + 4 b\, x_i (x_{i+1} - x_i^2)\) on the first \(n-1\)
        coordinates, minus \(2 b (x_i - x_{i-1}^2)\) on the last \(n-1\)."""
        x_i = x[:, :-1]
        gap = x[:, 1:] - x_i.pow(2)
        pad = nn.functional.pad
        return pad(2.0 * (self.a - x_i) + 4.0 * self.b * x_i * gap, (0, 1)) - pad(
            2.0 * self.b * gap, (1, 0)
        )


class AckleyModel(AnalyticModel):
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

    def _energy(self, x: torch.Tensor) -> torch.Tensor:
        r"""Computes the Ackley energy."""
        n = x.shape[-1]
        sum1 = torch.sum(x**2, dim=-1)
        sum2 = torch.sum(torch.cos(self.c * x), dim=-1)
        term1 = -self.a * torch.exp(-self.b * torch.sqrt(sum1 / n))
        term2 = -torch.exp(sum2 / n)
        return term1 + term2 + self.a + math.e

    def _score(self, x: torch.Tensor) -> torch.Tensor:
        r"""\(-\big[a b\, e^{-b r} x_i / (n r) + (c / n)\, e^{\bar c} \sin(c x_i)\big]\)
        with \(r = \sqrt{\sum_i x_i^2 / n}\) and \(\bar c = \sum_i \cos(c x_i) / n\)."""
        n = x.shape[-1]
        r = torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True) / n)
        radial = self.a * self.b * torch.exp(-self.b * r) / (n * r) * x
        periodic = (
            (self.c / n)
            * torch.exp(torch.sum(torch.cos(self.c * x), dim=-1, keepdim=True) / n)
            * torch.sin(self.c * x)
        )
        return -(radial + periodic)


class RastriginModel(AnalyticModel):
    r"""
    Energy-based model for the Rastrigin function.

    Args:
        a (float): The `a` parameter of the Rastrigin function.
    """
    def __init__(self, a: float = 10.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = a

    def _energy(self, x: torch.Tensor) -> torch.Tensor:
        r"""Computes the Rastrigin energy."""
        n = x.shape[-1]
        return self.a * n + torch.sum(
            x**2 - self.a * torch.cos(2 * math.pi * x), dim=-1
        )

    def _score(self, x: torch.Tensor) -> torch.Tensor:
        r"""\(-\big(2 x_i + 2 \pi a \sin(2 \pi x_i)\big)\)."""
        return -(2.0 * x + 2.0 * math.pi * self.a * torch.sin(2.0 * math.pi * x))


class HimmelblauModel(AnalyticModel):
    r"""Energy-based model for the two-dimensional Himmelblau function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 2:
            raise ValueError(
                f"Himmelblau energy function requires 2 dimensions, got {x.shape[-1]}"
            )
        return super().forward(x)

    def _energy(self, x: torch.Tensor) -> torch.Tensor:
        r"""Computes \((x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2\)."""
        x1, x2 = x.unbind(dim=-1)
        return (x1.pow(2) + x2 - 11).pow(2) + (x1 + x2.pow(2) - 7).pow(2)

    def _score(self, x: torch.Tensor) -> torch.Tensor:
        r"""\(-(4 x_1 u + 2 v,\; 2 u + 4 x_2 v)\) with \(u = x_1^2 + x_2 - 11\)
        and \(v = x_1 + x_2^2 - 7\)."""
        x1, x2 = x.unbind(dim=-1)
        u = x1.pow(2) + x2 - 11
        v = x1 + x2.pow(2) - 7
        return -torch.stack((4 * x1 * u + 2 * v, 2 * u + 4 * x2 * v), dim=-1)


class BoothModel(AnalyticModel):
    r"""Energy-based model for the two-dimensional Booth function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 2:
            raise ValueError(
                f"Booth energy function requires 2 dimensions, got {x.shape[-1]}"
            )
        return super().forward(x)

    def _energy(self, x: torch.Tensor) -> torch.Tensor:
        r"""Computes \((x_1 + 2x_2 - 7)^2 + (2x_1 + x_2 - 5)^2\)."""
        x1, x2 = x.unbind(dim=-1)
        return (x1 + 2 * x2 - 7).pow(2) + (2 * x1 + x2 - 5).pow(2)

    def _score(self, x: torch.Tensor) -> torch.Tensor:
        r"""\(-(2 u + 4 v,\; 4 u + 2 v)\) with \(u = x_1 + 2 x_2 - 7\) and
        \(v = 2 x_1 + x_2 - 5\)."""
        x1, x2 = x.unbind(dim=-1)
        u = x1 + 2 * x2 - 7
        v = 2 * x1 + x2 - 5
        return -torch.stack((2 * u + 4 * v, 4 * u + 2 * v), dim=-1)


class StyblinskiTangModel(AnalyticModel):
    r"""Energy-based model for the Styblinski-Tang function."""

    def _energy(self, x: torch.Tensor) -> torch.Tensor:
        r"""Computes \(\frac{1}{2}\sum_i(x_i^4 - 16x_i^2 + 5x_i)\)."""
        return 0.5 * (x.pow(4) - 16 * x.pow(2) + 5 * x).sum(dim=-1)

    def _score(self, x: torch.Tensor) -> torch.Tensor:
        r"""\(-(2 x_i^3 - 16 x_i + 5 / 2)\)."""
        return -(2.0 * x.pow(3) - 16.0 * x + 2.5)

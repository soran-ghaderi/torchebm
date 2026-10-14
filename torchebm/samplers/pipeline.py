r"""Sampling pipeline: a model run through a sampler's step contract."""

from typing import Any, Dict, Optional, Sequence, Union

import torch
from torch import nn

from torchebm.core import BaseSampler
from torchebm.models.functional import energy_gradient


class Pipeline:
    r"""Generate samples from `model` with `sampler`.

    Runs the `diffusers` loop over the sampler's step contract (see
    `BaseSampler`):

    ```python
    sampler.set_timesteps(num_inference_steps, device=x.device)
    for t in sampler.timesteps:
        x = sampler.step(model(sampler.scale_model_input(x, t), t), t, x)
    ```

    The model output is matched to `sampler.prediction_type` from
    `model.prediction_type`: equal types pass through; a tuple such as
    ``("energy", "score")`` is indexed; an ``"energy"`` model feeding a score
    sampler is differentiated, \(-\nabla_x E(x)\) by autograd, exactly as
    ``prediction_type="score"`` on the model does; a model without the
    attribute is assumed to output what the sampler consumes. The model is
    called as ``model(x, t, **model_kwargs)`` when
    `sampler.timestep_conditioned`, else ``model(x, **model_kwargs)``.

    Args:
        model: Trained model, or an analytic `BaseModel`.
        sampler: Sampler constructed without a model.

    Raises:
        TypeError: If `sampler` binds its model at construction (the HMC
            family) and has no step contract.
        ValueError: If `model.prediction_type` cannot feed the sampler.

    Example:
        ```python
        from torchebm.core import DoubleWellModel
        from torchebm.samplers import LangevinDynamics, Pipeline

        pipe = Pipeline(DoubleWellModel(), LangevinDynamics(step_size=0.01))
        x = pipe(batch_size=64, sample_shape=(2,), num_inference_steps=100)
        ```
    """

    def __init__(self, model: nn.Module, sampler: BaseSampler):
        if not hasattr(sampler, "set_timesteps"):
            raise TypeError(
                f"{type(sampler).__name__} binds its model at construction and "
                "advances with step(sample); Pipeline needs the "
                "set_timesteps/step contract"
            )
        consumed = sampler.prediction_type
        produced = getattr(model, "prediction_type", consumed)
        self._index: Optional[int] = None
        self._differentiate = False
        if isinstance(produced, tuple):
            if consumed not in produced:
                raise ValueError(
                    f"model outputs {produced!r}, none of which is the "
                    f"{consumed!r} that {type(sampler).__name__} consumes"
                )
            self._index = produced.index(consumed)
        elif produced == "energy" and consumed == "score":
            self._differentiate = True
        elif produced != consumed:
            raise ValueError(
                f"model outputs {produced!r} but {type(sampler).__name__} "
                f"consumes {consumed!r}; set prediction_type on the model or "
                "the sampler accordingly"
            )
        self.model = model
        self.sampler = sampler
        self._force_fp32 = bool(getattr(model, "force_fp32_gradient", False))

    @torch.no_grad()
    def __call__(
        self,
        sample: Optional[torch.Tensor] = None,
        *,
        batch_size: Optional[int] = None,
        sample_shape: Optional[Sequence[int]] = None,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        r"""Run the sampler for `num_inference_steps` updates.

        Args:
            sample: Initial state of shape ``(batch_size, *sample_shape)``.
                ``None`` draws it from \(\mathcal{N}(0, I)\).
            batch_size: Number of samples when `sample` is ``None``.
            sample_shape: Shape of one sample when `sample` is ``None``.
            num_inference_steps: Number of sampler updates.
            generator: RNG for the initial draw and the sampler's noise; the
                global RNG when ``None``.
            model_kwargs: Conditioning arguments forwarded to the model at
                every evaluation.
            return_trajectory: If True, return the state after every update,
                shape ``(batch_size, num_inference_steps, *sample_shape)``,
                instead of the final state. A sampler whose `timesteps` end
                with a correction (`FlowSampler` in SDE mode with a
                `last_step`) records it as one more entry.

        Returns:
            The final state, or the trajectory.

        Raises:
            ValueError: If `sample` is ``None`` and `batch_size` or
                `sample_shape` is missing.
        """
        sampler = self.sampler
        if sample is None:
            if batch_size is None or sample_shape is None:
                raise ValueError("pass sample, or batch_size and sample_shape")
            x = torch.randn(
                batch_size,
                *sample_shape,
                device=sampler.device,
                dtype=sampler.dtype,
                generator=generator,
            )
        else:
            x = sample.to(device=sampler.device, dtype=sampler.dtype)
        sampler.set_timesteps(num_inference_steps, device=x.device)
        timesteps = sampler.timesteps
        order = sampler.order
        model_kwargs = model_kwargs or {}
        trajectory = None
        if return_trajectory:
            n_kept = (timesteps.shape[0] + order - 1) // order
            trajectory = x.new_empty((x.shape[0], n_kept, *x.shape[1:]))
        last = timesteps.shape[0] - 1
        for i, t in enumerate(timesteps):
            model_input = sampler.scale_model_input(x, t)
            output = self._model_output(model_input, t, model_kwargs)
            x = sampler.step(output, t, x, generator=generator)
            if trajectory is not None and ((i + 1) % order == 0 or i == last):
                trajectory[:, i // order] = x
        return x if trajectory is None else trajectory

    def _model_output(
        self, x: torch.Tensor, t: torch.Tensor, model_kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        r"""The model output in the form the sampler consumes."""
        if self._differentiate:
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                energy = self._call(
                    x.float() if self._force_fp32 else x, t, model_kwargs
                )
                return -energy_gradient(energy, x).detach()
        output = self._call(x, t, model_kwargs)
        return output if self._index is None else output[self._index]

    def _call(
        self, x: torch.Tensor, t: torch.Tensor, model_kwargs: Dict[str, Any]
    ) -> Union[torch.Tensor, tuple]:
        if self.sampler.timestep_conditioned:
            return self.model(x, t.expand(x.shape[0]), **model_kwargs)
        return self.model(x, **model_kwargs)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model={type(self.model).__name__}, "
            f"sampler={type(self.sampler).__name__})"
        )


__all__ = ["Pipeline"]

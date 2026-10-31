r"""Model-induced couplings: pair \((x_0, \Phi(x_0))\) from a learned map.

Unlike cost couplings, these **generate** the target from the source under a
map \(\Phi\) (a `FlowSampler` or any callable), ignoring any incoming target
batch. This is the reflow / rectified-flow distillation shape: training on
pairs (noise, model-pushed noise) straightens the flow so it can be sampled
in few steps. The same machinery serves closed-form transport maps (e.g. a
Gaussian linear-OT map passed as a plain callable) and, in the future,
iterative-Markovian-fitting couplings (DSBM), which regenerate the pairing
from the current bridge each round.

Model couplings are instance-only: they need an injected model, so they are
not string-registrable via `get_coupling` (which constructs with defaults).
Pass instances directly; `resolve_coupling` accepts them.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

import torch

from torchebm._deprecation import deprecated
from torchebm.core import BaseModelCoupling


@deprecated(
    since="0.9.0",
    deprecated_on="2026-10-30",
    until="0.12.0",
    grace_months=4,
    replacement="x1 = sampler(model, x0, n_steps=...) in the training loop",
    message=(
        "ReflowCoupling is deprecated: generate the reflow targets in the "
        "training loop with x1 = sampler(model, x0, n_steps=50, "
        "model_kwargs=...) and train on the pairs (x0, x1)."
    ),
    removal=(
        "torchebm/couplings/model_induced.py, its export in "
        "torchebm/couplings/__init__.py and the ReflowCoupling tests in "
        "tests/couplings/test_model_couplings.py"
    ),
)
class ReflowCoupling(BaseModelCoupling):
    r"""
    Rectified-flow reflow coupling: \(x_1 = \Phi(x_0)\).

    Deprecated: the reflow targets come from a sampler call in the training
    loop, ``x1 = sampler(model, x0, n_steps=50)``, so no coupling holds the
    model.

    Pairs each source sample with its image under a learned flow, producing
    the (noise, generated-data) pairs that reflow training regresses on to
    straighten the flow (Rectified Flow, InstaFlow). `flow` is either a
    `FlowSampler` (its ODE `sample` maps source to target) or any callable
    ``x0 -> x1`` — which also covers closed-form transport maps such as the
    Gaussian linear-OT map.

    Args:
        flow: A `FlowSampler` instance or a callable mapping a source batch
            to a target batch.
        n_steps: ODE steps for the `FlowSampler` path (ignored for plain
            callables).

    Example:
        ```python
        from torchebm.samplers import FlowSampler

        x0 = torch.randn(256, 2)
        x1 = FlowSampler(integrator="heun")(model, x0, n_steps=50)
        ```
    """

    def __init__(self, flow: Union["FlowSampler", Callable], n_steps: int = 50):
        super().__init__(flow)
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        self.n_steps = n_steps

    def _generate(
        self,
        x0: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        from torchebm.samplers.flow import FlowSampler  # local: avoid cycle

        if isinstance(self.model, FlowSampler):
            return self.model.sample(
                x=x0, n_steps=self.n_steps, generator=generator, **kwargs
            )
        return self.model(x0, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_steps={self.n_steps})"


__all__ = ["ReflowCoupling"]

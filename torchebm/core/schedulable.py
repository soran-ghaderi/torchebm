r"""Schedulable mixin: uniform parameter scheduling for `nn.Module` subtrees.

Adds `register_scheduler` / `get_scheduled_value` / `step_schedulers` /
`reset_schedulers` and the `_register_param(name, value, validate=...)` helper
that accepts either a float (wrapped in `ConstantScheduler`) or a `BaseScheduler`.

`step_schedulers` and `reset_schedulers` traverse the host's `nn.Module`
subtree via `nn.Module.apply`, so a trainer can advance every scheduler in a
loss + sampler + integrator pipeline by calling once on the root.
"""

from typing import Dict, Union

from torchebm.core.base_scheduler import BaseScheduler, ConstantScheduler


class Schedulable:
    r"""Mixin: add scheduler registration + recursive stepping.

    Use alongside `TorchEBMModule` (or any `nn.Module` subclass). The host's
    MRO must invoke `super().__init__()` so this mixin can initialize the
    scheduler dict on construction.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.schedulers: Dict[str, BaseScheduler] = {}

    def register_scheduler(self, name: str, scheduler: BaseScheduler) -> None:
        r"""Register a scheduler under `name`. Replaces any existing entry."""
        self.schedulers[name] = scheduler

    def _register_param(
        self,
        name: str,
        value: Union[float, BaseScheduler],
        *,
        positive: bool = False,
    ) -> None:
        r"""Register a numeric param as either a `BaseScheduler` or a constant.

        For a float input, optionally enforce ``value > 0`` and wrap in
        `ConstantScheduler`. For a `BaseScheduler` input, store it directly.
        """
        if isinstance(value, BaseScheduler):
            self.schedulers[name] = value
            return
        if positive and value <= 0:
            raise ValueError(f"{name} must be positive")
        self.schedulers[name] = ConstantScheduler(float(value))

    def get_schedulers(self) -> Dict[str, BaseScheduler]:
        return self.schedulers

    def get_scheduled_value(self, name: str) -> float:
        if name not in self.schedulers:
            raise KeyError(f"No scheduler registered for parameter '{name}'")
        return self.schedulers[name].get_value()

    def step_schedulers(self) -> None:
        r"""Advance every scheduler in this module subtree by one step."""
        def _step(m):
            if isinstance(m, Schedulable):
                for s in m.schedulers.values():
                    s.step()
        # `apply` is an nn.Module method; the host class must inherit from nn.Module.
        self.apply(_step)

    def reset_schedulers(self) -> None:
        r"""Reset every scheduler in this module subtree to step 0."""
        def _reset(m):
            if isinstance(m, Schedulable):
                for s in m.schedulers.values():
                    s.reset()
        self.apply(_reset)

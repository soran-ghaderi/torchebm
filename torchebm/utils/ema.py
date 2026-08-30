r"""Exponential moving average of model weights."""

from __future__ import annotations

import copy
import weakref
from contextlib import contextmanager
from typing import Callable, List, Literal, Tuple, Union

import torch
import torch.nn as nn

from torchebm.utils.training import requires_grad

_SCHEDULES = ("constant", "warmup")


def _is_plain(t: torch.Tensor) -> bool:
    return t.__class__ in (torch.Tensor, nn.Parameter) and t.data.__class__ is torch.Tensor


def _copy_all_(dsts: List[torch.Tensor], srcs: List[torch.Tensor]) -> None:
    r"""Grouped copy: one fused kernel when every pair is plain and uniform."""
    if not dsts:
        return
    d0, s0 = dsts[0], srcs[0]
    uniform = all(
        _is_plain(d)
        and _is_plain(s)
        and d.device == s.device == d0.device
        and d.dtype == s.dtype == s0.dtype
        for d, s in zip(dsts, srcs)
    )
    if uniform:
        torch._foreach_copy_(dsts, srcs)
        return
    for d, s in zip(dsts, srcs):
        d.copy_(s)


class EMA(nn.Module):
    r"""Exponential moving average of a model's weights.

    Keeps a frozen, eval-mode copy of ``model`` and blends it toward the live
    weights on every `update`:

    \[
    \theta_{EMA} \leftarrow d \, \theta_{EMA} + (1 - d) \, \theta
    \]

    Parameters are averaged; buffers (running statistics, counters) are
    copied, so the averaged copy always carries the live model's buffer
    state. The copy is a full module: sample from ``ema.module`` (or call the
    `EMA` object, which delegates), export with `copy_to`, or evaluate the
    live model under averaged weights with `average_parameters`.

    Performance: parameter pairs are bucketed once at construction; plain
    same-device/dtype pairs update through a single fused
    ``torch._foreach_lerp_`` call per bucket and everything else (e.g.
    identically-sharded DTensor pairs) takes a per-tensor in-place fallback.
    No tensors are allocated per step and nothing synchronizes with the host:
    the step counter is a Python int, persisted through ``state_dict`` via
    the extra-state hooks.

    Contract: keep model structure, devices, and dtypes fixed after
    construction (move the `EMA` module alongside the model if you move
    either; mismatches fail loudly). Under DDP/FSDP wrappers, construct from
    and update with the same unwrapped module; for sharded checkpoint-style
    workflows the functional `torchebm.utils.update_ema` remains available.

    Args:
        model: Module to shadow. Deep-copied at construction.
        decay: EMA decay \(d\) (the cap under warmup). Must lie in [0, 1).
            Default 0.9999, the reference DiT/ADM setting.
        decay_schedule: ``"constant"`` (default) uses ``decay`` from step 0;
            ``"warmup"`` uses \(\min(d, (1+s)/(10+s))\) at step \(s\), the
            classic ramp that lets a fresh average track early training;
            a callable ``step -> decay`` is authoritative (values are
            validated into [0, 1]).

    Example:
        ```python
        ema = EMA(model, decay=0.9999)
        for batch in loader:
            train_step(model, batch)
            ema.update(model)
        samples = sampler_cls(ema.module).sample(...)
        ```
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        decay_schedule: Union[
            Literal["constant", "warmup"], Callable[[int], float]
        ] = "constant",
    ):
        super().__init__()
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        if not (decay_schedule in _SCHEDULES or callable(decay_schedule)):
            raise ValueError(
                f"decay_schedule must be one of {_SCHEDULES} or a callable, "
                f"got {decay_schedule!r}"
            )
        self.decay = float(decay)
        self.decay_schedule = decay_schedule
        self.step = 0
        self.module = copy.deepcopy(model)
        self.module.eval()
        requires_grad(self.module, False)
        self._build_pairs(model)

    def _build_pairs(self, model: nn.Module) -> None:
        self._source = weakref.ref(model)
        buckets: dict = {}
        fallback: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for e, p in zip(self.module.parameters(), model.parameters(), strict=True):
            if _is_plain(e) and _is_plain(p) and e.device == p.device and e.dtype == p.dtype:
                elist, plist = buckets.setdefault((e.device, e.dtype), ([], []))
                elist.append(e)
                plist.append(p)
            else:
                fallback.append((e, p))
        self._param_buckets = list(buckets.values())
        self._param_fallback = fallback
        self._buffer_dsts = list(self.module.buffers())
        self._buffer_srcs = list(model.buffers())
        if len(self._buffer_dsts) != len(self._buffer_srcs):
            raise ValueError("model and EMA copy disagree on the number of buffers")

    def _decay_value(self) -> float:
        if self.decay_schedule == "constant":
            return self.decay
        if self.decay_schedule == "warmup":
            return min(self.decay, (1 + self.step) / (10 + self.step))
        d = float(self.decay_schedule(self.step))
        if not 0.0 <= d <= 1.0:
            raise ValueError(f"decay_schedule returned {d}, expected a value in [0, 1]")
        return d

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        r"""Blend the shadow toward ``model`` and copy its buffers.

        Args:
            model: The live model. Passing a different (structurally
                identical) instance than last time re-binds the cached pairs
                to it.
        """
        if self._source() is not model:
            self._build_pairs(model)
        d = self._decay_value()
        w = 1.0 - d
        for elist, plist in self._param_buckets:
            torch._foreach_lerp_(elist, plist, w)
        for e, p in self._param_fallback:
            e.mul_(d).add_(p, alpha=w)
        _copy_all_(self._buffer_dsts, self._buffer_srcs)
        self.step += 1

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        r"""Write the averaged parameters and buffers into ``model``.

        ``requires_grad`` flags of the target are untouched. The target only
        needs to be structurally identical; it may be a different instance
        than the one being tracked (e.g. a fresh copy for export).
        """
        _copy_all_(
            list(model.parameters()), list(self.module.parameters())
        )
        _copy_all_(list(model.buffers()), list(self.module.buffers()))

    @contextmanager
    def average_parameters(self, model: nn.Module):
        r"""Temporarily run ``model`` under the averaged weights.

        Stores the live parameters and buffers, copies the averages in,
        and restores the originals bit-for-bit on exit, exceptions included.
        """
        params = list(model.parameters())
        buffers = list(model.buffers())
        backup = [t.detach().clone() for t in params + buffers]
        self.copy_to(model)
        try:
            yield model
        finally:
            with torch.no_grad():
                _copy_all_(params + buffers, backup)

    def forward(self, *args, **kwargs):
        r"""Delegate to the averaged module, so `EMA` is directly callable."""
        return self.module(*args, **kwargs)

    def get_extra_state(self) -> dict:
        return {"step": self.step}

    def set_extra_state(self, state: dict) -> None:
        self.step = int(state["step"])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"module={type(self.module).__name__}, decay={self.decay}, "
            f"decay_schedule={self.decay_schedule!r}, step={self.step})"
        )

r"""Base `nn.Module` for TorchEBM components.

`TorchEBMModule` is the single base class every TorchEBM component that holds
tensors should inherit from. It exposes `device` and `dtype` as read-only
properties resolved from the module's parameters/buffers (HuggingFace style),
with a zero-element non-persistent buffer (`_torchebm_probe`) acting as a
fallback for parameter-less modules. The probe is automatically tracked by
`nn.Module.to(...)`, so device/dtype stay consistent with no custom `.to()`
override and no cached state to drift.

Mixed-precision helpers (`setup_mixed_precision`, `autocast_context`) are
provided here as a thin compatibility shim. They will move to a dedicated
`AMPConfig` utility in a follow-up release.
"""

from contextlib import nullcontext
from typing import Optional, Union
import warnings

import torch
from torch import nn


def _normalize(device: torch.device) -> torch.device:
    if device.type == "cuda" and device.index == 0:
        return torch.device("cuda")
    return device


def substitute_condition(y, mask, null):
    r"""Replace conditioning rows selected by `mask` with the null condition.

    Shared semantics for classifier-free-guidance label dropout (losses) and
    the guidance wrapper's unconditional half (sampling).

    Args:
        y: Conditioning tensor of shape (batch_size, ...).
        mask: Bool tensor of shape (batch_size,); True rows become null.
        null: Null condition: an int/float filled into y's dtype (the
            ``num_classes`` label convention), a tensor broadcast over y's
            non-batch dims (e.g. a zero or learned null embedding), or a
            callable ``(y, mask) -> y``.

    Returns:
        Tensor of y's shape with masked rows nulled.
    """
    if callable(null):
        return null(y, mask)
    mask = mask.view(-1, *([1] * (y.ndim - 1)))
    if isinstance(null, torch.Tensor):
        return torch.where(mask, null.to(device=y.device, dtype=y.dtype), y)
    return torch.where(mask, torch.full_like(y, null), y)


_WARNED_ONCE: set = set()


def warn_once(
    key: str,
    message: str,
    category: type = DeprecationWarning,
    stacklevel: int = 3,
) -> None:
    r"""Emit a warning at most once per process, keyed by `key`.

    Deprecation paths on hot code (per-step sampler loops, per-batch losses)
    must not call `warnings.warn` every iteration: even when the filter shows a
    warning only once, the per-call filter processing adds avoidable overhead.
    This guard skips the call entirely after the first hit.
    """
    if key in _WARNED_ONCE:
        return
    _WARNED_ONCE.add(key)
    warnings.warn(message, category, stacklevel=stacklevel)


class TorchEBMModule(nn.Module):
    r"""Base `nn.Module` with cached, parameter-derived device/dtype access."""

    def __init__(
        self,
        device: Union[str, torch.device, None] = None,
        dtype: Optional[torch.dtype] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        probe_dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.register_buffer(
            "_torchebm_probe",
            torch.empty(0, dtype=probe_dtype, device=device),
            persistent=False,
        )
        self.use_mixed_precision: bool = False
        self.autocast_available: bool = False
        self._amp_dtype: torch.dtype = torch.float16
        self._cached_device: Optional[torch.device] = None
        self._cached_dtype: Optional[torch.dtype] = None

    @staticmethod
    def _merge_condition(
        model_kwargs: Optional[dict], y: Optional[torch.Tensor]
    ) -> Optional[dict]:
        r"""Fold the explicit ``y=`` conditioning into `model_kwargs`.

        ``y=None`` returns `model_kwargs` untouched, keeping the unconditional
        path identical. Passing ``y`` while `model_kwargs` already carries a
        ``'y'`` key is ambiguous and raises.

        Raises:
            ValueError: If both ``y`` and ``model_kwargs['y']`` are given.
        """
        if y is None:
            return model_kwargs
        if model_kwargs and "y" in model_kwargs:
            raise ValueError(
                "y was passed both as y= and inside model_kwargs['y']; "
                "provide it once"
            )
        return {**(model_kwargs or {}), "y": y}

    def _resolve_device_dtype(self) -> None:
        try:
            p = next(self.parameters())
            self._cached_device = _normalize(p.device)
            self._cached_dtype = p.dtype
            return
        except StopIteration:
            pass
        probe = self._torchebm_probe
        self._cached_device = _normalize(probe.device)
        self._cached_dtype = probe.dtype

    @property
    def device(self) -> torch.device:
        if self._cached_device is None:
            self._resolve_device_dtype()
        return self._cached_device

    @property
    def dtype(self) -> torch.dtype:
        if self._cached_dtype is None:
            self._resolve_device_dtype()
        return self._cached_dtype

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        self._cached_device = None
        self._cached_dtype = None
        return result

    def _prepare_model_kwargs(
        self, model_kwargs: Optional[dict]
    ) -> dict:
        r"""Normalize conditioning `model_kwargs` once at a call boundary.

        The single entry point for the library-wide conditioning convention:
        every ``sample()``/``forward()`` that forwards conditioning to the model
        calls this once, then reuses the returned dict (e.g. captured by a
        per-step drift closure) without re-normalizing.

        GPU-first: tensor values are moved to `self.device` a single time with
        ``non_blocking=True`` and are **not** dtype-cast (integer class labels /
        token ids must stay integral for embedding lookups). Non-tensor values
        pass through untouched. A fresh dict is always returned, so callers may
        mutate it without aliasing the caller's mapping.

        Args:
            model_kwargs: Conditioning mapping forwarded to the model, or None.

        Returns:
            A new dict with tensor values on `self.device`; ``{}`` when
            `model_kwargs` is None or empty.

        Raises:
            TypeError: If `model_kwargs` is neither None nor a mapping.
        """
        if not model_kwargs:
            return {}
        if not isinstance(model_kwargs, dict):
            raise TypeError(
                f"model_kwargs must be a dict, got {type(model_kwargs).__name__}"
            )
        device = self.device
        return {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in model_kwargs.items()
        }

    def setup_mixed_precision(
        self,
        use_mixed_precision: bool,
        amp_dtype: torch.dtype = torch.float16,
    ) -> None:
        r"""Configure autocast for `autocast_context()`.

        Args:
            use_mixed_precision: Enable autocast.
            amp_dtype: Autocast dtype (`torch.float16` default, `torch.bfloat16` supported).
        """
        self.use_mixed_precision = bool(use_mixed_precision)
        self._amp_dtype = amp_dtype
        if not self.use_mixed_precision:
            self.autocast_available = False
            return
        if self.device.type != "cuda":
            warnings.warn(
                f"Mixed precision requested but device is {self.device}. "
                f"Requires CUDA. Falling back to full precision.",
                UserWarning,
            )
            self.use_mixed_precision = False
            self.autocast_available = False
            return
        self.autocast_available = True

    def autocast_context(self):
        r"""Return an autocast context if mixed precision is enabled, else `nullcontext`."""
        if self.use_mixed_precision and self.autocast_available:
            return torch.amp.autocast(
                device_type=self.device.type, dtype=self._amp_dtype
            )
        return nullcontext()

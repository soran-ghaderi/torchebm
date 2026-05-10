"""Device and dtype management for TorchEBM modules."""

from contextlib import nullcontext
from typing import Optional, Union
import warnings

import torch


def normalize_device(device: Union[str, torch.device, None]) -> torch.device:
    r"""Normalize a device identifier to a concrete `torch.device`.

    Resolves `None` to the current CUDA device if available else CPU. String
    identifiers are converted to `torch.device`. CUDA devices that match the
    current device index are collapsed to indexless form.
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda" and device.index is not None and torch.cuda.is_available():
        if device.index == torch.cuda.current_device():
            return torch.device("cuda")
    return device


def safe_to(
    obj,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    r"""Move `obj` to `device`/`dtype` if it supports `.to()`. No-op otherwise."""
    if not hasattr(obj, "to") or not callable(obj.to):
        return obj
    if device is None and dtype is None:
        return obj
    try:
        return obj.to(device=device, dtype=dtype)
    except TypeError:
        if device is not None:
            try:
                return obj.to(device)
            except Exception:
                pass
        if dtype is not None:
            try:
                return obj.to(dtype)
            except Exception:
                pass
        return obj


class DeviceMixin:
    r"""Mixin for consistent device/dtype management across modules.

    .. deprecated::
        Use `TorchEBMModule` (`torchebm.core.base_module`) instead. This mixin
        is kept for backwards compatibility and will be removed in a future
        release. The new base class delegates `device`/`dtype` to actual
        parameters/buffers (HuggingFace style), avoiding duplicate state.

    Provides cached `device` and `dtype` properties resolved once at init time
    (or lazily inferred from parameters/buffers when not specified). The `.to()`
    override keeps the cache in sync with PyTorch movement semantics.
    """

    def __init__(
        self,
        device: Union[str, torch.device, None] = None,
        dtype: Optional[torch.dtype] = None,
        *args,
        **kwargs,
    ):
        warnings.warn(
            "DeviceMixin is deprecated; subclass TorchEBMModule "
            "(torchebm.core.TorchEBMModule) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
        self._device: torch.device = normalize_device(device)
        self._dtype: Optional[torch.dtype] = dtype
        self.use_mixed_precision: bool = False
        self.autocast_available: bool = False
        self._amp_dtype: torch.dtype = torch.float16

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        if self._dtype is not None:
            return self._dtype
        params = getattr(self, "parameters", None)
        if callable(params):
            try:
                p_dtype = next(self.parameters()).dtype
                self._dtype = p_dtype
                return p_dtype
            except StopIteration:
                pass
        bufs = getattr(self, "buffers", None)
        if callable(bufs):
            try:
                b_dtype = next(self.buffers()).dtype
                self._dtype = b_dtype
                return b_dtype
            except StopIteration:
                pass
        return torch.float32

    @dtype.setter
    def dtype(self, value: torch.dtype):
        self._dtype = value

    def to(self, *args, **kwargs):
        r"""Move module and update cached device/dtype using PyTorch's parser."""
        target_device, target_dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        parent_to = getattr(super(), "to", None)
        result = parent_to(*args, **kwargs) if callable(parent_to) else self
        if target_device is not None:
            self._device = normalize_device(target_device)
        if target_dtype is not None:
            self._dtype = target_dtype
        return result

    # Backwards-compatible alias; prefer the module-level `safe_to`.
    safe_to = staticmethod(safe_to)

    def setup_mixed_precision(
        self,
        use_mixed_precision: bool,
        amp_dtype: torch.dtype = torch.float16,
    ) -> None:
        r"""Configure mixed-precision autocast.

        Args:
            use_mixed_precision: Enable autocast when entering `autocast_context()`.
            amp_dtype: Autocast dtype. `torch.float16` (default) or `torch.bfloat16`.
        """
        self.use_mixed_precision = bool(use_mixed_precision)
        self._amp_dtype = amp_dtype
        if not self.use_mixed_precision:
            self.autocast_available = False
            return
        if self._device.type != "cuda":
            warnings.warn(
                f"Mixed precision requested but device is {self._device}. "
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
                device_type=self._device.type, dtype=self._amp_dtype
            )
        return nullcontext()

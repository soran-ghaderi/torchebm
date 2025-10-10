"""
Author: Soran Ghaderi
Email: soran.gdr.cs@gmail.com
This module handles the device management or TorchEBM modules
"""

from typing import Union, Optional
import warnings
import torch


def normalize_device(device):
    """
    Normalize the device for consistent usage across the library.

    - str to device object
    - remove unnecessary device indices
    - default device to 'cuda:0' if not specified
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda" and device.index is not None:
        if (
            device.index == torch.cuda.current_device()
            if torch.cuda.is_available()
            else 0
        ):
            return torch.device("cuda")

    return device


class DeviceMixin:
    """Consistent device management across all modules.

    This should be inherited by all classes.
    """

    def __init__(self, device: Union[str, torch.device, None] = None, dtype: Optional[torch.dtype] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._device = normalize_device(device)
        self._dtype: Optional[torch.dtype] = dtype

    @property
    def device(self) -> torch.device:
        if self._device is not None:
            return normalize_device(self._device)
        if self._device is None:
            if hasattr(self, "parameters") and callable(getattr(self, "parameters")):
                try:
                    param_device = next(self.parameters()).device
                    return normalize_device(param_device)
                except StopIteration:
                    pass

            if hasattr(self, "buffers") and callable(getattr(self, "buffers")):
                try:
                    buffer_device = next(self.buffers()).device
                    return normalize_device(buffer_device)
                except StopIteration:
                    pass

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def dtype(self) -> torch.dtype:
        if self._dtype is not None:
            return self._dtype
        # Try infer from parameters/buffers if available
        if hasattr(self, "parameters") and callable(getattr(self, "parameters")):
            try:
                param_dtype = next(self.parameters()).dtype
                return param_dtype
            except StopIteration:
                pass
        if hasattr(self, "buffers") and callable(getattr(self, "buffers")):
            try:
                buffer_dtype = next(self.buffers()).dtype
                return buffer_dtype
            except StopIteration:
                pass
        return torch.float32

    @dtype.setter
    def dtype(self, value: torch.dtype):
        self._dtype = value

    def to(self, *args, **kwargs):
        """Override to() to update internal device tracking."""
        # Call parent's to() if it exists (e.g., nn.Module); otherwise, operate in-place
        parent_to = getattr(super(), "to", None)
        result = self
        if callable(parent_to):
            result = parent_to(*args, **kwargs)

        # Update internal device tracking based on provided args/kwargs
        target_device = None
        target_dtype = None
        if args and isinstance(args[0], (str, torch.device)):
            target_device = normalize_device(args[0])
        elif args and isinstance(args[0], torch.dtype):
            target_dtype = args[0]
        if "device" in kwargs:
            target_device = normalize_device(kwargs["device"])
        if "dtype" in kwargs and isinstance(kwargs["dtype"], torch.dtype):
            target_dtype = kwargs["dtype"]
        if target_device is not None:
            self._device = target_device
        if target_dtype is not None:
            self._dtype = target_dtype

        return result

    @staticmethod
    def safe_to(obj, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """
        Safely move an object supporting .to(...) to device/dtype, handling different signatures.
        """
        if not hasattr(obj, "to") or not callable(getattr(obj, "to")):
            return obj
        try:
            if device is not None or dtype is not None:
                return obj.to(device=device, dtype=dtype)
            return obj
        except TypeError:
            # Fallbacks for custom signatures
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

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

    def __init__(self, device: Union[str, torch.device, None] = None, **kwargs):
        super().__init__(**kwargs)
        self._device = normalize_device(device)

    def device(self) -> torch.device:

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

        return self._device

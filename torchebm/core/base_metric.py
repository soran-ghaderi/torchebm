import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple


class BaseMetric(ABC):
    """
    Abstract base class for all evaluation metrics in torchebm.

    This class defines the standard interface for implementing evaluation metrics
    for energy-based models. All metrics should inherit from this class and implement
    the required methods.

    Attributes:
        name (str): The name of the metric
        lower_is_better (bool): Whether lower values indicate better performance
    """

    def __init__(self, name: str, lower_is_better: bool = True):
        """
        Initialize the base metric.

        Args:
            name (str): Name of the metric
            lower_is_better (bool): Whether lower values indicate better performance
        """
        self.name = name
        self.lower_is_better = lower_is_better
        self._device = None

    @property
    def device(self) -> Optional[torch.device]:
        """Returns the device associated with the metric."""
        return self._device

    def to(self, device: Union[str, torch.device]) -> "BaseMetric":
        """
        Move the metric to the specified device.

        Args:
            device: The device to move the metric to

        Returns:
            self: The metric instance moved to the device
        """
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        return self

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute the metric value.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the computed metric values
        """
        pass

    def compute(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Alias for __call__ to match sklearn and other libraries' conventions.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the computed metric values
        """
        return self(*args, **kwargs)

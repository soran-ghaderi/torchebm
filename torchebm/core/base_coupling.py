r"""Base class for minibatch couplings between source and target samples."""

from abc import ABC, abstractmethod
from typing import Tuple

import torch


class BaseCoupling(ABC):
    r"""
    Abstract base class for minibatch couplings.

    A coupling defines how source samples \(x_0\) (typically noise) are paired
    with target samples \(x_1\) (data) before interpolation. The independent
    coupling keeps the incoming pairing; optimal-transport couplings reorder
    the batch to (approximately) minimize the transport cost
    \(\sum_i \|x_0^{(i)} - x_1^{(i)}\|^2\), which straightens flow-matching
    paths (OT-CFM, rectified flow, Energy Matching).

    Couplings only reorder or resample batch indices; they are computed under
    `torch.no_grad()` and never propagate gradients.

    Subclasses must implement `couple`.
    """

    @abstractmethod
    def couple(
        self, x0: torch.Tensor, x1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Pair source and target samples.

        Args:
            x0: Source samples of shape (batch_size, ...).
            x1: Target samples of shape (batch_size, ...).

        Returns:
            Tuple of (x0, x1) with aligned pairing; same shapes as the inputs.

        Raises:
            ValueError: If batch sizes differ.
        """
        raise NotImplementedError

    def _check_batch(self, x0: torch.Tensor, x1: torch.Tensor) -> None:
        r"""Validate that both batches have the same leading dimension."""
        if x0.shape[0] != x1.shape[0]:
            raise ValueError(
                f"Coupling requires equal batch sizes, got {x0.shape[0]} and {x1.shape[0]}"
            )

    def __call__(
        self, x0: torch.Tensor, x1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.couple(x0, x1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

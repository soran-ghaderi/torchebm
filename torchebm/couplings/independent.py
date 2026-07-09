r"""Independent (identity) coupling."""

from __future__ import annotations

from typing import Any, Optional

import torch

from torchebm.core import BaseCoupling, CouplingResult


class IndependentCoupling(BaseCoupling):
    r"""
    Identity coupling: keep the incoming pairing.

    This is the classical flow-matching setup where noise and data are paired
    independently, and equally the paired-dataset setting (image restoration,
    domain translation) where the incoming pairing is already meaningful.

    Example:
        ```python
        from torchebm.couplings import IndependentCoupling

        coupling = IndependentCoupling()
        x0, x1 = coupling(torch.randn(64, 2), data_batch)  # unchanged
        ```
    """

    def couple(
        self, x0: torch.Tensor, x1: Optional[torch.Tensor] = None, **kwargs: Any
    ) -> CouplingResult:
        x1 = self._require_x1(x1)
        self._check_batch(x0, x1)
        return CouplingResult(x0, x1)


__all__ = ["IndependentCoupling"]

r"""Base classes for couplings between source and target samples."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import torch


@dataclass(frozen=True)
class CouplingResult:
    r"""
    Result of a coupling.

    Iterates as the pair, so the idiomatic unpacking always works:

    ```python
    x0, x1 = coupling(x0, x1)
    ```

    Extra information rides as attributes and is never part of the iteration,
    so appending optional fields can never break unpacking. Consumers that
    need extras access them by name:

    ```python
    res = coupling(x0, x1)
    x0, x1 = res
    if res.weights is not None:
        ...  # per-pair importance weights (unbalanced / reweighted OT)
    ```

    Note:
        `tuple(result)` and `list(result)` yield the pair only; extras are
        attribute-access. Future extras (e.g. transport plans, alignment
        transforms) are appended as `Optional[...] = None` fields.

    Attributes:
        x0: Source samples of shape (batch_size, ...).
        x1: Target samples of shape (batch_size, ...).
        weights: Optional per-pair weights of shape (batch_size,). ``None``
            means uniform. Produced by unbalanced/reweighted couplings;
            weight-aware consumers use them as importance weights.
    """

    x0: torch.Tensor
    x1: torch.Tensor
    weights: Optional[torch.Tensor] = None

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter((self.x0, self.x1))


class BaseCoupling(ABC):
    r"""
    Abstract base class for couplings.

    A coupling is a rule that pairs a batch of source samples \(x_0\)
    (typically noise) with target samples \(x_1\) (data) before
    interpolation. Depending on the family it may **reorder** or **resample**
    an incoming target batch (independent, minibatch OT), **transform** it
    (equivariant alignment, closed-form maps), or **generate** it from the
    source (model-induced couplings such as reflow and DSBM, where
    \(x_1 = \Phi(x_0)\)).

    Couplings are computed under `torch.no_grad()` and never propagate
    gradients. Equal batch sizes are not enforced at this level; families
    that require them call the `_check_batch` helper (all cost-based
    couplings do), and families that require a target batch call
    `_require_x1`.

    Two extension channels keep this contract closed against future families:

    - **Conditioning in:** `couple`/`__call__` accept `**kwargs` (class
      labels, prompts, geometry) that conditional and structure-aware
      couplings consume; unconditional couplings ignore them.
    - **Extras out:** results are `CouplingResult` objects that unpack as the
      `(x0, x1)` pair while carrying optional extras as attributes (per-pair
      `weights` for unbalanced/reweighted OT today; more may be appended
      without breaking any consumer).

    Subclasses must implement `couple`. Cost-based couplings should instead
    subclass `BaseCostCoupling`, which supplies the template and asks only
    for a solver; model-induced couplings subclass `BaseModelCoupling`.
    """

    @abstractmethod
    def couple(
        self, x0: torch.Tensor, x1: Optional[torch.Tensor] = None, **kwargs: Any
    ) -> CouplingResult:
        r"""
        Pair source and target samples.

        Args:
            x0: Source samples of shape (batch_size, ...).
            x1: Target samples of shape (batch_size, ...). Optional at this
                level: generate-family couplings produce the target from the
                source and ignore (or do not need) an incoming batch;
                pairing families require it via `_require_x1`.
            **kwargs: Optional conditioning forwarded by the caller; ignored
                by unconditional couplings.

        Returns:
            A `CouplingResult` that unpacks as the aligned pair
            ``(x0, x1)``; weighted couplings also set its ``weights``.
        """
        raise NotImplementedError

    def _check_batch(self, x0: torch.Tensor, x1: torch.Tensor) -> None:
        r"""Validate that both batches have the same leading dimension."""
        if x0.shape[0] != x1.shape[0]:
            raise ValueError(
                f"Coupling requires equal batch sizes, got {x0.shape[0]} and {x1.shape[0]}"
            )

    def _require_x1(self, x1: Optional[torch.Tensor]) -> torch.Tensor:
        r"""Validate that a target batch was provided (pairing families)."""
        if x1 is None:
            raise ValueError(
                f"{self.__class__.__name__} pairs against an existing target "
                f"batch; x1 must not be None"
            )
        return x1

    def __call__(
        self, x0: torch.Tensor, x1: Optional[torch.Tensor] = None, **kwargs: Any
    ) -> CouplingResult:
        return self.couple(x0, x1, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class BaseCostCoupling(BaseCoupling):
    r"""
    Family base for couplings that pair minibatches by minimizing a cost.

    This implements the shared pipeline once as a template method and asks
    concretes for a single piece, the assignment solver, playing the role
    Butcher tableaus play for `BaseRungeKuttaIntegrator`. `couple`:

    1. runs under `torch.no_grad()`,
    2. validates that a target batch exists and batch sizes match,
    3. passes a single-sample batch through unchanged,
    4. builds a pairwise cost matrix via `compute_cost` (overridable), and
    5. delegates pairing to the abstract `_solve`.

    `x0` order and marginal are always preserved; only `x1` is reindexed.
    Subclasses supply `_solve`; they may also override `compute_cost` for a
    non-Euclidean or conditioning-aware ground cost. Weighted cost variants
    (unbalanced OT) override `couple` itself, reusing `compute_cost`, to
    attach per-pair `weights` to the result.
    """

    @torch.no_grad()
    def couple(
        self, x0: torch.Tensor, x1: Optional[torch.Tensor] = None, **kwargs: Any
    ) -> CouplingResult:
        x1 = self._require_x1(x1)
        self._check_batch(x0, x1)
        if x0.shape[0] == 1:
            return CouplingResult(x0, x1)
        cost = self.compute_cost(x0, x1, **kwargs)
        idx = self._solve(cost)
        return CouplingResult(x0, x1[idx])

    def compute_cost(
        self, x0: torch.Tensor, x1: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        r"""
        Pairwise ground-cost matrix between source and target samples.

        The default is the max-normalized squared Euclidean cost on flattened
        samples:

        \[
        C_{ij} = \frac{\|x_0^{(i)} - x_1^{(j)}\|^2}{\max_{kl} C_{kl}}
        \]

        Args:
            x0: Source samples of shape (batch_size, ...).
            x1: Target samples of shape (batch_size, ...).
            **kwargs: Optional conditioning (unused by the default cost).

        Returns:
            Cost matrix of shape (batch_size, batch_size).
        """
        batch = x0.shape[0]
        cost = torch.cdist(x0.reshape(batch, -1), x1.reshape(batch, -1)).square()
        return cost / cost.max().clamp(min=1e-12)

    @abstractmethod
    def _solve(self, cost: torch.Tensor) -> torch.Tensor:
        r"""
        Solve the pairing problem on a square cost matrix.

        Args:
            cost: Cost matrix of shape (n, n).

        Returns:
            Long tensor `idx` of shape (n,) pairing `x0[i]` with `x1[idx[i]]`.
            Assignment solvers return a permutation; entropic solvers return
            row-conditional draws (the `x0` marginal is preserved exactly).
        """
        raise NotImplementedError


class BaseModelCoupling(BaseCoupling):
    r"""
    Family base for couplings that generate the target from a model map.

    The template `couple` runs under `torch.no_grad()` and returns
    \((x_0, \Phi(x_0))\); any incoming ``x1`` is ignored (documented family
    behavior; the argument stays optional for standalone use:
    ``coupling(x0)``). Concretes supply `_generate`, the map evaluation.
    This is the reflow / rectified-flow shape; iterative-Markovian-fitting
    couplings (DSBM) reuse the same machinery.

    Args:
        model: The map object (sampler, module, callable) `_generate` uses.
    """

    def __init__(self, model: Any):
        self.model = model

    @torch.no_grad()
    def couple(
        self, x0: torch.Tensor, x1: Optional[torch.Tensor] = None, **kwargs: Any
    ) -> CouplingResult:
        # x1 (if any) is ignored: the target is generated as Phi(x0).
        return CouplingResult(x0, self._generate(x0, **kwargs))

    @abstractmethod
    def _generate(self, x0: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        r"""
        Generate the target batch from the source batch.

        Args:
            x0: Source samples of shape (batch_size, ...).
            **kwargs: Optional conditioning forwarded from `couple`.

        Returns:
            Target samples of shape (batch_size, ...).
        """
        raise NotImplementedError

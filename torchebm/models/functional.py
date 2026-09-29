r"""Functional model operations."""

import torch


def energy_gradient(
    energy: torch.Tensor,
    x: torch.Tensor,
    create_graph: bool = False,
) -> torch.Tensor:
    r"""Differentiate a batch of scalar energies with respect to its inputs.

    ``energy`` must be differentiably connected to ``x``. The corresponding
    log-density score is ``-energy_gradient(energy, x)``.

    Args:
        energy: Per-sample energies.
        x: Inputs used to compute ``energy``.
        create_graph: Retain a differentiable gradient graph for training.

    Returns:
        The summed energy gradient with the same shape as ``x``.
    """
    if not energy.requires_grad:
        raise RuntimeError("energy must be differentiably connected to x")

    return torch.autograd.grad(
        outputs=energy,
        inputs=x,
        grad_outputs=torch.ones_like(energy),
        create_graph=create_graph,
    )[0]


__all__ = ["energy_gradient"]

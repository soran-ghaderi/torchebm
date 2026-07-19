r"""Guarded `torch.distributed` helpers for distributed-aware components.

TorchEBM components never require an initialized process group: every helper
here degrades to a no-op or identity in single-process runs, so behavior is
unchanged when `torch.distributed` is not in use. Components accept an
optional ``process_group`` only where the math is batch-global (e.g. minibatch
OT couplings); no default ``forward()``/``sample()`` path issues a collective.
"""

from typing import Any, Optional

import torch
import torch.distributed as dist

__all__ = [
    "is_distributed",
    "get_rank",
    "get_world_size",
    "all_gather_cat",
    "broadcast_object",
]


def is_distributed() -> bool:
    r"""Whether `torch.distributed` is available and a process group is initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank(group: Optional["dist.ProcessGroup"] = None) -> int:
    r"""Rank of this process in `group`; 0 when not distributed."""
    return dist.get_rank(group) if is_distributed() else 0


def get_world_size(group: Optional["dist.ProcessGroup"] = None) -> int:
    r"""Number of processes in `group`; 1 when not distributed."""
    return dist.get_world_size(group) if is_distributed() else 1


def all_gather_cat(
    x: torch.Tensor,
    group: Optional["dist.ProcessGroup"] = None,
    dim: int = 0,
) -> torch.Tensor:
    r"""Gather `x` from every rank and concatenate along `dim`.

    Identity when not distributed. Requires equal shapes on every rank (the
    library convention of equal per-rank batches). The result carries no
    gradient; callers that need differentiable gathers must handle that
    themselves.

    Args:
        x: Local tensor to gather.
        group: Process group; the default group when None.
        dim: Concatenation dimension.

    Returns:
        Tensor of shape `x.shape` with `dim` scaled by the world size, ordered
        by rank; `x` itself when not distributed.
    """
    world = get_world_size(group)
    if world == 1:
        return x
    x = x.detach().contiguous()
    out = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(out, x, group=group)
    return torch.cat(out, dim=dim)


def broadcast_object(
    obj: Any,
    src: int = 0,
    group: Optional["dist.ProcessGroup"] = None,
) -> Any:
    r"""Broadcast a picklable object from rank `src`; identity when not distributed.

    Args:
        obj: Object to broadcast (significant on `src` only).
        src: Source rank.
        group: Process group; the default group when None.

    Returns:
        The object from rank `src` on every rank.
    """
    if get_world_size(group) == 1:
        return obj
    buf = [obj if get_rank(group) == src else None]
    dist.broadcast_object_list(buf, src=src, group=group)
    return buf[0]

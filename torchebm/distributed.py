r"""Guarded `torch.distributed` helpers for distributed-aware components.

TorchEBM components never require an initialized process group: every helper
here degrades to a no-op or identity in single-process runs, so behavior is
unchanged when `torch.distributed` is not in use. Components accept an
optional ``process_group`` only where the math is batch-global (e.g. minibatch
OT couplings); no default ``forward()``/``sample()`` path issues a collective.
"""

from contextlib import contextmanager
from typing import Any, Iterator, Optional

import torch
import torch.distributed as dist
from torch import nn

__all__ = [
    "is_distributed",
    "get_rank",
    "get_world_size",
    "all_gather_cat",
    "broadcast_object",
    "broadcast_tensor",
    "unsharded",
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

    Requires numpy: `torch.distributed`'s object collectives deserialize
    through numpy buffers. For tensors of a shape known on every rank, use
    `broadcast_tensor`, which needs neither pickle nor numpy.

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


def broadcast_tensor(
    t: torch.Tensor,
    src: int = 0,
    group: Optional["dist.ProcessGroup"] = None,
) -> torch.Tensor:
    r"""Broadcast a tensor from rank `src`; identity when not distributed.

    The shape and dtype must match on every rank (no pickling, so this works
    without numpy). Under a CUDA backend a CPU tensor round-trips through the
    current device and is returned on its original device.

    Args:
        t: Tensor to broadcast (values significant on `src` only).
        src: Source rank.
        group: Process group; the default group when None.

    Returns:
        The tensor from rank `src` on every rank, on `t`'s device.
    """
    if get_world_size(group) == 1:
        return t
    device = t.device
    if "nccl" in dist.get_backend(group) and not t.is_cuda:
        t = t.cuda()
    t = t.contiguous()
    dist.broadcast(t, src=src, group=group)
    return t.to(device)


@contextmanager
def unsharded(module: nn.Module, recurse: bool = True) -> Iterator[nn.Module]:
    r"""Keep an FSDP2 module's parameters unsharded for the duration of the block.

    A k-step MCMC chain calls the energy model k times; with the default
    reshard-after-forward, every step re-runs the parameter all-gather. This
    holds the parameters gathered across the whole block, trading memory for
    k-1 fewer all-gathers, and restores reshard-after-forward on exit.

    Duck-typed on `set_reshard_after_forward`, so it is a no-op for plain
    modules, `DistributedDataParallel`, and single-process runs; no FSDP import
    and no distributed initialization are required.

    Use it around inference-style loops (sampling, evaluation), not around a
    forward whose activations feed a backward pass.

    Args:
        module: Module to run unsharded.
        recurse: Apply to nested FSDP modules as well.

    Yields:
        `module`, unchanged.

    Example:
        ```python
        from torchebm.distributed import unsharded

        with unsharded(model):
            samples = sampler.sample(x=x0, n_steps=100)
        ```
    """
    set_reshard = getattr(module, "set_reshard_after_forward", None)
    if set_reshard is None:
        yield module
        return
    set_reshard(False, recurse=recurse)
    try:
        yield module
    finally:
        set_reshard(True, recurse=recurse)

from typing import Callable

import torch

def _integrate_time_grid(
    x: torch.Tensor,
    t: torch.Tensor,
    step_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    if t.ndim != 1:
        raise ValueError("t must be a 1D tensor")
    if t.numel() < 2:
        raise ValueError("t must have length >= 2")
    for i in range(t.numel() - 1):
        dt = t[i + 1] - t[i]
        t_batch = t[i].expand(x.size(0))
        x = step_fn(x, t_batch, dt)
    return x

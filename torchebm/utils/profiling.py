r"""Lightweight profiling utilities for TorchEBM.

Wraps ``torch.profiler`` with sensible defaults for energy-based model
profiling. Use ``profile_context`` for ad-hoc profiling and
``record_function`` annotations in hot loops for per-region breakdowns.

Example:
    ```python
    from torchebm.utils.profiling import profile_context
    from torchebm.samplers import GradientDescentSampler
    from torchebm.core import DoubleWellModel

    model = DoubleWellModel()
    sampler = GradientDescentSampler(model, step_size=0.01, device="cuda")

    with profile_context(export_trace="trace.json") as prof:
        sampler.sample(dim=128, n_samples=1024, n_steps=200)
    ```
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, Sequence

import torch
from torch.profiler import ProfilerActivity, profile, record_function  # noqa: F401


def _default_activities() -> list[ProfilerActivity]:
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    return activities


@contextmanager
def profile_context(
    activities: Optional[Sequence[ProfilerActivity]] = None,
    record_shapes: bool = True,
    profile_memory: bool = True,
    with_stack: bool = False,
    sort_by: Optional[str] = None,
    row_limit: int = 20,
    export_trace: Optional[str] = None,
    print_table: bool = True,
):
    r"""Context manager for profiling TorchEBM operations.

    Wraps ``torch.profiler.profile`` with automatic CUDA sync, table printing,
    and optional Chrome trace export.

    Args:
        activities: Profiler activities. Defaults to CPU + CUDA (if available).
        record_shapes: Record tensor shapes in profiler events.
        profile_memory: Track memory allocations.
        with_stack: Record Python call stacks (slower but more detailed).
        sort_by: Column to sort the summary table. Defaults to
            ``"cuda_time_total"`` if CUDA available, else ``"cpu_time_total"``.
        row_limit: Max rows in the summary table.
        export_trace: Path to export Chrome trace JSON (viewable at
            ``chrome://tracing`` or https://ui.perfetto.dev).
        print_table: Whether to print the summary table on exit.

    Yields:
        ``torch.profiler.profile`` instance for advanced access.
    """
    if activities is None:
        activities = _default_activities()

    if sort_by is None:
        sort_by = (
            "cuda_time_total"
            if ProfilerActivity.CUDA in activities
            else "cpu_time_total"
        )

    with profile(
        activities=activities,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
    ) as prof:
        yield prof
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    if print_table:
        print(prof.key_averages().table(sort_by=sort_by, row_limit=row_limit))

    if export_trace is not None:
        prof.export_chrome_trace(export_trace)


__all__ = ["profile_context", "record_function"]

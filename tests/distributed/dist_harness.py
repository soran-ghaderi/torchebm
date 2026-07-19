r"""Spawn harness for the 2-process gloo/CPU distributed suite.

Lessons this harness hard-codes (learned from the design spikes):

- The DeviceMesh device heuristic silently picks cuda when a GPU is visible,
  and two gloo CPU processes then drive one GPU and segfault inside functional
  collectives. Children therefore hide CUDA, and meshes are created explicitly
  on cpu via `cpu_mesh`.
- Children run faulthandler into per-rank crash logs so a SIGSEGV still
  produces a stack instead of a bare ProcessExitedException.

Worker functions must be top-level in their test module (spawn pickles them by
reference; children inherit sys.path, so pytest-imported test modules resolve).
"""

import faulthandler
import os
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD_SIZE = 2
_TIMEOUT_S = 300.0


def cpu_mesh(world_size: int = WORLD_SIZE):
    r"""Explicit CPU device mesh; never rely on the device heuristic here."""
    from torch.distributed.device_mesh import init_device_mesh

    return init_device_mesh("cpu", (world_size,))


def save_result(tmpdir: str, rank: int, obj) -> None:
    r"""Persist a worker's result for the parent to load after join."""
    torch.save(obj, os.path.join(tmpdir, f"result_rank{rank}.pt"))


def _entry(rank, fn, world_size, tmpdir, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    faulthandler.enable(open(os.path.join(tmpdir, f"crash_rank{rank}.log"), "w"))
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{os.path.join(tmpdir, 'init')}",
        rank=rank,
        world_size=world_size,
    )
    try:
        fn(rank, world_size, tmpdir, *args)
    finally:
        dist.destroy_process_group()


def _crash_logs(tmpdir: str, world_size: int) -> str:
    parts = []
    for r in range(world_size):
        path = os.path.join(tmpdir, f"crash_rank{r}.log")
        if os.path.exists(path):
            with open(path) as f:
                text = f.read().strip()
            if text:
                parts.append(f"--- rank {r} crash log ---\n{text}")
    return "\n".join(parts)


def spawn_dist(fn, world_size: int = WORLD_SIZE, timeout: float = _TIMEOUT_S, args=()):
    r"""Run `fn(rank, world_size, tmpdir, *args)` on `world_size` gloo processes.

    Returns the per-rank objects stored via `save_result` (None where a rank
    saved nothing). Raises with the per-rank crash logs attached on child
    failure or timeout.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = mp.start_processes(
            _entry,
            args=(fn, world_size, tmpdir, args),
            nprocs=world_size,
            join=False,
            start_method="spawn",
        )
        try:
            deadline = time.monotonic() + timeout
            done = False
            while not done:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"distributed workers timed out after {timeout}s"
                    )
                done = ctx.join(remaining)
        except Exception as e:
            for p in ctx.processes:
                if p.is_alive():
                    p.terminate()
            logs = _crash_logs(tmpdir, world_size)
            raise RuntimeError(f"distributed workers failed: {e}\n{logs}") from e
        results = []
        for r in range(world_size):
            path = os.path.join(tmpdir, f"result_rank{r}.pt")
            results.append(torch.load(path) if os.path.exists(path) else None)
        return results

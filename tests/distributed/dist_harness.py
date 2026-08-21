r"""Spawn harness for the multi-process distributed suite.

The backend is selected by the ``TORCHEBM_DIST_DEVICE`` environment variable:
the default (``cpu``) runs 2 gloo processes with CUDA hidden, which is the CI
configuration; ``cuda`` runs one NCCL process per GPU for hardware validation.
Tests stay device-generic by building meshes via `dist_mesh`, placing tensors
on `dist_device()`, and seeding RNG through `make_generator` (generators are
device-bound).

On CPU, children hide CUDA and meshes are created explicitly on cpu: the
DeviceMesh device heuristic selects cuda whenever a GPU is visible, and
multiple gloo CPU processes driving one device crash inside functional
collectives. Children also run faulthandler into per-rank crash logs so a
SIGSEGV produces a stack trace instead of a bare ProcessExitedException.

Worker functions must be top-level in their test module: spawn pickles them by
reference, and children inherit sys.path, so pytest-imported test modules
resolve. Results saved via `save_result` are moved to CPU so parent-side
assertions never mix devices.
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

DEVICE_KIND = os.environ.get("TORCHEBM_DIST_DEVICE", "cpu")
BACKEND = "nccl" if DEVICE_KIND == "cuda" else "gloo"


def dist_device() -> torch.device:
    r"""This process's device: `cuda:<current>` under NCCL, else cpu."""
    if DEVICE_KIND == "cuda":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def dist_mesh(world_size: int = WORLD_SIZE):
    r"""Explicit device mesh on the harness device; never the heuristic."""
    from torch.distributed.device_mesh import init_device_mesh

    return init_device_mesh(DEVICE_KIND, (world_size,))


def make_generator(seed: int) -> torch.Generator:
    r"""Seeded generator bound to the harness device."""
    return torch.Generator(device=dist_device()).manual_seed(seed)


def _to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def save_result(tmpdir: str, rank: int, obj) -> None:
    r"""Persist a worker's result (tensors moved to CPU) for the parent."""
    torch.save(_to_cpu(obj), os.path.join(tmpdir, f"result_rank{rank}.pt"))


def _entry(rank, fn, world_size, tmpdir, args):
    if DEVICE_KIND == "cuda":
        torch.cuda.set_device(rank)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    faulthandler.enable(open(os.path.join(tmpdir, f"crash_rank{rank}.log"), "w"))
    torch.set_num_threads(1)
    dist.init_process_group(
        BACKEND,
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
    r"""Run `fn(rank, world_size, tmpdir, *args)` on `world_size` processes.

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

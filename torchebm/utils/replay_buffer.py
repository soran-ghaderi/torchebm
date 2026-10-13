r"""Replay buffer of persistent sampling chains."""

from typing import Optional, Tuple

import torch
from torch import nn

from torchebm.utils.distributed import (
    all_gather_cat,
    broadcast_tensor,
    get_rank,
    get_world_size,
)


class ReplayBuffer(nn.Module):
    r"""Replay buffer of persistent sampling chains.

    Holds the last ``capacity`` samples in a FIFO and serves start points for
    new chains: each draw is a uniformly random stored sample, replaced with
    probability ``new_sample_ratio`` by a fresh \(\mathcal{N}(0, \sigma^2 I)\)
    sample so the chains keep exploring. This is persistent contrastive
    divergence (Tieleman, 2008) with the sample replay buffer of Du and
    Mordatch (2019). The storage starts filled with the same noise, so the
    first draws are fresh chains.

    Distributed runs keep the buffer rank-local: every rank holds independent
    chains and no default path issues a collective. Chains are exchanged only
    through `mix_across_ranks`. Checkpoints are rank-local too: save one file
    per rank and restore with the same world size. Keep this state out of
    `torch.distributed.checkpoint`, whose planner deduplicates non-sharded
    tensors and would keep rank 0's chains only.

    Args:
        capacity: Number of stored samples.
        sample_shape: Shape of one sample.
        new_sample_ratio: Probability that a drawn sample is replaced by a
            fresh one.
        init_noise_scale: \(\sigma\) of the fresh samples and of the initial
            storage.
        device: Storage device.
        dtype: Storage dtype.
        generator: RNG for the initial storage; the global RNG when ``None``.

    Example:
        ```python
        from torchebm.utils import ReplayBuffer

        buffer = ReplayBuffer(capacity=8192, sample_shape=(2,))
        x_neg = sampler.sample(x=buffer.sample(256), n_steps=10)
        buffer.add(x_neg)
        ```
    """

    def __init__(
        self,
        capacity: int,
        sample_shape: Tuple[int, ...],
        *,
        new_sample_ratio: float = 0.05,
        init_noise_scale: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if not 0.0 <= new_sample_ratio <= 1.0:
            raise ValueError(
                f"new_sample_ratio must be in [0, 1], got {new_sample_ratio}"
            )
        if init_noise_scale < 0:
            raise ValueError(f"init_noise_scale must be >= 0, got {init_noise_scale}")
        self.capacity = int(capacity)
        self.sample_shape = tuple(sample_shape)
        self.new_sample_ratio = float(new_sample_ratio)
        self.init_noise_scale = float(init_noise_scale)
        storage = torch.randn(
            self.capacity,
            *self.sample_shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        self.register_buffer("storage", storage.mul_(self.init_noise_scale))
        self._position = 0

    def sample(
        self, batch_size: int, *, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        r"""Draw ``batch_size`` start points.

        Args:
            batch_size: Number of samples.
            generator: RNG for the index draw, the replacement mask and the
                fresh samples; the global RNG when ``None``. A generator is
                bound to its device, which must match the storage device.

        Returns:
            Tensor of shape (batch_size, *sample_shape), independent of the
            storage.
        """
        device = self.storage.device
        index = torch.randint(
            self.capacity, (batch_size,), device=device, generator=generator
        )
        samples = self.storage[index]
        if self.new_sample_ratio > 0:
            fresh = torch.randn_like(samples, generator=generator)
            fresh.mul_(self.init_noise_scale)
            replace = torch.rand(batch_size, device=device, generator=generator)
            replace = replace < self.new_sample_ratio
            samples = torch.where(
                replace.view(-1, *([1] * len(self.sample_shape))), fresh, samples
            )
        return samples

    def add(self, samples: torch.Tensor) -> None:
        r"""Append a batch, overwriting the oldest entries.

        Args:
            samples: Tensor of shape (n, *sample_shape); detached and moved to
                the storage device and dtype. A batch larger than the capacity
                keeps its last ``capacity`` rows.

        Raises:
            ValueError: If the trailing shape differs from ``sample_shape``.
        """
        if tuple(samples.shape[1:]) != self.sample_shape:
            raise ValueError(
                f"samples must have sample_shape {self.sample_shape}, "
                f"got {tuple(samples.shape[1:])}"
            )
        samples = samples.detach().to(
            device=self.storage.device, dtype=self.storage.dtype
        )
        n = samples.shape[0]
        if n >= self.capacity:
            self.storage.copy_(samples[-self.capacity :])
            self._position = 0
            return
        end = self._position + n
        if end <= self.capacity:
            self.storage[self._position : end] = samples
        else:
            first = self.capacity - self._position
            self.storage[self._position :] = samples[:first]
            self.storage[: end - self.capacity] = samples[first:]
        self._position = end % self.capacity

    def mix_across_ranks(
        self,
        process_group: Optional["torch.distributed.ProcessGroup"] = None,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        r"""Redistribute the pooled chains uniformly across ranks.

        A collective: every rank in the group calls it together, outside the
        training step. The pooled storage is re-dealt with one permutation
        drawn on rank 0 and broadcast, so no chain is duplicated or lost and
        per-rank generators cannot desynchronize the shuffle. Costs one
        all-gather of the full storage (transiently the world size times the
        buffer memory) plus a small broadcast, and requires equal ``capacity``
        on every rank. The write position is unchanged. No-op in
        single-process runs.

        Args:
            process_group: Group to mix over; the default group when ``None``.
            generator: CPU generator for the permutation; used on rank 0 only.
        """
        if get_world_size(process_group) == 1:
            return
        gathered = all_gather_cat(self.storage, group=process_group)
        permutation = torch.randperm(gathered.shape[0], generator=generator)
        permutation = broadcast_tensor(permutation, src=0, group=process_group)
        start = get_rank(process_group) * self.capacity
        self.storage.copy_(gathered[permutation[start : start + self.capacity]])

    def get_extra_state(self) -> dict:
        return {"position": self._position}

    def set_extra_state(self, state: dict) -> None:
        self._position = int(state["position"])

    def extra_repr(self) -> str:
        return (
            f"capacity={self.capacity}, sample_shape={self.sample_shape}, "
            f"new_sample_ratio={self.new_sample_ratio}, "
            f"init_noise_scale={self.init_noise_scale}"
        )


__all__ = ["ReplayBuffer"]

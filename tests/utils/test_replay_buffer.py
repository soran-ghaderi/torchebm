import pytest
import torch

from torchebm.utils import ReplayBuffer
from tests.conftest import requires_cuda


def rows_in_storage(samples, buffer):
    return (samples[:, None] == buffer.storage[None]).all(dim=-1).any(dim=1)


def test_storage_is_noise_at_init_noise_scale():
    buffer = ReplayBuffer(
        16, (3,), init_noise_scale=0.1, generator=torch.Generator().manual_seed(0)
    )
    expected = torch.randn(16, 3, generator=torch.Generator().manual_seed(0)) * 0.1
    assert torch.equal(buffer.storage, expected)
    assert not buffer.storage.requires_grad


@pytest.mark.parametrize(
    "kwargs",
    [
        {"capacity": 0},
        {"new_sample_ratio": 1.5},
        {"new_sample_ratio": -0.1},
        {"init_noise_scale": -1.0},
    ],
)
def test_invalid_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        ReplayBuffer(**{"capacity": 4, "sample_shape": (2,), **kwargs})


def test_sample_returns_stored_rows_without_replacement():
    buffer = ReplayBuffer(8, (2,), new_sample_ratio=0.0)
    samples = buffer.sample(32, generator=torch.Generator().manual_seed(0))
    assert samples.shape == (32, 2)
    assert rows_in_storage(samples, buffer).all()


def test_sample_replaces_rows_with_fresh_noise():
    buffer = ReplayBuffer(8, (2,), new_sample_ratio=1.0, init_noise_scale=0.5)
    samples = buffer.sample(32, generator=torch.Generator().manual_seed(0))
    assert not rows_in_storage(samples, buffer).any()


def test_replacement_follows_new_sample_ratio():
    buffer = ReplayBuffer(8, (2,), new_sample_ratio=0.25)
    samples = buffer.sample(4000, generator=torch.Generator().manual_seed(0))
    fresh_fraction = 1.0 - rows_in_storage(samples, buffer).float().mean().item()
    assert 0.2 < fresh_fraction < 0.3


def test_sample_is_independent_of_storage():
    buffer = ReplayBuffer(4, (2,), new_sample_ratio=0.0)
    before = buffer.storage.clone()
    buffer.sample(4).zero_()
    assert torch.equal(buffer.storage, before)


def test_add_is_fifo_with_wraparound():
    buffer = ReplayBuffer(5, (1,))
    first = torch.arange(3.0).view(3, 1)
    buffer.add(first)
    assert torch.equal(buffer.storage[:3], first)
    second = torch.arange(10.0, 14.0).view(4, 1)
    buffer.add(second)
    assert torch.equal(buffer.storage[3:], second[:2])
    assert torch.equal(buffer.storage[:2], second[2:])
    assert torch.equal(buffer.storage[2], first[2])
    buffer.add(torch.full((2, 1), 7.0))
    assert torch.equal(buffer.storage[2:4], torch.full((2, 1), 7.0))


def test_add_larger_than_capacity_keeps_latest_rows():
    buffer = ReplayBuffer(3, (1,))
    buffer.add(torch.arange(7.0).view(7, 1))
    assert torch.equal(buffer.storage, torch.tensor([[4.0], [5.0], [6.0]]))
    buffer.add(torch.tensor([[9.0]]))
    assert buffer.storage[0].item() == 9.0


def test_add_detaches_and_casts_to_storage():
    buffer = ReplayBuffer(4, (2,), dtype=torch.float64)
    buffer.add(torch.randn(2, 2, requires_grad=True))
    assert buffer.storage.dtype == torch.float64
    assert not buffer.storage.requires_grad


def test_add_rejects_wrong_sample_shape():
    with pytest.raises(ValueError, match="sample_shape"):
        ReplayBuffer(4, (2,)).add(torch.zeros(2, 3))


def test_state_dict_round_trip_restores_storage_and_position():
    buffer = ReplayBuffer(5, (1,), generator=torch.Generator().manual_seed(0))
    buffer.add(torch.arange(3.0).view(3, 1))
    restored = ReplayBuffer(5, (1,))
    restored.load_state_dict(buffer.state_dict())
    assert torch.equal(restored.storage, buffer.storage)
    restored.add(torch.tensor([[42.0]]))
    assert restored.storage[3].item() == 42.0


def test_mix_across_ranks_is_noop_single_process():
    buffer = ReplayBuffer(4, (2,))
    before = buffer.storage.clone()
    buffer.mix_across_ranks()
    assert torch.equal(buffer.storage, before)


def test_repr():
    assert repr(ReplayBuffer(8, (2,))) == (
        "ReplayBuffer(capacity=8, sample_shape=(2,), new_sample_ratio=0.05, "
        "init_noise_scale=1.0)"
    )


@requires_cuda
def test_to_moves_storage_and_samples():
    buffer = ReplayBuffer(4, (2,)).to("cuda")
    assert buffer.storage.device.type == "cuda"
    assert buffer.sample(2).device.type == "cuda"

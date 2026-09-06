import math

import pytest
import torch

from torchebm.models.components.embeddings import LabelEmbedder, MLPTimestepEmbedder


def _reference_freq_embedding(t, dim, max_period=10000):
    """Straightforward per-call computation used before the buffer was cached."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, device=t.device, dtype=torch.float32)
        / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


@pytest.mark.parametrize("freq_dim", [16, 128, 7])
def test_mlp_timestep_embedder_freq_buffer_matches_recompute(freq_dim):
    """The cached freq table must produce output identical to recomputing it."""
    torch.manual_seed(0)
    emb = MLPTimestepEmbedder(out_dim=32, frequency_embedding_size=freq_dim)
    t = torch.linspace(0.0, 999.0, steps=8)

    expected = emb.mlp(_reference_freq_embedding(t, freq_dim))
    got = emb(t)
    assert torch.allclose(got, expected, rtol=0, atol=0)


def test_mlp_timestep_embedder_registers_nonpersistent_freq_buffer():
    emb = MLPTimestepEmbedder(out_dim=16, frequency_embedding_size=32)
    # Buffer is registered and enumerated by .buffers()/.named_buffers().
    assert "freqs" in dict(emb.named_buffers())
    assert any(b is emb.freqs for b in emb.buffers())
    # Non-persistent: excluded from the state dict so checkpoints stay clean.
    assert "freqs" not in emb.state_dict()
    # Expected precomputed values.
    expected = torch.exp(
        -math.log(10000) * torch.arange(0, 16, dtype=torch.float32) / 16
    )
    assert torch.equal(emb.freqs, expected)


def test_mlp_timestep_embedder_freq_buffer_is_not_reallocated_per_call():
    emb = MLPTimestepEmbedder(out_dim=16, frequency_embedding_size=32)
    freq_id = id(emb.freqs)
    emb(torch.rand(4))
    emb(torch.rand(6))
    assert id(emb.freqs) == freq_id


def test_mlp_timestep_embedder_freq_buffer_moves_with_module():
    emb = MLPTimestepEmbedder(out_dim=16, frequency_embedding_size=32)
    # .to(dtype) moves the buffer just like parameters.
    emb64 = emb.to(torch.float64)
    assert emb64.freqs.dtype == torch.float64
    if torch.cuda.is_available():
        emb_cuda = MLPTimestepEmbedder(out_dim=16).cuda()
        assert emb_cuda.freqs.is_cuda


@pytest.mark.parametrize("out_dim", [32, 64])
@pytest.mark.parametrize("freq_dim", [16, 128])
def test_mlp_timestep_embedder_shape(out_dim, freq_dim):
    emb = MLPTimestepEmbedder(out_dim=out_dim, frequency_embedding_size=freq_dim)
    t = torch.rand(8)
    y = emb(t)
    assert y.shape == (8, out_dim)
    assert torch.isfinite(y).all()


def test_mlp_timestep_embedder_non_1d_input_is_reshaped():
    emb = MLPTimestepEmbedder(out_dim=16)
    t = torch.rand(4, 1)
    y = emb(t)
    assert y.shape == (4, 16)


def test_mlp_timestep_embedder_odd_frequency_dim():
    emb = MLPTimestepEmbedder(out_dim=8, frequency_embedding_size=7)
    t = torch.rand(3)
    y = emb(t)
    assert y.shape == (3, 8)


def test_mlp_timestep_embedder_gradient_flows():
    emb = MLPTimestepEmbedder(out_dim=16)
    t = torch.rand(4)
    y = emb(t).sum()
    y.backward()
    grads = [p.grad for p in emb.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any((g.abs() > 0).any() for g in grads)


def test_label_embedder_no_dropout_returns_embeddings():
    emb = LabelEmbedder(num_classes=10, out_dim=16, dropout_prob=0.0)
    labels = torch.randint(0, 10, (8,))
    out = emb(labels, training=True)
    assert out.shape == (8, 16)
    assert emb.null_label_id is None


def test_label_embedder_with_dropout_has_null_token():
    emb = LabelEmbedder(num_classes=10, out_dim=16, dropout_prob=0.5)
    assert emb.null_label_id == 10
    assert emb.embedding.num_embeddings == 11


def test_label_embedder_force_drop_uses_null_id():
    emb = LabelEmbedder(num_classes=4, out_dim=8, dropout_prob=0.1)
    labels = torch.tensor([0, 1, 2, 3])
    drop = torch.tensor([True, False, True, False])
    dropped = emb.maybe_drop_labels(labels, force_drop_mask=drop)
    assert dropped[0].item() == emb.null_label_id
    assert dropped[2].item() == emb.null_label_id
    assert dropped[1].item() == 1
    assert dropped[3].item() == 3


def test_label_embedder_no_dropout_maybe_drop_is_identity():
    emb = LabelEmbedder(num_classes=5, out_dim=8, dropout_prob=0.0)
    labels = torch.tensor([0, 1, 2, 3, 4])
    out = emb.maybe_drop_labels(labels)
    assert torch.equal(out, labels)


def test_label_embedder_forward_eval_no_force_mask_skips_drop():
    emb = LabelEmbedder(num_classes=4, out_dim=8, dropout_prob=1.0)
    labels = torch.tensor([0, 1, 2, 3])
    out = emb(labels, training=False)
    assert out.shape == (4, 8)


def test_label_embedder_dropout_raises_without_null_id():
    emb = LabelEmbedder(num_classes=4, out_dim=8, dropout_prob=0.0)
    emb.dropout_prob = 0.5
    labels = torch.tensor([0, 1])
    with pytest.raises(RuntimeError):
        emb.maybe_drop_labels(labels)


def test_label_embedder_null_token_without_dropout():
    emb = LabelEmbedder(num_classes=4, out_dim=8, dropout_prob=0.0, null_token=True)
    assert emb.null_label_id == 4
    assert emb.embedding.num_embeddings == 5
    labels = torch.tensor([0, 1, 2, 3])
    out = emb.maybe_drop_labels(labels)
    assert torch.equal(out, labels)


def test_label_embedder_force_drop_works_without_dropout():
    emb = LabelEmbedder(num_classes=4, out_dim=8, dropout_prob=0.0, null_token=True)
    labels = torch.tensor([0, 1, 2, 3])
    drop = torch.tensor([True, True, False, False])
    dropped = emb.maybe_drop_labels(labels, force_drop_mask=drop)
    assert dropped.tolist() == [4, 4, 2, 3]


def test_label_embedder_force_drop_without_null_raises():
    emb = LabelEmbedder(num_classes=4, out_dim=8, dropout_prob=0.0)
    labels = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="null token"):
        emb.maybe_drop_labels(labels, force_drop_mask=torch.tensor([True, False]))


def test_label_embedder_null_token_false_with_dropout_raises():
    with pytest.raises(ValueError, match="null token"):
        LabelEmbedder(num_classes=4, out_dim=8, dropout_prob=0.5, null_token=False)

import pytest
import torch

from torchebm.models.components.embeddings import LabelEmbedder, MLPTimestepEmbedder


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

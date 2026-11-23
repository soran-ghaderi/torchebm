import pytest
import torch
from torch import nn
from torch.testing import assert_close

from torchebm.losses import (
    EquilibriumMatchingLoss,
    MixedFieldMatchingLoss,
    compute_eqm_ct,
)
from torchebm.losses.functional import mixed_field_matching_loss

A, LAM, W = 0.8, 4.0, 0.1
BATCH, K = 16, 4


class Field(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, 3))

    def forward(self, x, t):
        return self.net(torch.cat([x, t[:, None]], dim=-1))


def _c_weight(t, a):
    r"""Port of ``c_weight`` (mixed-field e2e04f00, ``sampling/mixed.py``)."""
    return torch.where(t <= a, torch.ones_like(t), (1.0 - t) / (1.0 - a))


def _reference_field(model, x, t):
    r"""The reference image field: torchebm's EqM sign (away from the data), with
    the time input held at \(\min(t, a)\) (``_FixedTimeInput``, ``methods.py``)."""
    return -model(x, t.clamp(max=A))


def _reference_in_batch_loss(model, x0, x1, t):
    r"""Port of the in-batch estimator of ``MixedImageAdapter`` (``methods.py``):
    the EqM loss with target \(\lambda c_a(t)(x_0 - x_1)\) and the per-sample
    weights of ``_EndpointWeights``."""
    b = t.shape[0]
    x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
    target = (LAM * _c_weight(t, A))[:, None] * (x0 - x1)
    weight = torch.where(
        t >= 1.0, torch.full_like(t, W * b / K), torch.full_like(t, b / (b - K))
    )
    return (weight * (_reference_field(model, x_t, t) - target).square().mean(1)).mean()


def _reference_separate_batch_loss(model, x0, x1, t):
    r"""Port of the separate-batch estimator of ``EqMImageAdapter.loss_step``
    (``methods.py``): the EqM loss plus \(w\) times the mean square of the field
    at the clean examples with time input \(a\)."""
    x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
    target = (LAM * _c_weight(t, A))[:, None] * (x0 - x1)
    loss = (_reference_field(model, x_t, t) - target).square().mean()
    penalty = _reference_field(model, x1, torch.ones_like(t)).square().mean()
    return loss + W * penalty


def _batch(n=BATCH, clean=K):
    g = torch.Generator().manual_seed(0)
    x1 = torch.randn(n, 3, generator=g, dtype=torch.float64)
    x0 = torch.randn(n, 3, generator=g, dtype=torch.float64)
    t = MixedFieldMatchingLoss(endpoint_penalty="in_batch", n_clean=clean).sample_times(
        n, dtype=torch.float64, generator=g
    )
    return x0, x1, t


def _model():
    torch.manual_seed(0)
    return Field().double()


def _grads(model):
    return [p.grad.clone() for p in model.parameters()]


def _loss(input, target, t, **kwargs):
    ct = compute_eqm_ct(t, threshold=A, multiplier=LAM)
    return mixed_field_matching_loss(
        input, target, t, ct, endpoint_penalty="in_batch", endpoint_weight=W, **kwargs
    )


def test_in_batch_penalty_matches_reference():
    x0, x1, t = _batch()
    model = _model()
    x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
    loss_fn = MixedFieldMatchingLoss(endpoint_penalty="in_batch")
    loss = loss_fn(model(x_t, t.clamp(max=A)), x1 - x0, t)
    loss.backward()
    grads = _grads(model)
    model.zero_grad()
    ref = _reference_in_batch_loss(model, x0, x1, t)
    ref.backward()
    assert_close(loss, ref, rtol=1e-12, atol=0)
    for g, r in zip(grads, _grads(model)):
        assert_close(g, r, rtol=1e-10, atol=1e-14)


def test_separate_penalty_matches_reference():
    x0, x1, t = _batch(clean=0)
    model = _model()
    x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
    loss_fn = MixedFieldMatchingLoss(endpoint_penalty="separate")
    clean = model(x1, torch.full_like(t, A))
    loss = loss_fn(model(x_t, t.clamp(max=A)), x1 - x0, t, clean=clean)
    loss.backward()
    grads = _grads(model)
    model.zero_grad()
    ref = _reference_separate_batch_loss(model, x0, x1, t)
    ref.backward()
    assert_close(loss, ref, rtol=1e-12, atol=0)
    for g, r in zip(grads, _grads(model)):
        assert_close(g, r, rtol=1e-10, atol=1e-14)


def test_sample_times_put_the_clean_rows_last():
    g = torch.Generator().manual_seed(0)
    t = MixedFieldMatchingLoss(endpoint_penalty="in_batch", n_clean=K).sample_times(
        BATCH, generator=g
    )
    assert (t[-K:] == 1.0).all() and (t[:-K] < 1.0).all()
    for endpoint_penalty in ("separate", None):
        loss_fn = MixedFieldMatchingLoss(endpoint_penalty=endpoint_penalty)
        assert (loss_fn.sample_times(BATCH, generator=g) < 1.0).all()


def test_without_clean_rows_is_truncated_equilibrium_matching():
    x0, x1, t = _batch(clean=0)
    input = torch.randn_like(x1)
    expected = EquilibriumMatchingLoss(ct_threshold=A, ct_multiplier=LAM)(
        input, x1 - x0, t
    )
    assert_close(_loss(input, x1 - x0, t), expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("ct", ["truncated", "linear", "constant", torch.cos])
def test_ct_matches_equilibrium_matching(ct):
    x0, x1, t = _batch(clean=0)
    input = torch.randn_like(x1)
    kwargs = dict(ct=ct, ct_threshold=A, ct_multiplier=LAM)
    expected = EquilibriumMatchingLoss(**kwargs)(input, x1 - x0, t)
    loss = MixedFieldMatchingLoss(**kwargs)(input, x1 - x0, t)
    assert_close(loss, expected, rtol=1e-12, atol=0)


def test_in_batch_penalty_has_a_zero_target_for_every_ct():
    x0, x1, t = _batch()
    input = torch.randn_like(x1)
    noisy = t < 1.0
    expected = (input - LAM * (x1 - x0))[noisy].square().mean()
    expected = expected + W * input[~noisy].square().mean()
    loss_fn = MixedFieldMatchingLoss(
        ct="constant", ct_multiplier=LAM, endpoint_penalty="in_batch"
    )
    assert_close(loss_fn(input, x1 - x0, t), expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("clean", [1, 4, 15])
def test_each_term_is_averaged_over_its_own_rows(clean):
    x0, x1, t = _batch(clean=clean)
    input = torch.randn_like(x1)
    error = (input - (LAM * _c_weight(t, A))[:, None] * (x1 - x0)).square().mean(1)
    noisy = t < 1.0
    expected = error[noisy].mean() + W * error[~noisy].mean()
    assert_close(_loss(input, x1 - x0, t), expected, rtol=1e-12, atol=0)
    assert_close(
        _loss(input, x1 - x0, t, reduction="none").mean(), expected, rtol=1e-12, atol=0
    )


def test_clean_is_taken_by_the_separate_penalty_only():
    x0, x1, t = _batch()
    input = torch.randn_like(x1)
    with pytest.raises(TypeError, match="clean"):
        MixedFieldMatchingLoss(endpoint_penalty="separate")(input, x1 - x0, t)
    for endpoint_penalty in ("in_batch", None):
        with pytest.raises(TypeError, match="clean"):
            MixedFieldMatchingLoss(endpoint_penalty=endpoint_penalty)(
                input, x1 - x0, t, clean=input
            )


def test_bf16_field_gives_a_float32_loss():
    x0, x1, t = _batch()
    input = torch.randn_like(x1)
    loss = _loss(input.bfloat16(), (x1 - x0).float(), t.float())
    assert loss.dtype == torch.float32
    assert_close(loss.double(), _loss(input, x1 - x0, t), rtol=1e-2, atol=0)


@pytest.mark.parametrize("endpoint_penalty", ["in_batch", "separate", None])
def test_compiles_without_graph_breaks(endpoint_penalty):
    x0, x1, t = _batch()
    input = torch.randn_like(x1)
    clean = torch.randn_like(x1) if endpoint_penalty == "separate" else None
    loss_fn = MixedFieldMatchingLoss(endpoint_penalty=endpoint_penalty)
    compiled = torch.compile(loss_fn, backend="eager", fullgraph=True)
    assert_close(compiled(input, x1 - x0, t, clean), loss_fn(input, x1 - x0, t, clean))


def test_invalid_arguments_raise():
    with pytest.raises(ValueError, match="ct must"):
        MixedFieldMatchingLoss(ct="cosine")
    with pytest.raises(ValueError, match="ct_threshold"):
        MixedFieldMatchingLoss(ct_threshold=1.0)
    with pytest.raises(ValueError, match="endpoint_penalty"):
        MixedFieldMatchingLoss(endpoint_penalty="unweighted")

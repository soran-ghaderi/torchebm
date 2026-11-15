import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

from torchebm.losses import FlowMatchingLoss, ShortcutLoss
from torchebm.losses.functional import shortcut_loss

M = 128
LOG2_M = 7


class AverageVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 2))

    def forward(self, x, t, s):
        return 20.0 * self.net(torch.cat([x, s[:, None], (t - s)[:, None]], dim=-1))


def _reference_loss(model, ema, x0, x1, t_bst, dt_base, t_flow, bootstrap_every):
    r"""`get_targets` and `loss_fn` of kvfrans/shortcut-models (targets_shortcut.py,
    train.py) in PyTorch, with the model called on (x, t, dt_base)."""

    def call(net, x, t, base):
        return net(x, t + 0.5**base, t)

    n = len(x1) // bootstrap_every
    dt = 1 / 2**dt_base
    dt_base_bootstrap = dt_base + 1
    dt_bootstrap = dt / 2
    x_t = (1 - (1 - 1e-5) * t_bst[:, None]) * x0[:n] + t_bst[:, None] * x1[:n]
    with torch.no_grad():
        v_b1 = call(ema, x_t, t_bst, dt_base_bootstrap)
        x_t2 = (x_t + dt_bootstrap[:, None] * v_b1).clamp(-4, 4)
        v_b2 = call(ema, x_t2, t_bst + dt_bootstrap, dt_base_bootstrap)
        v_target = ((v_b1 + v_b2) / 2).clamp(-4, 4)
    x_f = (1 - (1 - 1e-5) * t_flow[:, None]) * x0[n:] + t_flow[:, None] * x1[n:]
    v_f = x1[n:] - (1 - 1e-5) * x0[n:]
    x_all = torch.cat([x_t, x_f])
    t_all = torch.cat([t_bst, t_flow])
    base_all = torch.cat([dt_base, torch.full_like(t_flow, LOG2_M)])
    v_all = torch.cat([v_target, v_f])
    v_prime = call(model, x_all, t_all, base_all)
    return ((v_prime - v_all) ** 2).mean(dim=1).mean()


def _batch(batch_size=64, bootstrap_every=8):
    g = torch.Generator().manual_seed(0)
    x1 = torch.randn(batch_size, 2, generator=g, dtype=torch.float64) * 2
    x0 = torch.randn(batch_size, 2, generator=g, dtype=torch.float64)
    n = batch_size // bootstrap_every
    dt_base = (LOG2_M - 1 - torch.arange(LOG2_M)).repeat_interleave(n // LOG2_M)
    dt_base = torch.cat([dt_base, dt_base.new_zeros(n - len(dt_base))]).double()
    sections = 2.0**dt_base
    t_bst = torch.floor(torch.rand(n, generator=g, dtype=torch.float64) * sections) / sections
    t_flow = torch.randint(M, (batch_size - n,), generator=g).double() / M
    return x0, x1, t_bst, dt_base, t_flow


def test_matches_reference_loss_and_gradients():
    torch.manual_seed(0)
    model = AverageVelocity().double()
    ema = AverageVelocity().double()
    x0, x1, t_bst, dt_base, t_flow = _batch()
    reference = _reference_loss(model, ema, x0, x1, t_bst, dt_base, t_flow, 8)
    reference_grads = torch.autograd.grad(reference, list(model.parameters()))

    n = len(t_bst)
    d = 0.5 ** (dt_base + 1)
    x_t = (1 - (1 - 1e-5) * t_bst[:, None]) * x0[:n] + t_bst[:, None] * x1[:n]
    with torch.no_grad():
        u1 = ema(x_t, t_bst + d, t_bst)
        u2 = ema((x_t + d[:, None] * u1).clamp(-4, 4), t_bst + 2 * d, t_bst + d)
    x_f = (1 - (1 - 1e-5) * t_flow[:, None]) * x0[n:] + t_flow[:, None] * x1[n:]
    v = x1[n:] - (1 - 1e-5) * x0[n:]
    loss_fn = ShortcutLoss(target_clip=4.0, reduction="none")
    fm_loss_fn = FlowMatchingLoss(reduction="none")
    loss = torch.cat(
        [
            loss_fn(model(x_t, t_bst + 2 * d, t_bst), u1, u2),
            fm_loss_fn(model(x_f, t_flow + 1 / M, t_flow), v),
        ]
    ).mean()
    grads = torch.autograd.grad(loss, list(model.parameters()))

    assert u1.abs().max() > 4
    assert_close(loss, reference)
    for grad, reference_grad in zip(grads, reference_grads):
        assert_close(grad, reference_grad)


def test_target_is_detached_mean():
    input = torch.randn(8, 3, requires_grad=True)
    first = torch.randn(8, 3, requires_grad=True)
    second = torch.randn(8, 3, requires_grad=True)
    loss = shortcut_loss(input, first, second)
    loss.backward()
    assert first.grad is None and second.grad is None
    target = 0.5 * (first + second)
    assert_close(loss, ((input - target) ** 2).mean())


def test_weight_and_reduction():
    input, first, second = torch.randn(3, 8, 2, 2).unbind(0)
    weight = torch.rand(8)
    per_sample = ShortcutLoss(reduction="none")(input, first, second)
    assert per_sample.shape == (8,)
    assert_close(
        ShortcutLoss(reduction="sum")(input, first, second, weight=weight),
        (per_sample * weight).sum(),
    )


def test_target_clip():
    input = torch.zeros(4, 2)
    first = torch.full((4, 2), 10.0)
    loss = ShortcutLoss(target_clip=4.0)(input, first, first)
    assert_close(loss, torch.tensor(16.0))


def test_compile():
    input, first, second = torch.randn(3, 8, 2).unbind(0)
    loss_fn = ShortcutLoss(target_clip=4.0)
    compiled = torch.compile(loss_fn, fullgraph=True)
    assert_close(compiled(input, first, second), loss_fn(input, first, second))


def test_unknown_argument_raises():
    with pytest.raises(TypeError):
        ShortcutLoss(clip=4.0)

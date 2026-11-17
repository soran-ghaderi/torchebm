import pytest
import torch
from torch.testing import assert_close

from torchebm.interpolants import LinearInterpolant
from torchebm.losses import MeanFlowLoss
from torchebm.losses.functional import adaptive_weight, meanflow_loss
from torchebm.models import DiT, TimeConditionedMLP
from torchebm.utils import average_velocity_jvp, sample_guidance

NUM_CLASSES = 4
EQUAL_RATIO = 0.25
IN_CONTEXT_OPTIONS = dict(
    conditioning="in_context",
    pos_embed=None,
    qk_norm=True,
    swiglu=True,
    rope="1d",
    rms_norm=True,
    block_bias=False,
    time_invariant=True,
)


def _model(kind, **options):
    torch.manual_seed(0)
    if kind == "mlp":
        model = TimeConditionedMLP(
            in_dim=6, num_classes=NUM_CLASSES, prediction_type="average_velocity", **options
        )
    else:
        if kind == "in_context_dit":
            options = {**IN_CONTEXT_OPTIONS, **options}
        model = DiT(
            input_size=8,
            in_channels=3,
            embed_dim=32,
            depth=2,
            num_heads=2,
            num_classes=NUM_CLASSES,
            prediction_type="average_velocity",
            **options,
        )
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return model.double()


def _batch(kind, n=12):
    g = torch.Generator().manual_seed(1)
    shape = (n, 6) if kind == "mlp" else (n, 3, 8, 8)
    x = torch.randn(shape, generator=g, dtype=torch.float64)
    e = torch.randn(shape, generator=g, dtype=torch.float64)
    t = torch.randint(65, (n,), generator=g).double() / 64
    r = torch.randint(65, (n,), generator=g).double() / 64
    t, r = torch.maximum(t, r), torch.minimum(t, r)
    t[-1], r[-1] = 0.75, 0.25
    r[: n // 4] = t[: n // 4]
    labels = torch.randint(NUM_CLASSES, (n,), generator=g)
    drop = torch.arange(n) < 2
    return x, e, t, r, labels, drop


def _view(a, x):
    return a.reshape(-1, *(1,) * (x.ndim - 1))


def _reference_net(model):
    r"""``u_fn(z, t, h)`` of the reference code (data at t = 0) from a torchebm model."""

    def net(z, t, h, y):
        return -model(z, 1 - t + h, y, s=1 - t)

    return net


def _adaptive_mean(loss, norm_p, norm_eps=0.01):
    adp_wt = (loss + norm_eps) ** norm_p
    return (loss / adp_wt.detach()).mean()


def _meanflow_reference(net, x, e, t, r, labels, drop, omega, kappa, t_start, t_end, norm_p):
    r"""`MeanFlow.forward` of Gsunshine/meanflow (d70cb55), in PyTorch, with the draws passed in."""
    tb = _view(t, x)
    z_t = (1 - tb) * x + tb * e
    v = e - x
    zeros = torch.zeros_like(t)
    y_null = torch.full_like(labels, NUM_CLASSES)
    v_uncond = net(z_t, t, zeros, y_null)
    v_cond = net(z_t, t, zeros, labels)
    inside = _view((t >= t_start) & (t <= t_end), x)
    omega = torch.where(inside, x.new_tensor(omega), x.new_tensor(1.0))
    kappa = torch.where(inside, x.new_tensor(kappa), x.new_tensor(0.0))
    v_g = omega * v + (1 - omega - kappa) * v_uncond + kappa * v_cond
    y_inp = torch.where(drop, NUM_CLASSES, labels)
    v_g = torch.where(_view(drop, x), v, v_g)
    u, du_dt = torch.func.jvp(
        lambda z, t, r: net(z, t, t - r, y_inp),
        (z_t, t, r),
        (v_g, torch.ones_like(t), torch.zeros_like(r)),
    )
    u_tgt = (v_g - _view(torch.clip(t - r, 0.0, 1.0), x) * du_dt).detach()
    loss = ((u - u_tgt) ** 2).flatten(1).sum(1)
    return _adaptive_mean(loss, norm_p)


def _imf_reference(net, x, e, t, r, labels, norm_p):
    r"""`iMeanFlow.forward` of Lyy-iiis/imeanflow (bf60cd7) without guidance, with
    \(v_\theta(z_t) = u_\theta(z_t, t, t)\) (Algorithm 1) in place of the auxiliary head."""
    tb = _view(t, x)
    z_t = (1 - tb) * x + tb * e
    v_g = e - x
    v_c = net(z_t, t, torch.zeros_like(t), labels)
    u, du_dt = torch.func.jvp(
        lambda z, t, r: net(z, t, t - r, labels),
        (z_t, t, r),
        (v_c, torch.ones_like(t), torch.zeros_like(r)),
    )
    V = u + _view(t - r, x) * du_dt.detach()
    loss_u = ((V - v_g.detach()) ** 2).flatten(1).sum(1)
    return _adaptive_mean(loss_u, norm_p)


def _guided_reference_net(model):
    r"""``net(x, t, h, w, t_min, t_max, y)`` of the reference code (data at t = 0) from a torchebm model."""

    def net(z, t, h, w, t_min, t_max, y):
        interval = torch.stack([1 - t_max, 1 - t_min], dim=-1)
        return -model(z, 1 - t + h, y, s=1 - t, guidance_scale=w, guidance_interval=interval)

    return net


def _imf_guidance_reference(net, x, e, t, r, labels, drop, omega, t_min, t_max, norm_p):
    r"""`iMeanFlow.forward` and `guidance_fn` of Lyy-iiis/imeanflow (bf60cd7c) with
    \(v_\theta(z_t) = u_\theta(z_t, t, t)\) in place of the auxiliary head and the draws passed in."""
    tb = _view(t, x)
    z_t = (1 - tb) * x + tb * e
    v_t = e - x
    fm_mask = _view(torch.arange(len(x)) < int(len(x) * EQUAL_RATIO), x)

    def v_cond_fn(z, t, w, y):
        return net(z, t, torch.zeros_like(t), w, torch.zeros_like(t), torch.ones_like(t), y)

    v_c, v_u = v_cond_fn(
        torch.cat([z_t, z_t]),
        torch.cat([t, t]),
        torch.cat([omega, torch.ones_like(omega)]),
        torch.cat([labels, torch.full_like(labels, NUM_CLASSES)]),
    ).chunk(2)
    v_g_fm = v_t + _view(1 - 1 / omega, x) * (v_c - v_u)
    w = torch.where((t >= t_min) & (t <= t_max), omega, 1.0)
    v_c = v_cond_fn(z_t, t, w, labels)
    v_g = v_t + _view(1 - 1 / w, x) * (v_c - v_u)
    v_g = torch.where(fm_mask, v_g_fm, v_g)
    labels = torch.where(drop, NUM_CLASSES, labels)
    v_g = torch.where(_view(drop, x), v_t, v_g)
    u, du_dt = torch.func.jvp(
        lambda z, t, r: net(z, t, t - r, omega, t_min, t_max, labels),
        (z_t, t, r),
        (v_c, torch.ones_like(t), torch.zeros_like(r)),
    )
    V = u + _view(t - r, x) * du_dt.detach()
    loss_u = ((V - v_g.detach()) ** 2).flatten(1).sum(1)
    return _adaptive_mean(loss_u, norm_p)


def _grads(model, loss):
    return torch.autograd.grad(loss, list(model.parameters()))


@pytest.mark.parametrize("ratio", [0.0, EQUAL_RATIO])
@pytest.mark.parametrize("p", [0.0, 1.0])
@pytest.mark.parametrize("kind", ["mlp", "dit"])
def test_matches_meanflow_reference_with_guidance(kind, p, ratio):
    model = _model(kind)
    x, e, t_ref, r_ref, labels, drop = _batch(kind)
    omega, kappa, t_end = 0.2, 0.92, 0.75
    reference = _meanflow_reference(
        _reference_net(model), x, e, t_ref, r_ref, labels, drop, omega, kappa, 0.0, t_end, p
    )

    s, t = 1 - t_ref, 1 - r_ref
    x_s, v = LinearInterpolant()(e, x, s)
    null = torch.full_like(labels, NUM_CLASSES)
    with torch.no_grad():
        v_cond = model(x_s, s, labels, s=s)
        v_uncond = model(x_s, s, null, s=s)
    guided = _view(s >= 1 - t_end, x)
    v_guided = torch.where(
        guided, omega * v + kappa * v_cond + (1 - omega - kappa) * v_uncond, v
    )
    y = torch.where(drop, null, labels)
    v_guided = torch.where(_view(drop, x), v, v_guided)
    u, du_ds = average_velocity_jvp(model, x_s, s, t, v_guided, y, flow_matching_ratio=ratio)
    loss = MeanFlowLoss(p=p)(u, du_ds, v_guided, s, t)

    assert_close(loss, reference)
    for grad, reference_grad in zip(_grads(model, loss), _grads(model, reference)):
        assert_close(grad, reference_grad)


@pytest.mark.parametrize("ratio", [0.0, EQUAL_RATIO])
@pytest.mark.parametrize("p", [0.0, 1.0])
def test_matches_imf_reference(p, ratio):
    model = _model("mlp")
    x, e, t_ref, r_ref, labels, _ = _batch("mlp")
    reference = _imf_reference(_reference_net(model), x, e, t_ref, r_ref, labels, p)

    s, t = 1 - t_ref, 1 - r_ref
    x_s, v = LinearInterpolant()(e, x, s)
    with torch.no_grad():
        v_model = model(x_s, s, labels, s=s)
    u, du_ds = average_velocity_jvp(model, x_s, s, t, v_model, labels, flow_matching_ratio=ratio)
    loss = MeanFlowLoss(p=p)(u, du_ds, v, s, t)

    assert_close(loss, reference)
    for grad, reference_grad in zip(_grads(model, loss), _grads(model, reference)):
        assert_close(grad, reference_grad)


@pytest.mark.parametrize("ratio", [0.0, EQUAL_RATIO])
@pytest.mark.parametrize("p", [0.0, 1.0])
@pytest.mark.parametrize("kind", ["mlp", "dit", "in_context_dit"])
def test_matches_imf_reference_with_guidance_conditioning(kind, p, ratio):
    model = _model(kind, guidance_conditioning="scale_and_interval")
    x, e, t_ref, r_ref, labels, drop = _batch(kind)
    n = len(x)
    scale, interval = sample_guidance(
        n,
        flow_matching_ratio=EQUAL_RATIO,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(2),
    )
    t_min, t_max = 1 - interval[:, 1], 1 - interval[:, 0]
    reference = _imf_guidance_reference(
        _guided_reference_net(model), x, e, t_ref, r_ref, labels, drop, scale, t_min, t_max, p
    )

    s, t = 1 - t_ref, 1 - r_ref
    x_s, v = LinearInterpolant()(e, x, s)
    null = torch.full_like(labels, NUM_CLASSES)
    inside = (s >= interval[:, 0]) & (s <= interval[:, 1])
    w = torch.where(inside, scale, 1.0)
    with torch.no_grad():
        v_c, v_u = model(
            x_s.repeat(2, *(1,) * (x.ndim - 1)),
            s.repeat(2),
            torch.cat([labels, null]),
            s=s.repeat(2),
            guidance_scale=torch.cat([w, torch.ones_like(w)]),
            guidance_interval=x.new_tensor([0.0, 1.0]).expand(2 * n, 2),
        ).chunk(2)
    v_guided = v + _view(1 - 1 / w, x) * (v_c - v_u)
    y = torch.where(drop, null, labels)
    v_guided = torch.where(_view(drop, x), v, v_guided)
    u, du_ds = average_velocity_jvp(
        model,
        x_s,
        s,
        t,
        v_c,
        y,
        flow_matching_ratio=ratio,
        model_kwargs={"guidance_scale": scale, "guidance_interval": interval},
    )
    loss = MeanFlowLoss(p=p)(u, du_ds, v_guided, s, t)

    assert_close(loss, reference)
    for grad, reference_grad in zip(_grads(model, loss), _grads(model, reference)):
        assert_close(grad, reference_grad)


@pytest.mark.parametrize("kind", ["mlp", "dit"])
def test_average_velocity_jvp_matches_full_batch_jvp(kind):
    model = _model(kind)
    x, e, t_ref, r_ref, labels, _ = _batch(kind)
    s, t = 1 - t_ref, 1 - r_ref
    x_s, v = LinearInterpolant()(e, x, s)
    u_ref, du_ref = torch.func.jvp(
        lambda x_s, s: model(x_s, t, labels, s=s), (x_s, s), (v, torch.ones_like(s))
    )

    u, du_ds = average_velocity_jvp(model, x_s, s, t, v, labels, flow_matching_ratio=EQUAL_RATIO)
    n = int(len(x) * EQUAL_RATIO)

    assert not du_ds.requires_grad
    assert_close(u, u_ref)
    assert_close(du_ds[n:], du_ref[n:])
    assert torch.equal(du_ds[:n], torch.zeros_like(du_ds[:n]))


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def _relative_error(a, b):
    return ((a.double().cpu() - b).norm() / b.norm()).item()


@requires_cuda
@pytest.mark.usefixtures("highest_matmul_precision")
@pytest.mark.parametrize(
    "autocast, compile, rtol", [(False, False, 1e-4), (True, False, 5e-2), (False, True, 1e-4)]
)
def test_cuda_matches_meanflow_reference(autocast, compile, rtol):
    model = _model("dit")
    x, e, t_ref, r_ref, labels, drop = _batch("dit")
    no_drop = torch.zeros_like(drop)
    reference = _meanflow_reference(
        _reference_net(model), x, e, t_ref, r_ref, labels, no_drop, 1.0, 0.0, 0.0, 1.0, 0.0
    )
    reference_grads = torch.cat([g.flatten() for g in _grads(model, reference)])

    model = model.to("cuda", torch.float32)
    x, e, s, t = (a.to("cuda", torch.float32) for a in (x, e, 1 - t_ref, 1 - r_ref))
    labels = labels.cuda()
    path, loss_fn = LinearInterpolant(), MeanFlowLoss(p=0.0)

    def step(x, e, s, t):
        x_s, v = path(e, x, s)
        u, du_ds = average_velocity_jvp(model, x_s, s, t, v, labels, flow_matching_ratio=EQUAL_RATIO)
        return loss_fn(u, du_ds, v, s, t)

    step = torch.compile(step, fullgraph=True) if compile else step
    with torch.autocast("cuda", torch.bfloat16, enabled=autocast):
        loss = step(x, e, s, t)
    grads = torch.cat([g.flatten() for g in _grads(model, loss)])

    assert _relative_error(loss, reference) < rtol
    assert _relative_error(grads, reference_grads) < rtol


def _outputs():
    g = torch.Generator().manual_seed(2)
    u, du_ds, v = torch.randn(3, 8, 3, 4, 4, generator=g, dtype=torch.float64)
    s = torch.rand(8, generator=g, dtype=torch.float64)
    t = s + (1 - s) * torch.rand(8, generator=g, dtype=torch.float64)
    return u.requires_grad_(), du_ds.requires_grad_(), v.requires_grad_(), s, t


def test_equal_times_is_flow_matching():
    u, du_ds, v, s, _ = _outputs()
    squared_error = (u - v).square().flatten(1).sum(1)
    expected = (adaptive_weight(squared_error) * squared_error).mean()
    assert_close(meanflow_loss(u, du_ds, v, s, s), expected)


def test_target_is_detached():
    u, du_ds, v, s, t = _outputs()
    grads = torch.autograd.grad(meanflow_loss(u, du_ds, v, s, t), (u, du_ds, v), allow_unused=True)
    assert grads[0] is not None
    assert grads[1] is None and grads[2] is None


def test_p_zero_weight_and_reduction():
    u, du_ds, v, s, t = _outputs()
    target = v + (t - s).view(-1, 1, 1, 1) * du_ds
    squared_error = (u - target).square().flatten(1).sum(1)
    per_sample = meanflow_loss(u, du_ds, v, s, t, p=0.0, reduction="none")
    assert per_sample.shape == (8,)
    assert_close(per_sample, squared_error)
    weight = torch.rand(8, dtype=torch.float64)
    assert_close(
        MeanFlowLoss(p=0.0, reduction="sum")(u, du_ds, v, s, t, weight=weight),
        (squared_error * weight).sum(),
    )


def test_compile():
    u, du_ds, v, s, t = (a.detach() for a in _outputs())
    compiled = torch.compile(meanflow_loss, fullgraph=True)
    assert_close(compiled(u, du_ds, v, s, t), meanflow_loss(u, du_ds, v, s, t))

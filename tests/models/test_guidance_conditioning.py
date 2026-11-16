import pytest
import torch

from torchebm.models import DiT, TimeConditionedMLP
from torchebm.models.components import MLPTimestepEmbedder
from torchebm.samplers import FlowMapSampler
from torchebm.utils import average_velocity_jvp, sample_guidance

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def _dit(**overrides):
    cfg = dict(
        input_size=8,
        patch_size=2,
        in_channels=3,
        embed_dim=32,
        depth=2,
        num_heads=2,
        num_classes=4,
        prediction_type="average_velocity",
        guidance_conditioning="scale_and_interval",
    )
    cfg.update(overrides)
    torch.manual_seed(0)
    model = DiT(**cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return model


def _inputs(batch=4):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(batch, 3, 8, 8, generator=g)
    s = torch.rand(batch, generator=g)
    t = s + (1 - s) * torch.rand(batch, generator=g)
    y = torch.arange(batch) % 4
    scale, interval = sample_guidance(batch, flow_matching_ratio=0.0, generator=g)
    return x, s, t, y, scale, interval


def test_guidance_inputs_change_the_output():
    model = _dit()
    x, s, t, y, scale, interval = _inputs()
    with torch.no_grad():
        out = model(x, t, y, s=s, guidance_scale=scale, guidance_interval=interval)
        other_scale = model(x, t, y, s=s, guidance_scale=scale + 1, guidance_interval=interval)
        other_interval = model(x, t, y, s=s, guidance_scale=scale, guidance_interval=interval.flip(-1))
    assert not torch.allclose(out, other_scale)
    assert not torch.allclose(out, other_interval)


def test_default_guidance_embedder_embeds_one_minus_inverse_scale():
    model = _dit()
    scale = torch.tensor([1.0, 1.5, 7.5])
    expected = MLPTimestepEmbedder.forward(model.guidance_embedder, 1 - 1 / scale)
    assert torch.equal(model.guidance_embedder(scale), expected)


def test_custom_guidance_embedder_is_used_and_keeps_its_init():
    embedder = MLPTimestepEmbedder(32)
    with torch.no_grad():
        embedder.mlp[0].weight.fill_(0.5)
    model = DiT(
        input_size=8, in_channels=3, patch_size=2, embed_dim=32, depth=1, num_heads=2,
        guidance_conditioning="scale", guidance_embedder=embedder,
    )
    assert model.guidance_embedder is embedder
    assert torch.all(embedder.mlp[0].weight == 0.5)
    assert model.interval_start_embedder is None


def test_defaults_build_no_guidance_embedders():
    model = DiT(input_size=8, in_channels=3, patch_size=2, embed_dim=32, depth=1, num_heads=2)
    assert model.guidance_embedder is None and model.interval_start_embedder is None
    assert not any("guidance" in k or "interval" in k for k in model.state_dict())


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(guidance_conditioning="interval"), "guidance_conditioning must be one of"),
        (dict(guidance_embedder=MLPTimestepEmbedder(32)), "requires guidance_conditioning"),
    ],
)
def test_invalid_constructor_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        DiT(input_size=8, in_channels=3, patch_size=2, embed_dim=32, depth=1, num_heads=2, **kwargs)


def test_unconditional_mlp_rejects_guidance_conditioning():
    with pytest.raises(ValueError, match="requires cond_dim > 0"):
        TimeConditionedMLP(in_dim=2, conditioning="concat", cond_dim=0, guidance_conditioning="scale")


def test_forward_validates_guidance_inputs():
    x, s, t, y, scale, interval = _inputs()
    model = _dit()
    with pytest.raises(ValueError, match="requires guidance_scale"):
        model(x, t, y, s=s, guidance_interval=interval)
    with pytest.raises(ValueError, match="requires guidance_interval"):
        model(x, t, y, s=s, guidance_scale=scale)
    with pytest.raises(ValueError, match="requires guidance_conditioning='scale_and_interval'"):
        _dit(guidance_conditioning="scale")(x, t, y, s=s, guidance_scale=scale, guidance_interval=interval)
    with pytest.raises(ValueError, match="has no guidance conditioning"):
        _dit(guidance_conditioning=None)(x, t, y, s=s, guidance_scale=scale)


@pytest.mark.parametrize("n_steps", [1, 2, 4])
def test_sampling_at_a_fixed_scale_takes_one_forward_pass_per_step(n_steps):
    model = _dit().eval()
    calls = []
    model.register_forward_pre_hook(lambda module, args: calls.append(1))
    y = torch.arange(4) % 4
    model_kwargs = {
        "y": y,
        "guidance_scale": torch.full((4,), 7.5),
        "guidance_interval": torch.tensor([0.35, 0.6]).expand(4, 2),
    }
    samples = FlowMapSampler(prediction_type="average_velocity")(
        model, torch.randn(4, 3, 8, 8), n_steps=n_steps, model_kwargs=model_kwargs
    )
    assert samples.shape == (4, 3, 8, 8)
    assert len(calls) == n_steps


@pytest.mark.usefixtures("highest_matmul_precision")
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
def test_average_velocity_jvp_with_guidance_compiles(device):
    model = _dit().to(device)
    x, s, t, y, scale, interval = (a.to(device) for a in _inputs())
    t = torch.cat([s[:2], t[2:]])
    v = torch.randn_like(x)
    model_kwargs = {"guidance_scale": scale, "guidance_interval": interval}

    def step(x, s, t, v):
        return average_velocity_jvp(
            model, x, s, t, v, y, flow_matching_ratio=0.5, model_kwargs=model_kwargs
        )

    results = []
    for fn in (step, torch.compile(step, fullgraph=True)):
        u, du_ds = fn(x, s, t, v)
        grads = torch.autograd.grad(u.square().mean(), list(model.parameters()))
        results.append((u, du_ds, grads))
    (u, du_ds, grads), (c_u, c_du_ds, c_grads) = results
    torch.testing.assert_close(c_u, u, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(c_du_ds, du_ds, atol=1e-4, rtol=1e-4)
    assert torch.equal(du_ds[:2], torch.zeros_like(du_ds[:2]))
    for grad, c_grad in zip(grads, c_grads):
        torch.testing.assert_close(c_grad, grad, atol=1e-5, rtol=1e-4)

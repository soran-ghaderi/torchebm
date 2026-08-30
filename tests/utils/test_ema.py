import pytest
import torch
import torch.nn as nn

from torchebm.utils import EMA


def _model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 4))


class _Stateful(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.register_buffer("running", torch.zeros(4))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    def forward(self, x):
        return self.linear(x)


def _set_params(model, value):
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(value)


def test_copy_is_frozen_and_eval():
    model = _model()
    model.train()
    ema = EMA(model)
    assert not ema.module.training
    assert all(not p.requires_grad for p in ema.module.parameters())
    assert all(p.requires_grad for p in model.parameters())


def test_decay_validation():
    with pytest.raises(ValueError, match="decay"):
        EMA(_model(), decay=1.0)
    with pytest.raises(ValueError, match="decay"):
        EMA(_model(), decay=-0.1)
    with pytest.raises(ValueError, match="decay_schedule"):
        EMA(_model(), decay_schedule="linear")


def test_constant_decay_closed_form():
    model = _model()
    _set_params(model, 1.0)
    ema = EMA(model, decay=0.9)
    _set_params(model, 3.0)
    k = 5
    for _ in range(k):
        ema.update(model)
    expected = 3.0 + 0.9**k * (1.0 - 3.0)
    for e in ema.module.parameters():
        assert torch.allclose(e, torch.full_like(e, expected))
    assert ema.step == k


def test_update_matches_naive_loop():
    model = _model(seed=1)
    ema = EMA(model, decay=0.75)
    reference = [p.detach().clone() for p in model.parameters()]
    for step in range(3):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))
        ema.update(model)
        for r, p in zip(reference, model.parameters()):
            r.mul_(0.75).add_(p, alpha=0.25)
    for e, r in zip(ema.module.parameters(), reference):
        assert torch.allclose(e, r)


def test_warmup_schedule():
    model = _model()
    _set_params(model, 0.0)
    ema = EMA(model, decay=0.9999, decay_schedule="warmup")
    _set_params(model, 1.0)
    ema.update(model)  # step 0: d = min(0.9999, 1/10) = 0.1
    for e in ema.module.parameters():
        assert torch.allclose(e, torch.full_like(e, 0.9))
    assert ema._decay_value() == pytest.approx(2 / 11)  # step 1


def test_callable_schedule():
    model = _model()
    _set_params(model, 0.0)
    ema = EMA(model, decay_schedule=lambda step: 0.5)
    _set_params(model, 1.0)
    ema.update(model)
    for e in ema.module.parameters():
        assert torch.allclose(e, torch.full_like(e, 0.5))

    bad = EMA(_model(), decay_schedule=lambda step: 2.0)
    with pytest.raises(ValueError, match="decay_schedule returned"):
        bad.update(_model())


def test_buffers_are_copied_not_averaged():
    model = _Stateful()
    ema = EMA(model, decay=0.9)
    with torch.no_grad():
        model.running.fill_(5.0)
        model.count.fill_(7)
    ema.update(model)
    assert torch.equal(ema.module.running, model.running)
    assert torch.equal(ema.module.count, model.count)


def test_copy_to_preserves_requires_grad():
    model = _model()
    ema = EMA(model)
    target = _model(seed=2)
    ema.copy_to(target)
    for t, e in zip(target.parameters(), ema.module.parameters()):
        assert torch.equal(t, e)
        assert t.requires_grad


def test_average_parameters_restores_bitwise():
    model = _model()
    ema = EMA(model, decay=0.5)
    _set_params(model, 2.0)
    ema.update(model)
    original = [p.detach().clone() for p in model.parameters()]
    with ema.average_parameters(model):
        for p, e in zip(model.parameters(), ema.module.parameters()):
            assert torch.equal(p, e)
    for p, o in zip(model.parameters(), original):
        assert torch.equal(p, o)


def test_average_parameters_restores_on_exception():
    model = _model()
    ema = EMA(model, decay=0.5)
    _set_params(model, 2.0)
    ema.update(model)
    original = [p.detach().clone() for p in model.parameters()]
    with pytest.raises(RuntimeError, match="boom"):
        with ema.average_parameters(model):
            raise RuntimeError("boom")
    for p, o in zip(model.parameters(), original):
        assert torch.equal(p, o)


def test_state_dict_round_trip():
    model = _model()
    ema = EMA(model, decay=0.9)
    _set_params(model, 2.0)
    for _ in range(3):
        ema.update(model)
    state = ema.state_dict()

    restored = EMA(_model(seed=3), decay=0.9)
    restored.load_state_dict(state)
    assert restored.step == 3
    for a, b in zip(restored.module.parameters(), ema.module.parameters()):
        assert torch.equal(a, b)


def test_forward_delegates_to_module():
    model = _model()
    ema = EMA(model)
    x = torch.randn(2, 4)
    assert torch.equal(ema(x), ema.module(x))


def test_update_rebinds_to_new_instance():
    model_a = _model()
    ema = EMA(model_a, decay=0.5)
    model_b = _model(seed=4)
    _set_params(model_b, 1.0)
    _set_params(model_a, 0.0)
    ema2 = EMA(model_a, decay=0.5)
    ema2.update(model_b)
    for e in ema2.module.parameters():
        assert torch.allclose(e, torch.full_like(e, 0.5))
    assert ema is not ema2


def test_update_with_mismatched_structure_raises():
    ema = EMA(_model())
    with pytest.raises(ValueError):
        ema.update(nn.Linear(4, 4))


def test_buffer_only_model():
    class BufferOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("state", torch.zeros(3))

    model = BufferOnly()
    ema = EMA(model)
    with torch.no_grad():
        model.state.fill_(4.0)
    ema.update(model)
    assert torch.equal(ema.module.state, model.state)


def test_fallback_bucket_math_matches_foreach():
    model = _model(seed=5)
    ema = EMA(model, decay=0.8)
    twin = EMA(model, decay=0.8)
    twin._param_fallback = [
        (e, p) for elist, plist in twin._param_buckets for e, p in zip(elist, plist)
    ]
    twin._param_buckets = []
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p))
    ema.update(model)
    twin.update(model)
    for a, b in zip(ema.module.parameters(), twin.module.parameters()):
        assert torch.allclose(a, b)

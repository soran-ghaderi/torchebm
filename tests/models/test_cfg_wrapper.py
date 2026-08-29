"""ClassifierFreeGuidance wrapper: one batched forward, every sampler."""

import pytest
import torch
import torch.nn as nn

from torchebm.core import BaseModel
from torchebm.models import ClassifierFreeGuidance
from torchebm.samplers import FlowSampler, LangevinDynamics

NUM_CLASSES = 5


class CountingField(nn.Module):
    """Conditional field with a strong y dependence; counts calls."""

    def __init__(self, dim=2):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.embed = nn.Embedding(NUM_CLASSES + 1, dim)
        self.calls = 0
        self.batch_sizes = []

    def forward(self, x, t=None, y=None, **kwargs):
        self.calls += 1
        self.batch_sizes.append(x.shape[0])
        out = self.linear(x)
        if y is not None:
            out = out + self.embed(y)
        return out


class ConditionalEnergy(BaseModel):
    def __init__(self, dim=2):
        super().__init__()
        self.linear = nn.Linear(dim, 1)
        self.embed = nn.Embedding(NUM_CLASSES + 1, dim)

    def forward(self, x, y=None, **kwargs):
        shift = self.embed(y) if y is not None else 0.0
        return 0.5 * (x - shift).square().sum(dim=1) + self.linear(x).squeeze(-1)


def test_w_zero_equals_null_condition_forward():
    field = CountingField()
    guided = ClassifierFreeGuidance(
        field, guidance_scale=0.0, null_condition=NUM_CLASSES
    )
    x = torch.randn(6, 2)
    t = torch.rand(6)
    y = torch.randint(0, NUM_CLASSES, (6,))
    out = guided(x, t, y=y)
    expected = field(x, t, y=torch.full_like(y, NUM_CLASSES))
    assert torch.equal(out, expected)


def test_w_one_matches_conditional_forward():
    field = CountingField()
    guided = ClassifierFreeGuidance(
        field, guidance_scale=1.0, null_condition=NUM_CLASSES
    )
    x = torch.randn(6, 2)
    t = torch.rand(6)
    y = torch.randint(0, NUM_CLASSES, (6,))
    assert torch.allclose(guided(x, t, y=y), field(x, t, y=y), atol=1e-6)


def test_single_doubled_forward():
    field = CountingField()
    guided = ClassifierFreeGuidance(
        field, guidance_scale=2.0, null_condition=NUM_CLASSES
    )
    x = torch.randn(6, 2)
    guided(x, torch.rand(6), y=torch.randint(0, NUM_CLASSES, (6,)))
    assert field.calls == 1
    assert field.batch_sizes == [12]


def test_guidance_formula():
    field = CountingField()
    w = 3.0
    guided = ClassifierFreeGuidance(field, guidance_scale=w, null_condition=NUM_CLASSES)
    x = torch.randn(6, 2)
    t = torch.rand(6)
    y = torch.randint(0, NUM_CLASSES, (6,))
    out = guided(x, t, y=y)
    cond = field(x, t, y=y)
    uncond = field(x, t, y=torch.full_like(y, NUM_CLASSES))
    assert torch.allclose(out, uncond + w * (cond - uncond), atol=1e-6)


def test_unconditional_passthrough_without_y():
    field = CountingField()
    guided = ClassifierFreeGuidance(
        field, guidance_scale=2.0, null_condition=NUM_CLASSES
    )
    x = torch.randn(6, 2)
    t = torch.rand(6)
    out = guided(x, t)
    assert field.batch_sizes == [6]
    assert torch.equal(out, field(x, t))


def test_tensor_null_condition():
    class EmbField(nn.Module):
        def forward(self, x, t=None, y=None, **kwargs):
            return x + y

    guided = ClassifierFreeGuidance(
        EmbField(), guidance_scale=0.0, null_condition=torch.zeros(2)
    )
    x = torch.randn(6, 2)
    out = guided(x, torch.rand(6), y=torch.randn(6, 2))
    assert torch.equal(out, x)


def test_flow_sampler_w_zero_equals_null_labeled_sampling():
    field = CountingField()
    guided = ClassifierFreeGuidance(
        field, guidance_scale=0.0, null_condition=NUM_CLASSES
    )
    y = torch.randint(0, NUM_CLASSES, (4,))
    null_y = torch.full_like(y, NUM_CLASSES)

    s_guided = FlowSampler(guided, integrator="euler")
    s_plain = FlowSampler(field, integrator="euler")

    x0 = torch.randn(4, 2)
    out_guided = s_guided.sample(
        x=x0.clone(), n_steps=5, model_kwargs={"y": y}
    )
    out_plain = s_plain.sample(
        x=x0.clone(), n_steps=5, model_kwargs={"y": null_y}
    )
    assert torch.allclose(out_guided, out_plain, atol=1e-6)


def test_langevin_w_zero_equals_null_labeled_sampling():
    energy = ConditionalEnergy()
    guided = ClassifierFreeGuidance(
        energy, guidance_scale=0.0, null_condition=NUM_CLASSES
    )
    y = torch.randint(0, NUM_CLASSES, (4,))
    null_y = torch.full_like(y, NUM_CLASSES)

    x0 = torch.randn(4, 2)
    out_guided = LangevinDynamics(guided, step_size=0.01).sample(
        x=x0.clone(),
        n_steps=10,
        model_kwargs={"y": y},
        generator=torch.Generator().manual_seed(3),
    )
    out_plain = LangevinDynamics(energy, step_size=0.01).sample(
        x=x0.clone(),
        n_steps=10,
        model_kwargs={"y": null_y},
        generator=torch.Generator().manual_seed(3),
    )
    assert torch.allclose(out_guided, out_plain, atol=1e-6)


def test_langevin_guided_sampling_runs_with_positive_scale():
    energy = ConditionalEnergy()
    guided = ClassifierFreeGuidance(
        energy, guidance_scale=2.5, null_condition=NUM_CLASSES
    )
    y = torch.randint(0, NUM_CLASSES, (4,))
    out = LangevinDynamics(guided, step_size=0.01).sample(
        x=torch.randn(4, 2), n_steps=10, model_kwargs={"y": y}
    )
    assert torch.isfinite(out).all()

import torch
from torch import nn

from torchebm.models.wrappers import LabelClassifierFreeGuidance


class _DummyBase(nn.Module):
    def __init__(self, null_id: int, out_channels: int = 4):
        super().__init__()
        self.null_id = null_id
        self.out_channels = out_channels

    def forward(self, x, t, *, y):
        b, _, h, w = x.shape
        base = torch.zeros(b, self.out_channels, h, w)
        cond_marker = (y != self.null_id).float().view(b, 1, 1, 1)
        return base + cond_marker


def test_cfg_scale_one_returns_base_forward():
    base = _DummyBase(null_id=10)
    wrapped = LabelClassifierFreeGuidance(
        base, null_label_id=10, cfg_scale=1.0, guide_channels=3
    )
    x = torch.randn(2, 3, 4, 4)
    t = torch.rand(2)
    y = torch.tensor([1, 2])
    out = wrapped(x, t, y=y)
    assert torch.allclose(out, base(x, t, y=y))


def test_cfg_applies_guidance_to_first_channels_only():
    base = _DummyBase(null_id=10, out_channels=4)
    wrapped = LabelClassifierFreeGuidance(
        base, null_label_id=10, cfg_scale=2.0, guide_channels=3
    )
    x = torch.randn(1, 3, 4, 4)
    t = torch.rand(1)
    y = torch.tensor([1])
    out = wrapped(x, t, y=y)
    assert out.shape == (1, 4, 4, 4)
    # First 3 channels: uncond(0) + 2.0 * (cond(1) - uncond(0)) = 2.0
    assert torch.allclose(out[:, :3], torch.full_like(out[:, :3], 2.0))
    # Last channel: uncond == 0
    assert torch.allclose(out[:, 3:], torch.zeros_like(out[:, 3:]))


def test_cfg_guide_channels_exceeds_output_clamps():
    base = _DummyBase(null_id=5, out_channels=2)
    wrapped = LabelClassifierFreeGuidance(
        base, null_label_id=5, cfg_scale=3.0, guide_channels=10
    )
    x = torch.randn(1, 3, 2, 2)
    t = torch.rand(1)
    y = torch.tensor([0])
    out = wrapped(x, t, y=y)
    assert out.shape == (1, 2, 2, 2)
    assert torch.allclose(out, torch.full_like(out, 3.0))

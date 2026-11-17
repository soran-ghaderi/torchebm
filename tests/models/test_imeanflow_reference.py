r"""DiT with in-context conditioning against a port of the Improved MeanFlow model.

The reference is ``imfDiT`` of Lyy-iiis/imeanflow (bf60cd7c), with
``models/imfDiT.py``, ``models/embedder.py`` and ``models/torch_models.py``,
in PyTorch: the u branch (``shared_blocks`` then ``u_heads``) without the
auxiliary v head. Rotations and sinusoids run in the input precision instead
of float32, so the float64 comparison is exact up to rounding.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import assert_close

from torchebm.models import DiT
from torchebm.models.components import RotaryEmbedding1d

IMF_OPTIONS = dict(
    conditioning="in_context",
    pos_embed=None,
    qk_norm=True,
    swiglu=True,
    rope="1d",
    rms_norm=True,
    block_bias=False,
    guidance_conditioning="scale_and_interval",
    time_invariant=True,
    prediction_type="average_velocity",
)


class _RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        mean_square = x.square().mean(-1, keepdim=True)
        return (x * torch.reciprocal(torch.sqrt(mean_square + self.eps))).to(x.dtype) * self.weight


def _precompute_rope_freqs(dim, seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)
    return torch.complex(torch.cos(angles), torch.sin(angles))


def _apply_rotary_pos_emb(x, freqs_cis):
    dtype = torch.promote_types(x.dtype, torch.float32)
    x_complex = torch.view_as_complex(x.to(dtype).reshape(*x.shape[:-1], -1, 2).contiguous())
    x_rotated = x_complex * freqs_cis[None, :, None, :]
    return torch.view_as_real(x_rotated).flatten(-2).to(x.dtype)


class _TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        args = t[:, None] * freqs.to(t.dtype)[None]
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


class _RoPEAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = _RMSNorm(self.head_dim)
        self.k_norm = _RMSNorm(self.head_dim)

    def forward(self, x, rope_freqs):
        batch, seq_len, hidden_size = x.shape
        shape = (batch, seq_len, self.num_heads, self.head_dim)
        q = self.q_norm(self.q_proj(x).reshape(shape))
        k = self.k_norm(self.k_proj(x).reshape(shape))
        v = self.v_proj(x).reshape(shape)
        q = _apply_rotary_pos_emb(q, rope_freqs)
        k = _apply_rotary_pos_emb(k, rope_freqs)
        attn = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.out_proj(attn.transpose(1, 2).reshape(batch, seq_len, hidden_size))


class _SwiGLUMlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w3 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(hidden_features, in_features, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class _TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=8 / 3):
        super().__init__()
        self.norm1 = _RMSNorm(hidden_size)
        self.attn = _RoPEAttention(hidden_size, num_heads)
        self.norm2 = _RMSNorm(hidden_size)
        self.mlp = _SwiGLUMlp(hidden_size, int(hidden_size * mlp_ratio))
        self.attn_scale = nn.Parameter(torch.zeros(hidden_size))
        self.mlp_scale = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, rope_freqs):
        x = x + self.attn(self.norm1(x), rope_freqs) * self.attn_scale
        return x + self.mlp(self.norm2(x)) * self.mlp_scale


class _ImfDiT(nn.Module):
    r"""``imfDiT.__call__`` up to the u output, with the patch embedding of a torchebm `DiT`."""

    def __init__(self, patch_embed, input_size, patch_size, in_channels, hidden_size, depth, num_heads, num_classes):
        super().__init__()
        self.patch_embed = patch_embed
        self.patch_size = patch_size
        self.out_channels = in_channels
        self.h_embedder = _TimestepEmbedder(hidden_size)
        self.omega_embedder = _TimestepEmbedder(hidden_size)
        self.cfg_t_start_embedder = _TimestepEmbedder(hidden_size)
        self.cfg_t_end_embedder = _TimestepEmbedder(hidden_size)
        self.y_embedder = nn.Embedding(num_classes + 1, hidden_size)
        self.time_tokens = nn.Parameter(torch.zeros(4, hidden_size))
        self.class_tokens = nn.Parameter(torch.zeros(8, hidden_size))
        self.omega_tokens = nn.Parameter(torch.zeros(4, hidden_size))
        self.t_min_tokens = nn.Parameter(torch.zeros(2, hidden_size))
        self.t_max_tokens = nn.Parameter(torch.zeros(2, hidden_size))
        self.prefix_tokens = 8 + 4 + 2 * 2 + 4
        num_patches = (input_size // patch_size) ** 2
        self.rope_freqs = _precompute_rope_freqs(hidden_size // num_heads, num_patches + self.prefix_tokens)
        self.blocks = nn.ModuleList(_TransformerBlock(hidden_size, num_heads) for _ in range(depth))
        self.u_final_norm = _RMSNorm(hidden_size)
        self.u_final_linear = nn.Linear(hidden_size, patch_size * patch_size * in_channels)

    def unpatchify(self, x):
        c, p = self.out_channels, self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nhpwqc", x)
        return x.reshape(x.shape[0], h * p, w * p, c)

    def forward(self, x, h, w, t_min, t_max, y):
        seq = torch.cat(
            [
                self.class_tokens + self.y_embedder(y)[:, None],
                self.omega_tokens + self.omega_embedder(1 - 1 / w)[:, None],
                self.t_min_tokens + self.cfg_t_start_embedder(t_min)[:, None],
                self.t_max_tokens + self.cfg_t_end_embedder(t_max)[:, None],
                self.time_tokens + self.h_embedder(h)[:, None],
                self.patch_embed(x),
            ],
            dim=1,
        )
        for block in self.blocks:
            seq = block(seq, self.rope_freqs)
        u = self.unpatchify(self.u_final_linear(self.u_final_norm(seq[:, self.prefix_tokens :])))
        return u.permute(0, 3, 1, 2)


def _load_reference(reference, model):
    r"""Copy the weights of an in-context `DiT` into the reference layout."""
    with torch.no_grad():
        for name, embedder in (
            ("h_embedder", model.dt_embedder),
            ("omega_embedder", model.guidance_embedder),
            ("cfg_t_start_embedder", model.interval_start_embedder),
            ("cfg_t_end_embedder", model.interval_end_embedder),
        ):
            getattr(reference, name).mlp.load_state_dict(embedder.mlp.state_dict())
        reference.y_embedder.weight.copy_(model.y_embedder.embedding.weight)
        class_tokens, omega_tokens, t_min_tokens, t_max_tokens, time_tokens = model.condition_tokens.split(
            [8, 4, 2, 2, 4]
        )
        for name, value in (
            ("class_tokens", class_tokens),
            ("omega_tokens", omega_tokens),
            ("t_min_tokens", t_min_tokens),
            ("t_max_tokens", t_max_tokens),
            ("time_tokens", time_tokens),
        ):
            getattr(reference, name).copy_(value)
        for block, ref in zip(model.blocks, reference.blocks):
            q, k, v = block.attn.qkv.weight.chunk(3)
            ref.attn.q_proj.weight.copy_(q)
            ref.attn.k_proj.weight.copy_(k)
            ref.attn.v_proj.weight.copy_(v)
            ref.attn.out_proj.weight.copy_(block.attn.out_proj.weight)
            ref.attn.q_norm.weight.copy_(block.attn.q_norm.weight)
            ref.attn.k_norm.weight.copy_(block.attn.k_norm.weight)
            ref.norm1.weight.copy_(block.norm1.weight)
            ref.norm2.weight.copy_(block.norm2.weight)
            w1, w3 = block.mlp.fc1.weight.chunk(2)
            ref.mlp.w1.weight.copy_(w1)
            ref.mlp.w3.weight.copy_(w3)
            ref.mlp.w2.weight.copy_(block.mlp.fc2.weight)
            ref.attn_scale.copy_(block.attn_scale)
            ref.mlp_scale.copy_(block.mlp_scale)
        reference.u_final_norm.weight.copy_(model.head.norm.weight)
        reference.u_final_linear.load_state_dict(model.head.proj.state_dict())


def _imf_model(**overrides):
    torch.manual_seed(0)
    cfg = dict(input_size=8, in_channels=4, patch_size=2, embed_dim=48, depth=2, num_heads=4, num_classes=10)
    cfg.update(IMF_OPTIONS)
    cfg.update(overrides)
    model = DiT(**cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return model.double().eval()


def test_in_context_dit_matches_imfdit():
    model = _imf_model()
    reference = _ImfDiT(model.patch_embed, 8, 2, 4, 48, 2, 4, 10).double()
    _load_reference(reference, model)

    g = torch.Generator().manual_seed(1)
    x = torch.randn(3, 4, 8, 8, generator=g, dtype=torch.float64)
    s = torch.rand(3, generator=g, dtype=torch.float64)
    h = torch.rand(3, generator=g, dtype=torch.float64) * (1 - s)
    w = 1 + 7 * torch.rand(3, generator=g, dtype=torch.float64)
    interval = torch.stack([0.5 * torch.rand(3, generator=g), 0.5 + 0.5 * torch.rand(3, generator=g)], -1).double()
    y = torch.tensor([0, 7, 10])
    with torch.no_grad():
        out = model(x, s + h, y, s=s, guidance_scale=w, guidance_interval=interval)
        expected = reference(x, h, w, interval[:, 0], interval[:, 1], y)
    assert_close(out, expected)


def test_rotary_embedding_1d_matches_reference():
    rope = RotaryEmbedding1d(head_dim=16, seq_len=20)
    x = torch.randn(2, 20, 3, 16, generator=torch.Generator().manual_seed(0))
    expected = _apply_rotary_pos_emb(x, _precompute_rope_freqs(16, 20))
    assert_close(rope(x.transpose(1, 2)).transpose(1, 2), expected)


def test_in_context_initialization_follows_imfdit():
    model = DiT(input_size=8, in_channels=4, patch_size=2, embed_dim=256, depth=2, num_heads=4, num_classes=10, **IMF_OPTIONS)
    block = model.blocks[0]
    assert_close(block.attn.qkv.weight.std(), torch.tensor(0.32 / 256**0.5), rtol=0.05, atol=0)
    assert_close(block.mlp.fc2.weight.std(), torch.tensor(0.32 / block.mlp.fc2.in_features**0.5), rtol=0.05, atol=0)
    assert_close(model.dt_embedder.mlp[0].weight.std(), torch.tensor(1 / 256**0.5), rtol=0.05, atol=0)
    assert_close(model.condition_tokens.std(), torch.tensor(1 / 256**0.5), rtol=0.1, atol=0)
    assert_close(model.y_embedder.embedding.weight.std(), torch.tensor(1 / 256**0.5), rtol=0.1, atol=0)
    assert block.attn.qkv.bias is None and block.mlp.fc1.bias is None
    assert torch.all(block.attn_scale == 0) and torch.all(block.mlp_scale == 0)
    assert torch.all(model.head.proj.weight == 0)


def test_two_time_time_invariant_model_sees_the_interval_only():
    model = _imf_model()
    assert model.t_embedder is None
    g = torch.Generator().manual_seed(2)
    x = torch.randn(2, 4, 8, 8, generator=g, dtype=torch.float64)
    s, h = torch.tensor([0.1, 0.3], dtype=torch.float64), torch.tensor([0.2, 0.5], dtype=torch.float64)
    kwargs = dict(guidance_scale=torch.full((2,), 2.0).double(), guidance_interval=torch.tensor([[0.0, 1.0]] * 2).double())
    with torch.no_grad():
        out = model(x, s + h, torch.tensor([1, 2]), s=s, **kwargs)
        shifted = model(x, s + h + 0.25, torch.tensor([1, 2]), s=s + 0.25, **kwargs)
    assert_close(out, shifted)


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(rope="2d"), "takes rope='1d' or None"),
        (dict(cond_dim=32), "requires cond_dim == embed_dim"),
        (dict(conditioning="cross_attention"), "conditioning must be one of"),
        (dict(rope="3d"), "rope must be one of"),
        (dict(t_embedder=nn.Linear(1, 48)), "takes no t_embedder"),
    ],
)
def test_invalid_in_context_configurations(overrides, match):
    with pytest.raises(ValueError, match=match):
        _imf_model(**overrides)


def test_in_context_forward_needs_every_conditioning_input():
    model = _imf_model()
    x = torch.randn(2, 4, 8, 8, dtype=torch.float64)
    s, t = torch.zeros(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64)
    kwargs = dict(guidance_scale=torch.ones(2).double(), guidance_interval=torch.tensor([[0.0, 1.0]] * 2).double())
    with pytest.raises(ValueError, match=r"needs \['y'\]"):
        model(x, t, s=s, **kwargs)
    with pytest.raises(ValueError, match="takes no cond"):
        model(x, t, torch.tensor([1, 2]), s=s, cond=torch.zeros(2, 48).double(), **kwargs)

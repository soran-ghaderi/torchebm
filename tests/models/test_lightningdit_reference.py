r"""DiT with the LightningDiT blocks against a port of the reference code.

The reference is hustvl/LightningDiT (f315f25b): ``Attention``,
``LightningDiTBlock`` and ``FinalLayer`` of ``models/lightningdit.py``,
``RMSNorm`` of ``models/rmsnorm.py``, ``SwiGLUFFN`` of
``models/swiglu_ffn.py`` and ``VisionRotaryEmbeddingFast`` of
``models/pos_embed.py``, with the einops patterns written as reshapes and
timm's ``Mlp`` as its two linears.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import assert_close

from torchebm.models import DiT
from torchebm.models.components import RotaryEmbedding2d


class _RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def _rotate_half(x):
    x1, x2 = x.reshape(*x.shape[:-1], -1, 2).unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class _VisionRotaryEmbeddingFast(nn.Module):
    def __init__(self, dim, pt_seq_len=16, theta=10000):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(pt_seq_len) / pt_seq_len * pt_seq_len
        freqs = torch.einsum("..., f -> ... f", t, freqs).repeat_interleave(2, dim=-1)
        n = freqs.shape[0]
        freqs = torch.cat(
            [freqs[:, None, :].expand(n, n, -1), freqs[None, :, :].expand(n, n, -1)], dim=-1
        )
        self.register_buffer("freqs_cos", freqs.cos().view(-1, freqs.shape[-1]))
        self.register_buffer("freqs_sin", freqs.sin().view(-1, freqs.shape[-1]))

    def forward(self, t):
        return t * self.freqs_cos + _rotate_half(t) * self.freqs_sin


class _Attention(nn.Module):
    def __init__(self, dim, num_heads, qk_norm, use_rmsnorm):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        norm_layer = _RMSNorm if use_rmsnorm else nn.LayerNorm
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, rope=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q = rope(q)
            k = rope(k)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class _Mlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class _SwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features)
        self.w3 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


def _modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _LightningDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio, use_qknorm, use_swiglu, use_rmsnorm):
        super().__init__()
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = _RMSNorm(hidden_size)
            self.norm2 = _RMSNorm(hidden_size)
        self.attn = _Attention(hidden_size, num_heads, use_qknorm, use_rmsnorm)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = _SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        else:
            self.mlp = _Mlp(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, x, c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            _modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class _FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = _RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(_modulate(self.norm_final(x), shift, scale))


def _block_state(block, swiglu):
    renames = [("attn.out_proj.", "attn.proj."), ("modulation.", "adaLN_modulation.")]
    if swiglu:
        renames += [("mlp.fc1.", "mlp.w12."), ("mlp.fc2.", "mlp.w3.")]
    else:
        renames += [("mlp.net.0.", "mlp.fc1."), ("mlp.net.3.", "mlp.fc2.")]
    state = {}
    for name, value in block.state_dict().items():
        for old, new in renames:
            name = name.replace(old, new)
        state[name] = value
    return state


def _head_state(head):
    renames = [("norm.", "norm_final."), ("modulation.", "adaLN_modulation."), ("proj.", "linear.")]
    state = {}
    for name, value in head.state_dict().items():
        for old, new in renames:
            name = name.replace(old, new)
        state[name] = value
    return state


def _reference_forward(model, blocks, final_layer, rope, x, t, y):
    r"""``LightningDiT.forward`` with the embedders of ``model``, which match the DiT reference."""
    tokens = model.patch_embed(x) + model.pos_embed
    c = model.t_embedder(t) + model.y_embedder(y, training=False)
    for block in blocks:
        tokens = block(tokens, c, rope)
    tokens = final_layer(tokens, c)
    p, channels = model.patch_size, model.out_channels
    h = int(tokens.shape[1] ** 0.5)
    tokens = tokens.reshape(tokens.shape[0], h, h, p, p, channels)
    return torch.einsum("nhwpqc->nchpwq", tokens).reshape(tokens.shape[0], channels, h * p, h * p)


OPTIONS = {
    "final_config": dict(qk_norm=False, swiglu=True, rope="2d", rms_norm=True),
    "all": dict(qk_norm=True, swiglu=True, rope="2d", rms_norm=True),
    "qk_layer_norm": dict(qk_norm=True, swiglu=False, rope=None, rms_norm=False),
    "rope": dict(qk_norm=False, swiglu=False, rope="2d", rms_norm=False),
    "none": dict(qk_norm=False, swiglu=False, rope=None, rms_norm=False),
}


@pytest.mark.usefixtures("highest_matmul_precision")
@pytest.mark.parametrize("options", OPTIONS.values(), ids=OPTIONS.keys())
def test_dit_matches_reference(options):
    embed_dim, num_heads, patch_size, in_channels = 64, 4, 2, 4
    torch.manual_seed(0)
    model = DiT(
        input_size=8,
        in_channels=in_channels,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=2,
        num_heads=num_heads,
        num_classes=10,
        **options,
    ).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * torch.randn_like(p))

    blocks = nn.ModuleList(
        _LightningDiTBlock(
            embed_dim, num_heads, 4.0, options["qk_norm"], options["swiglu"], options["rms_norm"]
        )
        for _ in model.blocks
    )
    for block, reference in zip(model.blocks, blocks):
        reference.load_state_dict(_block_state(block, options["swiglu"]))
    final_layer = _FinalLayer(embed_dim, patch_size, in_channels, options["rms_norm"])
    final_layer.load_state_dict(_head_state(model.head))
    grid = 8 // patch_size
    rope = _VisionRotaryEmbeddingFast(embed_dim // num_heads // 2, grid) if options["rope"] else None

    g = torch.Generator().manual_seed(1)
    x = torch.randn(3, in_channels, 8, 8, generator=g)
    t = torch.rand(3, generator=g)
    y = torch.tensor([0, 3, 10])
    with torch.no_grad():
        assert_close(model(x, t, y), _reference_forward(model, blocks, final_layer, rope, x, t, y))


@pytest.mark.parametrize("head_dim, grid", [(16, 4), (64, 16), (72, 16), (64, 32), (64, 12)])
def test_rope_matches_reference(head_dim, grid):
    rope = RotaryEmbedding2d(head_dim, grid)
    reference = _VisionRotaryEmbeddingFast(head_dim // 2, grid)
    x = torch.randn(2, 3, grid * grid, head_dim, generator=torch.Generator().manual_seed(0))
    assert_close(rope.cos, reference.freqs_cos)
    assert_close(rope(x), reference(x))


def test_rope_rectangular_grid_is_a_crop_of_the_square_grid():
    square = RotaryEmbedding2d(16, 4)
    for gh, gw in [(2, 4), (4, 3)]:
        rect = RotaryEmbedding2d(16, (gh, gw))
        assert torch.equal(rect.cos, square.cos.view(4, 4, 16)[:gh, :gw].reshape(-1, 16))
        assert torch.equal(rect.sin, square.sin.view(4, 4, 16)[:gh, :gw].reshape(-1, 16))


def test_rope_scores_depend_on_relative_position_only():
    rope = RotaryEmbedding2d(16, 5).double()
    g = torch.Generator().manual_seed(0)
    q, k = torch.randn(2, 16, generator=g, dtype=torch.float64)
    scores = rope(q.expand(25, 16)) @ rope(k.expand(25, 16)).T
    scores = scores.view(5, 5, 5, 5)
    assert_close(scores[1:, 1:, 1:, 1:], scores[:-1, :-1, :-1, :-1])


def test_rope_rejects_head_dim_not_multiple_of_four():
    with pytest.raises(ValueError, match="multiple of 4"):
        DiT(input_size=8, in_channels=3, patch_size=2, embed_dim=36, depth=1, num_heads=6, rope="2d")

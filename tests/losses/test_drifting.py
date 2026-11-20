import math

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from torchebm.losses import DriftingLoss, drifting_features
from torchebm.losses.functional import drifting_loss

GROUPS, N_GEN, N_POS, N_NEG, LOCATIONS, DIM = 3, 6, 5, 4, 7, 8
TEMPERATURES = (0.02, 0.05, 0.2)


def _cdist(x, y, eps=1e-8):
    xydot = torch.einsum("bnd,bmd->bnm", x, y)
    xnorms = torch.einsum("bnd,bnd->bn", x, x)
    ynorms = torch.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot
    return torch.sqrt(torch.clamp(sq_dist, min=eps))


def _reference_drift_loss(gen, fixed_pos, fixed_neg=None, weight_neg=None, R_list=TEMPERATURES):
    r"""Port of ``drift_loss`` of lambertae/drifting (88610664), ``drift_loss.py``."""
    B, C_g, S = gen.shape
    C_p = fixed_pos.shape[1]
    if fixed_neg is None:
        fixed_neg = torch.zeros_like(gen[:, :0, :])
    C_n = fixed_neg.shape[1]
    weight_gen = torch.ones_like(gen[:, :, 0])
    weight_pos = torch.ones_like(fixed_pos[:, :, 0])
    if weight_neg is None:
        weight_neg = torch.ones_like(fixed_neg[:, :, 0])
    old_gen = gen.detach()
    targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
    targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)
    with torch.no_grad():
        dist = _cdist(old_gen, targets)
        weighted_dist = dist * targets_w[:, None, :]
        scale = weighted_dist.mean() / targets_w.mean()
        scale_inputs = torch.clamp(scale / math.sqrt(S), min=1e-3)
        old_gen_scaled = old_gen / scale_inputs
        targets_scaled = targets / scale_inputs
        dist_normed = dist / torch.clamp(scale, min=1e-3)
        diag_mask = torch.eye(C_g, dtype=dist.dtype, device=dist.device)
        block_mask = F.pad(diag_mask, (0, C_n + C_p)).unsqueeze(0)
        dist_normed = dist_normed + block_mask * 100.0
        force_across_R = torch.zeros_like(old_gen_scaled)
        for R in R_list:
            logits = -dist_normed / R
            affinity = torch.softmax(logits, dim=-1)
            aff_transpose = torch.softmax(logits, dim=-2)
            affinity = torch.sqrt(torch.clamp(affinity * aff_transpose, min=1e-6))
            affinity = affinity * targets_w[:, None, :]
            split_idx = C_g + C_n
            aff_neg = affinity[:, :, :split_idx]
            aff_pos = affinity[:, :, split_idx:]
            sum_pos = torch.sum(aff_pos, dim=-1, keepdim=True)
            r_coeff_neg = -aff_neg * sum_pos
            sum_neg = torch.sum(aff_neg, dim=-1, keepdim=True)
            r_coeff_pos = aff_pos * sum_neg
            R_coeff = torch.cat([r_coeff_neg, r_coeff_pos], dim=2)
            total_force_R = torch.einsum("biy,byx->bix", R_coeff, targets_scaled)
            total_coeffs = R_coeff.sum(dim=-1)
            total_force_R = total_force_R - total_coeffs[..., None] * old_gen_scaled
            f_norm_val = (total_force_R**2).mean()
            force_scale = torch.sqrt(torch.clamp(f_norm_val, min=1e-8))
            force_across_R = force_across_R + total_force_R / force_scale
        goal_scaled = old_gen_scaled + force_across_R
    gen_scaled = gen / scale_inputs
    diff = gen_scaled - goal_scaled
    return torch.mean(diff**2, dim=(-1, -2))


def _reference_feature_loss(gen, pos, neg=None, cfg=None):
    r"""``feature_loss`` of ``train_step`` (``train.py``) on (groups, n, locations, D) features.

    ``rearrange(f, "b x f d -> (b f) x d")`` is ``f.transpose(1, 2).flatten(0, 1)`` and
    ``repeat(w, "b -> (b f) k")`` is ``w.repeat_interleave(f)`` expanded over ``k``.
    """
    gen_per_label = gen.shape[1]
    gen, pos = (f.transpose(1, 2).flatten(0, 1) for f in (gen, pos))
    weight_neg = None
    if neg is not None:
        n_uncond = neg.shape[1]
        neg = neg.transpose(1, 2).flatten(0, 1)
        uncond_w = (cfg - 1) * (gen_per_label - 1) / max(1, n_uncond)
        weight_neg = uncond_w.repeat_interleave(gen.shape[0] // uncond_w.shape[0])[:, None].expand(-1, n_uncond)
    return _reference_drift_loss(gen, pos, neg, weight_neg).mean()


def _features(locations=LOCATIONS, dtype=torch.float64, device="cpu"):
    torch.manual_seed(0)
    shape = (GROUPS, -1, locations, DIM)
    gen = (torch.randn(GROUPS * N_GEN * locations * DIM, dtype=dtype) * 3).view(shape)
    pos = (torch.randn(GROUPS * N_POS * locations * DIM, dtype=dtype) * 3 + 1).view(shape)
    neg = (torch.randn(GROUPS * N_NEG * locations * DIM, dtype=dtype) * 3 - 1).view(shape)
    cfg = 1 + 3 * torch.rand(GROUPS, dtype=dtype)
    return [t.to(device) for t in (gen, pos, neg, cfg)]


@pytest.mark.parametrize("guidance", [False, True], ids=["no_negatives", "negatives"])
def test_matches_reference(guidance):
    gen, pos, neg, cfg = _features()
    gen.requires_grad_(True)
    weight = (cfg - 1) * (N_GEN - 1) / N_NEG
    if guidance:
        expected = _reference_feature_loss(gen, pos, neg, cfg)
        loss = drifting_loss(gen, pos, neg, weight)
    else:
        expected = _reference_feature_loss(gen, pos)
        loss = drifting_loss(gen, pos)
    (expected_grad,) = torch.autograd.grad(expected, gen)
    (grad,) = torch.autograd.grad(loss, gen)
    assert_close(loss, expected, rtol=1e-12, atol=1e-12)
    assert_close(grad, expected_grad, rtol=1e-10, atol=1e-14)


def test_features_without_locations_match_reference():
    gen, pos, neg, cfg = _features(locations=1)
    weight = (cfg - 1) * (N_GEN - 1) / N_NEG
    expected = _reference_feature_loss(gen, pos, neg, cfg)
    assert_close(drifting_loss(gen[:, :, 0], pos[:, :, 0], neg[:, :, 0], weight), expected, rtol=1e-12, atol=1e-12)


def test_loss_sums_features_and_reduces_per_sample():
    gen, pos, neg, cfg = _features()
    weight = (cfg - 1) * (N_GEN - 1) / N_NEG
    features = [gen, gen[:, :, 0]], [pos, pos[:, :, 0]], [neg, neg[:, :, 0]]
    expected = _reference_feature_loss(gen, pos, neg, cfg) + _reference_feature_loss(
        gen[:, :, :1], pos[:, :, :1], neg[:, :, :1], cfg
    )
    assert_close(DriftingLoss()(*features, negative_weight=weight), expected, rtol=1e-12, atol=1e-12)
    per_sample = DriftingLoss(reduction="none")(*features, negative_weight=weight)
    assert per_sample.shape == (GROUPS, N_GEN)
    assert_close(per_sample.mean(), expected, rtol=1e-12, atol=1e-12)


def test_loss_value_is_the_squared_field():
    gen, pos, _, _ = _features()
    loss = DriftingLoss(temperatures=(0.05,))(gen, pos)
    assert_close(loss, torch.ones_like(loss), rtol=1e-12, atol=1e-12)


def _reference_process_feat(feat, sizes=(2, 4)):
    r"""``process_feat`` of ``get_activations`` (``models/mae_model.py``) on a (b, h, w, c) map.

    ``rearrange(feat, "b (h s1) (w s2) c -> b (h w) (s1 s2) c")`` is written with
    ``unflatten``, ``permute`` and ``reshape``.
    """

    def safe_std(x, axis):
        mean = x.mean(dim=axis, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=axis)
        return torch.sqrt(torch.clamp(var, min=0.0) + 1e-6)

    b, h, w, c = feat.shape
    out = [feat.flatten(1, 2), feat.mean(dim=(1, 2))[:, None], safe_std(feat, (1, 2))[:, None]]
    reshaped = [
        feat.unflatten(1, (h // s, s)).unflatten(3, (w // s, s)).permute(0, 1, 3, 2, 4, 5).reshape(b, -1, s * s, c)
        for s in sizes
        if h % s == 0 and w % s == 0
    ]
    return out + [r.mean(dim=2) for r in reshaped] + [safe_std(r, 2) for r in reshaped]


def test_drifting_features_match_reference():
    torch.manual_seed(0)
    maps = [torch.randn(2, 5, 8, 8, dtype=torch.float64), torch.randn(2, 3, 6, 6, dtype=torch.float64)]
    expected = [f for m in maps for f in _reference_process_feat(m.permute(0, 2, 3, 1))]
    features = drifting_features(maps)
    assert len(features) == len(expected) == 12
    for f, e in zip(features, expected):
        assert_close(f, e, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("compile", [False, True], ids=["eager", "compiled"])
def test_cuda_bf16_autocast_matches_float64(compile):
    torch._dynamo.reset()
    gen, pos, neg, cfg = _features(locations=16, dtype=torch.float32, device="cuda")
    gen, pos, neg = (t.bfloat16() for t in (gen, pos, neg))
    gen.requires_grad_(True)
    weight = (cfg - 1) * (N_GEN - 1) / N_NEG
    loss_fn = DriftingLoss()
    step = torch.compile(loss_fn, fullgraph=True) if compile else loss_fn
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = step(gen, pos, neg, negative_weight=weight)
    (grad,) = torch.autograd.grad(loss, gen)
    gen64 = gen.detach().double().requires_grad_(True)
    expected = _reference_feature_loss(gen64, pos.double(), neg.double(), cfg.double())
    (expected_grad,) = torch.autograd.grad(expected, gen64)
    assert loss.dtype == torch.float32
    assert_close(loss.double(), expected, rtol=1e-4, atol=0)
    assert_close(grad.double(), expected_grad, rtol=2e-2, atol=1e-2 * expected_grad.abs().max().item())

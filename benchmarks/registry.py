"""
Auto-discovery benchmark registry for TorchEBM.

Components are discovered from ``torchebm.*`` __init__.py exports by
checking against known base classes.  Per-component overrides (special
init kwargs, dimension caps, custom bench callables, etc.) are declared
in ``COMPONENT_OVERRIDES``.

Adding a new component to torchebm only requires:
  1. Export it in the subpackage __init__.py
  2. (Optional) Add an entry to COMPONENT_OVERRIDES if it has non-default
     construction / benchmark requirements.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Spec dataclass — one per benchmarked component
# ---------------------------------------------------------------------------


@dataclass
class BenchSpec:
    """Describes how to instantiate and benchmark a single component."""

    name: str
    module: str  # category: integrators, interpolants, losses, samplers, models
    cls: Type
    # -- construction helpers --
    model_type: str = "none"  # none | ebm | velocity | ebm_double_well
    init_kwargs: Dict[str, Any] = field(default_factory=dict)
    needs_diffusion: bool = False  # integrators: requires diffusion= kwarg
    needs_momentum: bool = False  # integrators: state includes "p"
    needs_sampler: bool = False  # losses: needs an MCMC sampler
    needs_grad: bool = False  # losses: x.requires_grad_(True)
    returns_tuple: bool = False  # loss forward returns (loss, extras)
    max_dim: Optional[int] = None  # cap dim for O(dim²) methods
    max_batch: Optional[int] = None  # cap batch_size
    # -- custom benchmark function --
    bench_fn: Optional[str] = None  # method name to call instead of default
    bench_kwargs: Dict[str, Any] = field(default_factory=dict)
    # -- models with completely custom configs --
    model_configs: Optional[Dict[str, Dict]] = None
    bench_backward: bool = False  # models: include backward pass


# ---------------------------------------------------------------------------
# Per-component overrides (only needed when defaults are insufficient)
# ---------------------------------------------------------------------------

COMPONENT_OVERRIDES: Dict[str, Dict[str, Any]] = {
    # ── Integrators ──
    "LeapfrogIntegrator": {"needs_momentum": True},
    "EulerMaruyamaIntegrator": {"needs_diffusion": True},
    "HeunIntegrator": {"needs_diffusion": True},

    # ── Losses ──
    "ScoreMatching": {
        "model_type": "ebm",
        "needs_grad": True,
        "variants": {
            "score_matching_exact": {"init_kwargs": {"hessian_method": "exact"}, "max_dim": 16, "max_batch": 128},
            "score_matching_approx": {"init_kwargs": {"hessian_method": "approx"}},
        },
    },
    "SlicedScoreMatching": {
        "model_type": "ebm",
        "needs_grad": True,
        "init_kwargs": {"n_projections": 5},
    },
    "DenoisingScoreMatching": {
        "model_type": "ebm",
        "needs_grad": True,
    },
    "ContrastiveDivergence": {
        "model_type": "ebm",
        "needs_sampler": True,
        "returns_tuple": True,
        "init_kwargs": {"k_steps": 10, "persistent": False},
    },
    "EquilibriumMatchingLoss": {
        "model_type": "velocity",
        "init_kwargs": {"prediction": "velocity", "energy_type": "none", "interpolant": "linear"},
    },
    # Stubs — skip
    "PersistentContrastiveDivergence": {"skip": True},
    "ParallelTemperingCD": {"skip": True},

    # ── Samplers ──
    "LangevinDynamics": {
        "model_type": "ebm_double_well",
        "init_kwargs": {"noise_scale": 1.0},
    },
    "HamiltonianMonteCarlo": {
        "model_type": "ebm_double_well",
        "init_kwargs": {"n_leapfrog_steps": 10},
    },
    "GradientDescentSampler": {"model_type": "ebm_double_well"},
    "NesterovSampler": {
        "model_type": "ebm_double_well",
        "init_kwargs": {"momentum": 0.9},
    },
    "FlowSampler": {
        "model_type": "velocity",
        "init_kwargs": {"interpolant": "linear", "prediction": "velocity"},
        "bench_fn": "sample_ode",
        "bench_kwargs": {"method": "euler"},
    },

    # ── Interpolants (defaults are fine — no overrides needed) ──
    "VariancePreservingInterpolant": {
        "init_kwargs": {"sigma_min": 0.1, "sigma_max": 20.0},
    },

    # ── Models (completely custom configs) ──
    "ConditionalTransformer2D": {
        "model_configs": {
            "small": {"input_size": 16, "patch_size": 4, "embed_dim": 128, "depth": 4, "num_heads": 4, "bs": 16},
            "medium": {"input_size": 32, "patch_size": 4, "embed_dim": 256, "depth": 8, "num_heads": 8, "bs": 8},
            "large": {"input_size": 32, "patch_size": 4, "embed_dim": 512, "depth": 12, "num_heads": 8, "bs": 4},
        },
        "variants": {
            "transformer_fwd": {},
            "transformer_fwd_bwd": {"bench_backward": True},
        },
    },
}


# ---------------------------------------------------------------------------
# Category → (package path, base class import path)
# ---------------------------------------------------------------------------

_CATEGORY_MAP = {
    "integrators": ("torchebm.integrators", "torchebm.core.base_integrator.BaseIntegrator"),
    "interpolants": ("torchebm.interpolants", "torchebm.core.base_interpolant.BaseInterpolant"),
    "losses": ("torchebm.losses", "torchebm.core.base_loss.BaseLoss"),
    "samplers": ("torchebm.samplers", "torchebm.core.base_sampler.BaseSampler"),
    "models": ("torchebm.models", None),  # no single base; use __all__ directly
}

# Names to exclude from auto-discovery (utilities, enums, functions, etc.)
_SKIP_NAMES = {
    "PredictionType",
    "mean_flat",
    "get_interpolant",
    "compute_eqm_ct",
    "dispersive_loss",
    "_integrate_time_grid",
    "expand_t_like_x",
    "LabelClassifierFreeGuidance",
    "AdaLNZeroBlock",
    "AdaLNZeroPatchHead",
    "ConvPatchEmbed2d",
    "FeedForward",
    "LabelEmbedder",
    "MLPTimestepEmbedder",
    "MultiheadSelfAttention",
    "build_2d_sincos_pos_embed",
    "patchify2d",
    "unpatchify2d",
}


def _resolve_class(dotted: str):
    mod_path, cls_name = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(mod_path), cls_name)


def _is_benchmarkable(obj, base_cls) -> bool:
    return isinstance(obj, type) and issubclass(obj, base_cls) and obj is not base_cls


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

def discover_components() -> List[BenchSpec]:
    """Scan torchebm subpackages and return BenchSpec for every exported component."""
    specs: List[BenchSpec] = []

    for category, (pkg_path, base_cls_path) in _CATEGORY_MAP.items():
        pkg = importlib.import_module(pkg_path)
        base_cls = _resolve_class(base_cls_path) if base_cls_path else None
        export_names = getattr(pkg, "__all__", [])

        for name in export_names:
            if name in _SKIP_NAMES:
                continue
            obj = getattr(pkg, name, None)
            if obj is None or not isinstance(obj, type):
                continue
            if base_cls and not _is_benchmarkable(obj, base_cls):
                continue

            overrides = COMPONENT_OVERRIDES.get(name, {})
            if overrides.get("skip"):
                continue

            # Handle "variants" (e.g. ScoreMatching with exact/approx, Transformer fwd/fwd_bwd)
            variants = overrides.pop("variants", None) if "variants" in overrides else None
            if variants:
                for variant_name, variant_overrides in variants.items():
                    merged = {**overrides, **variant_overrides}
                    merged_init = {**overrides.get("init_kwargs", {}), **variant_overrides.get("init_kwargs", {})}
                    specs.append(_build_spec(variant_name, category, obj, {**merged, "init_kwargs": merged_init}))
            else:
                specs.append(_build_spec(name, category, obj, overrides))

    return specs


def _build_spec(name: str, module: str, cls: Type, overrides: Dict) -> BenchSpec:
    return BenchSpec(
        name=name,
        module=module,
        cls=cls,
        model_type=overrides.get("model_type", "none"),
        init_kwargs=overrides.get("init_kwargs", {}),
        needs_diffusion=overrides.get("needs_diffusion", False),
        needs_momentum=overrides.get("needs_momentum", False),
        needs_sampler=overrides.get("needs_sampler", False),
        needs_grad=overrides.get("needs_grad", False),
        returns_tuple=overrides.get("returns_tuple", False),
        max_dim=overrides.get("max_dim"),
        max_batch=overrides.get("max_batch"),
        bench_fn=overrides.get("bench_fn"),
        bench_kwargs=overrides.get("bench_kwargs", {}),
        model_configs=overrides.get("model_configs"),
        bench_backward=overrides.get("bench_backward", False),
    )


# ---------------------------------------------------------------------------
# Template factories — build (callable, extra_info) for each category
# ---------------------------------------------------------------------------

_STEP_SIZE = 1e-3
_DIFFUSION_COEFF = 0.1
_BARRIER_HEIGHT = 2.0
_CD_STEP_SIZE = 1e-2
_MLP_HIDDEN = 128


class _MLPEnergy(nn.Module):
    def __init__(self, dim, hidden=_MLP_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class _MLPVelocity(nn.Module):
    def __init__(self, dim, hidden=_MLP_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        self._dim = dim

    def forward(self, x, t=None, **kwargs):
        if t is None:
            t = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        xt = torch.cat([x.view(x.shape[0], -1), t], dim=-1)
        return self.net(xt).view_as(x)


def _wrap_ebm(net, device, dtype):
    from torchebm.core.base_model import BaseModel

    class _W(BaseModel):
        def __init__(self, n, d, dt):
            super().__init__(dtype=dt)
            self.net = n
            self._device = d
            self.to(d)

        def forward(self, x):
            return self.net(x)

    return _W(net, device, dtype)


def _make_drift(net, device, dtype):
    def drift(x, t):
        x_req = x.detach().requires_grad_(True)
        e = net(x_req)
        return -torch.autograd.grad(e.sum(), x_req)[0]
    return drift


def _make_data(bs, dim, device, dtype):
    return torch.randn(bs, dim, device=device, dtype=dtype)


# ── Category templates ──


def build_integrator_bench(
    spec: BenchSpec, dim: int, bs: int, n_steps: int, device: torch.device, dtype: torch.dtype,
) -> Tuple[Callable, Dict]:
    net = _MLPEnergy(dim).to(device, dtype)
    integrator = spec.cls(device=device, dtype=dtype)
    drift = _make_drift(net, device, dtype)
    x0 = _make_data(bs, dim, device, dtype)
    step_size = torch.tensor(_STEP_SIZE, device=device, dtype=dtype)

    if spec.needs_momentum:
        p0 = torch.randn_like(x0)

        def fn():
            integrator.integrate(
                state={"x": x0.clone(), "p": p0.clone()},
                step_size=step_size, n_steps=n_steps, drift=drift,
            )
    elif spec.needs_diffusion:
        diffusion_val = torch.tensor(_DIFFUSION_COEFF, device=device, dtype=dtype)

        def fn():
            integrator.integrate(
                state={"x": x0.clone()}, step_size=step_size, n_steps=n_steps,
                drift=drift, diffusion=lambda x, t: diffusion_val,
            )
    else:
        def fn():
            integrator.integrate(
                state={"x": x0.clone()}, step_size=step_size, n_steps=n_steps,
                drift=drift, adaptive=False,
            )

    info = {"module": spec.module, "batch_size": bs, "dim": dim, "n_steps": n_steps}
    return fn, info


def build_interpolant_bench(
    spec: BenchSpec, dim: int, bs: int, n_steps: int, device: torch.device, dtype: torch.dtype,
) -> Tuple[Callable, Dict]:
    interp = spec.cls(**spec.init_kwargs)
    x0 = _make_data(bs, dim, device, dtype)
    x1 = _make_data(bs, dim, device, dtype)
    t = torch.rand(bs, device=device, dtype=dtype)

    def fn():
        interp.interpolate(x0, x1, t)

    info = {"module": spec.module, "batch_size": bs, "dim": dim}
    return fn, info


def build_loss_bench(
    spec: BenchSpec, dim: int, bs: int, n_steps: int, device: torch.device, dtype: torch.dtype,
) -> Tuple[Callable, Dict]:
    eff_dim = min(dim, spec.max_dim) if spec.max_dim else dim
    eff_bs = min(bs, spec.max_batch) if spec.max_batch else bs

    # Build model
    if spec.model_type == "ebm":
        net = _MLPEnergy(eff_dim).to(device, dtype)
        model = _wrap_ebm(net, device, dtype)
    elif spec.model_type == "velocity":
        net = _MLPVelocity(eff_dim).to(device, dtype)
        model = net
    else:
        raise ValueError(f"Unknown model_type {spec.model_type!r} for loss {spec.name}")

    # Build loss
    init_kw: Dict[str, Any] = {"device": device, "dtype": dtype, **spec.init_kwargs}

    if spec.needs_sampler:
        from torchebm.samplers.langevin_dynamics import LangevinDynamics
        sampler = LangevinDynamics(model=model, step_size=_CD_STEP_SIZE, device=device, dtype=dtype)
        init_kw["sampler"] = sampler

    if spec.model_type in ("ebm", "velocity"):
        init_kw["model"] = model

    loss_fn = spec.cls(**init_kw)
    x = _make_data(eff_bs, eff_dim, device, dtype)
    if spec.needs_grad:
        x = x.requires_grad_(True)

    if spec.returns_tuple:
        def fn():
            loss, _ = loss_fn(x)
            loss.backward()
            net.zero_grad()
    else:
        def fn():
            loss = loss_fn(x)
            loss.backward()
            net.zero_grad()

    info = {"module": spec.module, "batch_size": eff_bs, "dim": eff_dim}
    return fn, info


def build_sampler_bench(
    spec: BenchSpec, dim: int, bs: int, n_steps: int, device: torch.device, dtype: torch.dtype,
) -> Tuple[Callable, Dict]:
    init_kw: Dict[str, Any] = {"device": device, "dtype": dtype, **spec.init_kwargs}

    if spec.model_type == "ebm_double_well":
        from torchebm.core.base_model import DoubleWellModel
        model = DoubleWellModel(barrier_height=_BARRIER_HEIGHT, dtype=dtype)
        model._device = device
        init_kw["model"] = model
        init_kw.setdefault("step_size", _STEP_SIZE)
    elif spec.model_type == "velocity":
        vel_model = _MLPVelocity(dim).to(device, dtype)
        init_kw["model"] = vel_model
    else:
        raise ValueError(f"Unknown model_type {spec.model_type!r} for sampler {spec.name}")

    sampler = spec.cls(**init_kw)
    x0 = _make_data(bs, dim, device, dtype)

    if spec.bench_fn:
        method = getattr(sampler, spec.bench_fn)
        call_kwargs = {"num_steps": n_steps, **spec.bench_kwargs}

        def fn():
            method(x0, **call_kwargs)
    else:
        def fn():
            sampler.sample(x=x0, n_steps=n_steps, n_samples=bs, dim=dim)

    info = {"module": spec.module, "batch_size": bs, "dim": dim, "n_steps": n_steps}
    return fn, info


def build_model_bench(
    spec: BenchSpec, dim: int, bs: int, n_steps: int, device: torch.device, dtype: torch.dtype,
    scale: str = "small",
) -> Tuple[Callable, Dict]:
    if spec.model_configs is None:
        raise ValueError(f"Model {spec.name} requires model_configs in overrides")

    cfg = spec.model_configs[scale]
    bs_model = cfg.pop("bs", bs)
    model = spec.cls(in_channels=3, out_channels=3, **cfg).to(device, dtype)

    input_size = cfg["input_size"]
    embed_dim = cfg["embed_dim"]
    x = torch.randn(bs_model, 3, input_size, input_size, device=device, dtype=dtype)
    cond = torch.randn(bs_model, embed_dim, device=device, dtype=dtype)

    if spec.bench_backward:
        target = torch.randn_like(x)

        def fn():
            out = model(x, cond)
            loss = (out - target).pow(2).mean()
            loss.backward()
            model.zero_grad()
    else:
        def fn():
            model(x, cond)

    info = {"module": spec.module, "batch_size": bs_model}
    return fn, info


# ── Dispatch table ──

TEMPLATE_BUILDERS = {
    "integrators": build_integrator_bench,
    "interpolants": build_interpolant_bench,
    "losses": build_loss_bench,
    "samplers": build_sampler_bench,
    "models": build_model_bench,
}

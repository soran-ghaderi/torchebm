import warnings

import pytest
import torch
from torch import nn

from torchebm.core.base_module import TorchEBMModule, _normalize

from tests.conftest import requires_cuda


class _Empty(TorchEBMModule):
    pass


class _Linear(TorchEBMModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fc = nn.Linear(4, 4)


def test_torchebm_module_default_device_is_cpu():
    m = _Empty()
    assert m.device.type == "cpu"
    assert m.dtype == torch.get_default_dtype()


def test_torchebm_module_with_dtype_override():
    m = _Empty(dtype=torch.float64)
    assert m.dtype == torch.float64


def test_torchebm_module_device_from_parameters():
    m = _Linear()
    assert m.device.type == "cpu"
    assert m.dtype == torch.float32


def test_torchebm_module_double_cast_updates_cache():
    m = _Linear()
    m.double()
    assert m.dtype == torch.float64


def test_normalize_cuda_index_zero_collapses():
    assert _normalize(torch.device("cuda:0")) == torch.device("cuda")


def test_normalize_cpu_passthrough():
    assert _normalize(torch.device("cpu")) == torch.device("cpu")


def test_setup_mixed_precision_disabled_is_noop():
    m = _Empty()
    m.setup_mixed_precision(False)
    assert not m.use_mixed_precision
    assert not m.autocast_available


def test_setup_mixed_precision_cpu_warns_and_disables():
    m = _Empty()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.setup_mixed_precision(True)
    assert any(issubclass(x.category, UserWarning) for x in w)
    assert not m.use_mixed_precision


def test_autocast_context_is_nullcontext_on_cpu():
    m = _Empty()
    m.setup_mixed_precision(True)
    ctx = m.autocast_context()
    with ctx:
        x = torch.randn(3)
    assert x.dtype == torch.float32


def test_loss_call_fires_forward_hook():
    """A registered forward_hook on a loss must fire when the loss is called.

    Regression for the architectural bug where `BaseLoss.__call__` /
    `BaseScoreMatching.__call__` returned `self.forward(x)` directly,
    bypassing `nn.Module._call_impl` and thus all hooks (forward,
    forward-pre, full-backward), DDP gradient sync, and profiler
    instrumentation.
    """
    from torchebm.core import GaussianModel
    from torchebm.losses import SlicedScoreMatching

    energy = GaussianModel(mean=torch.zeros(2), cov=torch.eye(2))
    loss_fn = SlicedScoreMatching(model=energy, n_projections=2)

    fired = {"pre": 0, "post": 0}

    def pre_hook(module, inputs):
        fired["pre"] += 1

    def post_hook(module, inputs, output):
        fired["post"] += 1

    h_pre = loss_fn.register_forward_pre_hook(pre_hook)
    h_post = loss_fn.register_forward_hook(post_hook)
    try:
        x = torch.randn(8, 2)
        loss_fn(x)
    finally:
        h_pre.remove()
        h_post.remove()

    assert fired["pre"] == 1, "forward_pre_hook did not fire"
    assert fired["post"] == 1, "forward_hook did not fire"


def test_model_call_fires_forward_hook():
    """A registered forward_hook on a BaseModel must fire when the model is called."""
    from torchebm.core import GaussianModel

    energy = GaussianModel(mean=torch.zeros(2), cov=torch.eye(2))

    fired = {"pre": 0, "post": 0}

    h_pre = energy.register_forward_pre_hook(lambda m, i: fired.__setitem__("pre", fired["pre"] + 1))
    h_post = energy.register_forward_hook(lambda m, i, o: fired.__setitem__("post", fired["post"] + 1))
    try:
        energy(torch.randn(4, 2))
    finally:
        h_pre.remove()
        h_post.remove()

    assert fired["pre"] == 1
    assert fired["post"] == 1
